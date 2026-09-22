"""
chunked_attention.py — Memory-safe attention for CogVideoX on small GPUs.

Problem
-------
CogVideoX (both 2b and 5b) uses *joint* / double-stream attention: the 226 text
tokens and the video tokens are concatenated into a single sequence and one full
attention over that joint sequence is computed by every one of the 30 blocks.

For a 480p × 49-frame clip the joint sequence is

    N = 226 text + 20,670 video = 20,896 tokens

so a single block's score matrix is

    heads × N² × 4 B = 30 × 20,896² × 4 B ≈ 52 GB   (fp32 softmax)

which cannot fit on a T4 (or on 2× T4) no matter how the weights are offloaded.
`enable_attention_slicing()` does not help — it only applies to the old UNet 2D
attention classes.

Solution
--------
Keep diffusers' own attention processor (it correctly does the Q/K/V
projections, 3D RoPE on the video tokens, the text/video concatenation, the
split at the end and the `(hidden_states, encoder_hidden_states)` tuple the
transformer block expects) and replace *only* the
`F.scaled_dot_product_attention` kernel it calls with a Q-chunked equivalent:

    for each block of queries (chunk_size rows):
        scores = Q_chunk @ Kᵀ           # (B, H, chunk, N)
        probs  = softmax(scores)        # fp32, released every iteration
        out[chunk] = probs @ V

Peak score memory becomes `chunk × heads × N × 4 B` instead of `heads × N² × 4 B`
(the chunk is additionally clamped so a single score block never exceeds
`max_score_mb`), taking the example above from ~52 GB down to well under 512 MB.

Delegating instead of re-implementing the processor is what keeps this correct
across diffusers versions: newer versions return a *tuple* of tensors
(text-stream output, video-stream output) while older ones return a single
tensor, and a hand-written processor that only returns one of those shapes
crashes the block with

    RuntimeError: The size of tensor a (226) must match the size of tensor b
    (20670) at non-singleton dimension 1

Usage
-----
    from chunked_attention import patch_cogvideox_attention, unpatch_cogvideox_attention

    patch_cogvideox_attention(pipe, chunk_size=512)
    # ... run inference ...
    unpatch_cogvideox_attention(pipe)   # optional cleanup
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F


DEFAULT_CHUNK_SIZE = 512
DEFAULT_MAX_SCORE_MB = 512


# ── Chunk sizing ──────────────────────────────────────────────────────────────

def _resolve_chunk_size(
    chunk_size: int,
    heads: int,
    seq_len: int,
    max_score_mb: int = DEFAULT_MAX_SCORE_MB,
) -> int:
    """Clamp the Q-chunk so a single score block fits in `max_score_mb` MiB.

    One score block is `chunk × heads × N` fp32 elements (4 bytes each), so the
    budget is what keeps this patch OOM-safe at any resolution / frame count
    instead of relying on a hard-coded chunk size.
    """
    if heads <= 0 or seq_len <= 0 or max_score_mb <= 0:
        return max(1, chunk_size)
    max_elems = (max_score_mb * 1024 * 1024) // 4
    budget_chunk = max(1, max_elems // (heads * seq_len))
    return max(1, min(chunk_size, budget_chunk))


# ── Chunked attention kernel ──────────────────────────────────────────────────

def _slice_attn_mask(
    attn_mask: Optional[torch.Tensor],
    start: int,
    end: int,
    query_len: int,
) -> Optional[torch.Tensor]:
    """Slice an attention mask along its *query* dimension, if it has one.

    Handles the shapes SDPA accepts: (L, S), (B, L, S), (B, H, L, S) as well as
    already-broadcast masks whose query dim is 1 (those are returned unchanged).
    """
    if attn_mask is None:
        return None
    if attn_mask.dim() == 2:
        return attn_mask[start:end, :] if attn_mask.shape[0] == query_len else attn_mask
    if attn_mask.dim() == 3:
        return attn_mask[:, start:end, :] if attn_mask.shape[1] == query_len else attn_mask
    if attn_mask.dim() == 4:
        return attn_mask[:, :, start:end, :] if attn_mask.shape[2] == query_len else attn_mask
    return attn_mask


def _chunked_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_score_mb: int = DEFAULT_MAX_SCORE_MB,
) -> torch.Tensor:
    """`F.scaled_dot_product_attention` with Q-chunking (never builds the full N² matrix).

    Memory peak: O(B × H × chunk × N) instead of O(B × H × N²).
    For N=20,896 (480p, 49 frames) and chunk=512 over 30 heads:
        full  : 30 × 20,896² × 4 B ≈ 52 GB  ✗
        chunk :  512 × 30 × 20,896 × 4 B ≈ 1.3 GB (auto-clamped to < 512 MB) ✓
    """
    if query.dtype == torch.float32 and key.dtype == torch.float32:
        # Already fp32 — nothing to upcast.
        key_t = key.transpose(-2, -1)
        value_f = value
    else:
        # Keys/values stay in fp32 for the whole loop: SDPA upcasts internally
        # too, and on Turing (T4) fp32 matmul is no slower than emulated bf16.
        key_t = key.transpose(-2, -1).to(torch.float32)
        value_f = value.to(torch.float32)

    batch_size, heads, query_len, head_dim = query.shape
    out_dtype = query.dtype
    scale_factor = scale if scale is not None else (head_dim ** -0.5)
    chunk = _resolve_chunk_size(chunk_size, heads, query_len, max_score_mb)

    output = torch.empty_like(query)

    for start in range(0, query_len, chunk):
        end = min(start + chunk, query_len)
        q_blk = query[:, :, start:end, :]
        q_f = q_blk if q_blk.dtype == torch.float32 else q_blk.to(torch.float32)

        # (B, H, chunk, N) — this is the only large tensor, freed each iteration.
        scores = torch.matmul(q_f, key_t) * scale_factor

        if is_causal:
            q_idx = torch.arange(start, end, device=query.device).unsqueeze(1)
            k_idx = torch.arange(key.shape[-2], device=query.device).unsqueeze(0)
            scores = scores.masked_fill(q_idx < k_idx, float("-inf"))

        mask_blk = _slice_attn_mask(attn_mask, start, end, query_len)
        if mask_blk is not None:
            if mask_blk.dtype == torch.bool:
                scores = scores.masked_fill(~mask_blk, float("-inf"))
            else:
                scores = scores + mask_blk.to(scores.dtype)

        probs = torch.softmax(scores, dim=-1)
        del scores

        if dropout_p > 0.0 and torch.is_grad_enabled():
            probs = F.dropout(probs, p=dropout_p)

        output[:, :, start:end, :] = torch.matmul(probs, value_f).to(out_dtype)
        del probs

    return output


# ── Kernel swap ───────────────────────────────────────────────────────────────

@contextmanager
def _chunked_sdpa_kernel(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_score_mb: int = DEFAULT_MAX_SCORE_MB,
) -> Iterator[None]:
    """Temporarily route `F.scaled_dot_product_attention` through the chunked kernel.

    diffusers' attention processors call `F.scaled_dot_product_attention` (the
    module attribute, looked up at call time), so swapping it for the duration of
    one processor call leaves every other part of the pipeline untouched.
    """
    real_sdpa = F.scaled_dot_product_attention

    def sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # GQA/MQA needs the kernel's own head expansion — not used by CogVideoX.
        if not kwargs.get("enable_gqa", False) and query.dim() == 4 and query.shape[2] > chunk_size:
            return _chunked_scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                chunk_size=chunk_size,
                max_score_mb=max_score_mb,
            )
        return real_sdpa(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs,
        )

    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = real_sdpa


# ── Processor patch ───────────────────────────────────────────────────────────

_ORIGINAL_CALLS: dict[type, Any] = {}    # class → original __call__ (for unpatch)


def _make_chunked_forward(
    original_call: Any,
    chunk_size: int,
    max_score_mb: int,
):
    """Wrap a processor's original `__call__` so its SDPA call is chunked.

    The signature intentionally spells out `image_rotary_emb`: diffusers filters
    `cross_attention_kwargs` by the *names* of the processor's parameters
    (`Attention.forward` -> `inspect.signature(self.processor.__call__)`), so a
    `**kwargs`-only wrapper silently drops the 3D rotary embedding — the model
    then runs without positional information (and logs "cross_attention_kwargs
    ['image_rotary_emb'] are not expected by CogVideoXAttnProcessor2_0").
    """
    accepted = set(inspect.signature(original_call).parameters)

    def chunked_call(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Any = None,
        **kwargs: Any,
    ) -> Any:
        forwarded = {k: v for k, v in kwargs.items() if k in accepted}
        if "image_rotary_emb" in accepted:
            forwarded["image_rotary_emb"] = image_rotary_emb

        with _chunked_sdpa_kernel(chunk_size, max_score_mb):
            return original_call(
                self,
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **forwarded,
            )

    chunked_call.__name__ = getattr(original_call, "__name__", "chunked_call")
    chunked_call.__doc__ = (
        "Chunked-attention wrapper around the original CogVideoX attention processor."
    )
    return chunked_call


def _iter_attention_modules(pipe: Any) -> list[tuple[str, Any]]:
    """Walk the pipeline's transformer and return every `Attention` module."""
    from diffusers.models.attention_processor import Attention

    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return []
    return [(name, mod) for name, mod in transformer.named_modules() if isinstance(mod, Attention)]


def patch_cogvideox_attention(
    pipe: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_score_mb: int = DEFAULT_MAX_SCORE_MB,
) -> None:
    """Route CogVideoX attention through the chunked SDPA kernel.

    Parameters
    ----------
    pipe          : loaded CogVideoXPipeline
    chunk_size    : max number of Q tokens per iteration (upper bound; the
                    kernel clamps it further so one score block stays under
                    `max_score_mb`). 256 for 8 GB VRAM, 512 on a T4, 1024-2048
                    on 24 GB+.
    max_score_mb  : per-chunk score-block memory ceiling in MiB.
    """
    modules = _iter_attention_modules(pipe)
    if not modules:
        print("[chunked_attn] No Attention modules found — patch not applied")
        return

    patched, classes = 0, set()
    for _name, module in modules:
        proc = getattr(module, "processor", None)
        if proc is None:
            continue

        cls_name = type(proc).__name__
        if "CogVideoX" not in cls_name and "Attn" not in cls_name:
            continue

        cls = type(proc)
        if cls not in _ORIGINAL_CALLS:
            original = cls.__call__
            _ORIGINAL_CALLS[cls] = original
            cls.__call__ = _make_chunked_forward(original, chunk_size, max_score_mb)
            classes.add(cls_name)

        patched += 1

    if patched == 0:
        print("[chunked_attn] No attention processors matched — patch not applied")
        return

    print(
        f"[chunked_attn] {', '.join(sorted(classes))} patched — "
        f"Q-chunks ≤ {chunk_size} tokens, score block ≤ {max_score_mb} MB "
        f"({patched} attention modules)"
    )


def unpatch_cogvideox_attention(pipe: Any) -> None:
    """Restore the original attention processors."""
    restored = 0
    for _name, module in _iter_attention_modules(pipe):
        cls = type(getattr(module, "processor", None))
        if cls in _ORIGINAL_CALLS:
            cls.__call__ = _ORIGINAL_CALLS.pop(cls)
            restored += 1

    if restored:
        print("[chunked_attn] Attention processors restored to original.")
