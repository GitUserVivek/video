"""
chunked_attention.py — Patch CogVideoX attention to use chunked computation.

Problem
-------
CogVideoX uses full 3D space-time attention. For a 480p × 49-frame video:
  sequence length N = (49 frames) × (60×106 spatial patches) = 311,640 tokens
  QKᵀ matrix = N² × 2 bytes = 97 GB   ← impossible on any single T4

PyTorch's F.scaled_dot_product_attention allocates the full matrix at once.
diffusers' enable_attention_slicing() does NOT help CogVideoX — it only
works on older UNet 2D attention.

Solution
--------
Replace the attention forward with a chunked implementation that processes
Q in blocks of `chunk_size` tokens at a time. Peak memory becomes:
  chunk_size × N × 2 bytes   (e.g. chunk_size=512 → 512 × 311640 × 2 = 300 MB)

This is applied by monkey-patching the CogVideoXAttnProcessor2_0 class
(or whichever attention processor CogVideoX uses) before inference.

Usage
-----
    from chunked_attention import patch_cogvideox_attention, unpatch_cogvideox_attention

    patch_cogvideox_attention(pipe, chunk_size=512)
    # ... run inference ...
    unpatch_cogvideox_attention(pipe)   # optional cleanup
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F


# ── Chunked attention kernel ──────────────────────────────────────────────────

def _chunked_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Memory-efficient attention via Q-chunking.

    Instead of computing the full (B, H, N, N) attention matrix, we iterate
    over Q in blocks of `chunk_size` and accumulate the output.

    Memory peak: O(B × H × chunk_size × N) instead of O(B × H × N²)

    For N=311640, chunk_size=512:
      Full matrix: 311640² × 2B ≈ 97 GB
      Chunked:     512 × 311640 × 2B ≈ 300 MB  ✓
    """
    B, H, N, D = query.shape
    scale_factor = scale if scale is not None else (D ** -0.5)

    output = torch.zeros_like(query)

    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        q_blk = query[:, :, start:end, :]          # (B, H, chunk, D)

        # Attention scores for this Q chunk against all K
        scores = torch.matmul(q_blk, key.transpose(-2, -1)) * scale_factor
        # shape: (B, H, chunk, N)

        if is_causal:
            # Build causal mask for this chunk
            q_idx  = torch.arange(start, end, device=query.device).unsqueeze(1)
            k_idx  = torch.arange(N,     device=query.device).unsqueeze(0)
            mask   = q_idx < k_idx                 # True where we should mask
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask[start:end, :]
            elif attn_mask.dim() == 4:
                scores = scores + attn_mask[:, :, start:end, :]

        attn_weights = F.softmax(scores, dim=-1)

        if dropout_p > 0.0 and torch.is_grad_enabled():
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        output[:, :, start:end, :] = torch.matmul(attn_weights, value)

    return output


# ── Processor patch ───────────────────────────────────────────────────────────

_ORIGINAL_FORWARDS: dict[str, Any] = {}    # stores originals for unpatch


def _make_chunked_forward(original_cls, chunk_size: int):
    """Create a __call__ method that uses chunked attention."""

    def chunked_call(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # --- reproduce the original processor's Q/K/V projection ---
        residual = hidden_states
        batch_size, sequence_length, _ = hidden_states.shape

        # Use the attn module's projection layers
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is not None:
            key   = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key   = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # Reshape to multi-head format (B, H, N, D_head)
        inner_dim  = key.shape[-1]
        head_dim   = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(  batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply norm if present (CogVideoX uses qk_norm)
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key   = attn.norm_k(key)

        # --- chunked attention instead of F.scaled_dot_product_attention ---
        hidden_states = _chunked_scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            chunk_size=chunk_size,
        )

        # Reshape back to (B, N, inner_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)   # dropout

        return hidden_states

    return chunked_call


def _find_attention_processors(pipe) -> list[tuple[str, Any]]:
    """Walk the pipeline and find all CogVideoX attention processor instances."""
    found = []
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return found

    for name, module in transformer.named_modules():
        proc = getattr(module, "processor", None)
        if proc is not None:
            cls_name = type(proc).__name__
            if "CogVideoX" in cls_name or "Attn" in cls_name:
                found.append((name, module))

    return found


def patch_cogvideox_attention(pipe: Any, chunk_size: int = 512) -> None:
    """Replace CogVideoX attention processors with chunked versions.

    Parameters
    ----------
    pipe       : loaded CogVideoXPipeline
    chunk_size : number of Q tokens processed per iteration.
                 Smaller = less VRAM, slower. Default 512 works on T4 15 GB.
                 Use 256 for 8 GB VRAM, 1024 for 24+ GB.
    """
    from diffusers.models.attention_processor import Attention

    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        print("[chunked_attn] No transformer found — skipping patch")
        return

    patched = 0
    for name, module in transformer.named_modules():
        if not isinstance(module, Attention):
            continue

        proc     = module.processor
        cls_name = type(proc).__name__

        # Store original __call__ keyed by class name (patch once per class)
        if cls_name not in _ORIGINAL_FORWARDS:
            _ORIGINAL_FORWARDS[cls_name] = proc.__class__.__call__
            new_call = _make_chunked_forward(proc.__class__, chunk_size)
            proc.__class__.__call__ = new_call
            print(f"[chunked_attn] Patched {cls_name} with chunk_size={chunk_size}")

        patched += 1

    if patched == 0:
        print("[chunked_attn] No Attention modules found — patch not applied")
    else:
        print(f"[chunked_attn] {patched} attention modules patched. "
              f"Peak attention VRAM: ~{chunk_size * 311640 * 2 / 1e6:.0f} MB "
              f"(vs ~97 GB unpatched)")


def unpatch_cogvideox_attention(pipe: Any) -> None:
    """Restore original attention processors."""
    from diffusers.models.attention_processor import Attention

    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return

    for name, module in transformer.named_modules():
        if not isinstance(module, Attention):
            continue
        proc     = module.processor
        cls_name = type(proc).__name__
        if cls_name in _ORIGINAL_FORWARDS:
            proc.__class__.__call__ = _ORIGINAL_FORWARDS.pop(cls_name)

    print("[chunked_attn] Attention processors restored to original.")
