"""
generator.py — Video generation pipeline.

Drives the loaded diffusion pipeline, handles resolution / frame-count
selection, exports the final MP4 via imageio / ffmpeg.

Prompt-driven overrides
-----------------------
Keywords detected in the prompt automatically adjust generation params:

  Resolution  : "4k", "1080p", "hd", "720p", "480p", "sd"
  Duration    : "5 seconds", "10s", "15 second", "30sec", etc.
  Frame rate  : "24fps", "30 fps", "60fps"
  Quality     : "high quality", "best quality", "ultra", "low quality", "draft"

Anything the prompt asks for wins over the defaults (480p, 10 s, model fps);
an explicitly passed CLI flag still wins over both.

Duration handling
-----------------
CogVideoX 1.0 is trained on 49 frames (6.1 s at its native 8 fps), and asking it
for more yields drift. So the requested duration is met in two steps: generate the
native clip, then *retime* it to `duration × fps` frames by blending adjacent
frames (see `_retime_frames`). A "10 second" prompt therefore still generates 49
frames, but the resulting file really is 10 s long at the requested frame rate.

Fast preset
-----------
`FAST_PRESET` (CLI `--fast`) swaps in the distilled LTX checkpoint with 8 steps and
no classifier-free guidance, which is the only configuration that produces tens of
seconds of video on 2× T4 in roughly a minute. Guidance 1.0 halves the batch; the
LTX VAE's 32×/8× compression cuts the token count ~5× against CogVideoX.

Resume / cache
--------------
Every finished pass is written losslessly to `<cache_root>/<run_key>/pass_XX.npz`,
keyed by a `cache.RunSignature` of the whole run (prompt, model, resolution, fps, steps,
guidance, seed, duration, per-pass frame counts). Re-running the same command after
an OOM, a Kaggle timeout or a Ctrl-C reuses those passes instead of regenerating
them, and the text embeddings are cached too (one T5 forward per run, not per pass).
Behaviour that changes the output lands on a different key, so two configurations can
never splice into one file. Checkpoint granularity is one pass — see cache.py.

Live output
-----------
`stream=True` splits a long clip into several generation passes and writes + shows
each one the moment it lands (`<name>_part01.mp4`, …), rather than making the user
wait for the full clip. Passes are cross-dissolved into the final file. Each pass is
an independent generation, so a boundary reads as a scene change — ideal for
timelapse/montage prompts, visible otherwise. `preview_every=N` additionally decodes
a single latent frame every N steps to show the clip forming.

OOM recovery
------------
When device_map="balanced" (multi-GPU) causes CUDA OOM during inference,
the generator automatically retries once with reduced resolution/frames
before giving up.
"""

from __future__ import annotations

import gc
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cache import RunCache, RunSignature, default_cache_root


# ── Resolution presets ────────────────────────────────────────────────────────
# All dimensions must be divisible by 32 (LTX-Video hard requirement; also
# satisfies CogVideoX's divisible-by-8 requirement).
# Standard broadcast heights (480, 720, 1080) are NOT divisible by 32, so we
# use the nearest valid values that preserve the 16:9 aspect ratio.

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480":  (832,  480),   # 832 = 26×32 ✓  480 = 15×32 ✓
    "720":  (1280, 736),   # 1280 = 40×32 ✓  736 = 23×32 ✓  (was 720 — not div by 32)
    "1080": (1920, 1088),  # 1920 = 60×32 ✓  1088 = 34×32 ✓  (was 1080 — not div by 32)
}

DEFAULT_FPS = 8            # CogVideoX native; LTX can do 24 fps
DEFAULT_RESOLUTION = "480" # 480p is the sweet spot on T4-class GPUs
DEFAULT_DURATION_SEC = 10.0
MAX_OUTPUT_FRAMES = 900    # hard cap on retimed output (30 s @ 24 fps fits)
MAX_SEGMENTS = 8           # cap on chained passes for one long clip
SEGMENT_FADE = 4           # frames of cross-dissolve hiding a segment boundary

# Resolution tiers, low→high (used to compare a request against GPU capability)
_RES_ORDER = ["480", "720", "1080"]

# `--fast` preset: use ltx-video (already downloaded, ~8 GB) with reduced steps.
# Distilled checkpoints are not bundled because they require a separate HF repo
# that changes frequently — point LTX_DISTILLED_REPO at one if you have it.
FAST_PRESET: dict[str, Any] = {
    "model":      "ltx-video",
    "steps":      20,
    "guidance":   3.0,
    "fps":        24,
    "resolution": "480",
    "stream":     True,
}


# ── Prompt parsing ────────────────────────────────────────────────────────────

# Resolution keywords → resolution key
_RES_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b4k\b|ultra\s*hd|2160p",         re.I), "1080"),  # cap at 1080 (no 4K model)
    (re.compile(r"\b1080p?\b|full\s*hd",             re.I), "1080"),
    (re.compile(r"\bhd\b(?!r)",                      re.I), "720"),   # HD but not HDR
    (re.compile(r"\b720p?\b",                        re.I), "720"),
    (re.compile(r"\b480p?\b|\bsd\b",                 re.I), "480"),
]

# Duration: "10 seconds", "10s", "10sec", "10-second"
_DUR_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*[-–]?\s*(?:seconds?|secs?|s)\b",
    re.I,
)

# FPS: "24fps", "24 fps", "30fps"
_FPS_PATTERN = re.compile(r"\b(\d+)\s*fps\b", re.I)

# Quality → step multiplier
_QUALITY_MAP: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bultimate\b|\bmaximum quality\b|\bbest quality\b", re.I), 1.4),
    (re.compile(r"\bhigh quality\b|\bcinematic\b|\b4k\b",             re.I), 1.2),
    (re.compile(r"\bdraft\b|\bfast\b|\blow quality\b|\bquick\b",      re.I), 0.5),
]


def parse_prompt_hints(prompt: str) -> dict:
    """Extract generation hints embedded in the prompt text.

    Returns a dict with any of: resolution, duration_sec, fps, step_multiplier.
    Only keys that were found are included — callers use .get() with defaults.
    """
    hints: dict = {}
    lower = prompt.lower()

    # Resolution
    for pattern, res in _RES_PATTERNS:
        if pattern.search(prompt):
            hints["resolution"] = res
            break

    # Duration
    dur_match = _DUR_PATTERN.search(prompt)
    if dur_match:
        hints["duration_sec"] = float(dur_match.group(1))

    # FPS
    fps_match = _FPS_PATTERN.search(prompt)
    if fps_match:
        hints["fps"] = int(fps_match.group(1))

    # Quality → step multiplier
    for pattern, mult in _QUALITY_MAP:
        if pattern.search(prompt):
            hints["step_multiplier"] = mult
            break

    return hints


# ── Prompt enhancement ────────────────────────────────────────────────────────

_QUALITY_SUFFIX = (
    ", cinematic lighting, sharp focus, high detail, photorealistic"
)

def _enhance_prompt(prompt: str) -> str:
    lower = prompt.lower()
    if "cinematic" not in lower and "photorealistic" not in lower:
        return prompt.rstrip(" ,") + _QUALITY_SUFFIX
    return prompt


def _negative_prompt() -> str:
    return (
        "blurry, low quality, distorted, watermark, text, logo, "
        "noise, overexposed, underexposed, duplicate frames"
    )


# ── Frame helpers ─────────────────────────────────────────────────────────────

def _clamp_frames(requested: int, model_id: str) -> int:
    """Snap a frame count onto what the model's VAE can actually decode.

    CogVideoX compresses 4× in time → num_frames = 4k+1 (49 = its trained length).
    LTX compresses 8× → num_frames = 8k+1 (121 / 257 are the published lengths).
    Anything else is rejected by the VAE or silently padded.
    """
    if "cogvideox" in model_id:
        k = max(2, min(12, round((requested - 1) / 4)))
        return 4 * k + 1

    k = max(1, min(25, round((requested - 1) / 8)))
    return 8 * k + 1


# Full pipeline footprint in fp16 (transformer + T5-XXL text encoder + VAE).
# With adaptive multi-GPU sharding each GPU only holds ~1/gpu_count of this.
_PIPELINE_WEIGHTS_GB: dict[str, float] = {
    "cogvideox-2b": 14.0,
    "cogvideox-5b": 20.0,
    "ltx-video": 6.0,
}


def _pipeline_weights_gb(model_id: str) -> float:
    return next((w for k, w in _PIPELINE_WEIGHTS_GB.items() if k in model_id), 12.0)


def _safe_frames_for_vram(
    total_frames: int,
    resolution: str,
    per_gpu_vram_gb: float,
    weights_gb_per_gpu: float,
    model_id: str,
) -> int:
    """Only reduce frames if estimated activation memory would actually OOM.

    Everything is measured on the *busiest single GPU* — with device_map sharding
    that is where the activations have to fit. Attention itself is chunked (a ~0.5-
    1 GB working set regardless of frame count), so the frame-dependent term is the
    latent/activation state: ~25 MB/frame at 480p, ~60 MB at 720p, ~140 MB at 1080p.
    On 2×15.6 GB this never fires for 480p or 720p.
    """
    if "cogvideox" not in model_id or per_gpu_vram_gb <= 0:
        return total_frames

    headroom_mb  = max(0, (per_gpu_vram_gb - weights_gb_per_gpu) * 1024)
    mb_per_frame = {"1080": 140, "720": 60, "480": 25}.get(resolution, 60)
    safe_frames  = int(headroom_mb / mb_per_frame) if mb_per_frame > 0 else total_frames
    safe_frames  = _clamp_frames(max(9, safe_frames), model_id)

    if safe_frames < total_frames:
        print(
            f"[gen] VRAM heuristic: capping frames {total_frames} → {safe_frames} "
            f"({resolution}p on {per_gpu_vram_gb:.0f} GB GPU with "
            f"~{weights_gb_per_gpu:.1f} GB of weights, "
            f"~{headroom_mb:.0f} MB activation headroom)"
        )
        return safe_frames
    return total_frames


def _retime_frames(frames: list[Image.Image], target_count: int) -> list[Image.Image]:
    """Blend-interpolate a generated clip to `target_count` frames.

    Used to honour a prompt's duration/fps when the model cannot generate that
    many frames natively (CogVideoX 1.0: 49 frames = 6.1 s @ 8 fps). Sampling with
    linear interpolation between neighbouring frames is a cheap temporal
    cross-dissolve — it keeps motion smooth and needs no extra dependencies like
    RIFE/FILM, which matters for an offline pipeline.
    """
    src_count = len(frames)
    if target_count <= 0 or src_count == 0 or target_count == src_count:
        return frames

    positions = np.linspace(0.0, src_count - 1, target_count)
    out: list[Image.Image] = []

    for pos in positions:
        lo = int(math.floor(pos))
        hi = min(lo + 1, src_count - 1)
        weight = float(pos - lo)

        if weight <= 1e-6 or lo == hi:
            out.append(frames[lo])
            continue

        a = np.asarray(frames[lo], dtype=np.float32)
        b = np.asarray(frames[hi], dtype=np.float32)
        blended = a * (1.0 - weight) + b * weight
        out.append(Image.fromarray(blended.astype(np.uint8)))

    return out


# ── Live / streaming output ───────────────────────────────────────────────────

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def _show_video(path: Path) -> None:
    """Play a finished file inline when running in a notebook (Kaggle/Colab).

    The path is always printed as well — on Kaggle that is what the Output panel
    serves, so the file is watchable (or downloadable) the moment it is written.
    """
    print(f"[live] Ready to watch → {path}")
    if not _in_notebook():
        return
    try:
        from IPython.display import Video, display
        try:
            display(Video(str(path), embed=False, html_attributes="controls loop"))
        except Exception:
            display(Video(str(path), embed=True, html_attributes="controls loop"))
    except Exception:
        pass


def _show_image(path: Path, label: str = "") -> None:
    """Show a preview still inline (stacked, so earlier previews stay visible)."""
    print(f"[live] {label}preview → {path}")
    if not _in_notebook():
        return
    try:
        from IPython.display import Image as IPyImage, display
        display(IPyImage(filename=str(path), width=480))
    except Exception:
        pass


def _decode_preview_frame(pipe: Any, latents: torch.Tensor) -> Image.Image | None:
    """Decode a *single* latent frame straight to RGB — a cheap live preview.

    Decoding one latent frame costs ~1-2 s at 480p versus ~1 min for the whole
    clip, and uses the same latent normalisation the pipeline applies at the end.
    """
    vae = getattr(pipe, "vae", None)
    if vae is None or latents is None or latents.dim() != 5:
        return None

    lat = latents[:, :, :1]                       # first latent frame only
    cfg  = getattr(vae, "config", None)
    mean = getattr(cfg, "latents_mean", None) if cfg is not None else None
    std  = getattr(cfg, "latents_std", None) if cfg is not None else None
    scale = getattr(cfg, "scaling_factor", 1.0) if cfg is not None else 1.0

    if mean is not None and std is not None:      # CogVideoX denormalisation
        m = torch.tensor(mean, device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
        s = torch.tensor(std, device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
        lat = lat * s + m
    else:
        lat = lat / scale

    with torch.no_grad():
        video = vae.decode(lat.to(vae.dtype), return_dict=False)[0]

    frame = video[0, :, 0].detach().float().cpu()           # (C, H, W), range [-1, 1]
    arr = ((frame.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def _make_preview_hook(pipe: Any, every: int, out_dir: Path) -> Any:
    """Build a `callback_on_step_end` that shows the clip forming step by step."""
    state = {"failed": False}

    def hook(_pipe: Any, step: int, _timestep: Any, callback_kwargs: dict) -> dict:
        if state["failed"] or (step + 1) % every != 0:
            return callback_kwargs
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs
        try:
            frame = _decode_preview_frame(_pipe, latents)
            if frame is not None:
                path = Path(out_dir) / "preview.jpg"
                frame.save(path, quality=88)
                _show_image(path, label=f"step {step + 1} ")
        except Exception as exc:
            # Previews are cosmetic: never let them break a long run.
            state["failed"] = True
            print(f"[live] live previews disabled ({type(exc).__name__}: {exc})")
        return callback_kwargs

    return hook


# ── Segment planning / assembly ───────────────────────────────────────────────

def _plan_segments(
    target_frames: int,
    native_cap: int,
    stream: bool,
    model_id: str,
) -> list[int]:
    """Split a long clip into per-pass frame counts.

    A single pass is the coherent option (and what retiming is for), but nothing is
    watchable until it finishes. In stream mode the clip is chained from several
    independent passes of at most `native_cap` frames each, so every piece is
    playable the moment it lands. Expect a scene change at each boundary — fine for
    timelapse/montage prompts, and cross-dissolved to soften it.
    """
    if target_frames <= 1:
        return [max(8, target_frames)]
    if not stream or target_frames <= native_cap:
        return [_clamp_frames(min(target_frames, native_cap), model_id)]

    count = min(MAX_SEGMENTS, -(-target_frames // native_cap))
    base, extra = divmod(target_frames, count)
    sizes = [base + (1 if i < extra else 0) for i in range(count)]
    return [_clamp_frames(size, model_id) for size in sizes]


def _crossfade_join(prev: list[Image.Image], nxt: list[Image.Image], fade: int = SEGMENT_FADE):
    """Append `nxt` to `prev`, dissolving the overlap so the cut is not jarring."""
    if not prev:
        return list(nxt)
    fade = max(0, min(fade, len(prev), len(nxt)))
    if fade == 0:
        return prev + list(nxt)

    blended = [
        Image.blend(prev[len(prev) - fade + i].convert("RGB"),
                    nxt[i].convert("RGB"),
                    (i + 1) / (fade + 1))
        for i in range(fade)
    ]
    return prev[:-fade] + blended + nxt[fade:]


def _model_default(model_info: dict, key: str, fallback: Any) -> Any:
    """Read a per-model generation default from the registry (model.py)."""
    value = model_info.get(key)
    return fallback if value is None else value


# ── Prompt embeddings (encoded once, reused by every pass/resume) ─────────────

def _is_cuda_oom(exc: BaseException) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(exc, oom)) or "out of memory" in str(exc).lower()


def _place_embeds(pipe: Any, positive: Any, negative: Any) -> dict:
    """Return embeddings on the dtype/device the transformer will be called with."""
    try:
        dtype  = getattr(getattr(pipe, "transformer", None), "dtype", None)
        device = getattr(pipe, "_execution_device", None)
        if dtype is not None:
            positive = positive.to(dtype=dtype)
            negative = None if negative is None else negative.to(dtype=dtype)
        if device is not None:
            positive = positive.to(device=device)
            negative = None if negative is None else negative.to(device=device)
    except Exception:                                       # noqa: BLE001
        pass
    return {"positive": positive, "negative": negative}


def _prepare_embeds(
    pipe: Any,
    cache: RunCache,
    prompt: str,
    negative: str,
    guidance_scale: float,
) -> dict | None:
    """Encode the prompt once (or load it from cache) for every pass to reuse.

    Text encoding is identical for every pass of a run, so doing it once saves a T5
    forward per pass — noticeable on the fast path, where a whole clip is only a few
    passes. Returns a mutable dict (so a rejected batch can be cleared for the rest of
    the run) or None to let the pipeline encode as usual.
    """
    cached = cache.load_embeds()
    if cached is not None:
        positive, negative_embeds = cached
        if negative_embeds is None and guidance_scale > 1.0:
            print("[cache] Cached embeddings lack the negative half CFG needs "
                  "— re-encoding")
        else:
            return _place_embeds(pipe, positive, negative_embeds)

    encoder = getattr(pipe, "encode_prompt", None)
    if not callable(encoder):
        return None

    try:
        import inspect
        params = inspect.signature(encoder).parameters
        kwargs: dict[str, Any] = {}
        if "do_classifier_free_guidance" in params:
            kwargs["do_classifier_free_guidance"] = guidance_scale > 1.0
        if "num_videos_per_prompt" in params:
            kwargs["num_videos_per_prompt"] = 1
        if "device" in params:
            kwargs["device"] = getattr(pipe, "_execution_device", None) or pipe.device

        with torch.inference_mode():
            result = encoder(prompt=prompt, negative_prompt=negative, **kwargs)
    except Exception as exc:                                # noqa: BLE001
        print(f"[gen] Prompt-embedding reuse unavailable "
              f"({type(exc).__name__}: {exc}) — using the pipeline's own encoding")
        return None

    if not isinstance(result, (tuple, list)) or len(result) < 2:
        return None
    positive, negative_embeds = result[0], result[1]
    if positive is None or (negative_embeds is None and guidance_scale > 1.0):
        return None

    cache.save_embeds(positive, negative_embeds)
    return _place_embeds(pipe, positive, negative_embeds)


# ── Core generation ───────────────────────────────────────────────────────────

def generate_video(
    pipe: Any,
    model_info: dict,
    hw_cfg: dict,
    prompt: str,
    duration_sec: float | None = None,
    fps: int | None = None,
    resolution: str | None = None,
    seed: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    output_path: str | None = None,
    stream: bool = False,
    preview_every: int = 0,
    segment_seconds: float | None = None,
    cache_dir: str | Path | None = None,
    resume: bool = True,
) -> Path:
    """Run the generation pipeline and write an MP4.

    Parameters
    ----------
    pipe                : loaded diffusers pipeline
    model_info          : dict from model.MODELS
    hw_cfg              : dict from hardware.detect_device()
    prompt              : user text prompt (may contain quality/res/duration hints)
    duration_sec        : target video length in seconds (None → prompt hint, else 10 s)
    fps                 : frames per second (None → prompt hint / model default)
    resolution          : '480'|'720'|'1080' (None → prompt hint, else 480p)
    seed                : RNG seed (None → random)
    num_inference_steps : denoising steps (None → per-model default)
    guidance_scale      : CFG scale (None → per-model default; 1.0 disables CFG)
    output_path         : destination .mp4 (None → auto-named)
    stream              : save + show each pass as soon as it finishes
    preview_every       : decode a live preview still every N steps (0 → off)
    segment_seconds     : cap a single pass to this many seconds of footage
    cache_dir           : where finished passes are kept (None → env/cwd default)
    resume              : reuse passes from an identical earlier run when present
    """
    device        = hw_cfg["device"]
    model_id      = (model_info.get("model_id")
                     or model_info.get("repo_id")
                     or "").lower()
    total_vram_gb = hw_cfg.get("total_vram_gb", hw_cfg.get("vram_gb", 0))
    use_device_map = hw_cfg.get("use_device_map", False)

    # ── Parse hints from the prompt ──────────────────────────────────────
    hints = parse_prompt_hints(prompt)
    if hints:
        found = ", ".join(f"{k}={v}" for k, v in hints.items())
        print(f"[gen] Prompt hints detected: {found}")

    # ── Resolution: CLI flag > prompt hint > default (480p) ──────────────
    if resolution is None:
        resolution = hints.get("resolution") or DEFAULT_RESOLUTION
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {list(RESOLUTIONS.keys())}")

    capability = hw_cfg.get("max_resolution")
    if capability and _RES_ORDER.index(resolution) > _RES_ORDER.index(capability):
        print(f"[gen] Note: {resolution}p is above the detected capability "
              f"({capability}p) — this will be slow on "
              f"{hw_cfg.get('vram_gb', 0):.0f} GB per GPU")

    # ── Duration: CLI flag > prompt hint > default (10 s) ────────────────
    if duration_sec is None:
        duration_sec = hints.get("duration_sec", DEFAULT_DURATION_SEC)
    duration_sec = max(1.0, min(60.0, duration_sec))

    # ── FPS: CLI flag > prompt hint > model default ──────────────────────
    if fps is None:
        fps = hints.get("fps") or _model_default(model_info, "default_fps", DEFAULT_FPS)

    # ── Inference steps: CLI flag > quality hint > model default ────────
    if num_inference_steps is None:
        base = _model_default(model_info, "default_steps",
                              50 if device == "cuda" else 25)
        if device == "cpu":
            base = min(base, 25)
        elif device == "mps":
            base = min(base, 30)
        mult = hints.get("step_multiplier", 1.0)
        num_inference_steps = max(4, int(base * mult))

    # ── Guidance: CLI flag > model default (1.0 = no CFG, e.g. distilled) ─
    if guidance_scale is None:
        guidance_scale = _model_default(model_info, "default_guidance", 6.0)

    # ── Frame count ───────────────────────────────────────────────────────
    # Frames the *file* must contain to play for the requested duration at the
    # requested fps.
    target_frames = max(1, min(MAX_OUTPUT_FRAMES, int(round(duration_sec * fps))))

    # A pass can only produce so many in-distribution frames (49 for CogVideoX 1.0,
    # ~121 for LTX). In stream mode a long clip is chained from several passes so each
    # piece is watchable the moment it lands; otherwise it is one retimed pass.
    native_cap = int(_model_default(model_info, "max_native_frames", 49))
    if segment_seconds:
        native_cap = min(native_cap, max(8, int(round(segment_seconds * fps))))
    segment_sizes = _plan_segments(target_frames, native_cap, stream, model_id)

    # Apply VRAM safety heuristic when running on CUDA (per pass, per GPU)
    use_shard = use_device_map and hw_cfg.get("gpu_count", 1) > 1
    if use_shard or (device == "cuda" and total_vram_gb > 0):
        gpus            = max(1, hw_cfg.get("gpu_count", 1)) if use_shard else 1
        per_gpu_vram    = hw_cfg.get("vram_gb", total_vram_gb) if use_shard else total_vram_gb
        weights_per_gpu = _pipeline_weights_gb(model_id) / gpus
        segment_sizes = [
            _safe_frames_for_vram(n, resolution, per_gpu_vram, weights_per_gpu, model_id)
            for n in segment_sizes
        ]

    generated_total = sum(segment_sizes)
    width, height   = RESOLUTIONS[resolution]

    # ── Output path (resolved up front so each pass can be written/played) ─
    if output_path is None:
        safe = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt)
        safe = safe[:40].strip().replace(" ", "_")
        output_path = f"output_{safe}_{int(time.time())}.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_path.parent

    # ── Seed ─────────────────────────────────────────────────────────────
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")   # cpu generator is device-map safe
        generator.manual_seed(seed)

    # ── Prompt ────────────────────────────────────────────────────────────
    enhanced_prompt = _enhance_prompt(prompt)
    negative        = _negative_prompt()
    preview_hook    = (_make_preview_hook(pipe, preview_every, out_dir)
                       if preview_every > 0 else None)

    # ── Resume cache + text embeddings ────────────────────────────────────
    signature = RunSignature(
        prompt=enhanced_prompt,
        negative=negative,
        model_id=model_id,
        resolution=resolution,
        fps=fps,
        steps=num_inference_steps,
        guidance=guidance_scale,
        seed=seed,
        duration_sec=duration_sec,
        segment_sizes=tuple(segment_sizes),
    )
    cache  = RunCache(default_cache_root(cache_dir), signature, enabled=resume)
    cache.register()
    embeds = _prepare_embeds(pipe, cache, enhanced_prompt, negative, guidance_scale)

    print(f"[gen] Prompt      : {enhanced_prompt[:80]}{'…' if len(enhanced_prompt) > 80 else ''}")
    print(f"[gen] Resolution  : {width}×{height}  ({resolution}p)")
    print(f"[gen] Generate    : {generated_total} frames in {len(segment_sizes)} pass(es) "
          f"({generated_total / fps:.1f}s @ {fps} fps native)")
    print(f"[gen] Output clip : {target_frames} frames  ({target_frames / fps:.1f}s @ {fps} fps)")
    print(f"[gen] Steps       : {num_inference_steps}   Guidance: {guidance_scale}")
    print(f"[gen] Device      : {device}  dtype={hw_cfg['dtype']}")
    if stream and len(segment_sizes) > 1:
        print(f"[gen] Streaming   : every pass is saved and played as it finishes "
              f"({SEGMENT_FADE}-frame dissolve at each boundary)")
    if preview_hook is not None:
        print(f"[gen] Live preview: one decoded frame every {preview_every} steps")
    print(cache.describe())
    print()

    if cache.is_complete(len(segment_sizes)) and output_path.exists():
        print(f"[gen] Already complete — all {len(segment_sizes)} pass(es) are cached and "
              f"{output_path.name} exists. Nothing to generate.\n"
              f"      (--clear-cache to start over.)")
        _show_video(output_path)
        return output_path

    # ── Generate pass by pass; show each one as it lands ─────────────────
    pil_frames: list[Image.Image] = []
    for index, seg_target in enumerate(segment_sizes, start=1):
        seg_frames = cache.load_pass(index)          # None → must generate this one

        if seg_frames is None:
            if len(segment_sizes) > 1:
                print(f"\n[gen] ── Pass {index}/{len(segment_sizes)}: {seg_target} frames ──")
                if generator is not None:
                    # A fixed seed would make every pass byte-identical; offset per
                    # pass so the clip is reproducible yet actually moves.
                    generator.manual_seed(seed + index - 1)

            t_pass = time.time()
            try:
                output = _run_with_oom_recovery(
                    pipe=pipe,
                    model_id=model_id,
                    hw_cfg=hw_cfg,
                    enhanced_prompt=enhanced_prompt,
                    negative=negative,
                    total_frames=seg_target,
                    height=height,
                    width=width,
                    resolution=resolution,
                    fps=fps,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback=preview_hook,
                    embeds=embeds,
                )
            except BaseException as exc:            # includes Ctrl-C / Kaggle timeout
                cache.report_interrupted(index, len(segment_sizes), exc)
                raise

            elapsed     = time.time() - t_pass
            frames_used = output["frames_used"]
            print(f"\n[gen] Pass {index} done in {elapsed:.1f}s  "
                  f"({elapsed/frames_used:.2f}s/frame)")

            seg_frames = _extract_frames(output["result"])
            cache.save_pass(index, seg_frames)

        if stream and len(segment_sizes) > 1:
            part_path = out_dir / f"{output_path.stem}_part{index:02d}.mp4"
            _write_mp4(seg_frames, part_path, fps)
            _show_video(part_path)

        pil_frames = _crossfade_join(pil_frames, seg_frames)

    # ── Retime to the requested duration (blend interpolation) ────────────
    if len(pil_frames) != target_frames:
        generated = len(pil_frames)
        pil_frames = _retime_frames(pil_frames, target_frames)
        print(f"[gen] Retimed {generated} generated frames → {len(pil_frames)} output frames "
              f"({len(pil_frames) / fps:.1f}s @ {fps} fps, blend interpolation)")

    _write_mp4(pil_frames, output_path, fps)
    cache.mark_complete(output_path)
    print(f"[gen] Saved → {output_path.resolve()}")
    _show_video(output_path)
    return output_path


# ── OOM-safe runner ───────────────────────────────────────────────────────────

def _run_pipeline(pipe, model_id, enhanced_prompt, negative, total_frames,
                  height, width, num_inference_steps, guidance_scale, generator,
                  callback=None, embeds=None):
    """Single pipeline call — shared by both CogVideoX and LTX."""
    kwargs = dict(
        prompt=enhanced_prompt,
        negative_prompt=negative,
        num_frames=total_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    if callback is not None:
        kwargs["callback_on_step_end"] = callback

    using_embeds = bool(embeds and embeds.get("positive") is not None)
    if using_embeds:
        kwargs["prompt_embeds"] = embeds["positive"]
        if embeds.get("negative") is not None:
            kwargs["negative_prompt_embeds"] = embeds["negative"]
        # Pipeline rejects having both prompt text and prompt_embeds at once
        kwargs.pop("prompt", None)
        kwargs.pop("negative_prompt", None)
        # LTXPipeline requires an explicit prompt_attention_mask alongside the
        # cached prompt_embeds (it cannot re-derive it from the dropped text).
        if "prompt_attention_mask" not in kwargs:
            kwargs["prompt_attention_mask"] = torch.ones(
                1, dtype=torch.long,
                device=(kwargs["generator"].device
                        if hasattr(kwargs.get("generator"), "device")
                        else getattr(pipe, "_execution_device", None)
                        or torch.device("cpu")),
            )
        if "negative_prompt_attention_mask" not in kwargs and embeds.get("negative") is not None:
            kwargs["negative_prompt_attention_mask"] = torch.ones(
                1, dtype=torch.long, device=kwargs["prompt_attention_mask"].device,
            )

    with torch.inference_mode():
        try:
            return pipe(**kwargs)
        except (TypeError, RuntimeError) as exc:
            # Cached embeddings can be rejected (unsupported kwarg, device or dtype
            # mismatch). Drop them for the whole run and let the pipeline encode
            # normally — but never swallow an OOM, that is the OOM handler's job.
            if not using_embeds or _is_cuda_oom(exc):
                raise
            print(f"[gen] Cached prompt embeddings rejected "
                  f"({type(exc).__name__}) — falling back to the pipeline's encoding")
            if embeds is not None:
                embeds.clear()
            kwargs.pop("prompt_embeds", None)
            kwargs.pop("negative_prompt_embeds", None)
            return pipe(**kwargs)


def _run_with_oom_recovery(
    pipe, model_id, hw_cfg,
    enhanced_prompt, negative,
    total_frames, height, width, resolution, fps,
    num_inference_steps, guidance_scale, generator,
    callback=None,
    embeds=None,
) -> dict:
    """Run pipeline and auto-retry with reduced params on CUDA OOM.

    Retry strategy (applied once):
      1. Reduce frames by ~40%
      2. Drop resolution one tier (1080 → 720 → 480)
      3. Clear VRAM cache before retrying

    Returns dict with keys: result, frames_used, t0
    """
    t0 = time.time()
    try:
        result = _run_pipeline(
            pipe, model_id, enhanced_prompt, negative,
            total_frames, height, width,
            num_inference_steps, guidance_scale, generator,
            callback=callback,
            embeds=embeds,
        )
        return {"result": result, "frames_used": total_frames, "t0": t0}

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[gen] CUDA OOM: {e}")
        print("[gen] Retrying with reduced resolution and frames …")

        # Free cache before retry
        gc.collect()
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            free = torch.cuda.mem_get_info(i)[0] / 1e9
            print(f"[gen]   GPU {i} free after cache clear: {free:.1f} GB")

        # Reduce resolution
        _res_order = ["1080", "720", "480"]
        new_res = _res_order[min(_res_order.index(resolution) + 1, len(_res_order) - 1)]
        new_w, new_h = RESOLUTIONS[new_res]

        # Reduce frames ~40%
        new_frames = _clamp_frames(max(9, int(total_frames * 0.6)), model_id)

        print(f"[gen] Retry: {new_w}×{new_h} ({new_res}p), {new_frames} frames, "
              f"{num_inference_steps} steps")

        t0 = time.time()
        result = _run_pipeline(
            pipe, model_id, enhanced_prompt, negative,
            new_frames, new_h, new_w,
            num_inference_steps, guidance_scale, generator,
            callback=callback,
            embeds=embeds,
        )
        return {"result": result, "frames_used": new_frames, "t0": t0}


# ── Frame extraction ──────────────────────────────────────────────────────────

def _extract_frames(output) -> list[Image.Image]:
    frames = output.frames
    if isinstance(frames, torch.Tensor):
        frames = frames[0]
        if frames.dtype != torch.uint8:
            frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
        return [Image.fromarray(f) for f in frames.cpu().numpy()]
    elif isinstance(frames, (list, tuple)) and len(frames) > 0:
        inner = frames[0]
        if isinstance(inner, (list, tuple)):
            return list(inner)
        return list(frames)
    raise RuntimeError(
        f"Unexpected pipeline output type: {type(frames)}. Cannot extract frames."
    )


# ── MP4 writers ───────────────────────────────────────────────────────────────

def _write_mp4(frames: list[Image.Image], path: Path, fps: int) -> None:
    try:
        _write_mp4_imageio(frames, path, fps)
    except Exception as exc:
        print(f"[gen] imageio writer failed ({exc}), trying OpenCV …")
        _write_mp4_opencv(frames, path, fps)


def _write_mp4_imageio(frames: list[Image.Image], path: Path, fps: int) -> None:
    import imageio
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(np.array(frame.convert("RGB")))
    writer.close()
    print(f"[gen] Written via imageio ({len(frames)} frames, {fps} fps)")


def _write_mp4_opencv(frames: list[Image.Image], path: Path, fps: int) -> None:
    import cv2
    frame0 = np.array(frames[0].convert("RGB"))
    h, w   = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    print(f"[gen] Written via OpenCV ({len(frames)} frames, {fps} fps)")
