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

CLI flags always win over prompt-inferred values if explicitly set.

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


# ── Resolution presets ────────────────────────────────────────────────────────

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480":  (854,  480),
    "720":  (1280, 720),
    "1080": (1920, 1080),
}

DEFAULT_FPS = 8   # CogVideoX native; LTX can do 24 fps


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
    """CogVideoX requires num_frames = 4k+1; LTX is flexible."""
    if "cogvideox" in model_id:
        k = max(2, min(12, round((requested - 1) / 4)))
        return 4 * k + 1
    return max(8, min(200, requested))


def _safe_frames_for_vram(
    total_frames: int,
    resolution: str,
    total_vram_gb: float,
    model_id: str,
) -> int:
    """Reduce frame count if estimated activation memory would OOM.

    Rough heuristic: CogVideoX-5b @ 1080p needs ~150 MB per frame of
    activation memory. On 31 GB total with 17 GB weights → ~14 GB headroom.
    """
    if "cogvideox" not in model_id:
        return total_frames

    mb_per_frame = {"1080": 150, "720": 65, "480": 30}.get(resolution, 65)
    headroom_mb  = max(1000, (total_vram_gb - 17) * 1024)  # subtract weight footprint
    safe_frames  = int(headroom_mb / mb_per_frame)
    safe_frames  = _clamp_frames(max(9, safe_frames), model_id)

    if safe_frames < total_frames:
        print(
            f"[gen] VRAM heuristic: reducing frames {total_frames} → {safe_frames} "
            f"to fit {resolution}p activations in {total_vram_gb:.0f} GB VRAM"
        )
    return min(total_frames, safe_frames)


# ── Core generation ───────────────────────────────────────────────────────────

def generate_video(
    pipe: Any,
    model_info: dict,
    hw_cfg: dict,
    prompt: str,
    duration_sec: float = 5.0,
    fps: int | None = None,
    resolution: str | None = None,
    seed: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float = 6.0,
    output_path: str | None = None,
) -> Path:
    """Run the generation pipeline and write an MP4.

    Parameters
    ----------
    pipe                : loaded diffusers pipeline
    model_info          : dict from model.MODELS
    hw_cfg              : dict from hardware.detect_device()
    prompt              : user text prompt (may contain quality/res/duration hints)
    duration_sec        : target video length in seconds
    fps                 : frames per second (None → auto / prompt hint)
    resolution          : '480'|'720'|'1080' (None → auto / prompt hint)
    seed                : RNG seed (None → random)
    num_inference_steps : denoising steps (None → auto)
    guidance_scale      : CFG scale
    output_path         : destination .mp4 (None → auto-named)
    """
    device        = hw_cfg["device"]
    model_id      = model_info.get("repo_id", "").lower()
    total_vram_gb = hw_cfg.get("total_vram_gb", hw_cfg.get("vram_gb", 0))
    use_device_map = hw_cfg.get("use_device_map", False)

    # ── Parse hints from the prompt ──────────────────────────────────────
    hints = parse_prompt_hints(prompt)
    if hints:
        found = ", ".join(f"{k}={v}" for k, v in hints.items())
        print(f"[gen] Prompt hints detected: {found}")

    # ── Resolution: CLI flag > prompt hint > hardware default ────────────
    if resolution is None:
        resolution = hints.get("resolution") or hw_cfg.get("max_resolution", "720")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {list(RESOLUTIONS.keys())}")

    # ── Duration: CLI flag > prompt hint > default ───────────────────────
    duration_sec = hints.get("duration_sec", duration_sec)
    duration_sec = max(1.0, min(60.0, duration_sec))

    # ── FPS: CLI flag > prompt hint > model default ──────────────────────
    if fps is None:
        fps = hints.get("fps") or (24 if "ltx" in model_id else DEFAULT_FPS)

    # ── Inference steps: CLI flag > quality hint > device default ────────
    if num_inference_steps is None:
        base = 25 if device == "cpu" else (30 if device == "mps" else 50)
        mult = hints.get("step_multiplier", 1.0)
        num_inference_steps = max(10, int(base * mult))

    # ── Frame count ───────────────────────────────────────────────────────
    raw_frames   = int(math.ceil(duration_sec * fps))
    total_frames = _clamp_frames(raw_frames, model_id)

    # Apply VRAM safety heuristic when using balanced multi-GPU
    if use_device_map or (device == "cuda" and total_vram_gb > 0):
        total_frames = _safe_frames_for_vram(
            total_frames, resolution, total_vram_gb, model_id
        )

    actual_duration = total_frames / fps
    width, height   = RESOLUTIONS[resolution]

    # ── Seed ─────────────────────────────────────────────────────────────
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")   # cpu generator is device-map safe
        generator.manual_seed(seed)

    # ── Prompt ────────────────────────────────────────────────────────────
    enhanced_prompt = _enhance_prompt(prompt)
    negative        = _negative_prompt()

    print(f"[gen] Prompt      : {enhanced_prompt[:80]}{'…' if len(enhanced_prompt) > 80 else ''}")
    print(f"[gen] Resolution  : {width}×{height}  ({resolution}p)")
    print(f"[gen] Frames      : {total_frames}  ({actual_duration:.1f}s @ {fps} fps)")
    print(f"[gen] Steps       : {num_inference_steps}   Guidance: {guidance_scale}")
    print(f"[gen] Device      : {device}  dtype={hw_cfg['dtype']}\n")

    # ── Run with OOM recovery ─────────────────────────────────────────────
    output = _run_with_oom_recovery(
        pipe=pipe,
        model_id=model_id,
        hw_cfg=hw_cfg,
        enhanced_prompt=enhanced_prompt,
        negative=negative,
        total_frames=total_frames,
        height=height,
        width=width,
        resolution=resolution,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    elapsed = time.time() - output["t0"]
    frames_count = output["frames_used"]
    print(f"\n[gen] Generation complete in {elapsed:.1f}s  "
          f"({elapsed/frames_count:.2f}s/frame)\n")

    # ── Extract PIL frames ────────────────────────────────────────────────
    pil_frames = _extract_frames(output["result"])

    # ── Output path ───────────────────────────────────────────────────────
    if output_path is None:
        safe = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt)
        safe = safe[:40].strip().replace(" ", "_")
        output_path = f"output_{safe}_{int(time.time())}.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _write_mp4(pil_frames, output_path, fps)
    print(f"[gen] Saved → {output_path.resolve()}")
    return output_path


# ── OOM-safe runner ───────────────────────────────────────────────────────────

def _run_pipeline(pipe, model_id, enhanced_prompt, negative, total_frames,
                  height, width, num_inference_steps, guidance_scale, generator):
    """Single pipeline call — shared by both CogVideoX and LTX."""
    with torch.inference_mode():
        return pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative,
            num_frames=total_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )


def _run_with_oom_recovery(
    pipe, model_id, hw_cfg,
    enhanced_prompt, negative,
    total_frames, height, width, resolution, fps,
    num_inference_steps, guidance_scale, generator,
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
