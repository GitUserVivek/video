"""
generator.py — Video generation pipeline.

Drives the loaded diffusion pipeline, handles resolution / frame-count
selection, and exports the final MP4 via imageio / ffmpeg.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


# ── Resolution presets (width, height) ───────────────────────────────────────

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480":  (854,  480),
    "720":  (1280, 720),
    "1080": (1920, 1080),
}

# Frames-per-second target
DEFAULT_FPS = 8       # CogVideoX native; LTX can do 24 fps


# ── Prompt helpers ────────────────────────────────────────────────────────────

_QUALITY_SUFFIX = (
    ", cinematic lighting, sharp focus, high detail, photorealistic, 4K"
)

def _enhance_prompt(prompt: str) -> str:
    """Append quality tokens if not already present."""
    lower = prompt.lower()
    if "cinematic" not in lower and "photorealistic" not in lower:
        return prompt.rstrip(" ,") + _QUALITY_SUFFIX
    return prompt


def _negative_prompt() -> str:
    return (
        "blurry, low quality, distorted, watermark, text, logo, "
        "noise, overexposed, underexposed, duplicate frames"
    )


# ── Frame-count helpers ───────────────────────────────────────────────────────

def _clamp_frames(requested: int, device: str, model_id: str) -> int:
    """CogVideoX requires num_frames = 4k+1 (e.g. 49); LTX is more flexible."""
    if "cogvideox" in model_id:
        # Round to nearest 4k+1, minimum 9, maximum 49
        k = max(2, min(12, round((requested - 1) / 4)))
        return 4 * k + 1
    # LTX: just clamp to a sane range
    return max(8, min(200, requested))


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
    pipe            : loaded diffusers pipeline
    model_info      : dict from model.MODELS
    hw_cfg          : dict from hardware.detect_device()
    prompt          : user text prompt
    duration_sec    : target video length (5–15 s recommended)
    fps             : frames per second (None → auto)
    resolution      : '480' | '720' | '1080' (None → auto from hw_cfg)
    seed            : RNG seed for reproducibility (None → random)
    num_inference_steps : denoising steps (None → auto)
    guidance_scale  : classifier-free guidance scale
    output_path     : destination .mp4 path (None → auto-generated)

    Returns
    -------
    Path to the written MP4 file.
    """
    device   = hw_cfg["device"]
    model_id = model_info.get("repo_id", "").lower()

    # --- Resolution ---
    if resolution is None:
        resolution = hw_cfg.get("max_resolution", "720")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {list(RESOLUTIONS.keys())}")
    width, height = RESOLUTIONS[resolution]

    # --- FPS ---
    if fps is None:
        fps = 24 if "ltx" in model_id else DEFAULT_FPS

    # --- Frame count ---
    total_frames = _clamp_frames(int(math.ceil(duration_sec * fps)), device, model_id)
    actual_duration = total_frames / fps

    # --- Inference steps (auto-tune for CPU) ---
    if num_inference_steps is None:
        if device == "cpu":
            num_inference_steps = 25   # fewer steps → faster on CPU
        elif device == "mps":
            num_inference_steps = 30
        else:
            num_inference_steps = 50   # GPU: full quality

    # --- Seed ---
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device if device != "cpu" else "cpu")
        generator.manual_seed(seed)

    # --- Prompt enhancement ---
    enhanced_prompt   = _enhance_prompt(prompt)
    negative          = _negative_prompt()

    print(f"[gen] Prompt      : {enhanced_prompt[:80]}{'…' if len(enhanced_prompt) > 80 else ''}")
    print(f"[gen] Resolution  : {width}×{height}  ({resolution}p)")
    print(f"[gen] Frames      : {total_frames}  ({actual_duration:.1f}s @ {fps} fps)")
    print(f"[gen] Steps       : {num_inference_steps}   Guidance: {guidance_scale}")
    print(f"[gen] Device      : {device}  dtype={hw_cfg['dtype']}\n")

    t0 = time.time()

    # --- Run pipeline ---
    with torch.inference_mode():
        if "cogvideox" in model_id:
            output = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative,
                num_frames=total_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            # LTX-Video
            output = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative,
                num_frames=total_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

    elapsed = time.time() - t0
    print(f"\n[gen] Generation complete in {elapsed:.1f}s  "
          f"({elapsed/total_frames:.2f}s/frame)\n")

    # --- Extract frames ---
    # diffusers video pipelines return output.frames as a list of PIL Images
    # or as a 5-D tensor (B, T, H, W, C)
    frames = output.frames
    if isinstance(frames, torch.Tensor):
        # (1, T, H, W, C) → list of numpy uint8
        frames = frames[0]  # (T, H, W, C)
        if frames.dtype != torch.uint8:
            frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
        frames_np = frames.cpu().numpy()
        pil_frames = [Image.fromarray(f) for f in frames_np]
    elif isinstance(frames, (list, tuple)) and len(frames) > 0:
        if isinstance(frames[0], (list, tuple)):
            # some pipelines nest: [[PIL, PIL, …]]
            pil_frames = list(frames[0])
        else:
            pil_frames = list(frames)
    else:
        raise RuntimeError(
            f"Unexpected pipeline output type: {type(frames)}.  "
            "Cannot extract frames."
        )

    # --- Output path ---
    if output_path is None:
        safe_prompt = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt)
        safe_prompt = safe_prompt[:40].strip().replace(" ", "_")
        output_path = f"output_{safe_prompt}_{int(time.time())}.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Write MP4 ---
    _write_mp4(pil_frames, output_path, fps)

    print(f"[gen] Saved → {output_path.resolve()}")
    return output_path


# ── MP4 writer ────────────────────────────────────────────────────────────────

def _write_mp4(frames: list[Image.Image], path: Path, fps: int) -> None:
    """Write a list of PIL Images to an MP4 file.

    Tries imageio[ffmpeg] first, then falls back to opencv.
    """
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
