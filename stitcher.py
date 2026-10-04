"""
stitcher.py — Join multiple MP4 scene clips into one final video.

Each boundary between clips is softened with a linear cross-dissolve
(configurable fade length). The stitcher reads each clip frame-by-frame
using imageio so it works without ffmpeg being installed — the same
writer used by the rest of the pipeline.

Usage
-----
    from stitcher import stitch_scenes

    final = stitch_scenes(
        clips       = [Path("scene_01.mp4"), Path("scene_02.mp4")],
        output_path = Path("final.mp4"),
        fps         = 8,
        fade_frames = 4,    # cross-dissolve length
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


FADE_FRAMES = 4   # default cross-dissolve length in frames


def _read_frames(path: Path) -> list[np.ndarray]:
    """Read all frames from an MP4 file as uint8 RGB arrays."""
    import imageio
    reader = imageio.get_reader(str(path))
    frames = [np.asarray(frame) for frame in reader]
    reader.close()
    return frames


def _crossfade(
    a: list[np.ndarray],
    b: list[np.ndarray],
    fade: int,
) -> list[np.ndarray]:
    """Blend the tail of `a` into the head of `b` over `fade` frames.

    Returns the combined frame list with the dissolve applied in place
    of the last `fade` frames of `a` and first `fade` frames of `b`.
    """
    fade = max(0, min(fade, len(a), len(b)))
    if fade == 0:
        return list(a) + list(b)

    blended = []
    for i in range(fade):
        alpha = (i + 1) / (fade + 1)
        f = (a[len(a) - fade + i].astype(np.float32) * (1 - alpha)
             + b[i].astype(np.float32) * alpha)
        blended.append(f.astype(np.uint8))

    return list(a[:-fade]) + blended + list(b[fade:])


def stitch_scenes(
    clips: Sequence[Path],
    output_path: Path,
    fps: int = 8,
    fade_frames: int = FADE_FRAMES,
) -> Path:
    """Concatenate clips with cross-dissolves and write a single MP4.

    Parameters
    ----------
    clips       : ordered list of scene .mp4 paths
    output_path : destination file
    fps         : output frame rate (should match the clips)
    fade_frames : number of dissolve frames between each pair of clips

    Returns
    -------
    Path to the written output file.
    """
    if not clips:
        raise ValueError("stitch_scenes: no clips provided")

    clips = [Path(c) for c in clips]
    for c in clips:
        if not c.exists():
            raise FileNotFoundError(f"Clip not found: {c}")

    print(f"\n[stitch] Joining {len(clips)} scene(s) with "
          f"{fade_frames}-frame cross-dissolve …")

    # Read all clips
    all_clip_frames: list[list[np.ndarray]] = []
    for i, clip in enumerate(clips, start=1):
        frames = _read_frames(clip)
        print(f"[stitch]   Scene {i:>2}: {len(frames)} frames  ({clip.name})")
        all_clip_frames.append(frames)

    # Merge with dissolves
    merged: list[np.ndarray] = all_clip_frames[0]
    for next_frames in all_clip_frames[1:]:
        merged = _crossfade(merged, next_frames, fade_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_mp4(merged, output_path, fps)

    duration = len(merged) / fps
    print(f"[stitch] Final video: {len(merged)} frames  "
          f"({duration:.1f}s @ {fps} fps)  → {output_path.resolve()}")
    return output_path


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    try:
        _write_imageio(frames, path, fps)
    except Exception as exc:
        print(f"[stitch] imageio failed ({exc}), trying OpenCV …")
        _write_opencv(frames, path, fps)


def _write_imageio(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for frame in frames:
        arr = frame if frame.ndim == 3 else np.stack([frame] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        writer.append_data(arr.astype(np.uint8))
    writer.close()


def _write_opencv(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import cv2
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
