"""
screenplay.py — Scene script parser for multi-scene video generation.

Script format
-------------
Lines starting with # are comments and ignored.
Each scene is one non-empty line in one of these formats:

    [scene N, Xs]  Visual description of the scene
    [scene N, Xs, guidance G]  Same with custom guidance scale
    [scene N, Xs, steps S]     Same with custom step count

Examples:
    [scene 1, 3s] Nobita running late, Japanese street, morning light, anime style
    [scene 2, 4s] Doraemon pulling out a glowing blue door, cartoon colorful
    [scene 3, 3s, steps 30] Nobita on a sunny beach, surprised, anime style

Short form — just duration, scene number auto-assigned:
    [3s] Nobita running late, Japanese street
    [4s] Doraemon pulling out a glowing blue door

Plain lines with no bracket header — duration defaults to DEFAULT_SCENE_DURATION:
    Nobita running late, Japanese street, morning light
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SCENE_DURATION = 4.0   # seconds per scene when not specified
DEFAULT_STEPS          = None  # inherit from CLI / model default
DEFAULT_GUIDANCE       = None  # inherit from CLI / model default


@dataclass
class Scene:
    index:       int             # 1-based
    duration:    float           # seconds
    prompt:      str             # visual description
    steps:       int | None      # override inference steps (None = use CLI default)
    guidance:    float | None    # override guidance scale (None = use CLI default)
    output_path: Path | None = field(default=None, repr=False)  # filled in at runtime


# ── Regex patterns ────────────────────────────────────────────────────────────

# Full form:  [scene 3, 5s]  or  [scene 3, 5s, steps 30]  or  [scene 3, 5s, guidance 8.0]
_FULL_RE = re.compile(
    r"^\s*\[scene\s+(\d+)\s*,\s*(\d+(?:\.\d+)?)\s*s"     # [scene N, Xs
    r"(?:\s*,\s*steps\s+(\d+))?"                          # optional , steps S
    r"(?:\s*,\s*guidance\s+(\d+(?:\.\d+)?))?"             # optional , guidance G
    r"\]\s*(.+)$",                                        # ] prompt
    re.I,
)

# Short form: [3s] or [3.5s]
_SHORT_RE = re.compile(
    r"^\s*\[(\d+(?:\.\d+)?)\s*s\]\s*(.+)$",
    re.I,
)


def parse_screenplay(path: str | Path) -> list[Scene]:
    """Parse a screenplay file and return a list of Scene objects.

    Raises
    ------
    ValueError  if the file is empty or contains no valid scenes.
    FileNotFoundError  if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Screenplay file not found: {path}")

    text = path.read_text(encoding="utf-8")
    scenes: list[Scene] = []
    auto_index = 0

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Try full form first
        m = _FULL_RE.match(line)
        if m:
            idx      = int(m.group(1))
            duration = float(m.group(2))
            steps    = int(m.group(3)) if m.group(3) else DEFAULT_STEPS
            guidance = float(m.group(4)) if m.group(4) else DEFAULT_GUIDANCE
            prompt   = m.group(5).strip()
            scenes.append(Scene(idx, duration, prompt, steps, guidance))
            continue

        # Try short form
        m = _SHORT_RE.match(line)
        if m:
            auto_index += 1
            duration = float(m.group(1))
            prompt   = m.group(2).strip()
            scenes.append(Scene(auto_index, duration, prompt,
                                DEFAULT_STEPS, DEFAULT_GUIDANCE))
            continue

        # Plain line — no header, auto-assign index and default duration
        auto_index += 1
        scenes.append(Scene(auto_index, DEFAULT_SCENE_DURATION, line,
                            DEFAULT_STEPS, DEFAULT_GUIDANCE))

    if not scenes:
        raise ValueError(f"No valid scenes found in screenplay: {path}")

    # Sort by scene index so out-of-order entries are handled gracefully
    scenes.sort(key=lambda s: s.index)

    return scenes


def print_screenplay(scenes: list[Scene]) -> None:
    """Print a human-readable summary of the parsed screenplay."""
    total = sum(s.duration for s in scenes)
    print(f"\n── Screenplay  ({len(scenes)} scenes, ~{total:.0f}s total) ────────────────")
    for s in scenes:
        opts = []
        if s.steps    is not None: opts.append(f"steps={s.steps}")
        if s.guidance is not None: opts.append(f"guidance={s.guidance}")
        opt_str = f"  [{', '.join(opts)}]" if opts else ""
        print(f"  Scene {s.index:>2}  {s.duration:.0f}s{opt_str}  {s.prompt[:72]}"
              f"{'…' if len(s.prompt) > 72 else ''}")
    print(f"  Total duration: ~{total:.0f}s\n")
