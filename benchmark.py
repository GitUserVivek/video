"""
benchmark.py — Side-by-side timing / memory report for the model paths.

The estimates people quote for diffusion video are unreliable because cost scales
with tokens² and with the number of denoising steps. So this module *measures*:

  * load time (including multi-GPU sharding)
  * seconds per denoising step (median of N probed steps, first step reported too
    since it carries the kernel warm-up)
  * VAE decode time (measured as the tail after the last step)
  * peak VRAM per GPU (to show whether both GPUs actually did work)

...then projects each path to real clip lengths using the same segment planner the
generator uses, so "10 s" and "30 s" mean the same thing for both models.

Usage
-----
    python main.py --benchmark
    python main.py --benchmark --benchmark-steps 3
    python main.py --benchmark --benchmark-models cogvideox-2b,ltx-video

or from Python:

    from benchmark import run_benchmark
    run_benchmark(hw_cfg, models=["cogvideox-2b", "ltx-video-distilled"])
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass, field

import torch

from generator import (
    MAX_OUTPUT_FRAMES,
    RESOLUTIONS,
    _clamp_frames,
    _model_default,
    _plan_segments,
)

DEFAULT_MODELS: tuple[str, ...] = ("cogvideox-2b", "ltx-video-distilled")
DEFAULT_PROBE_STEPS = 3
CLIP_TARGETS: tuple[tuple[str, float, int | None], ...] = (
    # (label, duration seconds, fps override — None = the model's own rate)
    ("10 s", 10.0, None),
    ("30 s", 30.0, None),
    ("30 s @ 8 fps", 30.0, 8),
    ("10 s @ 8 fps", 10.0, 8),
)


@dataclass
class Probe:
    """Everything measured for one model path."""

    model_id: str
    probe_steps: int
    schedule_steps: int
    guidance: float
    fps: int
    resolution: str
    width: int
    height: int
    pass_frames: int
    load_seconds: float = 0.0
    step_seconds: list[float] = field(default_factory=list)
    decode_seconds: float = 0.0
    peak_vram_gb: list[float] = field(default_factory=list)
    dtype: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.step_seconds)

    @property
    def s_per_step(self) -> float:
        """Median step time — robust against the slow first step."""
        return statistics.median(self.step_seconds) if self.step_seconds else 0.0

    @property
    def first_step_s(self) -> float:
        return self.step_seconds[0] if self.step_seconds else 0.0

    def pass_seconds(self) -> float:
        """One full generation pass of this model's own schedule."""
        return self.s_per_step * self.schedule_steps + self.decode_seconds

    def project(self, duration_sec: float, fps: int | None = None) -> dict:
        """Project a clip of `duration_sec` onto this path (streaming passes)."""
        rate   = fps or self.fps
        target = max(1, min(MAX_OUTPUT_FRAMES, int(round(duration_sec * rate))))
        passes = len(_plan_segments(target, self.pass_frames, True, self.model_id))
        per    = self.pass_seconds()
        return {
            "fps":     rate,
            "frames":  target,
            "passes":  passes,
            "first":   per,
            "total":   per * passes,
            "native":  target <= self.pass_frames,
        }


# ── Measurement ───────────────────────────────────────────────────────────────

def _probe(
    hw_cfg: dict,
    model_id: str,
    prompt: str,
    probe_steps: int,
    resolution: str,
) -> Probe:
    """Load a model and time `probe_steps` denoising steps + one VAE decode."""
    from model import load_pipeline

    width, height = RESOLUTIONS[resolution]
    probe = Probe(
        model_id=model_id,
        probe_steps=probe_steps,
        schedule_steps=0,
        guidance=0.0,
        fps=0,
        resolution=resolution,
        width=width,
        height=height,
        pass_frames=0,
        dtype=str(hw_cfg.get("dtype", "")),
    )

    pipe = None
    try:
        print(f"\n[bench] Loading {model_id} …")
        t0 = time.monotonic()
        pipe, info = load_pipeline(model_id, hw_cfg)
        probe.load_seconds = time.monotonic() - t0

        probe.schedule_steps = int(_model_default(info, "default_steps", 50))
        probe.guidance       = float(_model_default(info, "default_guidance", 6.0))
        probe.fps            = int(_model_default(info, "default_fps", 8))
        probe.pass_frames    = _clamp_frames(
            int(_model_default(info, "max_native_frames", 49)), model_id
        )
        probe.model_id = info.get("model_id", model_id)

        steps = max(1, min(probe_steps, probe.schedule_steps))
        marks: list[float] = []

        def _mark(_pipe, _step, _timestep, callback_kwargs):
            marks.append(time.monotonic())
            return callback_kwargs

        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)

        print(f"[bench] Probing {steps} step(s) of a {probe.pass_frames}-frame pass "
              f"at {width}×{height} with {probe.model_id} "
              f"({probe.schedule_steps}-step schedule, guidance {probe.guidance}) …")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

        t_start = time.monotonic()
        with torch.inference_mode():
            pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality",
                num_frames=probe.pass_frames,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=probe.guidance,
                generator=gen,
                callback_on_step_end=_mark,
            )
        t_end = time.monotonic()

        # Step boundaries: first mark − start, then mark-to-mark deltas.
        if marks:
            probe.step_seconds = [
                marks[0] - t_start,
                *[b - a for a, b in zip(marks, marks[1:])],
            ]
            probe.decode_seconds = t_end - marks[-1]

        if torch.cuda.is_available():
            probe.peak_vram_gb = [
                torch.cuda.max_memory_allocated(i) / 1e9
                for i in range(torch.cuda.device_count())
            ]

    except Exception as exc:                                   # noqa: BLE001
        probe.error = f"{type(exc).__name__}: {exc}"
        print(f"[bench] {model_id} failed: {probe.error}")

    finally:
        # Drop the reference before the next model so both fit in 31 GB.
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return probe


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _vram_str(probe: Probe) -> str:
    if not probe.peak_vram_gb:
        return "—"
    return " / ".join(f"{v:.1f}" for v in probe.peak_vram_gb) + " GB"


def _print_report(probes: list[Probe], resolution: str) -> None:
    width, height = RESOLUTIONS[resolution]

    print("\n" + "─" * 100)
    print(f"  Timing report — {resolution}p ({width}×{height})   "
          f"dtype={probes[0].dtype}   (VRAM column lists every GPU)")
    print("─" * 100)

    head = (f"  {'path':<24} {'steps':>5} {'s/step':>8} {'1st step':>9} "
            f"{'decode':>7} {'pass':>9} {'frames':>7} {'load':>6} {'peak VRAM':>16}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    for probe in probes:
        if not probe.ok:
            print(f"  {probe.model_id:<24} FAILED — {probe.error}")
            continue
        print(f"  {probe.model_id:<24} {probe.schedule_steps:>5} "
              f"{probe.s_per_step:>7.2f}s {probe.first_step_s:>8.2f}s "
              f"{probe.decode_seconds:>6.1f}s {_fmt(probe.pass_seconds()):>9} "
              f"{probe.pass_frames:>7} {probe.load_seconds:>5.0f}s "
              f"{_vram_str(probe):>16}")

    usable = [p for p in probes if p.ok]
    if not usable:
        print("\n  Nothing to compare — every path failed to run.\n"
              + "─" * 100)
        return

    # ── Projected clip times ──────────────────────────────────────────────
    print(f"\n  Measured with {usable[0].probe_steps} probed step(s); projected to whole "
          f"clips using the same per-pass planner as generation.")
    print(f"  (Pass sizes: the model's native limit — "
          f"{', '.join(f'{p.model_id}={p.pass_frames}f' for p in usable)})")

    label_w = max(len(lbl) for lbl, _, _ in CLIP_TARGETS) + 2
    col_w   = max(24, max(len(p.model_id) for p in usable) + 12)
    print(f"\n  {'clip':<{label_w}}{''.join(p.model_id.ljust(col_w) for p in usable)}")
    print("  " + "-" * (label_w + col_w * len(usable)))

    for label, duration, fps_override in CLIP_TARGETS:
        row = f"  {label:<{label_w}}"
        for probe in usable:
            proj = probe.project(duration, fps_override)
            detail = (f"{_fmt(proj['total'])}  ({proj['passes']}× pass of "
                      f"{_fmt(proj['first'])})")
            row += detail.ljust(col_w)
        print(row)

    # ── First watchable output + verdict ──────────────────────────────────
    print(f"\n  First watchable piece (streaming): "
          + ", ".join(f"{p.model_id} ≈ {_fmt(p.pass_seconds())}" for p in usable))

    if len(usable) >= 2:
        fastest, slowest = min(usable, key=lambda p: p.pass_seconds()), \
                           max(usable, key=lambda p: p.pass_seconds())
        ratio = slowest.pass_seconds() / fastest.pass_seconds() \
            if fastest.pass_seconds() else float("inf")
        print(f"  Verdict: {fastest.model_id} is ~{ratio:.0f}× faster per pass than "
              f"{slowest.model_id} at {resolution}p — plus it needs "
              f"{fastest.schedule_steps} vs {slowest.schedule_steps} steps per pass.")

    print("\n  Not included: retiming/MP4 encoding (≈1-5 s per 100 output frames), "
          "model download time.")
    print("─" * 100 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_benchmark(
    hw_cfg: dict,
    models: list[str] | tuple[str, ...] | None = None,
    steps: int = DEFAULT_PROBE_STEPS,
    resolution: str = "480",
    prompt: str = "a 480p sunset timelapse, high quality, cinematic lighting",
) -> list[Probe]:
    """Probe each path and print the side-by-side report. Returns the raw probes."""
    from model import MODELS

    wanted = list(models or DEFAULT_MODELS)
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Choose from: {list(MODELS)}")

    print(f"\n[bench] Probing {len(wanted)} path(s) at {resolution}p, "
          f"{steps} step(s) each — expect a few minutes "
          f"(model loads dominate).")

    probes = [
        _probe(hw_cfg, model_id, prompt, steps, resolution)
        for model_id in wanted
    ]
    _print_report(probes, resolution)
    return probes
