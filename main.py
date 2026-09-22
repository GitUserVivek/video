#!/usr/bin/env python3
"""
main.py — Offline AI Video Generator CLI

Usage
-----
  python main.py "a sunset over the ocean, timelapse"
  python main.py "a cat playing piano" --duration 10 --resolution 1080 --seed 42
  python main.py "drone shot of a city at night" --model cogvideox-5b --steps 50
  python main.py --list-models

Options
-------
  --model         Model ID (auto if omitted).  See --list-models.
  --duration      Video length in seconds [default: prompt hint, else 10]
  --fps           Frames per second [default: prompt hint, else model native]
  --resolution    Output resolution: 480 | 720 | 1080 [default: prompt hint, else 480]
  --seed          Random seed for reproducibility
  --steps         Denoising steps [default: auto]
  --guidance      CFG guidance scale [default: 6.0]
  --output        Output .mp4 file path [default: auto-named]
  --no-compile    Disable torch.compile even when available
  --idle-timeout  Seconds of inactivity before unloading model [default: 180]
  --list-models   Show available models and exit
  --summary       Print hardware summary and exit
"""

from __future__ import annotations

import argparse
import atexit
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-video",
        description="Offline AI video generator — GPU (CUDA/ROCm, multi-GPU) or CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("prompt",
                   nargs="?",
                   default=None,
                   help="Text description of the video to generate.")
    p.add_argument("--model",        default=None,
                   help="Model ID (e.g. cogvideox-2b, cogvideox-5b, ltx-video). "
                        "Auto-selected if omitted.")
    p.add_argument("--duration",     type=float, default=None,
                   help="Target video duration in seconds. Default: whatever the "
                        "prompt asks for (e.g. '10 seconds'), else 10.")
    p.add_argument("--fps",          type=int,   default=None,
                   help="Frames per second. Default: prompt hint, else the model's "
                        "native rate (8 for CogVideoX).")
    p.add_argument("--resolution",   default=None, choices=["480", "720", "1080"],
                   help="Output resolution. Default: prompt hint, else 480p.")
    p.add_argument("--seed",         type=int,   default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--steps",        type=int,   default=None, dest="num_steps",
                   help="Number of denoising steps (default: auto).")
    p.add_argument("--guidance",     type=float, default=6.0,
                   help="Classifier-free guidance scale (default: 6.0).")
    p.add_argument("--output",       default=None,
                   help="Output .mp4 path (default: auto-named in current directory).")
    p.add_argument("--no-compile",   action="store_true",
                   help="Disable torch.compile.")
    p.add_argument("--idle-timeout", type=int, default=180, dest="idle_timeout",
                   help="Seconds of inactivity before unloading model from memory "
                        "(default: 180). Set 0 to disable.")
    p.add_argument("--list-models",  action="store_true",
                   help="List available models and exit.")
    p.add_argument("--summary",      action="store_true",
                   help="Print hardware summary and exit.")
    return p


def _list_models() -> None:
    from model import MODELS
    print("\nAvailable models:")
    print(f"  {'ID':<20} {'Size':<8} {'Min VRAM':<12} {'HuggingFace repo'}")
    print(f"  {'─'*20} {'─'*8} {'─'*12} {'─'*40}")
    for mid, info in MODELS.items():
        vram = f"{info['min_vram']} GB" if info['min_vram'] else "—"
        print(f"  {mid:<20} {info['size_label']:<8} {vram:<12} {info['repo_id']}")
    print()


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Informational flags ────────────────────────────────────────────────
    if args.list_models:
        _list_models()
        return 0

    # ── Hardware detection ─────────────────────────────────────────────────
    from hardware import detect_device, print_device_summary
    hw_cfg = detect_device()

    if args.summary:
        print_device_summary(hw_cfg)
        return 0

    if args.no_compile:
        hw_cfg["use_compile"] = False

    print_device_summary(hw_cfg)

    # ── Prompt required from here ──────────────────────────────────────────
    if not args.prompt:
        parser.print_help()
        print("\nError: a prompt is required.\n"
              "  Example: python main.py \"a sunset over the ocean\"")
        return 1

    if args.duration is not None and not (1.0 <= args.duration <= 60.0):
        print(f"Error: --duration must be between 1 and 60 seconds (got {args.duration}).")
        return 1

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"[main] Prompt: \"{args.prompt}\"")
    print(f"[main] Loading model …\n")

    from model import load_pipeline, select_model
    t_load = time.time()
    pipe, model_info = load_pipeline(args.model, hw_cfg)
    model_id = args.model or select_model(hw_cfg)
    print(f"[main] Model loaded in {time.time() - t_load:.1f}s\n")

    # ── Idle watchdog ──────────────────────────────────────────────────────
    guard = None
    if args.idle_timeout > 0:
        from idle_guard import IdleGuard
        guard = IdleGuard(
            pipe=pipe,
            hw_cfg=hw_cfg,
            model_id=model_id,
            idle_seconds=args.idle_timeout,
        )
        guard.start()
        print(f"[main] Idle watchdog active — model unloads after "
              f"{args.idle_timeout}s of inactivity\n")

        # Ensure clean shutdown even on Ctrl+C or exception
        def _cleanup() -> None:
            if guard is not None:
                print("\n[main] Shutting down — releasing all resources …")
                guard.stop()
        atexit.register(_cleanup)

    # ── Generate ───────────────────────────────────────────────────────────
    from generator import generate_video

    # Ping the guard to mark activity start
    if guard:
        guard.ping()
        active_pipe = guard.get_pipe()
    else:
        active_pipe = pipe

    output_path = generate_video(
        pipe            = active_pipe,
        model_info      = model_info,
        hw_cfg          = hw_cfg,
        prompt          = args.prompt,
        duration_sec    = args.duration,
        fps             = args.fps,
        resolution      = args.resolution,
        seed            = args.seed,
        num_inference_steps = args.num_steps,
        guidance_scale  = args.guidance,
        output_path     = args.output,
    )

    # Ping again after generation so the idle clock resets
    if guard:
        guard.ping()

    print(f"\n✓ Done!  Video saved to: {output_path.resolve()}\n")

    if guard:
        print(f"[main] Model will be unloaded automatically after "
              f"{args.idle_timeout}s of inactivity.\n"
              f"       Press Ctrl+C to exit and release resources immediately.\n")
        # Keep the process alive so the watchdog can fire if the user
        # wants to run another generation interactively.
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[main] Interrupted — releasing resources …")
            guard.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
