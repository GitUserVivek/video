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
  --guidance      CFG guidance scale [default: per-model; 6.0 for CogVideoX]
  --output        Output .mp4 file path [default: auto-named]
  --fast          Fast preset: distilled LTX, 8 steps, no CFG, 24 fps, streaming
  --stream        Save + play each pass as soon as it finishes [--no-stream to disable]
  --segment-secs  Cap a single pass to N seconds of footage
  --preview-every Decode a live preview still every N steps [default: 0 = off]
  --no-compile    Disable torch.compile even when available
  --cache-dir     Where finished passes are cached [default: ./.gen_cache]
  --no-resume     Ignore cached passes and start from pass 1
  --clear-cache   Delete cached runs for this cache dir, then continue
  --benchmark     Measure + compare model paths at one resolution, then exit
  --idle-timeout  Seconds of inactivity before unloading model [default: 180]
  --list-models   Show available models and exit
  --summary       Print hardware summary and exit
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
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
    p.add_argument("--script",
                   default=None,
                   help="Path to a script file containing scene lines. "
                        "Use one line per scene; the file is joined into a single prompt.")
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
    p.add_argument("--guidance",     type=float, default=None,
                   help="Classifier-free guidance scale. Default: the model's own "
                        "value (6.0 for CogVideoX, 1.0 = off for distilled LTX).")
    p.add_argument("--output",       default=None,
                   help="Output .mp4 path (default: auto-named in current directory).")
    p.add_argument("--fast",         action="store_true",
                   help="Fast preset for long clips on small GPUs: distilled LTX-Video, "
                        "8 steps, no CFG, 480p @ 24 fps, streaming output. Tens of "
                        "seconds of video in about a minute on 2x T4.")
    p.add_argument("--stream",       action=argparse.BooleanOptionalAction, default=None,
                   help="Write and play each generated pass as soon as it finishes "
                        "instead of waiting for the whole clip (default: on with "
                        "--fast, off otherwise).")
    p.add_argument("--segment-secs", type=float, default=None, dest="segment_secs",
                   help="Cap a single generation pass to N seconds of footage "
                        "(default: the model's native limit).")
    p.add_argument("--preview-every", type=int, default=0, dest="preview_every",
                   help="Decode and show a live preview still every N denoising steps "
                        "(0 disables; costs ~1-2 s per preview).")
    p.add_argument("--no-compile",   action="store_true",
                   help="Disable torch.compile.")
    p.add_argument("--cache-dir",    default=None, dest="cache_dir",
                   help="Directory holding finished generation passes, so a failed "
                        "run can be resumed [default: ./.gen_cache, or $VIDEO_CACHE_DIR].")
    p.add_argument("--no-resume",    action="store_true", dest="no_resume",
                   help="Ignore cached passes for this run and regenerate everything.")
    p.add_argument("--clear-cache",  action="store_true", dest="clear_cache",
                   help="Delete cached runs in the cache dir before generating.")
    p.add_argument("--idle-timeout", type=int, default=180, dest="idle_timeout",
                   help="Seconds of inactivity before unloading model from memory "
                        "(default: 180). Set 0 to disable.")
    p.add_argument("--benchmark",    action="store_true",
                   help="Measure each model path (s/step, decode, peak VRAM) and print "
                        "a side-by-side projection, then exit. No video is written.")
    p.add_argument("--benchmark-steps", type=int, default=3, dest="benchmark_steps",
                   help="Denoising steps to probe per model (default: 3).")
    p.add_argument("--benchmark-models", default=None, dest="benchmark_models",
                   help="Comma-separated model ids to compare "
                        "(default: cogvideox-2b,ltx-video-distilled).")
    p.add_argument("--list-models",  action="store_true",
                   help="List available models and exit.")
    p.add_argument("--summary",      action="store_true",
                   help="Print hardware summary and exit.")

    # ── Screenplay / scene-by-scene flags ──────────────────────────────────
    p.add_argument("--screenplay",   default=None, metavar="FILE",
                   help="Path to a scene script file. Each line describes one "
                        "scene; clips are generated sequentially then stitched "
                        "into one final video. See examples/doraemon.txt.")
    p.add_argument("--scene-output-dir", default=None, dest="scene_output_dir",
                   help="Directory for per-scene clip files "
                        "(default: <output_stem>_scenes/).")
    p.add_argument("--fade-frames",  type=int, default=4, dest="fade_frames",
                   help="Cross-dissolve length in frames between scenes (default: 4).")

    # ── Persistent server flags ────────────────────────────────────────────
    p.add_argument("--release-resources", action="store_true",
                   dest="release_resources",
                   help="Tell the running model server to unload the pipeline from "
                        "GPU memory, then exit. The server process stays alive so "
                        "the next run does not need to restart it.")
    p.add_argument("--reset",        action="store_true",
                   help="Tell the running model server to release GPU memory AND "
                        "shut down completely, then exit.")
    p.add_argument("--server-status", action="store_true", dest="server_status",
                   help="Print the model server's current status and exit.")
    p.add_argument("--no-server",    action="store_true", dest="no_server",
                   help="Disable the persistent server — load the model inline "
                        "(old behaviour, model is lost on exit/crash).")
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


def _apply_fast_preset(args) -> bool:
    """Fill in --fast defaults for anything the user did not set explicitly.

    Returns the effective `stream` flag (--fast turns it on; --no-stream overrides).
    """
    stream = args.stream
    if args.fast:
        from generator import FAST_PRESET
        res_display = FAST_PRESET['resolution'] or "hw-auto"
        print(f"[main] --fast preset: {FAST_PRESET['model']}, {FAST_PRESET['steps']} steps, "
              f"guidance {FAST_PRESET['guidance']}, {res_display} @ "
              f"{FAST_PRESET['fps']} fps, streaming\n")
        if args.model is None:
            args.model = FAST_PRESET["model"]
            if args.num_steps is None: args.num_steps = FAST_PRESET["steps"]
            if args.guidance is None:  args.guidance = FAST_PRESET["guidance"]
            if args.fps is None:       args.fps = FAST_PRESET["fps"]
        elif args.model != FAST_PRESET["model"]:
            # e.g. --fast --model cogvideox-2b: an 8-step no-CFG schedule on a
            # non-distilled model looks broken, so keep that model's own schedule.
            print(f"[main] --fast with --model {args.model}: keeping its own "
                  f"step/guidance defaults (only resolution + streaming apply)\n")
        # Only apply the preset resolution when it is set AND the user did not
        # specify one explicitly.  When FAST_PRESET["resolution"] is None the
        # generator will fall through to hw_cfg["max_resolution"].
        preset_res = FAST_PRESET["resolution"]
        if args.resolution is None and preset_res is not None:
            args.resolution = preset_res
        if stream is None:          stream = FAST_PRESET["stream"]

    return bool(stream)


def _run_screenplay(args, hw_cfg: dict) -> int:
    """Generate a scene-by-scene video from a screenplay file.

    Each scene is generated as an independent clip (with its own cache entry),
    then all clips are cross-dissolved into one final video.  A failed scene
    can be re-run without regenerating the others — the cache handles it.
    """
    from screenplay import parse_screenplay, print_screenplay
    from stitcher import stitch_scenes
    from generator import generate_scene
    from model import load_pipeline, select_model

    # ── Parse screenplay ───────────────────────────────────────────────────
    try:
        scenes = parse_screenplay(args.screenplay)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print_screenplay(scenes)

    # ── Resolve output paths ───────────────────────────────────────────────
    if args.output:
        final_path = Path(args.output)
    else:
        stem = Path(args.screenplay).stem
        final_path = Path(f"output_{stem}_{int(time.time())}.mp4")

    scene_dir = Path(args.scene_output_dir) if args.scene_output_dir \
                else final_path.parent / f"{final_path.stem}_scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model once for all scenes ────────────────────────────────────
    model_id = args.model or select_model(hw_cfg)
    print(f"[screenplay] Loading model: {model_id} …\n")
    t_load = time.time()
    pipe, model_info = load_pipeline(model_id, hw_cfg)
    print(f"[screenplay] Model ready in {time.time() - t_load:.1f}s\n")

    # ── Resolve shared generation params ──────────────────────────────────
    fps      = args.fps or 8
    seed     = args.seed
    steps    = args.num_steps
    guidance = args.guidance
    res      = args.resolution or hw_cfg.get("max_resolution", "720")

    # ── Generate each scene ────────────────────────────────────────────────
    clip_paths: list[Path] = []
    failed: list[int]      = []

    for scene in scenes:
        try:
            clip = generate_scene(
                scene            = scene,
                pipe             = pipe,
                model_info       = model_info,
                hw_cfg           = hw_cfg,
                output_dir       = scene_dir,
                default_steps    = steps,
                default_guidance = guidance,
                default_resolution = res,
                fps              = fps,
                seed             = seed,
                cache_dir        = args.cache_dir,
                resume           = not args.no_resume,
                preview_every    = args.preview_every,
            )
            clip_paths.append(clip)
            print(f"[screenplay] ✓ Scene {scene.index} → {clip.name}\n")
        except Exception as exc:
            print(f"\n[screenplay] ✗ Scene {scene.index} FAILED: {exc}")
            print(f"[screenplay]   Re-run the same command to retry — "
                  f"completed scenes are cached.\n")
            failed.append(scene.index)

    if not clip_paths:
        print("[screenplay] No scenes completed — nothing to stitch.")
        return 1

    if failed:
        print(f"[screenplay] Warning: {len(failed)} scene(s) failed "
              f"(scenes {failed}). Stitching available clips only.\n")

    # ── Stitch ─────────────────────────────────────────────────────────────
    try:
        final = stitch_scenes(
            clips       = clip_paths,
            output_path = final_path,
            fps         = fps,
            fade_frames = args.fade_frames,
        )
        print(f"\n✓ Done!  Final video ({len(clip_paths)} scenes) → {final.resolve()}\n")
        if failed:
            print(f"  Re-run to fill in missing scenes: {failed}\n")
        return 0
    except Exception as exc:
        print(f"[screenplay] Stitch failed: {exc}")
        return 1


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Informational flags ────────────────────────────────────────────────
    if args.list_models:
        _list_models()
        return 0

    # ── Server control flags (no model loading needed) ─────────────────────
    if args.reset or args.release_resources or args.server_status:
        from model_server import is_server_running, server_pid_path
        from server_client import (
            print_status, release_resources, shutdown_server,
        )
        if not is_server_running():
            print("[main] Model server is not running.")
            return 0

        # Fetch current status once — used by multiple branches below
        try:
            from server_client import get_status
            s = get_status()
        except Exception:
            s = {}

        if args.server_status:
            print_status()
            return 0

        if args.release_resources:
            if s.get("busy"):
                print("[main] Server is busy — cannot release resources while generating.\n"
                      "       Use --reset to force-stop the server instead.")
                return 1
            print("[main] Releasing GPU memory (server stays alive) …")
            release_resources()
            print("[main] Done. Re-run any generation command to reload the model.")
            return 0

        if args.reset:
            pid_file = server_pid_path()
            pid = None
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                except ValueError:
                    pass

            if s.get("busy"):
                print("[main] Server is busy (generation running). Force-killing …")
            else:
                print("[main] Stopping server …")

            # Try graceful shutdown first (works when not busy)
            try:
                shutdown_server()
            except Exception:
                pass

            # If still alive (was busy), kill by PID
            if pid:
                import time as _t
                _t.sleep(1.0)
                try:
                    os.kill(pid, 0)   # check if still alive
                    print(f"[main] Server (PID {pid}) still running — sending SIGKILL …")
                    os.kill(pid, signal.SIGKILL)
                    pid_file.unlink(missing_ok=True)
                except ProcessLookupError:
                    pass  # already dead

            print("[main] Server stopped. GPU memory released.")
            return 0

    stream = _apply_fast_preset(args)

    # ── Hardware detection ─────────────────────────────────────────────────
    from hardware import detect_device, print_device_summary
    hw_cfg = detect_device()

    if args.summary:
        print_device_summary(hw_cfg)
        return 0

    if args.no_compile:
        hw_cfg["use_compile"] = False

    print_device_summary(hw_cfg)

    # ── Cache housekeeping ─────────────────────────────────────────────────
    if args.clear_cache:
        from cache import clear_cache
        clear_cache(args.cache_dir)

    # ── Screenplay mode ────────────────────────────────────────────────────
    if args.screenplay:
        return _run_screenplay(args, hw_cfg)

    # ── Build prompt from positional arg and/or script file ───────────────
    if args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: script file not found: {script_path}")
            return 1
        try:
            text = script_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"Error: cannot read script file {script_path}: {exc}")
            return 1
        # Lines starting with a timestamp-like marker are treated as scene lines.
        # Everything is joined into one prompt, preserving scene order.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            print(f"Error: script file is empty: {script_path}")
            return 1
        args.prompt = "\n".join(lines)
        if not args.prompt:
            print(f"Error: no prompt text found in script file: {script_path}")
            return 1

    # ── Strip timestamp markers from prompt ────────────────────────────────
    # Handles patterns like [00:00], [00:05], [1:30], (00:00), etc. that appear
    # in screenplay-style prompts.  These cause SyntaxErrors when Python tries
    # to evaluate the brackets, and they confuse the model anyway.
    if args.prompt:
        import re as _re
        _TS_RE = _re.compile(r"[\[\(]\d{1,2}:\d{2}[\]\)]\s*")
        cleaned = _TS_RE.sub("", args.prompt)
        if cleaned != args.prompt:
            # Collapse whitespace/newlines left behind, re-join into one sentence
            lines_clean = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
            args.prompt = " ".join(lines_clean)
            print(f"[main] Timestamp markers stripped from prompt.\n"
                  f"[main] Cleaned prompt: {args.prompt[:120]}"
                  f"{'…' if len(args.prompt) > 120 else ''}\n")

    # ── Prompt required from here ──────────────────────────────────────────
    if not args.prompt:
        parser.print_help()
        print("\nError: a prompt is required.\n"
              "  Example: python main.py \"a sunset over the ocean\"")
        return 1

    # ── Benchmark mode: measure instead of generating ──────────────────────
    if args.benchmark:
        from benchmark import run_benchmark
        run_benchmark(
            hw_cfg,
            models=(args.benchmark_models.split(",") if args.benchmark_models else None),
            steps=args.benchmark_steps,
            resolution=args.resolution or "480",
            prompt=args.prompt or "a 480p sunset timelapse, high quality, cinematic lighting",
        )
        return 0

    if args.duration is not None and not (1.0 <= args.duration <= 60.0):
        print(f"Error: --duration must be between 1 and 60 seconds (got {args.duration}).")
        return 1

    print(f"[main] Prompt: \"{args.prompt}\"")

    # ── Persistent server mode (default) ──────────────────────────────────
    if not args.no_server:
        return _run_via_server(args, stream, hw_cfg)

    # ── Inline mode (--no-server) — old behaviour ─────────────────────────
    return _run_inline(args, stream, hw_cfg)


def _run_via_server(args, stream: bool, hw_cfg: dict) -> int:
    """Delegate generation to the persistent model server."""
    from server_client import ensure_server_running, generate
    from model import select_model

    # Resolve model now (in the client process where CUDA is confirmed available)
    # so the server is told exactly which model to load — never auto-selects blind.
    model_id = args.model or select_model(hw_cfg)

    # Auto-start the server if it isn't running yet, passing the resolved model
    ensure_server_running(model_id=model_id)

    try:
        output_path = generate(
            prompt              = args.prompt,
            model               = model_id,
            duration_sec        = args.duration,
            fps                 = args.fps,
            resolution          = args.resolution,
            seed                = args.seed,
            num_inference_steps = args.num_steps,
            guidance_scale      = args.guidance,
            output_path         = args.output,
            stream              = stream,
            preview_every       = args.preview_every,
            segment_seconds     = args.segment_secs,
            cache_dir           = args.cache_dir,
            resume              = not args.no_resume,
        )
        print(f"\n✓ Done!  Video saved to: {output_path.resolve()}\n")
        print("[main] Model stays loaded in GPU memory for the next run.\n"
              "       python main.py --release-resources   — free GPU memory\n"
              "       python main.py --reset               — stop server entirely\n")
        return 0
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        return 1


def _run_inline(args, stream: bool, hw_cfg: dict) -> int:
    """Load model in-process and generate (original behaviour, model lost on exit)."""
    print(f"[main] Loading model …\n")

    from model import load_pipeline, select_model
    t_load = time.time()
    pipe, model_info = load_pipeline(args.model, hw_cfg)
    model_id = model_info.get("model_id") or args.model or select_model(hw_cfg)
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

        def _cleanup() -> None:
            if guard is not None:
                print("\n[main] Shutting down — releasing all resources …")
                guard.stop()
        atexit.register(_cleanup)

    # ── Generate ───────────────────────────────────────────────────────────
    from generator import generate_video

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
        preview_every   = args.preview_every,
        segment_seconds = args.segment_secs,
        cache_dir       = args.cache_dir,
        resume          = not args.no_resume,
    )

    if guard:
        guard.ping()

    print(f"\n✓ Done!  Video saved to: {output_path.resolve()}\n")

    if guard:
        print(f"[main] Model will be unloaded automatically after "
              f"{args.idle_timeout}s of inactivity.\n"
              f"       Press Ctrl+C to exit and release resources immediately.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[main] Interrupted — releasing resources …")
            guard.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
