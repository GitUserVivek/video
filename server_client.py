"""
server_client.py — Thin client for model_server.py.

Sends a generation request (or a control command) to the running server,
streams log lines back to stdout in real time, and returns the output path.

Used by main.py when the persistent server mode is active.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ── Auto-start helpers ────────────────────────────────────────────────────────

def _start_server_background(model_id: str | None = None) -> None:
    """Spawn model_server.py as a detached background process."""
    script = Path(__file__).parent / "model_server.py"
    cmd = [sys.executable, str(script)]
    if model_id:
        cmd += ["--model", model_id]

    # nohup-style: detach from current terminal, redirect output to a log file
    log_path = Path(os.environ.get("AI_VIDEO_SERVER_LOG",
                                   "/tmp/ai_video_server.log"))
    log_file = open(log_path, "a")  # noqa: WPS515 (intentional persistent handle)

    kwargs: dict[str, Any] = dict(
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,
    )

    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True   # detach from parent's process group

    proc = subprocess.Popen(cmd, **kwargs)
    print(f"[client] Server started (PID {proc.pid})  log → {log_path}")


def ensure_server_running(
    model_id: str | None = None,
    startup_timeout: float = 120.0,
) -> None:
    """Auto-start the server if it isn't running, then wait until it's reachable."""
    from model_server import is_server_running, connect_to_server

    if is_server_running():
        return

    print("[client] Model server not running — starting it now …")
    print(f"[client] The model will stay in GPU memory between runs.\n"
          f"[client] Use  python main.py --release-resources  to free GPU memory.\n"
          f"[client] Use  python main.py --reset              to stop the server.\n")

    _start_server_background(model_id)

    # Wait for the socket to become available
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        try:
            connect_to_server(timeout=2.0).close()
            print("[client] Server is ready.\n")
            return
        except Exception:
            time.sleep(1.0)

    raise RuntimeError(
        f"Server did not become ready within {startup_timeout:.0f}s.\n"
        f"Check the log: /tmp/ai_video_server.log"
    )


# ── Core send/receive ─────────────────────────────────────────────────────────

def _send_and_stream(
    req: dict,
    connect_timeout: float = 10.0,
) -> dict:
    """Send a request to the server and stream log lines to stdout.

    Returns the final message dict (type == "done" or "error").
    """
    from model_server import connect_to_server

    sock = connect_to_server(timeout=connect_timeout)
    try:
        # Send request
        data = (json.dumps(req) + "\n").encode()
        sock.sendall(data)

        # Stream response lines
        buf = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError("Server closed connection unexpectedly.")
            buf += chunk
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                if not line_bytes.strip():
                    continue
                try:
                    msg = json.loads(line_bytes.decode())
                except json.JSONDecodeError:
                    # Raw text — just print it
                    print(line_bytes.decode(), end="")
                    continue

                if msg.get("type") == "log":
                    print(msg.get("text", ""))
                elif msg.get("type") in ("done", "error", "status"):
                    return msg
    finally:
        sock.close()


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    model: str | None = None,
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
    cache_dir: str | None = None,
    resume: bool = True,
) -> Path:
    """Send a generation request to the server and return the output Path."""
    req: dict[str, Any] = {"action": "generate", "prompt": prompt}
    # Only include non-None values so server uses its own defaults
    _opt = dict(
        model=model,
        duration_sec=duration_sec,
        fps=fps,
        resolution=resolution,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_path=output_path,
        stream=stream,
        preview_every=preview_every,
        segment_seconds=segment_seconds,
        cache_dir=cache_dir,
        resume=resume,
    )
    req.update({k: v for k, v in _opt.items() if v is not None})

    result = _send_and_stream(req)
    if result.get("type") == "error":
        raise RuntimeError(f"Server error:\n{result.get('text', '(no details)')}")

    out = result.get("output_path", "")
    if not out:
        raise RuntimeError("Server returned no output path.")
    return Path(out)


def release_resources() -> None:
    """Tell the server to free GPU memory (pipeline unloaded, server stays alive)."""
    result = _send_and_stream({"action": "release"})
    if result.get("type") == "error":
        raise RuntimeError(result.get("text", "release failed"))


def shutdown_server() -> None:
    """Tell the server to stop completely."""
    try:
        _send_and_stream({"action": "shutdown"})
    except Exception:
        pass   # server may close the socket before sending done


def get_status() -> dict:
    """Return the server's current status dict."""
    result = _send_and_stream({"action": "status"})
    return result


def print_status() -> None:
    """Print a human-readable server status."""
    from model_server import is_server_running
    if not is_server_running():
        print("[client] Model server is NOT running.")
        return
    try:
        s = get_status()
        loaded = s.get("loaded", False)
        busy   = s.get("busy", False)
        mid    = s.get("model_id") or "none"
        pid    = s.get("pid", "?")
        device = s.get("device", "?").upper()
        vram   = s.get("vram", {})

        print(f"\n── Model Server Status ─────────────────────────────────")
        print(f"  PID             : {pid}")
        print(f"  Device          : {device}")
        print(f"  Model loaded    : {'yes  (' + mid + ')' if loaded else 'no'}")
        print(f"  Busy            : {'yes (generating)' if busy else 'no'}")
        if vram:
            for gpu, info in vram.items():
                print(f"  {gpu.upper()} memory   : "
                      f"{info['free_gb']:.1f} / {info['total_gb']:.1f} GB free")
        print(f"────────────────────────────────────────────────────────\n")
    except Exception as exc:
        print(f"[client] Could not get status: {exc}")
