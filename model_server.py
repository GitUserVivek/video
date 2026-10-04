"""
model_server.py — Persistent model server that keeps the pipeline in GPU/CPU
memory across CLI invocations and across crashes of the main.py client.

Architecture
------------
  ┌─────────────────────────────────────────────────────────────┐
  │  model_server.py  (long-lived daemon process)               │
  │                                                             │
  │   GPU: cogvideox-5b / ltx-video loaded once, stays in VRAM │
  │   Listens on a Unix domain socket (or TCP on Windows)       │
  │   Handles one request at a time (generation is serial)      │
  └──────────────────┬──────────────────────────────────────────┘
                     │  JSON over socket
  ┌──────────────────┴──────────────────────────────────────────┐
  │  main.py  (short-lived client process)                      │
  │  Auto-starts server on first run.  Just sends the prompt    │
  │  and streams back log lines + final output path.            │
  └─────────────────────────────────────────────────────────────┘

Protocol (newline-delimited JSON)
----------------------------------
Client → Server  (single message):
  {"action": "generate", "prompt": "...", "model": "...", ...all generate_video kwargs}
  {"action": "release"}    — free GPU memory, keep server alive
  {"action": "status"}     — returns current state as JSON
  {"action": "shutdown"}   — stop the server

Server → Client  (stream of messages until "done" or "error"):
  {"type": "log",    "text": "..."}          — forwarded stdout line
  {"type": "done",   "output_path": "..."}   — generation finished
  {"type": "status", ...state dict...}       — response to status query
  {"type": "error",  "text": "..."}          — unrecoverable error

Socket path
-----------
  Linux / macOS : /tmp/ai_video_server.sock   (Unix socket)
  Windows       : TCP 127.0.0.1:57321         (fallback)

PID file
--------
  /tmp/ai_video_server.pid  — lets clients detect stale sockets and the
  server detect duplicate launches.

Usage
-----
  # Start the server (blocks; run in a background terminal or via nohup):
  python model_server.py

  # Or let main.py start it automatically (--no-server disables this).

  # Release GPU memory without stopping:
  python main.py --release-resources

  # Full shutdown:
  python main.py --reset          # release + stop server
"""

from __future__ import annotations

import gc
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import torch

# ── Socket / PID paths ────────────────────────────────────────────────────────

def _socket_path() -> str:
    """Unix socket path (or empty string on Windows → use TCP)."""
    if sys.platform == "win32":
        return ""
    return os.environ.get("AI_VIDEO_SOCK", "/tmp/ai_video_server.sock")


def _pid_path() -> Path:
    return Path(os.environ.get("AI_VIDEO_PID", "/tmp/ai_video_server.pid"))


TCP_HOST = "127.0.0.1"
TCP_PORT = int(os.environ.get("AI_VIDEO_PORT", "57321"))

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[server] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("model_server")


# ── Global state (lives for the lifetime of the server process) ───────────────

class ServerState:
    def __init__(self) -> None:
        self.pipe:        Any   = None
        self.model_id:    str   = ""
        self.hw_cfg:      dict  = {}
        self.model_info:  dict  = {}
        self.loaded:      bool  = False
        self.busy:        bool  = False
        self.lock:        threading.Lock = threading.Lock()

    def as_dict(self) -> dict:
        vram = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = (v / 1e9 for v in torch.cuda.mem_get_info(i))
                vram[f"gpu{i}"] = {"free_gb": round(free, 2), "total_gb": round(total, 2)}
        return {
            "loaded":   self.loaded,
            "busy":     self.busy,
            "model_id": self.model_id,
            "device":   self.hw_cfg.get("device", "unknown"),
            "vram":     vram,
            "pid":      os.getpid(),
        }


STATE = ServerState()


# ── Log interceptor (captures print() → forwards to client socket) ────────────

class _SocketWriter:
    """Writes lines to a connected client socket as {"type":"log","text":"..."} JSON."""

    def __init__(self, sock: socket.socket, original_stdout) -> None:
        self._sock     = sock
        self._orig     = original_stdout
        self._buf      = ""
        self._closed   = False

    def write(self, text: str) -> int:
        if self._orig:
            self._orig.write(text)
            self._orig.flush()
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._send_log(line)
        return len(text)

    def flush(self) -> None:
        if self._orig:
            self._orig.flush()
        if self._buf:
            self._send_log(self._buf)
            self._buf = ""

    def _send_log(self, line: str) -> None:
        if self._closed:
            return
        try:
            _send_msg(self._sock, {"type": "log", "text": line})
        except OSError:
            self._closed = True

    def fileno(self):          # needed by some libraries that check sys.stdout.fileno()
        return self._orig.fileno() if self._orig else 1

    # Make it look enough like a real file for tqdm / diffusers progress bars
    @property
    def encoding(self):
        return getattr(self._orig, "encoding", "utf-8")

    def isatty(self):
        return False


# ── Socket helpers ────────────────────────────────────────────────────────────

def _send_msg(sock: socket.socket, obj: dict) -> None:
    data = (json.dumps(obj) + "\n").encode()
    sock.sendall(data)


def _recv_msg(sock: socket.socket) -> dict | None:
    """Read one newline-terminated JSON message from a socket."""
    buf = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            return None
        buf += chunk
        if b"\n" in buf:
            line, _ = buf.split(b"\n", 1)
            return json.loads(line.decode())


# ── Action handlers ───────────────────────────────────────────────────────────

def _handle_status(sock: socket.socket) -> None:
    _send_msg(sock, {"type": "status", **STATE.as_dict()})


def _handle_release(sock: socket.socket) -> None:
    """Free GPU memory but keep the server process alive."""
    log.info("Release requested — unloading pipeline …")
    with STATE.lock:
        if STATE.pipe is not None:
            _do_release()
        else:
            log.info("Pipeline already unloaded.")
    _send_msg(sock, {"type": "log",  "text": "[server] GPU memory released. Server still running."})
    _send_msg(sock, {"type": "done", "output_path": ""})


def _do_release() -> None:
    """Actually delete the pipeline and reclaim GPU memory. Call with STATE.lock held."""
    try:
        device = STATE.hw_cfg.get("device", "cpu")
        if device == "cuda" and not STATE.hw_cfg.get("use_device_map", False):
            try:
                STATE.pipe.to("cpu")
            except Exception:
                pass
        del STATE.pipe
        STATE.pipe   = None
        STATE.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            lines = []
            for i in range(torch.cuda.device_count()):
                free = torch.cuda.mem_get_info(i)[0] / 1e9
                lines.append(f"GPU {i}: {free:.1f} GB free")
            log.info("Memory released. %s", " | ".join(lines))
        else:
            log.info("Memory released (CPU mode).")
    except Exception as exc:
        log.warning("Error during release: %s", exc)


def _ensure_loaded(model_id: str | None) -> None:
    """Load (or reload) the pipeline if not already in memory. STATE.lock must be held."""
    if STATE.loaded and STATE.pipe is not None:
        # If a specific model is requested and it differs from the loaded one, reload
        if model_id and model_id != STATE.model_id:
            log.info("Model change requested (%s → %s) — reloading …",
                     STATE.model_id, model_id)
            _do_release()
        else:
            return

    from hardware import detect_device
    from model import load_pipeline, select_model

    if not STATE.hw_cfg:
        STATE.hw_cfg = detect_device()

    # Resolve model_id: explicit arg > AI_VIDEO_MODEL env > auto-select.
    # Auto-select is done AFTER hardware detection so vram_gb is always valid.
    if model_id is None:
        model_id = os.environ.get("AI_VIDEO_MODEL") or None
    if model_id is None:
        model_id = select_model(STATE.hw_cfg)
        log.info("Auto-selected model: %s (vram_gb=%.1f)",
                 model_id, STATE.hw_cfg.get("vram_gb", 0))

    log.info("Loading pipeline: %s …", model_id)
    t0 = time.time()
    pipe, model_info = load_pipeline(model_id, STATE.hw_cfg)
    STATE.pipe       = pipe
    STATE.model_info = model_info
    STATE.model_id   = model_info.get("model_id") or model_id or ""
    STATE.loaded     = True
    log.info("Pipeline loaded in %.1fs", time.time() - t0)


def _handle_generate(sock: socket.socket, req: dict) -> None:
    """Load model if needed, run generation, stream log lines back to client."""
    orig_stdout = sys.stdout
    writer = _SocketWriter(sock, orig_stdout)
    sys.stdout = writer  # type: ignore[assignment]

    try:
        with STATE.lock:
            if STATE.busy:
                _send_msg(sock, {"type": "error",
                                 "text": "Server is busy with another generation."})
                return
            STATE.busy = True

        try:
            model_id = req.get("model")
            with STATE.lock:
                _ensure_loaded(model_id)

            from generator import generate_video
            output_path = generate_video(
                pipe             = STATE.pipe,
                model_info       = STATE.model_info,
                hw_cfg           = STATE.hw_cfg,
                prompt           = req["prompt"],
                duration_sec     = req.get("duration_sec"),
                fps              = req.get("fps"),
                resolution       = req.get("resolution"),
                seed             = req.get("seed"),
                num_inference_steps = req.get("num_inference_steps"),
                guidance_scale   = req.get("guidance_scale"),
                output_path      = req.get("output_path"),
                stream           = req.get("stream", False),
                preview_every    = req.get("preview_every", 0),
                segment_seconds  = req.get("segment_seconds"),
                cache_dir        = req.get("cache_dir"),
                resume           = req.get("resume", True),
            )
            writer.flush()
            _send_msg(sock, {"type": "done", "output_path": str(output_path)})

        except Exception as exc:
            tb = traceback.format_exc()
            writer.flush()
            _send_msg(sock, {"type": "error", "text": f"{exc}\n{tb}"})
        finally:
            with STATE.lock:
                STATE.busy = False

    finally:
        sys.stdout = orig_stdout


# ── Connection handler ────────────────────────────────────────────────────────

def _handle_connection(conn: socket.socket, addr: Any) -> None:
    log.info("Client connected: %s", addr)
    try:
        req = _recv_msg(conn)
        if req is None:
            return
        action = req.get("action", "generate")

        if action == "generate":
            _handle_generate(conn, req)
        elif action == "release":
            _handle_release(conn)
        elif action == "status":
            _handle_status(conn)
        elif action == "shutdown":
            _send_msg(conn, {"type": "log",  "text": "[server] Shutting down …"})
            _send_msg(conn, {"type": "done", "output_path": ""})
            conn.close()
            log.info("Shutdown requested — exiting.")
            _cleanup_pid()
            os.kill(os.getpid(), signal.SIGTERM)
        elif action == "force_shutdown":
            # Hard kill — works even when busy (generation will be interrupted)
            _send_msg(conn, {"type": "log",  "text": "[server] Force shutdown — killing process."})
            _send_msg(conn, {"type": "done", "output_path": ""})
            conn.close()
            log.info("Force shutdown requested — killing now.")
            _cleanup_pid()
            os.kill(os.getpid(), signal.SIGKILL)
        else:
            _send_msg(conn, {"type": "error", "text": f"Unknown action: {action}"})
    except Exception as exc:
        log.error("Connection error: %s", exc)
        try:
            _send_msg(conn, {"type": "error", "text": str(exc)})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    log.info("Client disconnected: %s", addr)


# ── Server main loop ──────────────────────────────────────────────────────────

def _make_server_socket() -> socket.socket:
    sock_path = _socket_path()
    if sock_path:
        # Remove stale socket file
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        os.chmod(sock_path, 0o600)   # owner-only access
        log.info("Listening on Unix socket: %s", sock_path)
    else:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((TCP_HOST, TCP_PORT))
        log.info("Listening on TCP %s:%d", TCP_HOST, TCP_PORT)
    srv.listen(4)
    srv.settimeout(1.0)   # allow periodic shutdown checks
    return srv


def _write_pid() -> None:
    _pid_path().write_text(str(os.getpid()))


def _cleanup_pid() -> None:
    try:
        _pid_path().unlink(missing_ok=True)
    except Exception:
        pass


def _preload_model() -> None:
    """Pre-load the model at server startup.

    Only runs if AI_VIDEO_MODEL is explicitly set — never auto-selects,
    because the server process may start before CUDA is fully initialised
    and select_model() would return 'ltx-video' (vram_gb=0 fallback),
    triggering an unwanted 39 GB download.

    If AI_VIDEO_MODEL is not set, skip preload and wait for the first
    generation request (which always carries an explicit model_id from
    the client-side select_model() call).
    """
    model_id = os.environ.get("AI_VIDEO_MODEL")
    if not model_id:
        log.info("Preload skipped — model will load on first request.")
        return
    if os.environ.get("AI_VIDEO_NO_PRELOAD"):
        log.info("Preload skipped (AI_VIDEO_NO_PRELOAD set).")
        return
    try:
        with STATE.lock:
            _ensure_loaded(model_id)
    except Exception as exc:
        log.warning("Preload failed: %s — will load on first request.", exc)


def run_server(preload: bool = True) -> None:
    """Start the model server and block until shutdown."""
    _write_pid()
    log.info("Model server started  PID=%d", os.getpid())

    # Graceful shutdown on SIGTERM / SIGINT
    _shutdown = threading.Event()

    def _sig_handler(sig, frame):
        log.info("Signal %d received — shutting down …", sig)
        _shutdown.set()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT,  _sig_handler)

    if preload:
        preload_thread = threading.Thread(target=_preload_model, daemon=True,
                                          name="Preloader")
        preload_thread.start()

    srv = _make_server_socket()

    try:
        while not _shutdown.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            # Each connection is handled in its own thread so status/release
            # requests are not blocked by an ongoing generation.
            t = threading.Thread(
                target=_handle_connection,
                args=(conn, addr),
                daemon=True,
                name=f"client-{addr}",
            )
            t.start()
    finally:
        srv.close()
        sock_path = _socket_path()
        if sock_path:
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass
        _cleanup_pid()
        log.info("Server stopped.")


# ── Public helpers used by main.py / server_client.py ────────────────────────

def server_socket_path() -> str:
    return _socket_path()


def server_pid_path() -> Path:
    return _pid_path()


def server_tcp_addr() -> tuple[str, int]:
    return TCP_HOST, TCP_PORT


def is_server_running() -> bool:
    """Return True if a server process is alive and its socket is reachable."""
    pid_file = _pid_path()
    if not pid_file.exists():
        return False

    # Check that the PID is still alive
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)   # signal 0 = existence check, raises if dead
    except (ValueError, ProcessLookupError, PermissionError):
        pid_file.unlink(missing_ok=True)
        return False

    # Try connecting
    try:
        _connect_socket().close()
        return True
    except OSError:
        return False


def _connect_socket() -> socket.socket:
    """Return a connected socket (Unix or TCP). Raises OSError if unreachable."""
    sock_path = _socket_path()
    if sock_path:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_HOST, TCP_PORT))
    return s


def connect_to_server(timeout: float = 5.0) -> socket.socket:
    """Connect to a running server socket with retries. Raises RuntimeError on failure."""
    deadline = time.monotonic() + timeout
    last_exc: Exception = OSError("no attempt made")
    while time.monotonic() < deadline:
        try:
            return _connect_socket()
        except OSError as exc:
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(
        f"Could not connect to model server after {timeout:.0f}s: {last_exc}\n"
        f"Start the server with:  python model_server.py"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Persistent AI Video model server")
    ap.add_argument("--no-preload", action="store_true",
                    help="Do not load the model at startup — load on first request instead.")
    ap.add_argument("--model", default=None,
                    help="Model to preload (e.g. cogvideox-5b). Default: auto-select.")
    args = ap.parse_args()

    if args.model:
        os.environ["AI_VIDEO_MODEL"] = args.model
    if args.no_preload:
        os.environ["AI_VIDEO_NO_PRELOAD"] = "1"

    run_server(preload=not args.no_preload)
