"""
idle_guard.py — Automatically unload the pipeline from GPU/CPU when idle.

Usage
-----
    from idle_guard import IdleGuard

    guard = IdleGuard(pipe, hw_cfg, idle_seconds=180)
    guard.start()          # starts background watchdog thread

    guard.ping()           # call this whenever generation starts/finishes
    guard.get_pipe()       # returns pipe (reloads if previously unloaded)

    guard.stop()           # clean shutdown (called automatically at exit)

The watchdog thread checks every 30 s. If the pipeline has been idle for
longer than idle_seconds it:
  1. Deletes the pipeline object
  2. Calls torch.cuda.empty_cache() (no-op on CPU)
  3. Runs gc.collect()

On next get_pipe() call the pipeline is reloaded transparently.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class IdleGuard:
    """Background watchdog that unloads the pipeline after idle_seconds of inactivity."""

    CHECK_INTERVAL = 30   # seconds between idle checks

    def __init__(
        self,
        pipe: Any,
        hw_cfg: dict,
        model_id: str,
        idle_seconds: int = 180,
    ) -> None:
        self._pipe         = pipe
        self._hw_cfg       = hw_cfg
        self._model_id     = model_id
        self._idle_seconds = idle_seconds
        self._last_used    = time.monotonic()
        self._lock         = threading.Lock()
        self._stop_event   = threading.Event()
        self._thread: threading.Thread | None = None
        self._unloaded     = False

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background watchdog thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watchdog,
            name="IdleGuard",
            daemon=True,          # won't block process exit
        )
        self._thread.start()
        logger.debug(
            "[idle_guard] Watchdog started — unload after %ds idle", self._idle_seconds
        )

    def stop(self) -> None:
        """Stop the watchdog thread and release all resources immediately."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._release()

    def ping(self) -> None:
        """Reset the idle timer. Call before and after each generation."""
        self._last_used = time.monotonic()
        if self._unloaded:
            logger.debug("[idle_guard] Ping received but pipeline is unloaded — will reload on get_pipe()")

    def get_pipe(self) -> Any:
        """Return the pipeline, reloading from cache if it was unloaded."""
        with self._lock:
            if self._unloaded:
                self._reload()
            self._last_used = time.monotonic()
            return self._pipe

    @property
    def is_loaded(self) -> bool:
        return not self._unloaded

    # ── Internal ───────────────────────────────────────────────────────────

    def _watchdog(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.CHECK_INTERVAL)
            if self._stop_event.is_set():
                break
            idle = time.monotonic() - self._last_used
            if idle >= self._idle_seconds and not self._unloaded:
                print(
                    f"\n[idle_guard] Pipeline idle for {idle:.0f}s "
                    f"(> {self._idle_seconds}s) — unloading from memory …"
                )
                with self._lock:
                    self._release()

    def _release(self) -> None:
        """Delete the pipeline and free GPU/CPU memory. Must hold self._lock."""
        if self._pipe is None:
            return
        try:
            # Move sub-modules off GPU before deletion so CUDA frees immediately
            device = self._hw_cfg.get("device", "cpu")
            if device == "cuda" and not self._hw_cfg.get("use_device_map", False):
                try:
                    self._pipe.to("cpu")
                except Exception:
                    pass

            del self._pipe
            self._pipe    = None
            self._unloaded = True

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                freed_info = []
                for i in range(torch.cuda.device_count()):
                    free = torch.cuda.mem_get_info(i)[0] / 1e9
                    freed_info.append(f"GPU {i}: {free:.1f} GB free")
                print(f"[idle_guard] Memory released. {' | '.join(freed_info)}")
            else:
                print("[idle_guard] Memory released (CPU mode).")

        except Exception as exc:
            logger.warning("[idle_guard] Error during release: %s", exc)

    def _reload(self) -> None:
        """Reload the pipeline from the local HuggingFace cache. Must hold self._lock."""
        print("[idle_guard] Reloading pipeline from cache …")
        t0 = time.time()
        try:
            from model import load_pipeline
            pipe, _ = load_pipeline(self._model_id, self._hw_cfg)
            self._pipe     = pipe
            self._unloaded = False
            self._last_used = time.monotonic()
            print(f"[idle_guard] Pipeline reloaded in {time.time() - t0:.1f}s")
        except Exception as exc:
            raise RuntimeError(
                f"[idle_guard] Failed to reload pipeline: {exc}"
            ) from exc
