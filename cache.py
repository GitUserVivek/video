"""
cache.py — Crash-resilient cache for generation passes.

Why
---
One 480p/49-frame CogVideoX pass takes ~25 minutes on 2×T4, and a 30 s clip is
several passes. A CUDA OOM, a Kaggle timeout or a stray Ctrl-C used to throw all of
it away. Now every finished pass is written to disk together with a signature of the
exact run that produced it, so the next attempt only generates what is missing.

Layout
------
    <cache_root>/<run_key>/
        run.json           signature of the run + last-updated timestamp
        pass_01.npz        lossless uint8 frames of pass 1
        pass_02.npz        …
        prompt_embeds.npz  cached T5 output (skips text encoding on resume)
        done.json          written once the final file is assembled

`run_key` is a hash of the signature — prompt, model, resolution, fps, steps,
guidance, seed, duration and the per-pass frame counts. Change any of those and you
get a different key, so a resumed run can never splice together passes from two
different configurations. Because the per-pass noise is derived as `seed + index`,
a pass regenerated after a crash is the same pass that would have been produced
originally.

Granularity: one pass. There is no mid-pass checkpoint — resuming inside a single
denoising loop would mean re-implementing the scheduler loop that `pipe.__call__`
owns, which is not worth the fragility. Keep passes short (`--segment-secs`,
`--fast`) if you care about the worst-case loss window.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:                                              # pragma: no cover
    import torch

CACHE_VERSION = 1
DEFAULT_CACHE_DIRNAME = ".gen_cache"


# ── Run signature ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunSignature:
    """Everything that changes what a generated pass looks like.

    Any difference here means the cached passes are not the ones this run would
    produce, so the run gets its own cache directory.
    """

    prompt: str
    negative: str
    model_id: str
    resolution: str
    fps: int
    steps: int
    guidance: float
    seed: int | None
    duration_sec: float
    segment_sizes: tuple[int, ...] = field(default_factory=tuple)
    version: int = CACHE_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["segment_sizes"] = list(self.segment_sizes)      # tuples are not JSON
        return data

    @property
    def key(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def describe(self) -> str:
        return (f"{self.model_id} @ {self.resolution}p, {self.fps} fps, "
                f"{self.steps} steps, guidance {self.guidance:g}, "
                f"{self.duration_sec:.1f}s in {len(self.segment_sizes)} pass(es)")


# ── Cache ─────────────────────────────────────────────────────────────────────

class RunCache:
    """On-disk store for the passes of one run. Safe to construct when disabled."""

    def __init__(
        self,
        cache_root: str | Path,
        signature: RunSignature,
        enabled: bool = True,
    ) -> None:
        self.signature = signature
        self.enabled   = enabled
        self.root      = Path(cache_root)
        self.dir       = self.root / signature.key

    # ── setup / state ─────────────────────────────────────────────────────

    def register(self) -> None:
        """Create the run directory and refuse to reuse it if it does not match."""
        if not self.enabled:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            meta = self.dir / "run.json"
            if meta.exists():
                stored = json.loads(meta.read_text()).get("signature")
                if stored != self.signature.to_dict():
                    print(f"[cache] WARNING: {self.dir} holds a different run's passes "
                          f"— regenerating instead of reusing")
                    self.enabled = False
                    return
            meta.write_text(json.dumps(
                {"signature": self.signature.to_dict(), "updated": time.time()},
                indent=2,
            ))
        except Exception as exc:                               # noqa: BLE001
            print(f"[cache] Disabled ({type(exc).__name__}: {exc})")
            self.enabled = False

    def describe(self) -> str:
        if not self.enabled:
            return "[cache] Disabled — passes will not be reused (--no-resume)"
        done = self.cached_passes()
        total = len(self.signature.segment_sizes)
        if done:
            return (f"[cache] Resuming run {self.signature.key}: "
                    f"{len(done)}/{total} passes on disk in {self.dir}")
        return f"[cache] New run {self.signature.key} → {self.dir}"

    def cached_passes(self) -> list[int]:
        """Indices of passes whose frames are present on disk (1-based)."""
        if not self.enabled or not self.dir.exists():
            return []
        return [
            index
            for index in range(1, len(self.signature.segment_sizes) + 1)
            if self.path_for(index).exists()
        ]

    def is_complete(self, n_passes: int) -> bool:
        return self.enabled and len(self.cached_passes()) >= n_passes

    # ── passes ────────────────────────────────────────────────────────────

    def path_for(self, index: int) -> Path:
        return self.dir / f"pass_{index:02d}.npz"

    def load_pass(self, index: int) -> list[Image.Image] | None:
        """Return the cached frames of a pass, or None when it must be generated."""
        if not self.enabled:
            return None
        path = self.path_for(index)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                frames = data["frames"]
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"unexpected frame array shape {frames.shape}")
            images = [Image.fromarray(frame) for frame in frames]
            print(f"[cache] Pass {index} reused from disk — {len(images)} frames "
                  f"({path.stat().st_size / 1e6:.0f} MB)")
            return images
        except Exception as exc:                               # noqa: BLE001
            print(f"[cache] Pass {index} unreadable ({type(exc).__name__}: {exc}) "
                  f"— regenerating it")
            self._discard(path)
            return None

    def save_pass(self, index: int, frames: Sequence[Image.Image]) -> None:
        """Store a finished pass losslessly (never re-encode a 2nd generation).

        Written to a temporary file and renamed, so an interrupted save can never
        leave a half-pass that a later run would happily load.
        """
        if not self.enabled or not frames:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            array = np.stack([
                np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames
            ])
            final = self.path_for(index)
            tmp   = final.with_name(final.stem + ".partial.npz")
            np.savez_compressed(tmp, frames=array)
            tmp.replace(final)
            print(f"[cache] Pass {index} saved — {len(frames)} frames, "
                  f"{final.stat().st_size / 1e6:.0f} MB")
        except Exception as exc:                               # noqa: BLE001
            print(f"[cache] Could not save pass {index} "
                  f"({type(exc).__name__}: {exc}) — continuing without it")

    # ── prompt embeddings ─────────────────────────────────────────────────

    def load_embeds(self) -> tuple["torch.Tensor", "torch.Tensor | None"] | None:
        """Return cached (positive, negative) text embeddings if they are usable."""
        if not self.enabled:
            return None
        path = self.dir / "prompt_embeds.npz"
        if not path.exists():
            return None
        try:
            import torch
            with np.load(path) as data:
                positive = _to_tensor(torch, data["positive"], str(data["positive_dtype"]))
                negative = None
                if "negative" in data.files:
                    negative = _to_tensor(torch, data["negative"], str(data["negative_dtype"]))
            print("[cache] Reusing cached prompt embeddings (text encoder skipped)")
            return positive, negative
        except Exception as exc:                               # noqa: BLE001
            print(f"[cache] Cached embeddings unusable ({type(exc).__name__}: {exc})")
            self._discard(path)
            return None

    def save_embeds(self, positive: "torch.Tensor", negative: "torch.Tensor | None") -> None:
        """Store raw T5 output so a resumed run can skip text encoding entirely."""
        if not self.enabled or positive is None:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                # bfloat16 has no numpy equivalent → store fp32, remember the dtype
                "positive":       positive.detach().float().cpu().numpy(),
                "positive_dtype": str(positive.dtype),
            }
            if negative is not None:
                payload["negative"]       = negative.detach().float().cpu().numpy()
                payload["negative_dtype"] = str(negative.dtype)
            np.savez(self.dir / "prompt_embeds.npz", **payload)
            print("[cache] Prompt embeddings cached")
        except Exception as exc:                               # noqa: BLE001
            print(f"[cache] Could not cache embeddings ({type(exc).__name__}: {exc})")

    # ── completion + failure bookkeeping ──────────────────────────────────

    def mark_complete(self, output_path: str | Path) -> None:
        if not self.enabled:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "done.json").write_text(json.dumps(
                {"output": str(output_path), "finished": time.time()}, indent=2
            ))
        except Exception:                                      # noqa: BLE001
            pass

    def report_interrupted(self, index: int, total: int, exc: BaseException) -> None:
        """Tell the user exactly what survived and how to continue."""
        done = len(self.cached_passes())
        print(f"\n[gen] Pass {index}/{total} failed — "
              f"{type(exc).__name__}: {exc}")
        if not self.enabled:
            print("[gen] No resume cache in use, so nothing was kept "
                  "(--no-resume). Drop the flag to make long runs restartable.")
            return
        print(f"[cache] {done}/{total} finished passes are on disk in {self.dir}")
        print("[cache] Re-run the same command to continue from there — cached passes "
              "are reused as-is. (--no-resume ignores them, --clear-cache deletes them.)")

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _discard(path: Path) -> None:
        try:
            path.unlink()
        except Exception:                                      # noqa: BLE001
            pass


def _to_tensor(torch: Any, array: np.ndarray, dtype_name: str) -> "torch.Tensor":
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    try:
        return tensor.to(getattr(torch, dtype_name.replace("torch.", "")))
    except Exception:                                          # noqa: BLE001
        return tensor.to(torch.float32)


def default_cache_root(cache_dir: str | Path | None = None) -> Path:
    """Cache lives in an explicit --cache-dir, else VIDEO_CACHE_DIR, else cwd."""
    import os
    if cache_dir:
        return Path(cache_dir)
    return Path(os.environ.get("VIDEO_CACHE_DIR", Path.cwd() / DEFAULT_CACHE_DIRNAME))


def clear_cache(cache_dir: str | Path | None = None) -> None:
    """Delete every cached run under the cache root."""
    root = default_cache_root(cache_dir)
    if not root.exists():
        print(f"[cache] Nothing to clear ({root} does not exist)")
        return

    runs = [p for p in root.iterdir() if p.is_dir()]
    freed = 0
    for run in runs:
        freed += sum(f.stat().st_size for f in run.rglob("*") if f.is_file())
    shutil.rmtree(root, ignore_errors=True)
    print(f"[cache] Cleared {len(runs)} run(s), {freed / 1e6:.0f} MB from {root}")
