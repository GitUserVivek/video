"""
model.py — Model registry, auto-download (with progress bar), and device-aware loading.

Supported model families
------------------------
  CogVideoX  (THUDM/CogVideoX-2b, CogVideoX-5b)   – primary choice for GPU
  LTX-Video  (Lightricks/LTX-Video)                – fast, low-VRAM / CPU

Diffusers version compatibility
--------------------------------
  CogVideoXPipeline  → diffusers >= 0.29.0
  LTXPipeline        → diffusers >= 0.32.0
  Fallback           → DiffusionPipeline.from_pretrained() works with any version
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import diffusers
import torch
from packaging.version import Version

# ── Version-safe pipeline imports ─────────────────────────────────────────────

_DIFFUSERS_VER = Version(diffusers.__version__)
_HAS_COGVIDEOX = _DIFFUSERS_VER >= Version("0.29.0")
_HAS_LTX       = _DIFFUSERS_VER >= Version("0.32.0")

print(f"[model] diffusers {diffusers.__version__}  "
      f"(CogVideoX={'✓' if _HAS_COGVIDEOX else '✗'}  "
      f"LTX={'✓' if _HAS_LTX else '✗'})")

if _HAS_COGVIDEOX:
    from diffusers import CogVideoXPipeline
else:
    CogVideoXPipeline = None  # type: ignore[assignment,misc]

if _HAS_LTX:
    from diffusers import LTXPipeline
else:
    LTXPipeline = None  # type: ignore[assignment,misc]

from diffusers import DiffusionPipeline  # always available


# ── Model registry ────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "cogvideox-5b": {
        "repo_id":      "THUDM/CogVideoX-5b",
        "pipeline":     "CogVideoXPipeline",
        "min_vram":     14,
        "size_label":   "5B",
        "requires_ver": "0.29.0",
    },
    "cogvideox-2b": {
        "repo_id":      "THUDM/CogVideoX-2b",
        "pipeline":     "CogVideoXPipeline",
        "min_vram":     8,
        "size_label":   "2B",
        "requires_ver": "0.29.0",
    },
    "ltx-video": {
        "repo_id":      "Lightricks/LTX-Video",
        "pipeline":     "LTXPipeline",
        "min_vram":     4,
        "size_label":   "~2B",
        "requires_ver": "0.32.0",
    },
}

# Cache directory – mirrors HuggingFace default but explicit
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


# ── Version guard ─────────────────────────────────────────────────────────────

def _check_version_for_model(model_id: str) -> None:
    """Raise a clear error if diffusers is too old for the requested model."""
    required = MODELS[model_id].get("requires_ver", "0.0.0")
    if _DIFFUSERS_VER < Version(required):
        raise RuntimeError(
            f"Model '{model_id}' requires diffusers >= {required} "
            f"but {diffusers.__version__} is installed.\n\n"
            f"Fix: pip install 'diffusers>={required}'\n"
            f"  or: pip install --upgrade diffusers"
        )


# ── Model selection ───────────────────────────────────────────────────────────

def select_model(hw_cfg: dict) -> str:
    """Choose the best model ID for the detected hardware.

    VRAM budgets (bfloat16):
      cogvideox-5b  ~18 GB to reside fully on GPU — only select if >= 18 GB
      cogvideox-2b  ~10 GB — safe on 10–17 GB cards (T4, 3080, etc.)
      ltx-video     ~6 GB  — safe on anything < 10 GB, CPU, or MPS
    """
    device  = hw_cfg["device"]
    vram_gb = hw_cfg.get("vram_gb", 0)

    if device in ("cpu", "mps") or not _HAS_COGVIDEOX:
        return "ltx-video"

    if vram_gb >= 18:
        return "cogvideox-5b"
    if vram_gb >= 10:
        return "cogvideox-2b"
    return "ltx-video"


# ── Download helpers ──────────────────────────────────────────────────────────


# Files to skip per repo — large extras that are NOT needed for text-to-video inference
_IGNORE_PATTERNS: dict[str, list[str]] = {
    "Lightricks/LTX-Video": [
        # image-to-video variants (separate checkpoints, not needed for t2v)
        "ltx-video-2b-v0.9-image-to-video*",
        "ltx-video-2b-v0.9.1-image-to-video*",
        "ltxv-13b-*",               # 13B variant — too large for CPU
        # training / fine-tuning helpers
        "training/*",
        "finetrainers/*",
        # GGUF quantised weights (we use bfloat16 safetensors directly)
        "*.gguf",
        # old numbered-shard bin files (superseded by safetensors)
        "pytorch_model*.bin",
        # large video samples bundled in the repo
        "*.mp4",
        "*.gif",
    ],
    "THUDM/CogVideoX-2b": [
        "*.bin",          # safetensors are preferred
        "*.msgpack",
        "flax_model*",
    ],
    "THUDM/CogVideoX-5b": [
        "*.bin",
        "*.msgpack",
        "flax_model*",
    ],
}


def ensure_model_downloaded(repo_id: str) -> Path:
    """Download only the inference-required files from HuggingFace.

    Uses snapshot_download() with ignore_patterns to skip large extras
    (image-to-video checkpoints, training scripts, GGUF weights, etc.).
    Downloads always resume if interrupted.
    Returns the local snapshot directory path.
    """
    from huggingface_hub import snapshot_download

    safe_name    = repo_id.replace("/", "--")
    snapshot_dir = HF_CACHE / "hub" / f"models--{safe_name}"

    if snapshot_dir.exists():
        weights = (list(snapshot_dir.glob("**/*.safetensors")) +
                   list(snapshot_dir.glob("**/*.bin")))
        if weights:
            print(f"[model] Cache hit: {repo_id}")
            return snapshot_dir

    ignore = _IGNORE_PATTERNS.get(repo_id, [])

    # Estimate download size for the user
    _SIZE_HINTS = {
        "Lightricks/LTX-Video":  "~8 GB  (text-to-video weights only)",
        "THUDM/CogVideoX-2b":    "~16 GB",
        "THUDM/CogVideoX-5b":    "~30 GB",
    }
    size_hint = _SIZE_HINTS.get(repo_id, "")
    print(f"[model] Downloading {repo_id}  {size_hint}")
    print(f"[model] Cache dir → {snapshot_dir}")
    if ignore:
        print(f"[model] Skipping {len(ignore)} ignore pattern(s) to avoid large unneeded files")
    print("  Download will resume automatically if interrupted.\n")

    t0 = time.time()
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(HF_CACHE / "hub"),
        local_files_only=False,
        ignore_patterns=ignore if ignore else None,
    )
    elapsed = time.time() - t0
    print(f"\n[model] Download complete in {elapsed:.0f}s → {local_dir}")
    return Path(local_dir)


# ── Pipeline loading ──────────────────────────────────────────────────────────

def _apply_optimisations(pipe: Any, hw_cfg: dict, model_id: str) -> Any:
    """Apply memory / speed optimisations and return the pipeline.

    Three-tier VRAM strategy for CUDA:
      Tier 1 — plenty of VRAM (>= model_full_vram):
          .to("cuda")  — fastest, everything resident on GPU

      Tier 2 — tight but workable (>= model_min_vram):
          enable_model_cpu_offload()  — moves whole modules CPU↔GPU as needed,
          much faster than sequential offload, avoids OOM on T4/3080 class cards

      Tier 3 — low VRAM (< model_min_vram) or sequential_offload flag:
          enable_sequential_cpu_offload()  — layer-by-layer, slowest but safest
    """
    device  = hw_cfg["device"]
    vram_gb = hw_cfg.get("vram_gb", 0)

    # Per-model VRAM requirements (bfloat16, approx)
    _FULL_VRAM = {
        "cogvideox-5b": 18,   # GB needed to fully reside on GPU
        "cogvideox-2b": 12,
        "ltx-video":     8,
    }
    _MIN_VRAM = {
        "cogvideox-5b": 14,   # minimum for model_cpu_offload (rest streams)
        "cogvideox-2b":  8,
        "ltx-video":     5,
    }

    # Normalise model_id to a short key
    mid = model_id.lower()
    key = next((k for k in _FULL_VRAM if k in mid), None)
    full_vram = _FULL_VRAM.get(key, 12)
    min_vram  = _MIN_VRAM.get(key, 8)

    if device == "cuda":
        if vram_gb >= full_vram:
            print(f"[model] {vram_gb:.0f} GB VRAM ≥ {full_vram} GB — loading fully onto GPU")
            pipe = pipe.to(device)
        elif vram_gb >= min_vram:
            print(f"[model] {vram_gb:.0f} GB VRAM — using model CPU offload "
                  f"(need {full_vram} GB for full GPU load)")
            pipe.enable_model_cpu_offload()
        else:
            print(f"[model] {vram_gb:.0f} GB VRAM — using sequential CPU offload (low-VRAM mode)")
            pipe.enable_sequential_cpu_offload()

        # These are safe to call regardless of offload mode
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

    elif device == "mps":
        pipe = pipe.to(device)

    else:  # CPU
        pipe.enable_sequential_cpu_offload()

    # torch.compile — only the transformer sub-module, only on CUDA
    # Skip if model_cpu_offload is active (hooks conflict with compile)
    compile_ok = (
        hw_cfg.get("use_compile")
        and device == "cuda"
        and vram_gb >= full_vram   # only compile when fully on GPU
    )
    if compile_ok:
        transformer = getattr(pipe, "transformer", None)
        if transformer is not None:
            try:
                print("[model] Compiling transformer with torch.compile …")
                pipe.transformer = torch.compile(
                    transformer,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except Exception as exc:
                print(f"[model] torch.compile skipped: {exc}")

    return pipe


def _load_pipeline(repo_id: str, pipeline_cls: Any, hw_cfg: dict) -> Any:
    """Generic loader: from_pretrained → optimisations."""
    dtype = hw_cfg["dtype"]
    print(f"[model] Loading {pipeline_cls.__name__ if pipeline_cls else 'DiffusionPipeline'} "
          f"({dtype}) …")

    loader = pipeline_cls if pipeline_cls is not None else DiffusionPipeline
    pipe   = loader.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE / "hub"),
    )
    return pipe


# ── Public API ────────────────────────────────────────────────────────────────

def load_pipeline(model_id: str | None, hw_cfg: dict) -> tuple[Any, dict]:
    """Download (if needed) and load the appropriate video generation pipeline.

    Parameters
    ----------
    model_id : str | None
        One of the keys in MODELS, or None to auto-select.
    hw_cfg : dict
        Config dict from hardware.detect_device().

    Returns
    -------
    (pipeline, model_info_dict)
    """
    if model_id is None:
        model_id = select_model(hw_cfg)

    if model_id not in MODELS:
        raise ValueError(
            f"Unknown model '{model_id}'. Choose from: {list(MODELS.keys())}"
        )

    # Hard stop with a clear upgrade message if diffusers is too old
    _check_version_for_model(model_id)

    info    = MODELS[model_id]
    repo_id = info["repo_id"]

    print(f"[model] Selected: {model_id} ({info['size_label']})  repo={repo_id}")

    # Ensure weights are cached locally
    ensure_model_downloaded(repo_id)

    # Pick the right class (or None → DiffusionPipeline fallback)
    pipeline_name = info["pipeline"]
    if pipeline_name == "CogVideoXPipeline":
        cls = CogVideoXPipeline
    elif pipeline_name == "LTXPipeline":
        cls = LTXPipeline
    else:
        cls = None

    # Load from cache
    pipe = _load_pipeline(repo_id, cls, hw_cfg)

    # Memory / speed optimisations
    pipe = _apply_optimisations(pipe, hw_cfg, model_id)

    print(f"[model] Ready: {model_id}\n")
    return pipe, info
