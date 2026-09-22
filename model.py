"""
model.py — Model registry, auto-download (with progress bar), and device-aware loading.

Supported model families
------------------------
  CogVideoX  (THUDM/CogVideoX-2b, CogVideoX-5b)   – primary choice
  LTX-Video  (Lightricks/LTX-Video)                – fast, low-VRAM
  Open-Sora  (hpcai-tech/Open-Sora)                – research/CPU-friendly

Selection heuristic
-------------------
  ≥16 GB VRAM  → CogVideoX-5b   (best quality)
  ≥ 8 GB VRAM  → CogVideoX-2b
  <  8 GB VRAM / MPS → LTX-Video (most compact)
  CPU          → LTX-Video with sequential offload
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import (
    CogVideoXPipeline,
    LTXPipeline,
    LTXVideoTransformer3DModel,
)
from huggingface_hub import snapshot_download
from transformers import T5EncoderModel


# ── Model registry ────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "cogvideox-5b": {
        "repo_id":    "THUDM/CogVideoX-5b",
        "pipeline":   "CogVideoXPipeline",
        "min_vram":   14,   # GB – can use with sequential offload at ~10 GB
        "size_label": "5B",
    },
    "cogvideox-2b": {
        "repo_id":    "THUDM/CogVideoX-2b",
        "pipeline":   "CogVideoXPipeline",
        "min_vram":   8,
        "size_label": "2B",
    },
    "ltx-video": {
        "repo_id":    "Lightricks/LTX-Video",
        "pipeline":   "LTXPipeline",
        "min_vram":   4,
        "size_label": "~2B",
    },
}

# Cache directory – mirrors HuggingFace default but explicit
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


# ── Model selection ───────────────────────────────────────────────────────────

def select_model(hw_cfg: dict) -> str:
    """Choose the best model ID for the detected hardware."""
    device   = hw_cfg["device"]
    vram_gb  = hw_cfg.get("vram_gb", 0)
    max_size = hw_cfg.get("max_model_size", "1.3B")

    if device == "cpu" or max_size == "1.3B":
        return "ltx-video"

    if device in ("cuda", "mps"):
        if vram_gb >= 14:
            return "cogvideox-5b"
        if vram_gb >= 8:
            return "cogvideox-2b"
        return "ltx-video"

    return "ltx-video"   # safe fallback


# ── Download helpers ──────────────────────────────────────────────────────────

def _progress_callback(downloaded: int, total: int, bar_width: int = 40) -> None:
    """Simple ASCII progress bar printed to stdout."""
    if total <= 0:
        return
    frac   = min(downloaded / total, 1.0)
    filled = int(bar_width * frac)
    bar    = "█" * filled + "░" * (bar_width - filled)
    pct    = frac * 100
    dl_gb  = downloaded / 1e9
    tot_gb = total / 1e9
    print(f"\r  [{bar}] {pct:5.1f}%  {dl_gb:.2f}/{tot_gb:.2f} GB", end="", flush=True)


def ensure_model_downloaded(repo_id: str) -> Path:
    """Download model snapshot from HuggingFace if not already cached.

    Returns the local path to the snapshot directory.
    Uses huggingface_hub snapshot_download which resumes partial downloads.
    """
    safe_name = repo_id.replace("/", "--")
    snapshot_dir = HF_CACHE / "hub" / f"models--{safe_name}"

    if snapshot_dir.exists():
        # Check for at least one .safetensors or .bin file
        weights = list(snapshot_dir.glob("**/*.safetensors")) + \
                  list(snapshot_dir.glob("**/*.bin"))
        if weights:
            print(f"[model] Cache hit: {repo_id}")
            return snapshot_dir

    print(f"[model] Downloading {repo_id} → {snapshot_dir}")
    print("  This may take several minutes on first run …")

    t0 = time.time()
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(HF_CACHE / "hub"),
        local_files_only=False,
        resume_download=True,
    )
    elapsed = time.time() - t0
    print(f"\n[model] Download complete in {elapsed:.0f}s → {local_dir}")
    return Path(local_dir)


# ── Pipeline loading ──────────────────────────────────────────────────────────

def _load_cogvideox(repo_id: str, hw_cfg: dict) -> Any:
    """Load a CogVideoX pipeline with correct dtype and offloading strategy."""
    device    = hw_cfg["device"]
    dtype     = hw_cfg["dtype"]
    offload   = hw_cfg["sequential_offload"]
    vram_gb   = hw_cfg.get("vram_gb", 0)

    print(f"[model] Loading CogVideoXPipeline ({dtype}) …")
    pipe = CogVideoXPipeline.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE / "hub"),
    )

    if device == "cuda":
        if offload or vram_gb < 10:
            print("[model] Enabling sequential CPU offload (low-VRAM mode)")
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)
        # Slice attention to save VRAM
        pipe.enable_attention_slicing()
        if vram_gb >= 8:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
    elif device == "mps":
        pipe = pipe.to(device)
    else:
        # CPU: keep weights in RAM, use sequential offload
        pipe.enable_sequential_cpu_offload()

    if hw_cfg.get("use_compile") and device == "cuda":
        try:
            print("[model] Compiling transformer with torch.compile …")
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode="reduce-overhead",
                fullgraph=False,
            )
        except Exception as exc:
            print(f"[model] torch.compile skipped: {exc}")

    return pipe


def _load_ltx(repo_id: str, hw_cfg: dict) -> Any:
    """Load an LTX-Video pipeline."""
    device  = hw_cfg["device"]
    dtype   = hw_cfg["dtype"]
    offload = hw_cfg["sequential_offload"]

    print(f"[model] Loading LTXPipeline ({dtype}) …")

    # LTX uses a separate transformer + text encoder loading pattern
    pipe = LTXPipeline.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE / "hub"),
    )

    if device == "cuda":
        if offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    elif device == "mps":
        pipe = pipe.to(device)
    else:
        pipe.enable_sequential_cpu_offload()

    if hw_cfg.get("use_compile") and device == "cuda":
        try:
            print("[model] Compiling transformer with torch.compile …")
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode="reduce-overhead",
                fullgraph=False,
            )
        except Exception as exc:
            print(f"[model] torch.compile skipped: {exc}")

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

    info     = MODELS[model_id]
    repo_id  = info["repo_id"]
    pipeline = info["pipeline"]

    print(f"[model] Selected: {model_id} ({info['size_label']})  repo={repo_id}")

    # Ensure weights are present locally
    ensure_model_downloaded(repo_id)

    # Load into memory
    if pipeline == "CogVideoXPipeline":
        pipe = _load_cogvideox(repo_id, hw_cfg)
    elif pipeline == "LTXPipeline":
        pipe = _load_ltx(repo_id, hw_cfg)
    else:
        raise NotImplementedError(f"Pipeline type '{pipeline}' is not implemented.")

    print(f"[model] Ready: {model_id}\n")
    return pipe, info
