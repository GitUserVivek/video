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

Multi-GPU
---------
  When hw_cfg["use_device_map"] is True (multiple CUDA GPUs detected),
  from_pretrained() is called with device_map="auto". Accelerate then
  shards the model across all available GPUs automatically — no manual
  tensor splitting required. On 2× T4 (2× 15.6 GB = ~31 GB total) this
  allows running cogvideox-5b (~18 GB) fully in VRAM.
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

# Cache directory
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


# ── Version guard ─────────────────────────────────────────────────────────────

def _check_version_for_model(model_id: str) -> None:
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
    """Choose the best model for the detected hardware.

    Key insight: CogVideoX-5b's 3D attention materialises QKᵀ matrices that
    are O(T×H×W)² in size. On T4-class GPUs (15 GB each) this OOMs during the
    forward pass regardless of how many GPUs hold the weights.

    Safe model choices per GPU tier:
      Single GPU  ≥ 24 GB  (A100/H100)  → cogvideox-5b  (weights + activations fit)
      Multi-GPU   any config             → cogvideox-2b  (lighter attention, safe)
      Single GPU  10–23 GB              → cogvideox-2b
      < 10 GB / CPU / MPS               → ltx-video
    """
    device     = hw_cfg["device"]
    vram_gb    = hw_cfg.get("vram_gb", 0)        # single GPU VRAM
    gpu_count  = hw_cfg.get("gpu_count", 1)

    if device in ("cpu", "mps") or not _HAS_COGVIDEOX:
        return "ltx-video"

    # cogvideox-5b only safe on a single high-VRAM GPU (≥24 GB) where
    # both weights (~17 GB) AND attention activations (~6 GB+) fit together
    if gpu_count == 1 and vram_gb >= 24:
        return "cogvideox-5b"

    if vram_gb >= 10:
        return "cogvideox-2b"

    return "ltx-video"


# ── Download helpers ──────────────────────────────────────────────────────────

_IGNORE_PATTERNS: dict[str, list[str]] = {
    "Lightricks/LTX-Video": [
        "ltx-video-2b-v0.9-image-to-video*",
        "ltx-video-2b-v0.9.1-image-to-video*",
        "ltxv-13b-*",
        "training/*",
        "finetrainers/*",
        "*.gguf",
        "pytorch_model*.bin",
        "*.mp4",
        "*.gif",
    ],
    "THUDM/CogVideoX-2b": [
        "*.bin",
        "*.msgpack",
        "flax_model*",
    ],
    "THUDM/CogVideoX-5b": [
        "*.bin",
        "*.msgpack",
        "flax_model*",
    ],
}

_SIZE_HINTS = {
    "Lightricks/LTX-Video": "~8 GB  (text-to-video weights only)",
    "THUDM/CogVideoX-2b":   "~16 GB",
    "THUDM/CogVideoX-5b":   "~30 GB",
}


def ensure_model_downloaded(repo_id: str) -> Path:
    """Download only inference-required files. Resumes automatically if interrupted."""
    from huggingface_hub import snapshot_download

    safe_name    = repo_id.replace("/", "--")
    snapshot_dir = HF_CACHE / "hub" / f"models--{safe_name}"

    if snapshot_dir.exists():
        weights = (list(snapshot_dir.glob("**/*.safetensors")) +
                   list(snapshot_dir.glob("**/*.bin")))
        if weights:
            print(f"[model] Cache hit: {repo_id}")
            return snapshot_dir

    ignore    = _IGNORE_PATTERNS.get(repo_id, [])
    size_hint = _SIZE_HINTS.get(repo_id, "")
    print(f"[model] Downloading {repo_id}  {size_hint}")
    print(f"[model] Cache dir → {snapshot_dir}")
    if ignore:
        print(f"[model] Skipping {len(ignore)} pattern(s) (large unneeded variants)")
    print("  Download resumes automatically if interrupted.\n")

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

def _load_pipeline(repo_id: str, pipeline_cls: Any, hw_cfg: dict) -> Any:
    """Load pipeline from cache.

    device_map="balanced" splits weights but NOT attention activations —
    on T4-class GPUs cogvideox-5b's QKᵀ attention matrices still OOM on a
    single GPU during the forward pass. So we always load with plain
    from_pretrained() and let _apply_optimisations handle placement.
    """
    dtype    = hw_cfg["dtype"]
    loader   = pipeline_cls if pipeline_cls is not None else DiffusionPipeline
    cls_name = loader.__name__ if loader else "DiffusionPipeline"

    print(f"[model] Loading {cls_name} ({dtype}) …")
    pipe = loader.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE / "hub"),
    )
    return pipe


def _try_enable_xformers(pipe: Any) -> bool:
    """Enable xformers memory-efficient attention if available.

    Returns True if successfully enabled.
    """
    try:
        import xformers  # noqa: F401
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            print("[model] xformers memory-efficient attention enabled ✓")
            return True
    except ImportError:
        pass
    except Exception as exc:
        print(f"[model] xformers available but failed to enable: {exc}")
    return False


def _apply_chunked_attention(pipe: Any, vram_gb: float) -> bool:
    """Apply chunked attention patch as fallback when xformers is unavailable.

    chunk_size tuned to leave enough VRAM for weights + activations on T4.
      T4 15 GB  → chunk_size=512  → peak attn ~300 MB
      8 GB GPU  → chunk_size=256  → peak attn ~150 MB
      ≥24 GB    → chunk_size=2048 → peak attn ~1.2 GB (fast)
    """
    try:
        from chunked_attention import patch_cogvideox_attention
        if vram_gb >= 24:
            chunk = 2048
        elif vram_gb >= 12:
            chunk = 512
        else:
            chunk = 256
        patch_cogvideox_attention(pipe, chunk_size=chunk)
        return True
    except Exception as exc:
        print(f"[model] chunked attention patch failed: {exc}")
        return False


def _apply_optimisations(pipe: Any, hw_cfg: dict, model_id: str) -> Any:
    """Apply memory / speed optimisations and return the pipeline.

    Attention OOM fix (critical for CogVideoX on T4):
        1. Try xformers (O(N) attention) — best if installed
        2. Fall back to chunked attention patch (processes Q in 512-token blocks)
        Both prevent the 97 GB QKᵀ matrix allocation.

    Placement strategy:
        Multi-GPU any config         → sequential_cpu_offload on GPU 0
        Single GPU ≥ full_vram GB    → .to("cuda")
        Single GPU ≥ min_vram GB     → enable_model_cpu_offload()
        Single GPU < min_vram GB     → enable_sequential_cpu_offload()
        CPU                          → enable_sequential_cpu_offload()
    """
    device    = hw_cfg["device"]
    vram_gb   = hw_cfg.get("vram_gb", 0)
    gpu_count = hw_cfg.get("gpu_count", 1)
    mid       = model_id.lower()

    _FULL_VRAM = {"cogvideox-5b": 24, "cogvideox-2b": 12, "ltx-video": 8}
    _MIN_VRAM  = {"cogvideox-5b": 16, "cogvideox-2b":  8, "ltx-video": 5}
    key        = next((k for k in _FULL_VRAM if k in mid), None)
    full_vram  = _FULL_VRAM.get(key, 12)
    min_vram   = _MIN_VRAM.get(key, 8)

    if device == "cuda":
        # ── Step 1: fix attention BEFORE placement ─────────────────────────
        is_cogvideox = "cogvideox" in mid
        if is_cogvideox:
            xformers_ok = _try_enable_xformers(pipe)
            if not xformers_ok:
                chunked_ok = _apply_chunked_attention(pipe, vram_gb)
                if not chunked_ok:
                    print("[model] WARNING: no attention fix applied — OOM likely")

        # ── Step 2: placement ──────────────────────────────────────────────
        if gpu_count > 1:
            print(
                f"[model] Multi-GPU ({gpu_count}× GPU, {hw_cfg.get('total_vram_gb',0):.0f} GB): "
                f"sequential_cpu_offload on GPU 0"
            )
            pipe.enable_sequential_cpu_offload(gpu_id=0)
        elif vram_gb >= full_vram:
            print(f"[model] {vram_gb:.0f} GB VRAM — loading fully onto GPU")
            pipe = pipe.to(device)
        elif vram_gb >= min_vram:
            print(f"[model] {vram_gb:.0f} GB VRAM — model CPU offload")
            pipe.enable_model_cpu_offload()
        else:
            print(f"[model] {vram_gb:.0f} GB VRAM — sequential CPU offload")
            pipe.enable_sequential_cpu_offload()

        # VAE optimisations (always safe)
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

    elif device == "mps":
        pipe = pipe.to(device)
    else:
        pipe.enable_sequential_cpu_offload()

    # torch.compile — single GPU, fully resident, no offload hooks
    compile_ok = (
        hw_cfg.get("use_compile")
        and device == "cuda"
        and gpu_count == 1
        and vram_gb >= full_vram
    )
    if compile_ok:
        transformer = getattr(pipe, "transformer", None)
        if transformer is not None:
            try:
                print("[model] Compiling transformer with torch.compile …")
                pipe.transformer = torch.compile(
                    transformer, mode="reduce-overhead", fullgraph=False,
                )
            except Exception as exc:
                print(f"[model] torch.compile skipped: {exc}")

    return pipe


# ── Public API ────────────────────────────────────────────────────────────────

def load_pipeline(model_id: str | None, hw_cfg: dict) -> tuple[Any, dict]:
    """Download (if needed) and load the video generation pipeline.

    Parameters
    ----------
    model_id : str | None
        Key from MODELS, or None to auto-select based on hw_cfg.
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

    _check_version_for_model(model_id)

    info    = MODELS[model_id]
    repo_id = info["repo_id"]

    gpu_info = ""
    if hw_cfg.get("use_device_map"):
        gpu_info = f"  [sharded across {hw_cfg['gpu_count']} GPUs, " \
                   f"{hw_cfg['total_vram_gb']:.0f} GB total]"
    print(f"[model] Selected: {model_id} ({info['size_label']})  repo={repo_id}{gpu_info}")

    ensure_model_downloaded(repo_id)

    pipeline_name = info["pipeline"]
    if pipeline_name == "CogVideoXPipeline":
        cls = CogVideoXPipeline
    elif pipeline_name == "LTXPipeline":
        cls = LTXPipeline
    else:
        cls = None

    pipe = _load_pipeline(repo_id, cls, hw_cfg)
    pipe = _apply_optimisations(pipe, hw_cfg, model_id)

    print(f"[model] Ready: {model_id}\n")
    return pipe, info
