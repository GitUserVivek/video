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
  from_pretrained() is called with device_map="balanced". Accelerate then
  shards the weights evenly across all
  available GPUs — no manual tensor splitting required, and no per-layer
  streaming through a single GPU over PCIe. On 2× T4 (2× 15.6 GB = ~31 GB
  total) this allows running cogvideox-5b (~20 GB in fp16) fully in VRAM.
  Attention activations are kept small by the chunked attention patch, which
  is what makes the fully-resident layout possible.
"""

from __future__ import annotations

import gc
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


# ── HuggingFace token ─────────────────────────────────────────────────────────
# Set this to your HF token for authenticated downloads (higher rate limits,
# access to gated models). Reads from the HF_TOKEN variable below — edit here
# rather than setting an environment variable.

HF_TOKEN: str | None = None   # ← paste your token here, e.g. "hf_xxxxxxxxxxxx"

# If left as None, falls back to the HF_TOKEN environment variable if set.
if HF_TOKEN is None:
    HF_TOKEN = os.environ.get("HF_TOKEN") or "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"

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print(f"[model] HF_TOKEN set ({HF_TOKEN[:8]}…)")
else:
    print("[model] No HF_TOKEN — unauthenticated downloads (rate-limited)")


# ── Model registry ────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "cogvideox-5b": {
        "repo_id":           "THUDM/CogVideoX-5b",
        "pipeline":          "CogVideoXPipeline",
        "min_vram":          14,
        "size_label":        "5B",
        "requires_ver":      "0.29.0",
        "default_steps":     50,
        "default_guidance":  6.0,
        "default_fps":       8,
        "max_native_frames": 49,
    },
    "cogvideox-2b": {
        "repo_id":           "THUDM/CogVideoX-2b",
        "pipeline":          "CogVideoXPipeline",
        "min_vram":          8,
        "size_label":        "2B",
        "requires_ver":      "0.29.0",
        "default_steps":     50,
        "default_guidance":  6.0,
        "default_fps":       8,
        "max_native_frames": 49,
    },
    "ltx-video": {
        "repo_id":           "Lightricks/LTX-Video",
        "pipeline":          "LTXPipeline",
        "min_vram":          4,
        "size_label":        "~2B",
        "requires_ver":      "0.32.0",
        "default_steps":     30,
        "default_guidance":  3.0,
        "default_fps":       24,
        "max_native_frames": 121,
    },
}

# Cache directory
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _dir_size_gb(path: Path) -> float:
    """Approximate directory size in GB, gracefully degrading when permissions deny us."""
    try:
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return total / 1e9
    except Exception:                                  # noqa: BLE001
        return 0.0


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
        # ── Old 2B checkpoint versions (superseded by v0.9.5) ────────────
        "ltx-video-2b-v0.9.safetensors",
        "ltx-video-2b-v0.9.1.safetensors",
        "ltx-video-2b-v0.9-image-to-video*",
        "ltx-video-2b-v0.9.1-image-to-video*",
        # ── 2B distilled variants ─────────────────────────────────────────
        "ltxv-2b-0.9.6-dev-04-25.safetensors",
        "ltxv-2b-0.9.6-distilled-04-25.safetensors",
        "ltxv-2b-0.9.8-distilled.safetensors",
        "ltxv-2b-0.9.8-distilled-fp8.safetensors",
        # ── All 13B variants (too large) ─────────────────────────────────
        "ltxv-13b-*.safetensors",
        # ── Upscaler models (not needed for basic t2v) ────────────────────
        "ltxv-spatial-upscaler-*.safetensors",
        "ltxv-temporal-upscaler-*.safetensors",
        # ── Training / media extras ───────────────────────────────────────
        "training/*",
        "finetrainers/*",
        "media/*",
        "*.mp4",
        "*.gif",
        "*.gguf",
        "pytorch_model*.bin",
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


# Exact files needed per repo for text-to-video inference.
# Using explicit lists instead of snapshot_download + ignore_patterns because
# huggingface_hub's ignore_patterns filtering is unreliable on large repos —
# it queues all blobs first, then filters, so partial downloads still pull
# everything. hf_hub_download fetches one file at a time, guaranteed.
# _REQUIRED_FILES: dict[str, list[str]] = {
#     "Lightricks/LTX-Video": [
#         # Pipeline config
#         "model_index.json",
#         # Transformer config + weights (latest 2B checkpoint)
#         "transformer/config.json",
#         "ltx-video-2b-v0.9.5.safetensors",
#         # VAE
#         "vae/config.json",
#         # Text encoder (T5)
#         "text_encoder/config.json",
#         "tokenizer/special_tokens_map.json",
#         "tokenizer/spiece.model",
#         "tokenizer/tokenizer.json",
#         "tokenizer/tokenizer_config.json",
#         # Scheduler
#         "scheduler/scheduler_config.json",
#     ],
# }
_REQUIRED_FILES: dict[str, list[str]] = {
    "Lightricks/LTX-Video": [
        "model_index.json",
        "transformer/config.json",
        "ltx-video-2b-v0.9.5.safetensors",
        "vae/config.json",
        "text_encoder/config.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
    ],
}

def ensure_model_downloaded(repo_id: str) -> Path:
    """Download only the exact files needed for inference.

    For repos in _REQUIRED_FILES: uses hf_hub_download() one file at a time —
    guaranteed to skip all other files regardless of huggingface_hub version.

    For all other repos: falls back to snapshot_download() with ignore_patterns.
    """
    safe_name    = repo_id.replace("/", "--")
    snapshot_dir = HF_CACHE / "hub" / f"models--{safe_name}"

    # Cache hit check
    if snapshot_dir.exists():
        weights = (list(snapshot_dir.glob("**/*.safetensors")) +
                   list(snapshot_dir.glob("**/*.bin")))
        if weights:
            print(f"[model] Cache hit: {repo_id}")
            return snapshot_dir

    size_hint = _SIZE_HINTS.get(repo_id, "")
    print(f"[model] Downloading {repo_id}  {size_hint}")

    required = _REQUIRED_FILES.get(repo_id)
    if required:
        return _download_exact_files(repo_id, required)
    else:
        return _download_snapshot(repo_id)


def _download_exact_files(repo_id: str, files: list[str]) -> Path:
    """Download only explicitly requested files from Hugging Face."""

    from huggingface_hub import hf_hub_download

    print(
        f"[model] Fetching {len(files)} file(s) "
        f"(exact list — skips all other weights)"
    )

    downloaded = 0
    failed = []

    for filename in files:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(HF_CACHE / "hub"),
                token=HF_TOKEN or None,
            )

            size_mb = Path(path).stat().st_size / 1e6

            print(
                f"[model]   ✓  {filename}  "
                f"({size_mb:.0f} MB)"
            )

            downloaded += 1

        except Exception as exc:
            print(
                f"[model]   ✗  {filename}  "
                f"({type(exc).__name__}: {exc})"
            )
            failed.append(filename)

    if failed:
        raise RuntimeError(
            f"Failed to download required files from {repo_id}:\n"
            + "\n".join(f"  - {f}" for f in failed)
        )

    print(
        f"[model] Downloaded {downloaded} file(s) "
        f"to HF cache"
    )

    # Return the actual HF snapshot directory.
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(HF_CACHE / "hub"),
        local_files_only=True,
        token=HF_TOKEN or None,
    )

    return Path(snapshot_path)

def _download_exact_files_old(repo_id: str, files: list[str]) -> Path:
    """Download a specific list of files from a HuggingFace repo."""
    from huggingface_hub import hf_hub_download

    print(f"[model] Fetching {len(files)} file(s) (exact list — skips all other weights)")

    local_dir = HF_CACHE / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "main"
    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = []
    for filename in files:
        dest = local_dir / filename
        if dest.exists():
            print(f"[model]   skip  {filename} (already present)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(HF_CACHE / "hub"),
                token=HF_TOKEN or None,
            )
            # hf_hub_download returns a path inside the blob cache; symlink into our layout
            import shutil
            if not dest.exists():
                shutil.copy2(path, dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"[model]   ✓  {filename}  ({size_mb:.0f} MB)")
            downloaded += 1
        except Exception as exc:
            print(f"[model]   ✗  {filename}  ({type(exc).__name__}: {exc})")
            failed.append(filename)

    if failed:
        # Non-fatal: config files may be embedded differently in some versions
        print(f"[model] Warning: {len(failed)} file(s) not found: {failed}")
        print("[model] This may be fine if the pipeline loads from the blob cache.")

    print(f"[model] Downloaded {downloaded} new file(s) to {local_dir}")
    return local_dir


def _download_snapshot(repo_id: str) -> Path:
    """Fallback: snapshot_download with ignore patterns for non-listed repos."""
    from huggingface_hub import snapshot_download

    safe_name    = repo_id.replace("/", "--")
    snapshot_dir = HF_CACHE / "hub" / f"models--{safe_name}"
    ignore       = _IGNORE_PATTERNS.get(repo_id, [])

    if ignore:
        print(f"[model] Skipping {len(ignore)} pattern(s) (large unneeded variants)")
    print("  Download resumes automatically if interrupted (Ctrl-C).\n")

    t0 = time.time()
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(HF_CACHE / "hub"),
            local_files_only=False,
            ignore_patterns=ignore if ignore else None,
            token=HF_TOKEN or None,
        )
    except KeyboardInterrupt:
        print("\n[model] Download cancelled — partial snapshot left on disk.")
        raise
    print(f"[model] Download complete in {time.time() - t0:.0f}s → {local_dir}")
    return Path(local_dir)


# ── Pipeline loading ──────────────────────────────────────────────────────────

def _free_vram_report(hw_cfg: dict) -> str:
    """Describe the headroom accelerate will balance the weights over."""
    entries = []
    for i in range(hw_cfg.get("gpu_count", 1)):
        free_gb, total_gb = (v / 1e9 for v in torch.cuda.mem_get_info(i))
        entries.append(f"GPU {i}: {free_gb:.1f}/{total_gb:.1f} GB free")
    return ", ".join(entries)


def _report_device_map(pipe: Any) -> None:
    """Summarise where accelerate actually placed the pipeline's modules."""
    device_map = getattr(pipe, "hf_device_map", None)
    if not device_map:
        return

    buckets: dict[str, int] = {}
    for device in device_map.values():
        key = f"cuda:{device}" if isinstance(device, int) else str(device)
        buckets[key] = buckets.get(key, 0) + 1

    placed = ", ".join(f"{k}: {v} blocks" for k, v in sorted(buckets.items()))
    print(f"[model] Placement: {placed}")

    spilled = [k for k in buckets if k in ("cpu", "disk")]
    if spilled:
        print(f"[model] WARNING: {', '.join(spilled)} hold weights — that path will "
              f"be slow (raise VRAM or use a smaller model to keep everything on GPU)")


def _load_pipeline(repo_id: str, pipeline_cls: Any, hw_cfg: dict) -> Any:
    """Load pipeline from cache.

    Multi-GPU: weights are sharded across every visible GPU with accelerate's
    `device_map="balanced"`. Each `CogVideoXBlock` is a `_no_split_module`, so a
    block is never cut in half across GPUs — only the activation handoff at the
    shard boundary happens (one small copy per forward pass). Combined with the
    chunked attention patch this keeps both GPUs holding and computing weights,
    instead of streaming every layer through GPU 0 over PCIe.
    """
    dtype    = hw_cfg["dtype"]
    loader   = pipeline_cls if pipeline_cls is not None else DiffusionPipeline
    cls_name = loader.__name__ if loader else "DiffusionPipeline"
    shard    = hw_cfg.get("use_device_map", False) and hw_cfg.get("gpu_count", 0) > 1

    print(f"[model] Loading {cls_name} ({dtype}) …")
    base_kwargs: dict[str, Any] = {"torch_dtype": dtype, "cache_dir": str(HF_CACHE / "hub")}

    if shard:
        # No explicit `max_memory` map: accelerate sizes its balanced split from the
        # *currently free* VRAM of every GPU (plus RAM as a spill target), which
        # adapts to a Kaggle session that already has other allocations. Only CPU
        # spill would slow us down, and `_report_device_map` calls that out.
        print(f"[model] device_map=\"balanced\"  ({_free_vram_report(hw_cfg)})")
        try:
            pipe = loader.from_pretrained(repo_id, device_map="balanced", **base_kwargs)
            _report_device_map(pipe)
            return pipe
        except Exception as exc:
            # A failed sharded load must not cost the user a 30-minute session.
            print(f"[model] Sharded load failed ({type(exc).__name__}: {exc})")
            print("[model] Falling back to single-GPU residency + CPU offload")
            hw_cfg["use_device_map"] = False   # keep every downstream path consistent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return loader.from_pretrained(repo_id, **base_kwargs)


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

    `chunk_size` caps how many query tokens are scored per iteration; the kernel
    additionally clamps it so a single score block stays under ~512 MB (CogVideoX
    runs joint text+video attention, so a 480p/49-frame pass would otherwise need
    ~52 GB for one block's fp32 score matrix).
      T4 15 GB  → chunk_size=512
      8 GB GPU  → chunk_size=256
      ≥24 GB    → chunk_size=2048 (fast)
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
        Multi-GPU device_map        → weights already sharded, no offload needed
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
        if hw_cfg.get("use_device_map") and gpu_count > 1:
            print(
                f"[model] Multi-GPU: weights sharded across {gpu_count} GPUs "
                f"({hw_cfg.get('total_vram_gb', 0):.0f} GB total) — no CPU offload"
            )
        elif gpu_count > 1:
            print(
                f"[model] Multi-GPU without device_map — CPU offload on GPU 0"
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

    info    = dict(MODELS[model_id])
    repo_id = info["repo_id"]
    info["model_id"] = model_id

    gpu_info = ""
    if hw_cfg.get("use_device_map"):
        gpu_info = (f"  [sharded across {hw_cfg['gpu_count']} GPUs, "
                    f"{hw_cfg['total_vram_gb']:.0f} GB total]")
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
