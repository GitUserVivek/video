"""
hardware.py — Auto-detect best available compute device and return a config dict.

Supports:
  - NVIDIA GPU (CUDA) — single or multi-GPU
  - AMD GPU (ROCm via torch CUDA API)
  - Apple Silicon (MPS)
  - CPU fallback (optimised for many-core x86_64 with large RAM)
"""

import os
import torch


def detect_device() -> dict:
    """Auto-detect best available device and return config dict.

    Returns
    -------
    dict with keys:
        device              str        – 'cuda' | 'mps' | 'cpu'
        dtype               dtype      – preferred tensor dtype
        vram_gb             float      – VRAM of GPU 0 (used for single-GPU decisions)
        total_vram_gb       float      – sum of VRAM across all visible GPUs
        gpu_count           int        – number of CUDA devices visible
        use_compile         bool       – whether torch.compile is safe
        sequential_offload  bool       – layer-by-layer CPU offload
        use_device_map      bool       – shard weights across GPUs with device_map
        threads             int|None   – OMP/MKL thread count (None ⇒ PyTorch default)
        max_resolution      str        – '480' | '720' | '1080'
        max_model_size      str        – '1.3B' | '7B' | '14B'
    """

    # ── GPU detection ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name  = torch.cuda.get_device_name(0)
        vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Sum VRAM across all visible GPUs
        total_vram_gb = sum(
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(gpu_count)
        )

        vendor_tag = (
            "[ROCm]" if "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper()
            else "[CUDA]"
        )

        # Reduce memory fragmentation — recovers ~10-15% usable VRAM
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        if gpu_count > 1:
            names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            vrams = [
                torch.cuda.get_device_properties(i).total_memory / 1e9
                for i in range(gpu_count)
            ]
            print(f"{vendor_tag} Detected {gpu_count} GPUs:")
            for i, (n, v) in enumerate(zip(names, vrams)):
                print(f"  GPU {i}: {n} ({v:.1f} GB)")
            print(f"  Total VRAM: {total_vram_gb:.1f} GB")
        else:
            print(f"{vendor_tag} Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        # Use total VRAM for model/resolution selection when multi-GPU
        effective_vram = total_vram_gb

        # Turing (T4, sm_75) has fp16 tensor cores but only *emulates* bf16;
        # Ampere+ (sm_80+) runs bf16 natively. Picking the wrong one costs several
        # times the throughput on the matmul-heavy transformer, so key this off the
        # compute capability instead of the VRAM size.
        compute_major = torch.cuda.get_device_capability(0)[0]

        return {
            "device": "cuda",
            "dtype": torch.bfloat16 if compute_major >= 8 else torch.float16,
            "compute_capability": compute_major,
            "vram_gb": vram_gb,                   # single GPU (GPU 0) VRAM
            "total_vram_gb": total_vram_gb,        # all GPUs combined
            "gpu_count": gpu_count,
            "use_compile": gpu_count == 1,         # torch.compile + device_map don't mix well
            "sequential_offload": effective_vram < 8,
            # Shard the pipeline's weights across every GPU with accelerate
            # device_map="balanced" whenever more than one is present.
            "use_device_map": gpu_count > 1,
            "threads": None,
            "max_resolution": (
                "1080" if effective_vram >= 18
                else "720" if effective_vram >= 10
                else "480"
            ),
            "max_model_size": (
                "14B" if effective_vram >= 18
                else "7B" if effective_vram >= 10
                else "1.3B"
            ),
        }

    # ── Apple MPS ──────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        print("[MPS] Detected: Apple Silicon GPU")
        return {
            "device": "mps",
            "dtype": torch.float16,
            "vram_gb": 0,
            "total_vram_gb": 0,
            "gpu_count": 0,
            "use_compile": False,
            "sequential_offload": False,
            "use_device_map": False,
            "threads": None,
            "max_resolution": "720",
            "max_model_size": "1.3B",
        }

    # ── CPU fallback ───────────────────────────────────────────────────────
    cpu_cores = os.cpu_count() or 32
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or cpu_cores
        print(
            f"[CPU] No GPU detected. Using {cpu_cores} logical threads "
            f"({physical} physical cores), "
            f"{psutil.virtual_memory().total / 1e9:.0f} GB RAM."
        )
    except ImportError:
        print(f"[CPU] No GPU detected. Using {cpu_cores} threads.")

    os.environ["OMP_NUM_THREADS"]        = str(cpu_cores)
    os.environ["MKL_NUM_THREADS"]        = str(cpu_cores)
    os.environ["OPENBLAS_NUM_THREADS"]   = str(cpu_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_cores)
    os.environ["NUMEXPR_NUM_THREADS"]    = str(cpu_cores)

    torch.set_num_threads(cpu_cores)
    torch.set_num_interop_threads(min(4, cpu_cores))
    torch.set_float32_matmul_precision("high")

    return {
        "device": "cpu",
        "dtype": torch.bfloat16,
        "vram_gb": 0,
        "total_vram_gb": 0,
        "gpu_count": 0,
        "use_compile": True,
        "sequential_offload": True,
        "use_device_map": False,
        "threads": cpu_cores,
        "max_resolution": "720",
        "max_model_size": "14B",
    }


def print_device_summary(cfg: dict) -> None:
    """Pretty-print the hardware config for the user."""
    print("\n── Hardware Configuration ──────────────────────────────")
    print(f"  Device            : {cfg['device'].upper()}")
    print(f"  Dtype             : {cfg['dtype']}")
    if cfg.get("compute_capability"):
        print(f"  Compute capability: sm_{cfg['compute_capability']}")
    if cfg.get("gpu_count", 0) > 1:
        print(f"  GPUs              : {cfg['gpu_count']}× "
              f"(total {cfg['total_vram_gb']:.1f} GB VRAM, "
              f"{cfg['vram_gb']:.1f} GB each)")
        if cfg.get("use_device_map"):
            print(f"  Offload strategy  : device_map=\"balanced\" — weights sharded "
                  f"across all {cfg['gpu_count']} GPUs (no CPU streaming)")
        else:
            print("  Offload strategy  : CPU offload on GPU 0")
    elif cfg.get("vram_gb"):
        print(f"  VRAM              : {cfg['vram_gb']:.1f} GB")
    if cfg.get("threads"):
        print(f"  CPU threads       : {cfg['threads']}")
    print(f"  Max resolution    : {cfg['max_resolution']}p")
    print(f"  Max model size    : {cfg['max_model_size']}")
    print(f"  torch.compile     : {'yes' if cfg['use_compile'] else 'no'}")
    print(f"  Sequential offload: {'yes' if cfg['sequential_offload'] else 'no'}")
    print("────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    cfg = detect_device()
    print_device_summary(cfg)
