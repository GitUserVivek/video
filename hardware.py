"""
hardware.py — Auto-detect best available compute device and return a config dict.

Supports:
  - NVIDIA GPU (CUDA)
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
        device            str   – 'cuda' | 'mps' | 'cpu'
        dtype             dtype – preferred tensor dtype
        vram_gb           float – detected VRAM (0 for CPU/MPS without query)
        use_compile       bool  – whether torch.compile is safe to use
        sequential_offload bool – offload layers to RAM when VRAM is tight
        threads           int|None – OMP/MKL thread count (None ⇒ PyTorch default)
        max_resolution    str   – '480' | '720' | '1080'
        max_model_size    str   – '1.3B' | '7B' | '14B'
    """

    # ------------------------------------------------------------------ GPU --
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Distinguish NVIDIA vs AMD (ROCm surfaces as CUDA)
        vendor_tag = "[ROCm]" if "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper() else "[CUDA]"
        print(f"{vendor_tag} Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        return {
            "device": "cuda",
            # bfloat16 preferred on Ampere+/RDNA3+; fall back to float16 for older hardware
            "dtype": torch.bfloat16 if vram_gb >= 8 else torch.float16,
            "vram_gb": vram_gb,
            "use_compile": True,
            "sequential_offload": vram_gb < 8,   # offload only when VRAM < 8 GB
            "threads": None,
            "max_resolution": "1080" if vram_gb >= 16 else ("720" if vram_gb >= 8 else "480"),
            "max_model_size": "14B" if vram_gb >= 12 else ("7B" if vram_gb >= 8 else "1.3B"),
        }

    # --------------------------------------------------------------- Apple MPS --
    if torch.backends.mps.is_available():
        print("[MPS] Detected: Apple Silicon GPU")
        return {
            "device": "mps",
            "dtype": torch.float16,
            "vram_gb": 0,
            "use_compile": False,   # torch.compile support on MPS is limited
            "sequential_offload": False,
            "threads": None,
            "max_resolution": "720",
            "max_model_size": "1.3B",
        }

    # --------------------------------------------------------------- CPU fallback --
    cpu_cores = os.cpu_count() or 32
    # Prefer physical-core count if psutil is available
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or cpu_cores
        print(f"[CPU] No GPU detected. Using {cpu_cores} logical threads ({physical} physical cores), "
              f"{psutil.virtual_memory().total / 1e9:.0f} GB RAM.")
    except ImportError:
        print(f"[CPU] No GPU detected. Using {cpu_cores} threads.")

    # Tune threading environment before any BLAS call
    os.environ["OMP_NUM_THREADS"]           = str(cpu_cores)
    os.environ["MKL_NUM_THREADS"]           = str(cpu_cores)
    os.environ["OPENBLAS_NUM_THREADS"]      = str(cpu_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"]    = str(cpu_cores)
    os.environ["NUMEXPR_NUM_THREADS"]       = str(cpu_cores)

    torch.set_num_threads(cpu_cores)
    torch.set_num_interop_threads(min(4, cpu_cores))   # inter-op parallelism
    torch.set_float32_matmul_precision("high")          # use TF32-style fast path

    return {
        "device": "cpu",
        "dtype": torch.bfloat16,   # bfloat16 on CPU avoids range overflow vs float16
        "vram_gb": 0,
        "use_compile": True,       # torch.compile with LLVM back-end is safe on CPU
        "sequential_offload": True,  # not strictly needed but keeps peak RAM lower
        "threads": cpu_cores,
        "max_resolution": "720",
        # 248 GB RAM comfortably holds a 14B param model (~28 GB in bfloat16)
        "max_model_size": "14B",
    }


def print_device_summary(cfg: dict) -> None:
    """Pretty-print the hardware config for the user."""
    print("\n── Hardware Configuration ──────────────────────────────")
    print(f"  Device            : {cfg['device'].upper()}")
    print(f"  Dtype             : {cfg['dtype']}")
    if cfg["vram_gb"]:
        print(f"  VRAM              : {cfg['vram_gb']:.1f} GB")
    if cfg["threads"]:
        print(f"  CPU threads       : {cfg['threads']}")
    print(f"  Max resolution    : {cfg['max_resolution']}p")
    print(f"  Max model size    : {cfg['max_model_size']}")
    print(f"  torch.compile     : {'yes' if cfg['use_compile'] else 'no'}")
    print(f"  Sequential offload: {'yes' if cfg['sequential_offload'] else 'no'}")
    print("────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    cfg = detect_device()
    print_device_summary(cfg)
