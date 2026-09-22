#!/usr/bin/env bash
# setup.sh — Create a Python virtual environment and install all dependencies.
#
# Usage:
#   bash setup.sh           # auto-detect GPU / CPU
#   bash setup.sh --cuda    # force CUDA 12.1 torch wheel
#   bash setup.sh --rocm    # force ROCm 6.0 torch wheel
#   bash setup.sh --cpu     # force CPU-only torch wheel
#
# After setup:
#   source .venv/bin/activate
#   python main.py "your prompt here"

set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
TORCH_VERSION="2.3.1"
TORCHVISION_VERSION="0.18.1"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup]${NC} $*"; }
error()   { echo -e "${RED}[setup] ERROR:${NC} $*" >&2; exit 1; }

# ── Parse arguments ───────────────────────────────────────────────────────────
TORCH_VARIANT=""
for arg in "$@"; do
    case "$arg" in
        --cuda) TORCH_VARIANT="cuda" ;;
        --rocm) TORCH_VARIANT="rocm" ;;
        --cpu)  TORCH_VARIANT="cpu"  ;;
        *) warn "Unknown argument: $arg (ignored)" ;;
    esac
done

# ── Check Python ──────────────────────────────────────────────────────────────
command -v "$PYTHON" &>/dev/null || error "Python 3 not found. Install python3 and retry."
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python version: $PY_VER"

# Require Python ≥ 3.10
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" \
    || error "Python ≥ 3.10 required (found $PY_VER)."

# ── Create virtualenv ─────────────────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at $VENV_DIR — reusing."
else
    info "Creating virtual environment at $VENV_DIR …"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── Detect hardware (if not forced) ──────────────────────────────────────────
detect_torch_variant() {
    # NVIDIA CUDA
    if command -v nvidia-smi &>/dev/null; then
        info "nvidia-smi found → installing CUDA torch wheel"
        echo "cuda"
        return
    fi
    # AMD ROCm
    if command -v rocm-smi &>/dev/null || [[ -d /opt/rocm ]]; then
        info "ROCm detected → installing ROCm torch wheel"
        echo "rocm"
        return
    fi
    # Fallback
    warn "No GPU tooling detected → installing CPU-only torch wheel"
    echo "cpu"
}

if [[ -z "$TORCH_VARIANT" ]]; then
    TORCH_VARIANT=$(detect_torch_variant)
fi

# ── Install PyTorch ───────────────────────────────────────────────────────────
case "$TORCH_VARIANT" in
    cuda)
        INDEX_URL="https://download.pytorch.org/whl/cu121"
        info "Installing PyTorch $TORCH_VERSION (CUDA 12.1) …"
        ;;
    rocm)
        INDEX_URL="https://download.pytorch.org/whl/rocm6.0"
        info "Installing PyTorch $TORCH_VERSION (ROCm 6.0) …"
        ;;
    cpu)
        INDEX_URL="https://download.pytorch.org/whl/cpu"
        info "Installing PyTorch $TORCH_VERSION (CPU only) …"
        ;;
    *)
        error "Unknown torch variant: $TORCH_VARIANT"
        ;;
esac

pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "$INDEX_URL" \
    --quiet

# ── Install remaining dependencies ────────────────────────────────────────────
# Install packaging first — model.py uses it for version checks at import time
pip install "packaging>=24.0" --quiet

info "Installing remaining dependencies from requirements.txt …"
# diffusers >= 0.32.0 required for LTXPipeline (added in that release)
pip install -r requirements.txt --quiet

# ── Verify install ────────────────────────────────────────────────────────────
info "Verifying installation …"
"$PYTHON" - <<'EOF'
import torch, diffusers, transformers, huggingface_hub, imageio, PIL
print(f"  torch        {torch.__version__}")
print(f"  diffusers    {diffusers.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  hf_hub       {huggingface_hub.__version__}")
print(f"  imageio      {imageio.__version__}")
print(f"  PIL          {PIL.__version__}")

if torch.cuda.is_available():
    print(f"  CUDA device  {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print(f"  MPS (Apple Silicon) available")
else:
    print(f"  No GPU detected — CPU mode")
EOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Setup complete!"
echo ""
echo "  Activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Generate a video:"
echo "    python main.py \"a sunset over the ocean\""
echo "    python main.py \"a cat playing piano\" --duration 10 --resolution 1080"
echo "    python main.py --help"
echo ""
