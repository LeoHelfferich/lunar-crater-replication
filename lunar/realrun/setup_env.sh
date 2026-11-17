#!/usr/bin/env bash
# Separate Python 3.12 environment for the NASA repo (it requires Python >=3.11,<3.13;
# Kaggle's and Colab's kernels are 3.13). The notebook kernel itself is not touched, so no restart is needed.
# Usage: bash setup_env.sh [--force-torch]   (--force-torch: reinstall the CUDA torch wheels)
# Env overrides (for testing elsewhere): VENV, TORCH_INDEX, REQUIRE_CUDA=0 (accept a CPU torch build)
set -euo pipefail
export MPLBACKEND=Agg  # Colab exports an inline backend that crashes matplotlib outside the kernel
VENV="${VENV:-/tmp/py312}"  # outside /kaggle/working, so it is not saved as output
# cu126 wheels: contain kernels for the T4 (sm_75) and run on every driver >= 525, incl. newer 5xx drivers.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
UV_VERSION="${UV_VERSION:-0.13.8}"
REPO_REF="${REPO_REF:-d54c67aad513cb9daca444afa425cfb278e4fbf8}"  # pinned upstream commit (20.09.2026)
REPO="git+https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model.git@$REPO_REF"
PY="$VENV/bin/python"

status() {  # always printed, also on the "already done" path
  "$PY" - <<'PY'
import sys, torch, importlib.metadata as m
print("Python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA-Build", torch.version.cuda,
      "| terratorch", m.version("terratorch"), "| GPU sichtbar:", torch.cuda.is_available())
PY
}
repo_ok()  { [ -x "$PY" ] && "$PY" -c "import ni_lfm, terratorch_integration" 2>/dev/null; }
torch_ok() {  # importable torch; with REQUIRE_CUDA=1 it must be a CUDA build (CPU-only wheels have torch.version.cuda None)
  [ -x "$PY" ] && "$PY" -c "import torch, torchvision, sys; sys.exit(0 if ($REQUIRE_CUDA == 0 or torch.version.cuda) else 1)" 2>/dev/null; }
install_torch() {
  echo "Installiere PyTorch-Räder von $TORCH_INDEX ..."
  uv pip install --quiet --python "$PY" --reinstall-package torch --reinstall-package torchvision \
    --index-url "$TORCH_INDEX" "torch==2.12.1" "torchvision==0.27.1"
  # Pin these exact builds so the repo install cannot swap them for other CUDA builds.
  "$PY" -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > "$VENV/pins.txt"
}
install_repo() {
  uv pip install --quiet --python "$PY" --constraint "$VENV/pins.txt" \
    --index-url https://pypi.org/simple --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
    "$REPO" "terratorch[peft]==1.2.13"
}

FORCE=0; [ "${1:-}" = "--force-torch" ] && FORCE=1
if [ $FORCE = 0 ] && repo_ok && torch_ok; then
  echo "Umgebung $VENV ist schon fertig."; status; exit 0; fi
pip -q install "uv==${UV_VERSION}"
[ -x "$PY" ] || uv venv --quiet --python 3.12 "$VENV"
if [ $FORCE = 1 ] || ! torch_ok; then install_torch; fi
[ -f "$VENV/pins.txt" ] || install_torch
repo_ok || install_repo
torch_ok || { echo "FEHLER: torch in $VENV ist kein CUDA-Build."; status; exit 1; }
status
