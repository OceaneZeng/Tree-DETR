#!/usr/bin/env bash
# Create the Linux CUDA environment and build the MS-DeformAttn extension.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/environment.yml}"
ENV_NAME="${ENV_NAME:-tree-detr}"
CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-8.6}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found in PATH." >&2
    exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: environment file not found: ${ENV_FILE}" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Updating conda environment: ${ENV_NAME}"
    conda env update --name "${ENV_NAME}" --file "${ENV_FILE}"
else
    echo "Creating conda environment: ${ENV_NAME}"
    conda env create --name "${ENV_NAME}" --file "${ENV_FILE}"
fi

export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
echo "Building MS-DeformAttn for CUDA arch ${TORCH_CUDA_ARCH_LIST}"
conda run --no-capture-output --name "${ENV_NAME}" \
    python "${PROJECT_ROOT}/models/ops/setup.py" build_ext --inplace

conda run --no-capture-output --name "${ENV_NAME}" \
    python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__); print('cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

echo "Environment ready: ${ENV_NAME}"
echo "Activate with: conda activate ${ENV_NAME}"
