#!/usr/bin/env bash
# Remove generated artifacts from the retired Pet/graph-local experiments.
# Source code and result logs are intentionally preserved.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY=0
REMOVE_PET_DATA=0

usage() {
    echo "Usage: bash tools/cleanup_legacy_artifacts.sh [--apply] [--remove-pet-data]"
    echo "  default: print exact targets without deleting"
    echo "  --apply: delete old experiment checkpoints and Python caches"
    echo "  --remove-pet-data: with --apply, delete data/oxford-pet-tree-detr"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --remove-pet-data) REMOVE_PET_DATA=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

TARGET_EXP="${PROJECT_ROOT}/exps/pet_baseline_diagnostics"
TARGET_PET="${PROJECT_ROOT}/data/oxford-pet-tree-detr"

mapfile -t CHECKPOINTS < <(find "${TARGET_EXP}" -type f -name '*.pth' -print 2>/dev/null | sort)
mapfile -t CACHES < <(find "${PROJECT_ROOT}" -type d -name '__pycache__' -print 2>/dev/null | sort)

echo "Legacy checkpoints: ${#CHECKPOINTS[@]}"
printf '%s\n' "${CHECKPOINTS[@]}"
echo "Python caches: ${#CACHES[@]}"
printf '%s\n' "${CACHES[@]}"
if [[ "${REMOVE_PET_DATA}" == 1 ]]; then
    echo "Oxford-Pet data: ${TARGET_PET}"
fi

if [[ "${APPLY}" != 1 ]]; then
    echo "Dry run only. Add --apply to delete these generated artifacts."
    exit 0
fi

for path in "${CHECKPOINTS[@]}"; do
    [[ -n "${path}" ]] && rm -f -- "${path}"
done
for path in "${CACHES[@]}"; do
    [[ -n "${path}" && "${path}" == "${PROJECT_ROOT}"/* ]] && rm -rf -- "${path}"
done
if [[ "${REMOVE_PET_DATA}" == 1 && -d "${TARGET_PET}" ]]; then
    rm -rf -- "${TARGET_PET}"
fi
echo "Cleanup complete. Logs, summaries, source code, and pretrained/ were preserved."
