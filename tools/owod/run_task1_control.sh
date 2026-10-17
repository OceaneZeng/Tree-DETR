#!/usr/bin/env bash
# Fresh v2 run only. Activate tree-detr first. Optional argument: --dry-run.
set -euo pipefail
if (( $# > 1 )) || (( $# == 1 )) && [[ "${1:-}" != "--dry-run" ]]; then
  echo 'Fresh launch only: accepted argument is --dry-run. Resume and output overrides are disabled.' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p output/task1-control-v2-logs
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python -u tools/owod/run_benchmark.py \
  --task1-only --method control \
  --manifest data/owod/s-owodb/split_manifest.json \
  --recipe ow-detr-paper --backbone-weights pretrained/dino_resnet50.pth \
  --output-dir output/task1-control-v2-full \
  --gpus 0,1 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
  --seed 42 --activation-checkpointing "$@" \
  2>&1 | tee -a output/task1-control-v2-logs/console.log
