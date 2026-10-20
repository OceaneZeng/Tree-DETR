#!/usr/bin/env bash
# Fixed v2 experiment. Activate tree-detr first. Options: --resume, --dry-run.
set -euo pipefail
resume=false
for arg in "$@"; do
  case "$arg" in
    --resume) resume=true ;;
    --dry-run) ;;
    *) echo 'Accepted options: --resume --dry-run. Configuration/path overrides are disabled.' >&2; exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if "$resume"; then
  for artifact in \
    output/task1-control-v2-full/s-owodb/seed_42/benchmark_plan.json \
    output/task1-control-v2-full/s-owodb/seed_42/stage_0/increment/checkpoint.pth; do
    if [[ ! -s "$artifact" ]]; then
      echo "Cannot resume v2: missing or empty $artifact" >&2
      exit 2
    fi
  done
fi
mkdir -p output/task1-control-v2-logs
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python -u tools/owod/run_benchmark.py \
  --task1-only --method control \
  --manifest data/owod/s-owodb/split_manifest.json \
  --recipe ow-detr-paper --backbone-weights pretrained/dino_resnet50.pth \
  --output-dir output/task1-control-v2-full \
  --gpus 1,2 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
  --seed 42 --activation-checkpointing "$@" \
  2>&1 | tee -a output/task1-control-v2-logs/console.log
