#!/usr/bin/env bash
# Full S-OWODB discovery v4, independent of control and historical discovery v3.
set -euo pipefail
for arg in "$@"; do
  case "$arg" in
    --resume|--dry-run) ;;
    *) echo 'Accepted options: --resume --dry-run. Configuration overrides are disabled.' >&2; exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -c 'from models.graph_local.discovery import DISCOVERY_VERSION; assert DISCOVERY_VERSION == 4, "This launcher requires discovery v4"'
mkdir -p output/tree-detr-s-v4-logs
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python -u tools/owod/run_benchmark.py \
  --manifest data/owod/s-owodb/split_manifest.json \
  --recipe ow-detr-paper --method discovery \
  --backbone-weights pretrained/dino_resnet50.pth \
  --output-dir output/tree-detr-s-v4-full \
  --gpus 0,1 --nproc-per-node 2 --batch-size 2 --num-workers 4 \
  --seed 42 --activation-checkpointing \
  --graph-routing hard --graph-localization frozen \
  --discovery-scoring hierarchical "$@" \
  2>&1 | tee -a output/tree-detr-s-v4-logs/console.log
