#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MAN="$ROOT/data/coco-owod/m-owodb/order0/split_manifest_deus.invalid.json"
BASE="$ROOT/exps/owod/m-owodb/order0/vanilla_d_detr/stage_0_50ep/checkpoint.pth"
GNN="$ROOT/exps/owod/m-owodb/order0/pilot_unverified/full_gnn_calibration_stage0_v2/gnn_stage0.pt"
OUT="${OUT:-$ROOT/exps/owod/m-owodb/order0/pilot_unverified/full_gnn_stage1_retention_v3_smoke10}"
MASTER_PORT="${MASTER_PORT:-29565}"

for required in "$MAN" "$BASE" "$GNN"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

python tools/owod/run_graph_local_increment.py \
  --coco-path "$ROOT/data/coco" \
  --manifest "$MAN" \
  --allow-unverified-protocol \
  --stage 1 \
  --checkpoint "$BASE" \
  --gnn-checkpoint "$GNN" \
  --output-dir "$OUT" \
  --control graph \
  --graph-k 5 \
  --graph-aggregation top_mean \
  --graph-aggregation-top-n 3 \
  --base-exemplars-per-class 10 \
  --risk-extra-exemplars-per-class 40 \
  --replay-sampling-fraction 0.10 \
  --old-class-distillation \
  --distill-class-coef 2 \
  --distill-bbox-coef 5 \
  --distill-score-threshold 0.3 \
  --distill-max-queries 20 \
  --lr 5e-5 \
  --lr-backbone 5e-6 \
  --epochs 10 \
  --lr-drop 8 \
  --batch-size 2 \
  --num-workers 4 \
  --seed 42 \
  --eval-interval 2 \
  --gpus 0,1 \
  --nproc-per-node 2 \
  --master-port "$MASTER_PORT" \
  2>&1 | tee "$OUT/launcher.log"
