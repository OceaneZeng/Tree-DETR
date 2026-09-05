#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MAN="$ROOT/data/coco-owod/m-owodb/order0/split_manifest_deus.invalid.json"
BASE="$ROOT/exps/owod/m-owodb/order0/vanilla_d_detr/stage_0_50ep/checkpoint.pth"
GNN="$ROOT/exps/owod/m-owodb/order0/pilot_unverified/full_gnn_calibration_stage0_v2/gnn_stage0.pt"
OUT="${OUT:-$ROOT/exps/owod/m-owodb/order0/pilot_unverified/full_three_module_stage1_v1}"
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
export PYTHONPATH="$ROOT/models/ops:$ROOT:${PYTHONPATH:-}"
TORCH_LIB="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

python -c 'import torch; import MultiScaleDeformableAttention as msda; print("PyTorch:", torch.__version__); print("MSDA:", msda.__file__)'

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
  --neighbor-scoped-lora \
  --lora-rank 8 \
  --teacher-completion \
  --teacher-score-threshold 0.5 \
  --local-margin-coef 0.5 \
  --local-margin 1.0 \
  --off-projection-coef 0.1 \
  --off-basis-rank 8 \
  --lr 1e-4 \
  --epochs 20 \
  --lr-drop 15 \
  --batch-size 2 \
  --num-workers 4 \
  --seed 42 \
  --eval-interval 5 \
  --gpus 0,1 \
  --nproc-per-node 2 \
  --master-port "$MASTER_PORT"
