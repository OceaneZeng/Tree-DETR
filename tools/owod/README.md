# OWOD baselines

This directory replaces the removed IOD baseline launchers with the primary
OWOD experiment entry points.

## Build a protocol

```bash
python tools/owod/build_protocol.py \
  --coco-root data/coco \
  --output-root data/coco-owod/s-owodb/order0 \
  --protocol s-owodb --order random --seed 42
```

S-OWODB (Strict OWODB) keeps semantically related COCO categories in the same
super-category-based task partition, avoiding the mixed-supercategory leakage
of historical M-OWODB. For exact S-OWODB reproduction, the official class
grouping must be passed through `--groups-json`; the builder refuses to invent
a grouping unless `--allow-heuristic-groups` is explicitly used for a smoke
test. M-OWODB is the historical mixed-supercategory protocol with four
20-class stages.

## Run a baseline

```bash
python tools/owod/run_baseline.py \
  --method prob \
  --coco-path data/coco \
  --train-ann data/coco-owod/s-owodb/order0/stage_0/instances_train2017.json \
  --val-ann data/coco-owod/s-owodb/order0/stage_0/instances_val2017_full.json \
  --manifest data/coco-owod/s-owodb/order0/split_manifest.json \
  --stage 0 \
  --output-dir exps/owod/s-owodb/order0/prob/stage_0 \
  --gpus 0,1 --nproc-per-node 2
```

Available method names are `vanilla_d_detr`, `ore_star`, `ow_detr`, `prob`, and
`oracle`. They must only be reported as paper baselines after the corresponding
paper-faithful implementation is verified; a shared detector with a renamed
unknown score is not a valid reproduction. Every run writes the exact command and metadata to the output
directory. Every runner writes one human-readable `train.log`; `main.py` writes
compact per-epoch records to `metrics.jsonl` in the same experiment directory.
Progress logs show only primary losses while `metrics.jsonl` retains all
auxiliary losses. A fresh run archives stale logs under `log_archive/`; a run
with `--resume` appends to the active logs. External `tee` is not required.
When a full-label validation annotation and a
manifest are supplied, the same log also includes `owod_u_recall`,
`owod_a_ose`, `owod_wi`, `owod_udr`, and `owod_udp`.

## Graph-local OWOD idea

The main method is a trainable side-car GNN, not classifier-prototype cosine.
It does not run in the detector inference path. First, a completed earlier
detector is probed on that stage's training data. Decoder-FFN gradient sketches
are the class-node features and the measured increase in target-class training
loss after a short source-class LoRA update is the directed edge supervision.
No validation annotations are used for either feature extraction or harm
labels. Per-class sketches and per-source harm rows are cached, so an
interrupted calibration can be resumed by running the same command.

Before the full calibration, a pipeline-only smoke test can add
`--source-limit 2 --sketch-max-images 2 --probe-max-images 2 --gnn-epochs 5`
and use a separate output directory. Its checkpoint is marked
`production_ready=false` and is intentionally rejected by the Stage 1 runner.

```bash
CAL="$PWD/exps/owod/m-owodb/order0/graph_local/gnn_calibration_stage0"
mkdir -p "$CAL"
CUDA_VISIBLE_DEVICES=0 python tools/owod/calibrate_interference_gnn.py \
  --coco_path "$PWD/data/coco" \
  --manifest "$PWD/data/coco-owod/m-owodb/order0/split_manifest.json" \
  --stage 0 \
  --checkpoint "$PWD/exps/owod/m-owodb/order0/vanilla_d_detr/stage_0_50ep/checkpoint.pth" \
  --output-dir "$CAL" \
  --owod-baseline vanilla_d_detr --num_classes 91 \
  --batch_size 1 --num_workers 4 --device cuda \
  --sketch-max-images 12 --probe-max-images 12 --probe-steps 3 \
  --gnn-epochs 400
```

The calibration writes `empirical_stage0.pt`, `gnn_stage0.pt`,
`calibration.log`, `calibration_summary.json`, and
`calibration_complete.json`. The summary includes a held-out-source check;
inspect it before treating the GNN as a useful estimator.

For Stage 1, gradient sketches for old classes come from Stage 0 retained
training data and sketches for current classes come from the Stage 1 increment
training data. The GNN predicts all directed current-to-old edges. Each old
class receives its maximum predicted risk over current classes and
`--graph-k` selects the total stage-level Top-K replay neighborhood. At stage
`t > 0`, replay exemplars are drawn from stage `t - 1`, never from the current
increment annotation.

```bash
export CUDA_VISIBLE_DEVICES=0,1
OUT="$PWD/exps/owod/m-owodb/order0/graph_local/stage_1_gnn_k5_v1"
mkdir -p "$OUT"
python tools/owod/run_graph_local_increment.py \
  --coco-path "$PWD/data/coco" \
  --manifest "$PWD/data/coco-owod/m-owodb/order0/split_manifest.json" \
  --stage 1 \
  --checkpoint "$PWD/exps/owod/m-owodb/order0/vanilla_d_detr/stage_0_50ep/checkpoint.pth" \
  --output-dir "$OUT" \
  --owod-baseline vanilla_d_detr \
  --graph-estimator gnn \
  --gnn-checkpoint "$CAL/gnn_stage0.pt" \
  --graph-k 5 --replay-budget 256 \
  --epochs 20 --batch-size 2 --num-workers 4 --eval-interval 5 \
  --gpus 0,1 --nproc-per-node 2 --master-port 29561
```

`graph.json` records feature provenance, GNN checkpoint metadata, all
current-to-old scores, the aggregated ranking, and selected replay classes.
`gnn_node_features.pt` stores the exact inference-stage node features. Detector
logs are `graph/train.log` and `graph/metrics.jsonl`. Exact commands and
configuration are stored in `command.txt`, `run_config.json`, and
`run_history/`.

Prototype cosine remains available only as an ablation with an explicit flag:

```bash
python tools/owod/run_graph_local_increment.py ... \
  --graph-estimator cosine --graph-k 5
```

Do not use the legacy cosine-generated `gnn_stage.pt` files as GNN training
labels. The runner rejects checkpoints without
`supervision=empirical_train_loss_increase`. To continue an interrupted
detector arm, keep the same output directory and add
`--resume "$OUT/graph/checkpoint.pth"`, while leaving `--checkpoint` pointed at
the previous completed stage detector.
