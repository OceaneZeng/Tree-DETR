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
directory, while `main.py` writes both a human-readable `train.log` and
structured per-epoch `log.txt`. When a full-label validation annotation and a
manifest are supplied, the same log also includes `owod_u_recall`,
`owod_a_ose`, `owod_wi`, `owod_udr`, and `owod_udp`.

## Graph-local OWOD idea

Run one increment from an earlier detector checkpoint with the graph-local
replay controller. The first increment uses a deterministic cosine graph over
the checkpoint classifier prototypes; selected old classes become a balanced
replay set. Full-label validation keeps the normal COCO AP and OWOD metrics in
the detector log.

```bash
OUT="$PWD/exps/owod/m-owodb/order0/graph_local/stage_1_cosine"
mkdir -p "$OUT"
python tools/owod/run_graph_local_increment.py \
  --coco-path "$PWD/data/coco" \
  --manifest "$PWD/data/coco-owod/m-owodb/order0/split_manifest.json" \
  --stage 1 \
  --checkpoint "$PWD/exps/owod/m-owodb/order0/vanilla_d_detr/stage_0_50ep/checkpoint.pth" \
  --output-dir "$OUT" \
  --owod-baseline vanilla_d_detr --epochs 20 --batch-size 2 \
  --gpus 0,1 --nproc-per-node 2 --master-port 29561 \
  2>&1 | tee "$OUT/launcher.log"
```

Train the side-car GNN only from completed earlier graph artifacts:

```bash
python tools/graph_local/train_gnn.py \
  --stages "$PWD/exps/owod/m-owodb/order0/graph_local/stage_1_cosine/gnn_stage.pt" \
  --output "$PWD/exps/owod/m-owodb/order0/graph_local/gnn_stage_1.pt" \
  --log-file "$PWD/exps/owod/m-owodb/order0/graph_local/gnn_train.log"
```

Pass `--graph-estimator gnn --gnn-checkpoint ...` to the stage 2 command and
use the stage 1 detector checkpoint. The runner creates per-arm `train.log`
and `console.log`; create the parent directory before an external `tee`.
To continue an interrupted arm, keep the same output directory and add
`--resume "$OUT/graph/checkpoint0004.pth"` (or the latest saved checkpoint),
while leaving `--checkpoint` pointed at the previous completed stage.
