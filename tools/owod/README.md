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
  --output-dir exps/owod/s-owodb/order0/prob/stage_0
```

Available method names are `vanilla_d_detr`, `ore_star`, `ow_detr`, `prob`, and
`oracle`. They must only be reported as paper baselines after the corresponding
paper-faithful implementation is verified; a shared detector with a renamed
unknown score is not a valid reproduction. Every run writes the exact command and metadata to the output
directory, while `main.py` writes both a human-readable `train.log` and
structured per-epoch `log.txt`. When a full-label validation annotation and a
manifest are supplied, the same log also includes `owod_u_recall`,
`owod_a_ose`, `owod_wi`, `owod_udr`, and `owod_udp`.
