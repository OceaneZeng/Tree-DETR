# COCO incremental protocol

Build a reproducible, disjoint-image COCO split:

```bash
python tools/iod/coco_incremental.py \
  --coco-root data/coco \
  --output-root data/coco-iod/40+20x2/order0 \
  --protocol 40+20x2 --order random --seed 42
python tools/iod/validate_split.py \
  data/coco-iod/40+20x2/order0/split_manifest.json
```

For the three fixed category orders used in the experiments:

```bash
COCO_ROOT=data/coco OUTPUT_ROOT=data/coco-iod \
  bash tools/iod/build_coco_protocols.sh --protocol 40+20x2
```

The COCO images remain under `--coco-root`; generated annotation JSON files
contain relative image metadata and can be passed to the existing loader after
pointing each stage's annotation path at the generated file.  The manifest
contains the source category order, stage classes, disjoint training image
IDs, and the fixed 10% memory selection.  It is the source of truth for
reproducibility. Official COCO category IDs are preserved; the random order
only changes which stage owns each category.

Estimate the first-stage risk signal without changing the checkpoint:

```bash
python tools/iod/estimate_conflict_risk.py \
  --checkpoint pretrained/r50_deformable_detr-checkpoint.pth \
  --coco-root data/coco \
  --old-ann data/coco-iod/40+20x2/order0/stage_0/instances_train2017.json \
  --new-ann data/coco-iod/40+20x2/order0/stage_1/instances_train2017.json \
  --old-classes <stage-0-source-ids> --new-classes <stage-1-source-ids> \
  --output exps/iod/order0/risk.json --max-images 20
```

The output contains the positive-conflict matrix and one risk value per old
class. It must be evaluated against a later measured old-class AP drop before
being used for replay allocation.
