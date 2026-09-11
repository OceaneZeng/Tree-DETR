# PROB, CAT and OWOBJ reproduction status

External implementations live only under `baselines/`; shared data lives under
`data/`; every checkpoint, metric, manifest, and log lives under the root-level
`exps/`. See `project-layout.md` for the ownership rules.

The pinned repositories are PROB (`orrzohar/PROB`) and OWOBJ
(`AI4Math-ShanZhang/OWOBJ`). CAT's published repository URL currently returns
404, so CAT remains `pending_source` instead of being approximated locally.

Initialize sources and prepare the common official M-OWODB data view:

```bash
git submodule update --init --recursive

python tools/owod/prepare_shared_mowodb.py \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --train-coco "$PWD/data/coco/annotations/instances_train2017.json" \
  --val-coco "$PWD/data/coco/annotations/instances_val2017.json" \
  --image-root "$PWD/data/coco/train2017" \
  --image-root "$PWD/data/coco/val2017" \
  --image-mode symlink \
  --output "$PWD/data/derived/m-owodb-voc" \
  --baseline-root "$PWD/baselines" \
  --clean
```

Apply `baselines/patches/owobj_shared_mowodb.patch` to the pinned OWOBJ checkout.
It removes author-machine data paths without changing the model or losses.

Generate an auditable command manifest:

```bash
python tools/owod/prepare_external_baselines.py \
  --methods prob owobj cat \
  --source-root "$PWD/baselines" \
  --shared-mowodb \
  --gpus 0,1 \
  --output "$PWD/exps/owod/m-owodb/order0/official/baselines/run_manifest.json"
```

The shared scripts require `MOWODB_DATA_ROOT` and `MOWODB_OUTPUT_ROOT`. This
ensures the external source directories remain source-only and all durable run
artifacts are written beneath the project's canonical `exps/` tree.
