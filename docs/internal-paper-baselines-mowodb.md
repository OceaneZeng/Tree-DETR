# Internal PROB / CAT / OWOBJ baselines

The three comparison arms run inside Tree-DETR.  They share the repository
Deformable-DETR data loader, M-OWODB stage annotations, optimizer, evaluator,
checkpoint format, and logging.  The external `baselines/` checkouts are kept
only as paper/source references and are never imported by the training entry.
The default architecture is the repository recipe (ResNet-50, six encoder and
decoder layers, 300 queries), so the comparison is aligned with the own-idea
experiments rather than an author script's query count.

## Run

Use the existing `tree-detr` environment and the two 3090 devices.  The local
manifest may be used for a consistency run with `--allow-unverified`:

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
source /home/top/miniconda3/etc/profile.d/conda.sh
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
mkdir -p "$PWD/exps/owod/m-owodb/order0/pilot_unverified/internal_baselines"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
python -u tools/owod/run_mowodb_reimplementations.py \
  --methods ow-detr prob owobj cat \
  --manifest "$PWD/data/coco-owod/m-owodb/order0/split_manifest.json" \
  --coco-path "$PWD/data/coco" \
  --cat-proposals "$PWD/data/derived/m-owodb-cat/selective_search.sqlite3" \
  --output-dir "$PWD/exps/owod/m-owodb/order0/pilot_unverified/internal_baselines" \
  --gpus 0,1 \
  --allow-unverified \
  2>&1 | tee "$PWD/exps/owod/m-owodb/order0/pilot_unverified/internal_baselines/launcher.log"
```

Run a read-only preflight first by adding `--dry-run`.  The launcher records
the exact command and source/data hashes in `reimplementation_plan.json` and
refuses to mix a changed recipe into an existing output directory.  It also
checks that the active PyTorch build contains kernels for the visible GPUs.

Each method has `stage_0` through `stage_3`.  Every stage contains `train.log`,
`metrics.jsonl`, `run_config.json`, `checkpoint.pth`, `train.json`, and
`training_complete.json`.  CAT additionally needs the selective-search SQLite
cache.  Resume an interrupted queue with the same command plus `--resume`.

Summarize without loading a model:

```bash
python tools/owod/run_mowodb_reimplementations.py \
  --methods ow-detr prob owobj cat \
  --output-dir "$PWD/exps/owod/m-owodb/order0/pilot_unverified/internal_baselines" \
  --summarize
```

PROB uses the BN feature-energy objectness from the paper.  OWOBJ uses the
stochastic sketch energy and energy-margin loss.  Both are explicitly marked
as local paper reimplementations; their scores are not author-repository
results and should be compared with the same local Tree-DETR protocol.
