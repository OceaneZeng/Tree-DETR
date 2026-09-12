# CAT and EW-DETR paper reimplementations on M-OWODB

This repository contains from-paper reimplementations of CAT (CVPR 2023) and
EW-DETR (CVPR 2026). They use the same validated M-OWODB 20/20/20/20 class
order, COCO images, stage annotations, and local evaluator as Tree-DETR.

These are not author-code reproductions. CAT's published repository currently
returns HTTP 404. No EW-DETR author repository was located. Results must be
reported as `CAT (paper reimplementation)` and `EW-DETR (paper
reimplementation)`.

## Implemented paper components

CAT:

- equations (1)-(2): one shared Deformable Transformer decoder is executed
  twice; first-pass embeddings predict boxes and become second-pass class
  queries;
- equation (3): attention-driven confidence and maximum selective-search IoU
  are combined with adaptive exponents;
- equations (5)-(8) and Algorithm 1: a checkpointed loss-memory controller
  updates model-driven/input-driven weights, initially 0.8/0.2;
- equation (9): localization, identification, and objectness losses;
- 100 queries, five pseudo unknowns, and top-50 inference detections.

EW-DETR:

- rank-16 task and aggregate LoRA adapters on transformer linear layers;
- data-aware merge with bounds 0.2/0.8, truncated SVD, and task reset;
- Query-Norm Objectness Adapter;
- EUMix from main-paper equations (7)-(11) and supplement Appendix A;
- exemplar-free stage transitions through consolidated checkpoints.

## Explicit transfer assumptions

The comparison target is M-OWODB, not EW-DETR's cross-domain EWOD benchmark.
Consequently both methods use the same M-OWODB stages and validation files.
The default runner uses a shared 50/20/20/20 epoch schedule, two GPUs, batch 2
per GPU, six encoder/decoder layers, 100 queries, and seed 42. These values are
recorded in `reimplementation_plan.json` and can be overridden from the CLI.

The CAT paper does not publish the loss-memory length, recent window, update
period, momentum amplitudes, selective-search mode, replay budget, learning
rate, or epoch schedule. Defaults are therefore declared parameters rather
than attributed to the authors. The second decoder pass uses the final
location embeddings as decoder state with the original reference points.
CAT replay is implemented as a bounded, class-balanced memory mixed into
ordinary training batches (10% replay samples after stage 0 by default). It is
an explicit implementation assumption, not a claim that the paper specifies
an independent balanced fine-tuning pass.

The shared optimizer is AdamW with learning rate `2e-4` and weight decay
`1e-4`. Both are explicit runner arguments and are recorded in the generated
commands and run configurations.

The EW-DETR supplement uses 100 epochs, batch 16, one encoder/decoder layer,
and a DINO-pretrained backbone for its original EWOD experiments. The M-OWODB
runner defaults to the shared comparison schedule and architecture above.
The current repository backbone loader uses ImageNet ResNet-50 rather than an
author-supplied DINO detector checkpoint; this avoids future-class detector
supervision but remains a material reproduction difference.

## Server preparation

The official manifest must have `protocol=m-owodb`,
`official_annotations=true`, and `paper_comparable=true`:

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
source /home/top/miniconda3/etc/profile.d/conda.sh
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr

python -m pip install "opencv-contrib-python-headless>=4.8,<5"

python tools/owod/prepare_cat_proposals.py \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --coco-path "$PWD/data/coco" \
  --output "$PWD/data/derived/m-owodb-cat/selective_search.sqlite3" \
  --mode fast \
  --resume
```

Proposal generation is resumable and writes only to `data/derived/`. Training
checkpoints and logs are written only below the root `exps/` directory.

Before proposal generation or training, run the focused checks once:

```bash
python -m pytest \
  tools/owod/tests/test_cat_reimplementation.py \
  tools/owod/tests/test_paper_baselines.py -q

python tools/owod/smoke_paper_baselines.py
```

Run a complete read-only preflight before allocating GPUs:

```bash
python tools/owod/run_mowodb_reimplementations.py \
  --methods cat ew-detr \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --coco-path "$PWD/data/coco" \
  --cat-proposals "$PWD/data/derived/m-owodb-cat/selective_search.sqlite3" \
  --output-dir "$PWD/exps/owod/m-owodb/order0/official/baselines" \
  --gpus 0,1 \
  --dry-run
```

## tmux training commands

CAT and EW-DETR each use GPUs 0 and 1, so run them sequentially.

CAT:

```bash
tmux new -s mowodb-cat

cd /home/top/disks/new-hdd/zhy/Tree-DETR
source /home/top/miniconda3/etc/profile.d/conda.sh
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
set -o pipefail
mkdir -p "$PWD/exps/owod/m-owodb/order0/official/baselines/cat"

CUDA_VISIBLE_DEVICES=0,1 python -u tools/owod/run_mowodb_reimplementations.py \
  --methods cat \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --coco-path "$PWD/data/coco" \
  --cat-proposals "$PWD/data/derived/m-owodb-cat/selective_search.sqlite3" \
  --output-dir "$PWD/exps/owod/m-owodb/order0/official/baselines" \
  --gpus 0,1 \
  2>&1 | tee "$PWD/exps/owod/m-owodb/order0/official/baselines/cat/launcher.log"
```

Detach with `Ctrl-b`, then `d`. Reattach with `tmux attach -t mowodb-cat`.
The launcher is quiet while a child trainer is running. From another shell,
follow the active stage with:

```bash
tail -f "$PWD/exps/owod/m-owodb/order0/official/baselines/cat"/stage_*/console.log
```

EW-DETR, after CAT completes and both GPUs are idle:

```bash
tmux new -s mowodb-ew-detr

cd /home/top/disks/new-hdd/zhy/Tree-DETR
source /home/top/miniconda3/etc/profile.d/conda.sh
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
set -o pipefail
mkdir -p "$PWD/exps/owod/m-owodb/order0/official/baselines/ew-detr"

CUDA_VISIBLE_DEVICES=0,1 python -u tools/owod/run_mowodb_reimplementations.py \
  --methods ew-detr \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --coco-path "$PWD/data/coco" \
  --output-dir "$PWD/exps/owod/m-owodb/order0/official/baselines" \
  --gpus 0,1 \
  2>&1 | tee "$PWD/exps/owod/m-owodb/order0/official/baselines/ew-detr/launcher.log"
```

Monitor EW-DETR in another shell with:

```bash
tail -f "$PWD/exps/owod/m-owodb/order0/official/baselines/ew-detr"/stage_*/console.log
```

Resume an interrupted method by adding `--resume` to the identical command.
Do not change code, data, or configuration inside an existing plan; use a new
output directory for a changed experiment.

Summarize results:

```bash
python tools/owod/run_mowodb_reimplementations.py \
  --methods cat ew-detr \
  --output-dir "$PWD/exps/owod/m-owodb/order0/official/baselines" \
  --summarize
```

Primary outputs are `cat/stage_0` through `cat/stage_3` and
`ew-detr/stage_0` through `ew-detr/stage_3`. EW-DETR reports both task and
consolidated weights; use the consolidated row for the continual checkpoint
chain while retaining both rows in the experiment record.
