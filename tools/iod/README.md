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
  --checkpoint exps/iod/40+20x2/order0/stage0_base/checkpoint.pth \
  --coco-root data/coco \
  --old-ann data/coco-iod/40+20x2/order0/stage_0/instances_train2017.json \
  --new-ann data/coco-iod/40+20x2/order0/stage_1/instances_train2017.json \
  --old-classes <stage-0-source-ids> --new-classes <stage-1-source-ids> \
  --output exps/iod/order0/risk.json --max-images 20
```

The output contains the positive-conflict matrix and one risk value per old
class. It must be evaluated against a later measured old-class AP drop before
being used for replay allocation.

To measure actual forgetting before adding any RCGC protection, run the
unprotected increment probe. It uses the stage-1 annotation containing only
new-class images; no old exemplar or teacher consolidation is included:

```bash
GPU_LIST=0,1 \
SPLIT_ROOT="$PWD/data/coco-iod/40+20x2/order0" \
BASE_CHECKPOINT="$PWD/exps/iod/40+20x2/order0/stage0_base/checkpoint.pth" \
bash tools/iod/run_stage1_unprotected.sh
```

The matched-budget uniform-replay control can then be run from the same base
checkpoint. It selects old images from the stage-0 training split only and
keeps the total replay budget fixed:

```bash
GPU_LIST=0,1 REPLAY_BUDGET=400 \
SPLIT_ROOT="$PWD/data/coco-iod/40+20x2/order0" \
BASE_CHECKPOINT="$PWD/exps/iod/40+20x2/order0/stage0_base/checkpoint.pth" \
bash tools/iod/run_stage1_uniform_replay.sh
```

This is a baseline, not the RCGC claim: it uses uniform class-balanced replay
and the existing detector loss, without risk weighting or teacher
consolidation. Its output is kept under `stage1_uniform_replay` so it can be
compared with the later risk-conditioned arm at identical epochs and budget.

After `estimate_conflict_risk.py` has produced `risk_full.json`, run the
risk-weighted replay arm with the same budget:

```bash
GPU_LIST=0,1 \
RISK_JSON="$PWD/exps/iod/40+20x2/order0/risk_full.json" \
SPLIT_ROOT="$PWD/data/coco-iod/40+20x2/order0" \
BASE_CHECKPOINT="$PWD/exps/iod/40+20x2/order0/stage0_base/checkpoint.pth" \
bash tools/iod/run_stage1_risk_replay.sh
```

This arm isolates the replay-allocation effect. It deliberately does not
claim the full RCGC result until risk-weighted teacher consolidation is added
and compared at the same budget.

Evaluate its final checkpoint with the existing single-GPU evaluation pattern,
using stage-1's `instances_val2017.json`. The per-class AP drop from stage-0
to this evaluation is the target for validating the M1 risk vector.

After both evaluations, calculate the pre-registered diagnostic:

```bash
python tools/iod/analyze_risk_drop.py \
  --base-eval exps/iod/40+20x2/order0/stage0_base_eval/eval.pth \
  --increment-eval exps/iod/40+20x2/order0/stage1_unprotected_eval/eval.pth \
  --risk exps/iod/40+20x2/order0/risk_full.json \
  --output exps/iod/40+20x2/order0/risk_drop_analysis.json
```

The report includes risk/AP-drop Spearman correlation, risk/base-AP
correlation (a confounding check), top-k harm coverage, mean old-class AP50
before/after the increment, and a saturation diagnostic. Treat
`diagnostic_status=saturated_forgetting` as an invalid setting for ranking
class-specific harm: the unprotected update erased too many old classes to
leave a useful AP-drop distribution.

## Continuation after saturated forgetting

First evaluate the already-saved epoch-4 checkpoint. This costs only one
evaluation and shows whether the final epoch-9 result merely saturated an
earlier, more informative forgetting trajectory:

```bash
CHECKPOINT="$PWD/exps/iod/40+20x2/order0/stage1_unprotected/checkpoint0004.pth" \
OUTPUT_DIR="$PWD/exps/iod/40+20x2/order0/stage1_unprotected_epoch4_eval" \
bash tools/iod/eval_increment.sh
```

Then run a conservative uniform-replay pilot. Incremental learning rates are
explicit environment variables so all later arms can use the identical
configuration:

```bash
GPU_LIST=0,1 REPLAY_BUDGET=400 EPOCHS=5 \
LR=2e-5 LR_BACKBONE=2e-6 LR_DROP=4 \
OUTPUT_DIR="$PWD/exps/iod/40+20x2/order0/stage1_uniform_replay_lr2e5_e5" \
bash tools/iod/run_stage1_uniform_replay.sh
```

Evaluate the pilot before spending compute on more arms:

```bash
CHECKPOINT="$PWD/exps/iod/40+20x2/order0/stage1_uniform_replay_lr2e5_e5/checkpoint.pth" \
OUTPUT_DIR="$PWD/exps/iod/40+20x2/order0/stage1_uniform_replay_lr2e5_e5_eval" \
bash tools/iod/eval_increment.sh
```

If the pilot avoids the old-class AP50 floor, freeze this configuration and
run matched stabilized-unprotected and risk-replay arms under new output
directories. Do not compare an arm at `2e-5` with the original unprotected
result at `2e-4` as evidence for the replay allocation method.

Before estimating risk, train the stage-0 base detector. This uses the official
checkpoint only for shared detector initialization and discards its classifier
rows, preventing the future increment classes from leaking into the base model:

```bash
GPU_LIST=0,1 \
SPLIT_ROOT="$PWD/data/coco-iod/40+20x2/order0" \
CHECKPOINT="$PWD/pretrained/r50_deformable_detr-checkpoint.pth" \
bash tools/iod/run_stage0_baseline.sh
```

The DDP runner skips validation by default because this repository's legacy
COCO evaluator gathers large Python objects through NCCL. Evaluate the saved
checkpoint on one GPU after training:

```bash
SPLIT_ROOT="$PWD/data/coco-iod/40+20x2/order0" \
CHECKPOINT="$PWD/exps/iod/40+20x2/order0/stage0_base/checkpoint.pth" \
bash tools/iod/eval_stage0_baseline.sh
```
