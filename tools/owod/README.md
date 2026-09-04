# Deformable DETR + GNN on the DEUS OWOD protocol

This project keeps Deformable DETR as its detector. It adopts the experimental
shape of DEUS (CVPR 2026): four sequential tasks on M-OWODB and S-OWODB,
class-balanced exemplar replay, full-label evaluation, and the Table 1 columns
Previous/Current/Known mAP, U-Rec, and H-Score.

The local method is not a reimplementation of DEUS. DEUS uses OrthogonalDet,
ETF-Subspace Unknown Separation (EUS), and Energy-based Known Distinction
(EKD). Here a trainable GNN estimates directed class interference and selects
old-class exemplars for Deformable DETR. The detector backbone is unchanged.

## 1. Register official annotations

The repository no longer synthesizes an OWODB split. Obtain the official
four-task annotation files and arrange them under `stage_0` through `stage_3`.
Each stage must contain:

- `instances_increment_train2017.json`
- `instances_train2017.json` (current data plus the official replay memory)
- `instances_val2017.json`
- `instances_val2017_full.json`

Then validate and register them:

```bash
python tools/owod/prepare_protocol.py \
  --annotation-root "$PWD/data/coco-owod/m-owodb/official" \
  --protocol m-owodb \
  --source-reference "<official-release-url-or-commit>" \
  --output "$PWD/data/coco-owod/m-owodb/split_manifest.json"
```

The importer requires four disjoint increments, cumulative known classes, all
80 COCO classes in every full validation file, and no repeated increment images.
It records SHA-256 hashes and the supplied official source reference. The last check
guards against the M-OWODB annotation duplication bug noted in DEUS Table 1.

## 2. External Table 1 baselines

ORE, OW-DETR, CAT, PROB, OrthogonalDet, O1O, OWOBJ, and DEUS are external
methods. They are not aliases for this repository's detector. The values
transcribed from DEUS Table 1 are stored in `deus_table1.json`:

```bash
python tools/owod/table1_reference.py --protocol m-owodb
python tools/owod/table1_reference.py --protocol s-owodb
```

For a claimed reproduction, run each method's real implementation on the same
validated annotations. The paper marks the M-OWODB PROB and OrthogonalDet rows
with a dagger because those two were rerun after correcting duplicated
annotations. Values in `deus_table1.json` are literature references, not local
measurements.

## 3. Deformable DETR control

The only locally implemented detector control is named
`deformable_detr_control`:

```bash
python tools/owod/run_detector_control.py \
  --coco-path "$PWD/data/coco" \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --stage 0 \
  --output-dir "$PWD/exps/owod/m-owodb/deformable_detr_control/stage_0" \
  --pretrained "$PWD/pretrained/r50_deformable_detr-checkpoint.pth" \
  --epochs 50 --batch-size 2 --num-workers 4 \
  --gpus 0,1 --nproc-per-node 2
```

This control uses `1 - max(known probability)` as a simple unknown score. It
must not be reported as ORE, PROB, OW-DETR, or DEUS.

## 4. Calibrate the GNN

Calibrate directed interference only from completed training-stage data. No
validation labels are used. Decoder FFN gradient sketches form class-node
features; the increase in target-class detector loss after a short source-class
LoRA probe supervises each edge.

```bash
CAL="$PWD/exps/owod/m-owodb/gnn/calibration_stage0"
CUDA_VISIBLE_DEVICES=0 python tools/owod/calibrate_interference_gnn.py \
  --coco_path "$PWD/data/coco" \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --stage 0 \
  --checkpoint "$PWD/exps/owod/m-owodb/deformable_detr_control/stage_0/checkpoint.pth" \
  --output-dir "$CAL" --num_classes 91 \
  --batch_size 1 --num_workers 4 --device cuda \
  --sketch-max-images 12 --probe-max-images 12 --probe-steps 3 \
  --gnn-epochs 400
```

The calibration writes `gnn_stage0.pt`, empirical labels, a held-out-source
summary, and a completion marker. A smoke calibration made with
`--source-limit` is marked non-production and is rejected by the runner.

## 5. Run the GNN method

DEUS states that replay stores a fixed number of exemplars per class, but the
main paper defers the exact number and pseudo-label details to Appendix A. Set
`EPC` to the verified value from the official supplementary recipe; the runner
does not invent a default.

```bash
EPC=<official_exemplars_per_class>
OUT="$PWD/exps/owod/m-owodb/gnn/stage_1_k5"
python tools/owod/run_graph_local_increment.py \
  --coco-path "$PWD/data/coco" \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --stage 1 \
  --checkpoint "$PWD/exps/owod/m-owodb/deformable_detr_control/stage_0/checkpoint.pth" \
  --gnn-checkpoint "$CAL/gnn_stage0.pt" \
  --output-dir "$OUT" \
  --graph-k 5 --exemplars-per-class "$EPC" \
  --epochs 20 --batch-size 2 --num-workers 4 --eval-interval 5 \
  --gpus 0,1 --nproc-per-node 2 --master-port 29561
```

Optional replay sanity controls use the same split, detector checkpoint,
exemplar count, seed, and training schedule:

- `--control graph`: GNN Top-K old classes.
- `--control random`: Random-K old classes.
- `--control global`: all old classes.

The graph arm requires `--gnn-checkpoint`. Report at least three seeds before
claiming an improvement.

## 6. Primary three-component ablation

The main ablation is internal to the GNN. It does not compare against cosine
similarity. Train four GNN checkpoints from the same cached empirical harm
artifact:

```bash
for ABLATION in full no_node_encoder no_message_passing no_ranking_loss; do
  python tools/owod/calibrate_interference_gnn.py \
    --coco_path "$PWD/data/coco" \
    --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
    --stage 0 \
    --checkpoint "$PWD/exps/owod/m-owodb/deformable_detr_control/stage_0/checkpoint.pth" \
    --output-dir "$CAL" --num_classes 91 \
    --batch_size 1 --num_workers 4 --device cuda \
    --sketch-max-images 12 --probe-max-images 12 --probe-steps 3 \
    --gnn-epochs 400 --gnn-ablation "$ABLATION"
done
```

The four rows are:

- `full`: node encoder + directed message passing + ranking loss.
- `no_node_encoder`: a fixed parameter-free pooling replaces the learned node encoder.
- `no_message_passing`: directed edge MLP sees node pairs without graph aggregation.
- `no_ranking_loss`: retains continuous harm regression but removes pairwise ranking.

The first run creates detector sketches and empirical harm labels. Later runs
reuse those caches from the same `CAL` directory, so differences come from the
GNN component rather than a new detector probe. The checkpoints are
`gnn_stage0.pt`, `gnn_stage0_no_node_encoder.pt`,
`gnn_stage0_no_message_passing.pt`, and `gnn_stage0_no_ranking_loss.pt`.

Run the Stage 1 detector command once per checkpoint, changing only
`--gnn-checkpoint` and `--output-dir`. Keep the detector checkpoint, Top-K,
per-class exemplar count, seed, and training schedule identical.

`--control random` and `--control global` remain optional replay sanity checks;
they are not rows in the three-component GNN ablation table.

## 7. Logs and metrics

Each experiment directory contains:

- `train.log`: compact human-readable output.
- `metrics.jsonl`: complete per-epoch structured metrics.
- `run_config.json` and `run_history/`: exact configuration provenance.
- `training_complete.json`: written only after normal completion.

Evaluation prints `OWOD Table 1 (%)` in percentage points. A displayed value of
`53.1` corresponds to a raw metric of `0.531`; these are not different scores.
U-Rec and H-Score are omitted at Task 4 when no unknown ground truth remains.
A-OSE, WI, UDR, and UDP remain diagnostic metrics outside the main DEUS table.
