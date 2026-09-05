# Graph-local continual adapters on Deformable DETR

This project keeps Deformable DETR as its detector. It adopts the experimental
shape of DEUS (CVPR 2026): four sequential tasks on M-OWODB and S-OWODB,
class-balanced exemplar replay, full-label evaluation, and the Table 1 columns
Previous/Current/Known mAP, U-Rec, and H-Score.

The local method is not a reimplementation of DEUS. DEUS uses OrthogonalDet,
ETF-Subspace Unknown Separation (EUS), and Energy-based Known Distinction
(EKD). Here the local method has three modules around an unchanged Deformable
DETR: a Class-Interference GNN selects old classes needing extra protection, a
temporary Neighbor-Scoped LoRA updates the final two decoder FFNs, and a
Graph-Conditioned Continual Objective supplies teacher completion, graph-aware
replay, local discrimination, and off-neighborhood protection. LoRA is merged
after each stage, so inference has no adapter bank or router.

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

For an internal engineering pilot on an unverified manifest, the calibration,
GNN runner, and detector control accept `--allow-unverified-protocol`. This
explicitly records `paper_comparable=false`; results from that mode must not be
compared with published OWOD tables.

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

## 5. Run the full three-module method

DEUS states that replay stores a fixed number of exemplars per class, but the
main paper defers the exact number and pseudo-label details to Appendix A. Set
`EPC` to the verified value from the official supplementary recipe; the runner
does not invent a default.

```bash
EPC=<official_exemplars_per_class>
OUT="$PWD/exps/owod/m-owodb/full_three_module/stage_1_k5"
python tools/owod/run_graph_local_increment.py \
  --coco-path "$PWD/data/coco" \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --stage 1 \
  --checkpoint "$PWD/exps/owod/m-owodb/deformable_detr_control/stage_0/checkpoint.pth" \
  --gnn-checkpoint "$CAL/gnn_stage0.pt" \
  --output-dir "$OUT" \
  --graph-k 5 --exemplars-per-class "$EPC" \
  --neighbor-scoped-lora --lora-rank 8 \
  --teacher-completion --teacher-score-threshold 0.5 \
  --local-margin-coef 0.5 --local-margin 1.0 \
  --off-projection-coef 0.1 --off-basis-rank 8 \
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

### Retention-aware Stage 1 pilot

Top-K is an extra-budget controller, not a hard gate. Every previous class gets
a small exemplar floor, while the GNN-selected classes receive additional
images. The sampler fixes the replay exposure per epoch, and a frozen previous-
stage Deformable DETR completes missing old foreground. A rank-8 stage-level
LoRA and two graph-conditioned losses execute the constrained update. The
detector backbone itself is frozen and unchanged.

```bash
GNN="$CAL/gnn_stage0.pt"
python tools/owod/run_graph_local_increment.py \
  --coco-path "$PWD/data/coco" \
  --manifest "$MAN" --allow-unverified-protocol \
  --stage 1 --checkpoint "$BASE" --gnn-checkpoint "$GNN" \
  --output-dir "$OUT" --control graph \
  --graph-k 5 --graph-aggregation top_mean --graph-aggregation-top-n 3 \
  --base-exemplars-per-class 10 --risk-extra-exemplars-per-class 40 \
  --replay-sampling-fraction 0.10 \
  --neighbor-scoped-lora --lora-rank 8 \
  --teacher-completion --teacher-score-threshold 0.5 \
  --local-margin-coef 0.5 --local-margin 1.0 \
  --off-projection-coef 0.1 --off-basis-rank 8 \
  --lr 1e-4 --epochs 20 --lr-drop 15 --eval-interval 5 \
  --batch-size 2 --num-workers 4 --gpus 0,1 --nproc-per-node 2
```

Use a new output directory. `graph.json` records per-class quotas, robust GNN
ranking, and the protected off-neighborhood. `off_neighborhood_basis.pt` stores
the projection basis. `checkpoint.pth` is resumable with LoRA parameters, while
`checkpoint_merged.pth` is the plain Deformable DETR checkpoint for the next
stage. This command is a pilot when the manifest is passed
with `--allow-unverified-protocol` and must not be reported as paper-comparable.
The repository also provides the same guarded command as
`tools/owod/run_stage1_retention_pilot.sh`.

## 6. Primary three-module ablation

The main ablation removes one macro module at a time while keeping the split,
memory budget, seed, and training schedule fixed:

- `Full`: GNN + LoRA + graph-conditioned objective.
- `without GNN`: use cardinality-matched `--control random`, retaining the same
  LoRA and objective.
- `without LoRA`: use `--no-neighbor-scoped-lora --off-projection-coef 0`,
  retaining GNN selection, teacher completion, replay, and local margin. The
  LoRA-coordinate projection is necessarily absent with its parameterization.
- `without graph-conditioned objective`: retain GNN and LoRA, use
  `--no-teacher-completion --local-margin-coef 0 --off-projection-coef 0`.

Removing the GNN node encoder, message passing, or ranking loss is a secondary
internal diagnostic. Cosine similarity is an optional neighborhood baseline,
not a primary module ablation.

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

## 8. Audit a server workspace

Source cleanup on a development machine does not inspect experiments generated
on the training server. Run the server audit from the repository root before
removing old artifacts:

```bash
python tools/owod/audit_server_workspace.py \
  --project-root "$PWD" \
  --report-json "$PWD/server_cleanup_report.json"
```

The default is a read-only dry run. The report separates completed experiments,
interrupted checkpoints, periodic checkpoints, large logs, and safe deletion
candidates. To remove only rebuildable Python/build caches and logs whose bytes
and SHA-256 hash exactly match the canonical `train.log`, run:

```bash
python tools/owod/audit_server_workspace.py \
  --project-root "$PWD" \
  --report-json "$PWD/server_cleanup_report_after.json" \
  --apply-safe
```

This command never deletes datasets, pretrained weights, GNN checkpoints,
detector checkpoints, completed experiments, or nonidentical logs. Items under
`review_only` require an explicit manual decision.
