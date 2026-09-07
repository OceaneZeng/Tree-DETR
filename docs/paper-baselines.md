# OW-DETR and EW-DETR: D4-matched controlled runs

Updated: 2026-09-07. This queue implements the user's request for OW-DETR and
EW-DETR after D4. It supersedes the older PROB-first execution plan. It does not
stop D4, alter GLCA's GNN, or reuse GLCA's trained Stage 0 checkpoint.

## Status and interpretation

- OW-DETR: core components ported against the pinned author source below.
- EW-DETR: implementation from the supplied paper and official supplement.
  No author repository was located through the CVF page or GitHub repository
  search. This is not a claim that no author code exists elsewhere.
- These are controlled adaptations to the current internal COCO protocol,
  **not exact author-recipe or published-number reproductions**.
- Local CUDA forward/backward and synthetic training/checkpoint tests passed.
  Full COCO training, two-GPU NCCL execution, and baseline accuracy have not
  been verified on the server. The queue runs a CUDA smoke check there first.
- Neither paper provides evidence that its result on this adapted protocol
  must match its published result. Do not put these scores in an official
  M/S-OWODB table without resolving the data/protocol differences.

## Controlled settings

| Item | Configuration |
| --- | --- |
| Dataset | COCO2017 images, D4's `split_manifest_deus.invalid.json` |
| Protocol | Four disjoint 20-class increments; 20/40/60/80 known categories |
| Official status | Internal, unverified split; `paper_comparable=false` |
| Detector | Deformable DETR, ResNet50, 6 encoder + 6 decoder layers, 300 queries |
| Backbone | Entirely frozen in both baselines, including Stage 0 |
| Initialization | Torchvision ImageNet-1K V1 ResNet50; random detector/heads |
| Stage 0 | Each method independently trains 50 epochs; LR drop at 40 |
| Stages 1-3 | Each method loads its own previous-stage checkpoint; 20 epochs |
| Batch and seed | 2 GPUs x 2 images, seed 42 |
| Shared recipe | LR, drop at incremental stages, losses, optimizer multipliers, augmentation and evaluation interval from D4's recorded configuration |
| OW replay | Class-balanced random exemplars; total memory defaults to D4's tagged replay image count (398 in the supplied run); D4's sampling fraction (10%) |
| EW replay | Zero, preserving the method's exemplar-free constraint |
| Teacher/GNN | Neither baseline receives GLCA's graph, teacher completion or extra losses |
| Evaluation | Existing COCO AP50 subsets and local unknown metrics; same threshold as D4 |

The current D4 Stage 1 checkpoint descends from a historical Stage 0 run that
allowed partial backbone updates. That original initialization/training chain
has not been fully audited. These independently trained, fully frozen-backbone
Stage 0 baselines therefore do **not** isolate method alone against historical
D4. A final publication comparison requires rerunning GLCA through all four
stages under the same initial training policy. D4 currently supplies the Stage 1
settings and the queue dependency; it is not itself a completed four-stage run.

ImageNet initialization follows this repository's existing backbone builder.
The papers use DINO ResNet50. No supervised all-COCO detector checkpoint is
introduced, since it could contain future-category supervision. In EW, the
random base transformer, projections and queries stay frozen from Stage 0;
task adapters and detection/calibration heads learn from the Stage 0 data.
This initialization choice is a material limitation of the matched adaptation.

The launcher checks that Stage 1 current images/class lists match D4. Its OW
memory construction is online: each transition draws from current-stage
annotations plus the bounded previous memory. Dropped old images cannot be
retrieved. Class quotas are visited in round-robin order, randomly selecting
images within each class. Shared current/memory images are merged once, retaining
only annotation labels already available; actual replay image counts are logged.
There is no separate author-style balanced fine-tuning pass: balanced replay is
interleaved within the matched 20-epoch budget. This schedule adaptation must
remain visible in any reported OW-DETR result.

## Method fidelity and explicit assumptions

OW-DETR sources:

- Author repository: <https://github.com/akshitac8/OW-DETR>
- Commit: `3515ff8c36687a4f582ef6a2c83866966b56ce23`
- Reference files: `models/deformable_detr.py`,
  `configs/OWOD_our_proposed_split.sh`.
- Author README declares Apache 2.0. This integration reuses the local Apache
  Deformable DETR implementation and ports the OW components; no runtime
  dependency on the ignored downloaded source directory is required.

Implemented: average the 1024-channel ResNet feature map, upsample it, rank
unmatched query boxes by mean activation, select up to five pseudo unknowns,
supervise the dedicated unknown classifier slot, and train a foreground
objectness branch with sigmoid focal loss weighted by 0.1. Pseudo unknowns
receive no box regression supervision. Auxiliary decoder losses include the
same components. Pseudo-label warmup is nine zero-based epochs per stage.
Inference uses class scores, without multiplying them by objectness, following
the author postprocessor.

Compatibility changes: clip image rectangles and exclude invalid/GT-matched
queries even when fewer than five valid candidates exist; calculate means with
an integral image; fix the author's reused auxiliary loop-index issue; keep
sparse COCO category IDs; mask unused/future class slots before matching and
postprocessing. These improve validity but are not bitwise-equivalent to the
author implementation. The original has 100 queries, DINO pretraining, a
different class split, VOC evaluation and a different multi-pass training recipe.

EW-DETR sources:

- Main paper: <https://openaccess.thecvf.com/content/CVPR2026/html/Monga_EW-DETR_Evolving_World_Object_Detection_via_Incremental_Low-Rank_DEtection_TRansformer_CVPR_2026_paper.html>
- Official supplement: <https://openaccess.thecvf.com/content/CVPR2026/supplemental/Monga_EW-DETR_Evolving_World_CVPR_2026_supplemental.pdf>
- Main Sections 3.3-3.5; supplement Appendices A, F, G.1 and G.3.

Implemented: rank-16 task and aggregate adapters on every linear weight inside
encoder/decoder, including deformable attention projections and packed MHA
Q/K/V and output projections; freeze the base detector except class/box heads;
freeze aggregate adapters; learn only task factors and detection/calibration
heads. Consolidation uses a truncated rank-16 SVD, saves the aggregate factors,
and resets the task update. A is initialized nonzero and B to zero so adapters
can learn after reset. A single packed Q/K/V weight uses one rank-16 adapter.

Query-Norm applies LayerNorm and L2 normalization, mixes normalized and original
features, and predicts objectness from the scalar query norm. EUMix implements
the sigmoid known-confidence gap, positive learned exponent, classifier bias,
learned mixture, and inverse-sigmoid unknown logit from Appendix A. The ordinary
detection loss trains these modules; no additional objectness loss, replay,
teacher, or positive unknown ground-truth labels are used.

Choices not fully specified by the paper/supplement are explicit in this port:

| Item | Chosen behavior |
| --- | --- |
| Adapter forward | `W0 + B_aggregate A_aggregate + B_task A_task`, scaling 1 |
| Merging | Main Eq. 3, clamped to [0.2, 0.8]; first task beta=1 |
| Merging ambiguity | Appendix B's beta=0.317 example is consistent with a current/cumulative-including-current ratio; the printed main equation uses cumulative previous samples. The implementation follows the printed equation with its stated bounds |
| Unknown suppression | Subtract `softplus(theta_lambda) * p_unknown_objectness` from known logits; the paper states proportional suppression but omits its exact equation |
| Norm objectness head | MLP 1 -> 32 -> 1 with ReLU; temperature fixed at 1; epsilon=1e-6 |
| Initial calibration | Feature mix=0.5, classifier/unknown mix=0.9, gamma=1, lambda=1, bias=0 |
| Decoder auxiliary layers | Share Query-Norm/EUMix across all decoder outputs to preserve the existing auxiliary-loss schedule |
| Stage-end inference | Report both task and consolidated weights; use consolidated weights for the next task and the primary continual-result row |

The supplement G.1 says its D-DETR experiments use **one encoder and one decoder
layer**, reporting near-zero mAP with six layers in its EWOD setting. This
controlled queue retains D4's six layers. It is an OWOD transfer experiment,
without the original Pascal/weather domain shifts, not the published EWOD
experiment. EW's original schedule was 100 epochs per task and batch 16.
These distinctions and the assumptions above preclude claiming exact EW-DETR
reproduction accuracy from the queue alone.

## Server commands

First synchronize these code changes to the server using the user's existing
Git workflow. The commands below assume the `tree-detr` conda environment is
already active and GPU indices 0 and 1 are the intended devices.

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
python tools/owod/run_paper_baselines.py --dry-run
```

The dry run checks all four stages, actual image paths, the D4 configuration,
class mappings, and existing output plans. It creates no outputs. It does not
allocate a GPU or load/validate the still-changing D4 checkpoint.

Launch the queue now; it will wait for D4 and idle GPUs:

```bash
nohup python -u tools/owod/run_paper_baselines.py > paper_baselines_launcher.log 2>&1 < /dev/null &
echo "queue PID: $!"
```

The runner sets PYTHONPATH and the PyTorch library path for its child processes,
using the invoking conda Python. It retains the established NCCL settings. It
does not change the conda environment or install an alternative PyTorch.

```bash
tail -n 60 paper_baselines_launcher.log
python tools/owod/run_paper_baselines.py --summarize
```

Default output root:
`exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_v1/`

Each `ow-detr/stage_0` through `stage_3`, followed by `ew-detr/stage_0` through
`stage_3`, has `console.log`, `metrics.jsonl`, `run_config.json`, `train.json`,
`checkpoint.pth`, and a completion marker. EW additionally has
`checkpoint_consolidated.pth` and `consolidated_metrics.json`. Queue state is in
`queue_status.json`; CUDA smoke output is in `preflight.log`.

If a run fails, inspect its console log. After fixing an environment-only issue,
resume with the same code/configuration and the interrupted stage's checkpoint:

```bash
nohup python -u tools/owod/run_paper_baselines.py --resume >> paper_baselines_launcher.log 2>&1 < /dev/null &
echo "queue PID: $!"
```

Completed stages are verified and skipped. A failure before the first checkpoint
requires a new output directory, for example `--output-dir "$PWD/exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_v2"`.
That starts an independent queue. Never remove D4 or existing baseline outputs
to bypass a preflight error. A code/configuration change also requires a new
output directory; the runner deliberately rejects silently mixing recipes.

## Dependency and failure checks

The queue waits for a D4 marker with last_epoch=19, final evaluation metrics,
checkpoint existence, absence of D4 output-matching processes, and no compute
jobs on the selected GPUs. It then loads D4's checkpoint to validate its epoch,
stage, frozen-backbone policy and finite weights. A shell exit or lone marker
cannot release the queue. The default wait limit is seven days; a failed D4
without a completion marker leaves the queue waiting until timeout or manual
interruption. It never assumes that GPU idleness means D4 succeeded.

An OS lock prevents duplicate queues for one output directory. Subprocess
failures stop subsequent stages. Source annotations and implementation files
are hashed and rechecked before training. Stage transitions validate the method
and previous-stage index; EW additionally requires a consolidated checkpoint.
Resuming restores the unconsolidated optimizer checkpoint, not a merged one.

Local validation performed: isolated method tests, D1-D4 launcher regression
tests, real six-layer CUDA gradient/frozen-weight checks, and synthetic COCO
training/evaluation through Stage 0, resume, and Stage 1 for both methods. The
synthetic integration uses small images; it is not an accuracy result or a
server-scale throughput/memory test.

```bash
python -m pytest tools/owod/tests/test_paper_baselines.py tools/owod/tests/test_stage1_diagnostics.py -q
python tools/owod/smoke_paper_baselines.py
TREE_BASELINE_INTEGRATION=1 python -m pytest tools/owod/tests/test_paper_training_integration.py -q
```

## Reporting

Use COCO previous/current/known AP50 and the existing U-Recall/H for internal
comparison. The local WI/A-OSE/unknown handling is not the authors' evaluator;
do not label it official WI/A-OSE. Unknown ground truth is absent in Stage 3,
so unknown metrics and H have no useful open-world interpretation there.
For EW, keep the `task` and `consolidated` rows separate; do not silently select
the better row per stage. No FOGS or cross-domain generalization claim can be
made from this single-domain COCO sequence.
