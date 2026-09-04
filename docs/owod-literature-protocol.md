# DEUS-aligned OWOD research protocol

## Paper pipeline

DEUS addresses two separate OWOD problems. EUS constructs fixed Simplex ETF
known and unknown subspaces and optimizes energy-margin plus focal objectives
to separate known, unknown, and background proposals. EKD splits the known
classifier into previous-task and current-task subsets and applies a pairwise
energy loss during replay to reduce cross-task interference. Its full objective
is classification + box regression + EUS + EKD.

The paper uses OrthogonalDet in MMDetection, sets both EUS and EKD weights to
1.0, and uses a 128-vector ETF divided into 64 known-space and 64
unknown-space vectors. The present project does not replace its Deformable DETR
detector and does not claim EUS/EKD as implemented components.

## Experimental setting adopted here

- M-OWODB and S-OWODB, each with four non-overlapping incremental tasks.
- Official annotations only; generated random or heuristic splits are rejected.
- Full-label validation for evaluating classes not known at the current task.
- Previous, Current, and all Known mAP; unknown recall (U-Rec); harmonic mean
  between Known mAP and U-Rec (H-Score).
- Task 4 reports known mAP only because all benchmark classes have been seen.
- Fixed exemplars per replayed class, with the exact count supplied from the
  official supplementary recipe rather than a repository default.

The DEUS paper's Appendix A, which contains exact replay and improved
pseudo-label details, is not included in the supplied ten-page PDF. Those
values must be confirmed from the official supplementary material before
claiming an exact reproduction.

## Baseline policy

Table 1 compares ORE, OW-DETR, CAT, PROB, OrthogonalDet, O1O, OWOBJ, and DEUS.
All eight rows are preserved in `tools/owod/deus_table1.json`. PROB and
OrthogonalDet on M-OWODB are daggered reruns with the annotation duplication
bug corrected.

These are external baselines with different architectures and training
recipes. This repository does not rename one Deformable DETR implementation to
simulate them. The local comparison consists of:

- Deformable DETR control.
- GNN Top-K class-local replay (the proposed method).
- matched Random-K replay.
- Global old-class replay.

The primary method ablation is `Full`, `w/o Node Encoder`, `w/o Directed
Message Passing`, and `w/o Pairwise Ranking Loss`. Each row removes exactly one
GNN component while keeping the Deformable DETR checkpoint, replay quota,
Top-K, seed, and schedule fixed. Random/global replay remain optional sanity
controls; cosine similarity is not used as a component ablation.

External method numbers must either be cited as Table 1 references or produced
by the method's actual implementation on the validated annotations.

## Table 1 headline comparisons

On M-OWODB, DEUS reaches U-Rec/H-Score of 65.1/65.6, 66.2/59.0, and 69.0/58.0
for Tasks 1-3. The strongest non-DEUS H-Scores are O1O's 56.1, 51.6, and 47.4.
Task 4 Known mAP is 46.0 for DEUS versus 44.7 for OrthogonalDet and 42.4 for
O1O.

On S-OWODB, DEUS reaches U-Rec/H-Score of 68.7/70.1, 62.9/57.4, and 60.7/55.4
for Tasks 1-3. O1O is the strongest competing H-Score method at 59.1, 52.8,
and 47.4. Task 4 Known mAP is 48.8 for DEUS, 46.2 for OrthogonalDet, and 45.9
for O1O.
