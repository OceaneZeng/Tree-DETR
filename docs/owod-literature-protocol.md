# OWOD Literature Protocol and Experiment Reset

This document records the protocol extracted from the core OWOD papers in the
Zotero export. The Oxford-Pet runs remain engineering diagnostics; they are not
the primary evidence for an OWOD claim.

## What the papers actually evaluate

| Work | Detector and benchmark | Main baselines | Main metrics |
|---|---|---|---|
| ORE (Joseph et al., CVPR 2021) | Faster R-CNN R50; M-OWODB from VOC + COCO, four tasks of 20 classes; VOC 10+10, 15+5, 19+1 IOD | Faster R-CNN, fine-tuning, Oracle, ILOD/Faster ILOD | known mAP, U-Recall, WI, A-OSE |
| OW-DETR (Gupta et al., CVPR 2022) | Deformable DETR; M-OWODB and stricter S-OWODB (COCO super-category-separated); VOC IOD | Faster R-CNN, vanilla D-DETR, ORE without EBUI, Oracle | known mAP, U-Recall, WI, A-OSE |
| Revisiting OWOD (Zhao et al., 2023) | COCO-only fair benchmark; four tasks, disjoint data, full test labels for unknowns | Faster R-CNN, fine-tuning, ORE and ORE* | mAP, U-Recall, WI/A-OSE, UDR, UDP |
| PROB (Zohar et al., CVPR 2023) | D-DETR with DINO R50 FPN, M-OWODB and S-OWODB; VOC IOD | D-DETR, ORE*, UC-OWOD, OCPL, 2B-OCD, OW-DETR, Oracle | known mAP, U-Recall, A-OSE/WI; VOC old/new/all mAP |
| CL-DETR (Liu et al., CVPR 2023) | COCO 2017 IOD, `70+10`, `40+40`, `40+20x2`, `40+10x4`; three random orders, 10% exemplar memory | Deformable DETR, UP-DETR, LwF, RILOD, SID, ERD | AP/AP50/AP75/APs/m/l, old AP, forgetting percentage points |

The principal split distinction is important:

- **M-OWODB** is the mixed-superclass protocol introduced by ORE. It is useful
  for historical comparison but has cross-task semantic leakage and, in the
  original formulation, an unknown-validation leakage issue.
- **S-OWODB** groups COCO by super-category, so related classes are not spread
  across tasks. It is the stricter primary protocol for a new claim.
- **Revisiting OWOD** requires class openness, task increment, annotation
  specificity, label integrity, and data specificity. Its criticism of ORE is
  why `ORE*` (without EBUI) is the fair comparison.

## Consequence for this project

The current graph-local Pet runner is **class-incremental object detection**,
not full OWOD:

- it has no explicit `unknown` output;
- it does not report U-Recall, A-OSE/WI, or UDR/UDP;
- its six-class Pet split is not M-OWODB or S-OWODB;
- its `AP50 >= 0.40` gate is project-specific, not a literature standard.

The graph-local idea can first be tested honestly as an IOD method. Its central
claim is then: a gradient-conflict neighborhood predicts old-class forgetting
and beats a matched random neighborhood at the same replay/update budget.

## Revised experiment order

### Phase A: standard IOD, no unknown claim

Use COCO 2017 and the CL-DETR disjoint-image protocols:

1. `40+20x2` and `40+10x4` are the main multi-step settings.
2. `70+10` is a two-phase sanity setting.
3. Use three category/data orders and report mean and standard deviation.
4. Fix a 10% exemplar-memory budget.
5. Report AP, AP50, AP75, APs, APm, APl, old-class AP, new-class AP, all-class AP,
   and forgetting percentage points.

Required controls:

- vanilla Deformable DETR fine-tuning;
- replay or balanced fine-tuning;
- KD + replay baseline (CL-DETR-style control);
- global LoRA/replay;
- matched random-neighborhood replay;
- graph-local replay/update;
- graph-local component ablations.

The graph claim is supported only if graph-local beats the matched random
neighborhood across the fixed budget and multiple seeds. A lower loss or a
single improved seed is insufficient.

### Phase B: full OWOD, only if an unknown branch is implemented

Add an explicit class-agnostic objectness/unknown prediction path and evaluate
on S-OWODB first, then M-OWODB for historical comparability. Required controls
are vanilla D-DETR, ORE*, OW-DETR, and an Oracle upper bound. Report known mAP,
U-Recall, A-OSE/WI, and UDR/UDP where the benchmark implementation supports
them. Only this phase may be called an OWOD experiment.

## References checked

- ORE: <https://arxiv.org/abs/2103.02603>
- OW-DETR: <https://arxiv.org/abs/2112.01513>
- Revisiting OWOD: <https://arxiv.org/abs/2201.00471>
- PROB: <https://openaccess.thecvf.com/content/CVPR2023/html/Zohar_PROB_Probabilistic_Objectness_for_Open_World_Object_Detection_CVPR_2023_paper.html>
- CL-DETR: <https://arxiv.org/abs/2304.03110>
