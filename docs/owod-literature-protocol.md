# OWOD Literature Protocol and Experiment Reset

This document records the protocol extracted from the core OWOD papers in the
Zotero export. Only M-OWODB/S-OWODB experiments are used as evidence for the
current method.

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

The primary method is the trainable class-interference GNN evaluated directly
under OWOD. Detector decoder-gradient sketches are class-node features and
empirical source-update/target-loss increases from training data supervise the
directed edges. The GNN selects a fixed-size old-class replay neighborhood for
each increment and does not use validation labels.

The main benchmark order is S-OWODB, followed by M-OWODB for historical
comparison. Required controls are vanilla D-DETR, ORE*, OW-DETR, PROB, Oracle,
matched Random-K replay, Global replay, and prototype Cosine-K as an ablation.
Report Previous/Current/Known AP50 together with U-Recall, A-OSE, WI, UDR and
UDP. A GNN claim requires improvement over matched Random-K across multiple
seeds; lower training loss or one favorable run is insufficient.

## References checked

- ORE: <https://arxiv.org/abs/2103.02603>
- OW-DETR: <https://arxiv.org/abs/2112.01513>
- Revisiting OWOD: <https://arxiv.org/abs/2201.00471>
- PROB: <https://openaccess.thecvf.com/content/CVPR2023/html/Zohar_PROB_Probabilistic_Objectness_for_Open_World_Object_Detection_CVPR_2023_paper.html>
- CL-DETR: <https://arxiv.org/abs/2304.03110>
