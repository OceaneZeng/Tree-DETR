"""Global old-class label completion for incremental detection."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from util import box_ops


def _greedy_nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor,
                iou_threshold: float) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes_xyxy.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        ious = box_ops.box_iou(boxes_xyxy[current].unsqueeze(0), boxes_xyxy[order[1:]])[0][0]
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep)


def select_teacher_pseudo_labels(outputs: Dict[str, torch.Tensor],
                                 targets: Sequence[Dict[str, torch.Tensor]],
                                 old_num_classes: int,
                                 score_threshold: float = 0.5,
                                 duplicate_iou: float = 0.7,
                                 ground_truth_iou: float = 0.5,
                                 max_per_image: int = 20
                                 ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
    """Append confident old-class boxes that do not overlap current ground truth."""
    probabilities = outputs["pred_logits"][..., :old_num_classes].sigmoid()
    query_scores, query_labels = probabilities.max(dim=-1)
    query_boxes = outputs["pred_boxes"]
    completed = []
    counts = []

    for scores, labels, boxes, target in zip(query_scores, query_labels, query_boxes, targets):
        candidate = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()
        if candidate.numel():
            candidate = candidate[scores[candidate].argsort(descending=True)[:max_per_image]]
            candidate_boxes = box_ops.box_cxcywh_to_xyxy(boxes[candidate])
            keep = _greedy_nms(candidate_boxes, scores[candidate], duplicate_iou)
            candidate = candidate[keep]
        if candidate.numel() and target["boxes"].numel():
            candidate_boxes = box_ops.box_cxcywh_to_xyxy(boxes[candidate])
            target_boxes = box_ops.box_cxcywh_to_xyxy(target["boxes"])
            overlap = box_ops.box_iou(candidate_boxes, target_boxes)[0].amax(dim=1)
            candidate = candidate[overlap < ground_truth_iou]

        merged = {key: value.clone() if torch.is_tensor(value) else value
                  for key, value in target.items()}
        if candidate.numel():
            pseudo_boxes = boxes[candidate].detach()
            pseudo_labels = labels[candidate].detach()
            merged["boxes"] = torch.cat([merged["boxes"], pseudo_boxes], dim=0)
            merged["labels"] = torch.cat([merged["labels"], pseudo_labels], dim=0)
            if "iscrowd" in merged:
                merged["iscrowd"] = torch.cat([
                    merged["iscrowd"],
                    torch.zeros(candidate.numel(), dtype=merged["iscrowd"].dtype,
                                device=merged["iscrowd"].device),
                ])
            if "area" in merged:
                size = merged.get("size", merged.get("orig_size"))
                height, width = size.to(pseudo_boxes).unbind(0)
                pseudo_area = pseudo_boxes[:, 2] * width * pseudo_boxes[:, 3] * height
                merged["area"] = torch.cat([merged["area"], pseudo_area.to(merged["area"])])
        completed.append(merged)
        counts.append(int(candidate.numel()))
    return completed, counts


@torch.no_grad()
def complete_targets_with_teacher(teacher: nn.Module, samples,
                                  targets: Sequence[Dict[str, torch.Tensor]],
                                  old_num_classes: int,
                                  score_threshold: float = 0.5,
                                  duplicate_iou: float = 0.7,
                                  ground_truth_iou: float = 0.5,
                                  max_per_image: int = 20
                                  ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
    was_training = teacher.training
    teacher.eval()
    outputs = teacher(samples)
    completed, counts = select_teacher_pseudo_labels(
        outputs,
        targets,
        old_num_classes=old_num_classes,
        score_threshold=score_threshold,
        duplicate_iou=duplicate_iou,
        ground_truth_iou=ground_truth_iou,
        max_per_image=max_per_image,
    )
    teacher.train(was_training)
    return completed, counts
