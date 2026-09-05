"""Global old-class target completion for incremental detection."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch

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
        ious = box_ops.box_iou(
            boxes_xyxy[current].unsqueeze(0), boxes_xyxy[order[1:]])[0][0]
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep)


def _classwise_nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor,
                   labels: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    kept = []
    for label in labels.unique():
        positions = torch.nonzero(labels == label, as_tuple=False).flatten()
        selected = _greedy_nms(boxes_xyxy[positions], scores[positions], iou_threshold)
        kept.append(positions[selected])
    if not kept:
        return torch.empty(0, dtype=torch.long, device=boxes_xyxy.device)
    kept = torch.cat(kept)
    return kept[scores[kept].argsort(descending=True)]


def select_teacher_pseudo_labels(outputs: Dict[str, torch.Tensor],
                                 targets: Sequence[Dict[str, torch.Tensor]],
                                 old_class_ids: Iterable[int],
                                 score_threshold: float = 0.5,
                                 duplicate_iou: float = 0.7,
                                 ground_truth_iou: float = 0.5,
                                 max_per_image: int = 20
                                 ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
    """Append confident old-class boxes that do not overlap current targets."""
    class_ids = sorted({int(class_id) for class_id in old_class_ids})
    if not class_ids:
        raise ValueError("old_class_ids must contain at least one class")
    logits = outputs["pred_logits"]
    if class_ids[0] < 0 or class_ids[-1] >= logits.shape[-1]:
        raise ValueError("old_class_ids contains an index outside the classifier")
    probabilities = logits[..., class_ids].sigmoid()
    query_scores, query_labels = probabilities.max(dim=-1)
    class_id_tensor = torch.as_tensor(class_ids, device=query_labels.device)
    query_labels = class_id_tensor[query_labels]
    query_boxes = outputs["pred_boxes"]
    completed = []
    counts = []

    for scores, labels, boxes, target in zip(
            query_scores, query_labels, query_boxes, targets):
        candidate = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()
        if candidate.numel():
            candidate = candidate[scores[candidate].argsort(descending=True)[:max_per_image]]
            candidate_boxes = box_ops.box_cxcywh_to_xyxy(boxes[candidate])
            candidate = candidate[_classwise_nms(
                candidate_boxes, scores[candidate], labels[candidate], duplicate_iou)]
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
                    merged["iscrowd"], torch.zeros(
                        candidate.numel(), dtype=merged["iscrowd"].dtype,
                        device=merged["iscrowd"].device)])
            if "area" in merged:
                height, width = merged.get("size", merged.get("orig_size")).to(
                    pseudo_boxes).unbind(0)
                pseudo_area = pseudo_boxes[:, 2] * width * pseudo_boxes[:, 3] * height
                merged["area"] = torch.cat([
                    merged["area"], pseudo_area.to(merged["area"])])
        completed.append(merged)
        counts.append(int(candidate.numel()))
    return completed, counts
