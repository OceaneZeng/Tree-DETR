"""Lightweight OWOD metrics for full-label validation annotations."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import torch

from util.box_ops import box_cxcywh_to_xyxy, box_iou


def harmonic_score(known_map: float, unknown_recall: float) -> float:
    denominator = known_map + unknown_recall
    return 0.0 if denominator == 0 else 2.0 * known_map * unknown_recall / denominator


def table1_summary(class_ap50: Mapping[str, float], owod_metrics: Mapping[str, float]
                   ) -> dict[str, float]:
    """Format the DEUS Table 1 columns in percentage points."""
    result = {
        f"{name} mAP": 100.0 * float(value)
        for name, value in class_ap50.items()
        if name in {"Previous", "Current", "Known"}
    }
    if float(owod_metrics.get("unknown_gt", 0.0)) > 0:
        unknown_recall = float(owod_metrics["U-Recall"])
        known_map = float(class_ap50.get("Known", class_ap50.get("Current", 0.0)))
        result["U-Rec"] = 100.0 * unknown_recall
        result["H-Score"] = 100.0 * harmonic_score(known_map, unknown_recall)
    return result


def _absolute_target_boxes(target: Mapping[str, torch.Tensor]) -> torch.Tensor:
    boxes = box_cxcywh_to_xyxy(target["boxes"])
    height, width = target["orig_size"].tolist()
    scale = boxes.new_tensor([width, height, width, height])
    return boxes * scale


def compute_owod_metrics(predictions: Sequence[Mapping[str, torch.Tensor]],
                         targets: Sequence[Mapping[str, torch.Tensor]],
                         known_class_ids: Iterable[int], threshold: float = 0.5,
                         iou_threshold: float = 0.5) -> dict[str, float]:
    """Compute unknown recall/precision and A-OSE on full-label validation.

    The definitions follow the operational OWOD convention: an unknown ground
    truth is recovered by an unknown-scored prediction at IoU >= 0.5, while an
    unknown matched by a known-class prediction contributes to A-OSE. WI is
    reported as the fraction of known predictions that overlap an unknown GT.
    """
    known = {int(value) for value in known_class_ids}
    unknown_gt = known_hits = unknown_pred = unknown_hits = 0
    a_ose = 0
    unknown_on_known = 0
    known_prediction_count = 0
    for prediction, target in zip(predictions, targets):
        gt_boxes = _absolute_target_boxes(target)
        gt_labels = target["labels"].tolist()
        unknown_indices = [i for i, label in enumerate(gt_labels) if int(label) not in known]
        known_indices = [i for i, label in enumerate(gt_labels) if int(label) in known]
        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"].tolist()
        pred_scores = prediction.get("scores", torch.zeros(len(pred_boxes)))
        pred_unknown = prediction.get("unknown_scores", torch.zeros(len(pred_boxes)))
        unknown_mask = pred_unknown >= threshold
        unknown_prediction_indices = [i for i, value in enumerate(unknown_mask.tolist()) if value]
        known_prediction_indices = [i for i, label in enumerate(pred_labels) if int(label) in known]
        unknown_gt += len(unknown_indices)
        unknown_pred += len(unknown_prediction_indices)
        known_prediction_count += len(known_prediction_indices)
        if unknown_indices and len(pred_boxes):
            ious = box_iou(gt_boxes, pred_boxes)[0]
            for gt_index in unknown_indices:
                unknown_hits_for_gt = [j for j in unknown_prediction_indices
                                       if float(ious[gt_index, j]) >= iou_threshold]
                known_hits_for_gt = [j for j in known_prediction_indices
                                     if float(ious[gt_index, j]) >= iou_threshold]
                if unknown_hits_for_gt:
                    unknown_hits += 1
                if known_hits_for_gt:
                    a_ose += 1
        if known_indices and len(pred_boxes):
            ious = box_iou(gt_boxes, pred_boxes)[0]
            for gt_index in known_indices:
                if any(float(ious[gt_index, j]) >= iou_threshold
                       for j in unknown_prediction_indices):
                    unknown_on_known += 1
    u_recall = unknown_hits / max(1, unknown_gt)
    u_precision = unknown_hits / max(1, unknown_pred)
    return {
        "U-Recall": float(u_recall),
        "UDR": float(u_recall),
        "UDP": float(u_precision),
        "A-OSE": float(a_ose),
        "WI": float(unknown_on_known / max(1, known_prediction_count)),
        "unknown_gt": float(unknown_gt),
        "unknown_predictions": float(unknown_pred),
    }
