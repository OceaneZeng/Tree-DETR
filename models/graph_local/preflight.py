"""Detector-free feasibility gates for graph-local incremental adaptation.

These checks are intentionally synthetic.  They verify that every module has a
non-zero, observable effect before a costly detector experiment is attempted;
they do not establish that the locality hypothesis holds on real data.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .interference import build_conflict_matrix, evaluate_neighborhood, select_positive_neighbors
from .lora import LoRALinear
from .losses import local_margin_loss, projection_loss
from .pseudo_labels import select_teacher_pseudo_labels
from .replay import select_replay_images


class _SingleMatch:
    """Minimal matcher used to isolate the local-margin loss."""

    def __call__(self, _outputs, _targets):
        index = torch.tensor([0], dtype=torch.long)
        return [(index, index)]


def _gate(value: float, passed: bool, description: str) -> Dict[str, object]:
    return {"value": float(value), "passed": bool(passed), "gate": description}


def run_preflight() -> Dict[str, object]:
    """Run deterministic module-level feasibility gates and return a JSON-safe report."""
    torch.manual_seed(7)
    gates: Dict[str, Dict[str, object]] = {}

    # One new class has an anti-aligned gradient only with old class 0.  Its
    # measured harm is also deliberately concentrated in class 0.
    sketches = {
        0: torch.tensor([-1.0, 0.0, 0.0]),
        1: torch.tensor([0.0, 1.0, 0.0]),
        2: torch.tensor([1.0, 0.0, 0.0]),
    }
    class_ids, conflict = build_conflict_matrix(sketches)
    neighbors = [class_id for class_id, _ in select_positive_neighbors(class_ids, conflict, 2, k=1)]
    neighborhood = evaluate_neighborhood(neighbors, {0: 0.9, 1: 0.1}, k=1, random_trials=200, seed=7)
    graph_gain = neighborhood["predicted_concentration"] - neighborhood["random_concentration_mean"]
    gates["G1_graph_predicts_harm"] = _gate(
        graph_gain,
        neighbors == [0] and graph_gain >= 0.30,
        "top-1 graph neighbor is harmed class 0 and gain over random >= 0.30",
    )

    coco = {
        "images": [{"id": value} for value in range(1, 7)],
        "annotations": [
            {"image_id": 1, "category_id": 0}, {"image_id": 2, "category_id": 0},
            {"image_id": 3, "category_id": 0}, {"image_id": 4, "category_id": 1},
            {"image_id": 5, "category_id": 1}, {"image_id": 6, "category_id": 1},
        ],
    }
    replay_ids = select_replay_images(coco, [0, 1], budget=4, seed=7)
    replay_classes = {
        class_id: sum(any(a["image_id"] == image_id and a["category_id"] == class_id
                          for a in coco["annotations"]) for image_id in replay_ids)
        for class_id in (0, 1)
    }
    replay_imbalance = abs(replay_classes[0] - replay_classes[1])
    gates["R1_balanced_local_replay"] = _gate(
        replay_imbalance,
        len(replay_ids) == 4 and replay_imbalance == 0,
        "four replay images with zero class-count imbalance",
    )

    outputs = {
        "pred_logits": torch.tensor([[[6.0, -5.0, -5.0], [5.0, -5.0, -5.0], [-5.0, 6.0, -5.0]]]),
        "pred_boxes": torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.21, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]),
    }
    targets = [{
        "boxes": torch.tensor([[0.8, 0.8, 0.1, 0.1]]),
        "labels": torch.tensor([2]),
    }]
    completed, pseudo_counts = select_teacher_pseudo_labels(
        outputs, targets, old_num_classes=2, score_threshold=0.9,
        duplicate_iou=0.5, ground_truth_iou=0.5, max_per_image=5,
    )
    pseudo_added = pseudo_counts[0]
    gates["P1_pseudo_label_completion"] = _gate(
        pseudo_added,
        pseudo_added == 1 and int(completed[0]["labels"].numel()) == 2,
        "one confident non-duplicate old-class pseudo label is appended",
    )

    weak_outputs = {
        "pred_logits": torch.tensor([[[0.0, 0.0, 0.1]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
    }
    strong_outputs = {
        "pred_logits": torch.tensor([[[0.0, 0.0, 2.0]]]),
        "pred_boxes": weak_outputs["pred_boxes"],
    }
    margin_targets = [{"labels": torch.tensor([2]), "boxes": weak_outputs["pred_boxes"][0]}]
    weak_margin = local_margin_loss(weak_outputs, margin_targets, _SingleMatch(), [0, 2], margin=1.0)
    strong_margin = local_margin_loss(strong_outputs, margin_targets, _SingleMatch(), [0, 2], margin=1.0)
    margin_drop = float(weak_margin - strong_margin)
    gates["M1_local_margin_signal"] = _gate(
        margin_drop,
        float(weak_margin) > 0.0 and float(strong_margin) == 0.0,
        "raising the local target logit reduces a positive margin loss to zero",
    )

    delta = torch.tensor([1.0, 0.0], requires_grad=True)
    basis = torch.tensor([[1.0], [0.0]])
    optimizer = torch.optim.SGD([delta], lr=0.2)
    initial_projection = float(projection_loss(delta, basis).detach())
    for _ in range(12):
        optimizer.zero_grad()
        projection_loss(delta, basis).backward()
        optimizer.step()
    final_projection = float(projection_loss(delta, basis).detach())
    gates["O1_off_neighborhood_projection"] = _gate(
        initial_projection - final_projection,
        final_projection <= initial_projection * 0.01,
        "optimizing the projection penalty removes >= 99% off-neighborhood overlap",
    )

    base = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.eye(2))
    lora = LoRALinear(base, rank=1)
    inputs = torch.tensor([[1.0, 0.0]])
    initial_output = lora(inputs).detach().clone()
    lora_optimizer = torch.optim.SGD([lora.lora_a, lora.lora_b], lr=0.5)
    target = torch.tensor([[0.0, 1.0]])
    for _ in range(20):
        lora_optimizer.zero_grad()
        (lora(inputs) - target).square().mean().backward()
        lora_optimizer.step()
    adapted_output = lora(inputs).detach().clone()
    merged_output = lora.merge()(inputs).detach()
    merge_error = float((adapted_output - merged_output).abs().max())
    adaptation_distance = float((adapted_output - initial_output).abs().max())
    gates["L1_low_rank_adaptation"] = _gate(
        adaptation_distance,
        adaptation_distance > 0.1 and merge_error <= 1e-6,
        "LoRA changes output and merged dense layer preserves it (max error <= 1e-6)",
    )

    return {
        "schema_version": 1,
        "experiment": "graph_local_synthetic_preflight",
        "scientific_scope": "implementation feasibility only; not evidence for real-data locality",
        "gates": gates,
        "all_passed": all(result["passed"] for result in gates.values()),
        "diagnostics": {
            "graph_neighbors": neighbors,
            "graph_predicted_concentration": neighborhood["predicted_concentration"],
            "graph_random_concentration_mean": neighborhood["random_concentration_mean"],
            "replay_ids": replay_ids,
            "pseudo_labels_added": pseudo_added,
            "local_margin_before": float(weak_margin),
            "local_margin_after": float(strong_margin),
            "projection_before": initial_projection,
            "projection_after": final_projection,
            "lora_merge_max_abs_error": merge_error,
        },
    }
