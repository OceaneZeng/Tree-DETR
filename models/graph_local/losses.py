"""Losses used by graph-local incremental adaptation."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F


def weighted_detection_loss(losses: Dict[str, torch.Tensor],
                            weight_dict: Dict[str, float],
                            include_auxiliary: bool = True) -> torch.Tensor:
    terms = []
    for name, value in losses.items():
        if name not in weight_dict:
            continue
        if not include_auxiliary and (name.rsplit("_", 1)[-1].isdigit() or name.endswith("_enc")):
            continue
        terms.append(value * weight_dict[name])
    if not terms:
        raise ValueError("No weighted detector losses were found")
    return torch.stack([term if term.ndim == 0 else term.sum() for term in terms]).sum()


def local_margin_loss(outputs: Dict[str, torch.Tensor], targets: Sequence[Dict],
                      matcher, local_classes: Iterable[int], margin: float = 1.0
                      ) -> torch.Tensor:
    """Separate matched current/neighbor targets from their local competitors."""
    logits = outputs["pred_logits"]
    local = sorted({int(class_id) for class_id in local_classes})
    if len(local) < 2:
        return logits.sum() * 0.0
    if local[0] < 0 or local[-1] >= logits.shape[-1]:
        raise ValueError("A local class index exceeds the classifier size")
    positions = {class_id: index for index, class_id in enumerate(local)}
    indices = matcher(
        {"pred_logits": logits, "pred_boxes": outputs["pred_boxes"]}, targets)
    terms = []
    for batch_index, (source_indices, target_indices) in enumerate(indices):
        if source_indices.numel() == 0:
            continue
        labels = targets[batch_index]["labels"][target_indices]
        selected = [offset for offset, label in enumerate(labels.tolist())
                    if int(label) in positions]
        if not selected:
            continue
        # HungarianMatcher returns CPU indices (via scipy) even when targets
        # and logits live on CUDA. Index each source tensor on its own device,
        # then move only the resolved query indices to the logits device.
        source_selection = torch.as_tensor(
            selected, device=source_indices.device, dtype=torch.long)
        target_selection = torch.as_tensor(
            selected, device=labels.device, dtype=torch.long)
        matched_queries = source_indices[source_selection].to(logits.device)
        local_tensor = torch.as_tensor(local, device=logits.device, dtype=torch.long)
        matched_logits = logits[batch_index, matched_queries].index_select(1, local_tensor)
        matched_labels = labels[target_selection]
        true_positions = torch.as_tensor(
            [positions[int(label)] for label in matched_labels.tolist()],
            device=matched_logits.device, dtype=torch.long)
        true_logits = matched_logits.gather(1, true_positions[:, None]).squeeze(1)
        competitor_logits = matched_logits.clone()
        competitor_logits.scatter_(1, true_positions[:, None], float("-inf"))
        strongest_competitor = competitor_logits.max(dim=1).values
        terms.append(F.relu(margin - (true_logits - strongest_competitor)))
    if not terms:
        return logits.sum() * 0.0
    return torch.cat(terms).mean()


def projection_loss(delta_vector: torch.Tensor, off_basis: torch.Tensor) -> torch.Tensor:
    """Penalize the LoRA update inside non-neighbor gradient directions."""
    if off_basis.ndim != 2:
        raise ValueError("off_basis must be a [parameters, rank] matrix")
    if off_basis.shape[0] != delta_vector.numel():
        raise ValueError(
            f"Basis dimension {off_basis.shape[0]} != delta dimension {delta_vector.numel()}")
    if off_basis.shape[1] == 0:
        return delta_vector.sum() * 0.0
    basis = off_basis.to(device=delta_vector.device, dtype=delta_vector.dtype)
    return basis.t().matmul(delta_vector).square().mean()
