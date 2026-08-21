"""Checkpoint loading helpers that preserve only architecture-compatible tensors."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import torch
from torch import nn


def matching_state_dict(model: nn.Module, source: Mapping[str, torch.Tensor]
                        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Keep tensors whose normalized key and shape match ``model`` exactly.

    COCO detector checkpoints have a different classification-head shape from a
    remapped Pet task.  Filtering avoids accidentally loading a partial or
    malformed head while retaining the transferable detector components.
    """
    target = model.state_dict()
    matched: Dict[str, torch.Tensor] = {}
    skipped = {"missing_key": 0, "shape_mismatch": 0, "non_tensor": 0}
    for raw_key, value in source.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if not torch.is_tensor(value):
            skipped["non_tensor"] += 1
        elif key not in target:
            skipped["missing_key"] += 1
        elif tuple(value.shape) != tuple(target[key].shape):
            skipped["shape_mismatch"] += 1
        else:
            matched[key] = value
    return matched, skipped


def initialize_pet_classifier_from_coco(model: nn.Module,
                                        source: Mapping[str, torch.Tensor],
                                        categories: Iterable[Mapping[str, object]]) -> int:
    """Seed Pet breed logits from the COCO cat/dog classifier rows.

    COCO category IDs 17 and 18 are cat and dog.  The Pet split retains the
    species in each category's ``supercategory`` field, so copying these rows
    gives every breed a detector-aware starting point while leaving subsequent
    fine-tuning to learn breed-specific separation.
    """
    normalized = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in source.items()
    }
    source_weight = normalized.get("class_embed.0.weight",
                                   normalized.get("class_embed.weight"))
    source_bias = normalized.get("class_embed.0.bias",
                                 normalized.get("class_embed.bias"))
    class_embed = getattr(model, "class_embed", None)
    head = class_embed[0] if isinstance(class_embed, nn.ModuleList) else class_embed
    if not isinstance(head, nn.Linear):
        raise ValueError("model has no compatible classification head")
    if not torch.is_tensor(source_weight) or not torch.is_tensor(source_bias):
        raise ValueError("COCO checkpoint has no compatible class_embed tensors")

    coco_species_ids = {"cat": 17, "dog": 18}
    copied = 0
    with torch.no_grad():
        for category in categories:
            species = str(category.get("supercategory", "")).lower()
            if species not in coco_species_ids:
                species = str(category.get("name", "")).lower()
            category_id = int(category["id"])
            source_ids = ([coco_species_ids["cat"], coco_species_ids["dog"]]
                          if species in {"pet", "animal"}
                          else [coco_species_ids.get(species)])
            source_ids = [source_id for source_id in source_ids if source_id is not None]
            if not source_ids or not 0 <= category_id < head.out_features:
                continue
            if any(source_id >= source_weight.shape[0] for source_id in source_ids):
                raise ValueError("COCO checkpoint classifier does not contain cat/dog rows")
            head.weight[category_id].copy_(source_weight[source_ids].mean(dim=0).to(
                device=head.weight.device, dtype=head.weight.dtype))
            head.bias[category_id].copy_(source_bias[source_ids].mean().to(
                device=head.bias.device, dtype=head.bias.dtype))
            copied += 1
    return copied
