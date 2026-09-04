"""Checkpoint loading helpers that preserve only architecture-compatible tensors."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
from torch import nn


def matching_state_dict(model: nn.Module, source: Mapping[str, torch.Tensor]
                        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Keep tensors whose normalized key and shape match ``model`` exactly.

    Detector checkpoints can have a different classification-head shape from
    the active OWOD stage. Filtering keeps transferable detector components
    without partially loading an incompatible head.
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
