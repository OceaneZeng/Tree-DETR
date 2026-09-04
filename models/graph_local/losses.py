"""Losses used by graph-local incremental adaptation."""

from __future__ import annotations

from typing import Dict

import torch


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
