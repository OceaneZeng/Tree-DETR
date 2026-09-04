"""OWOD scoring for the local Deformable DETR control.

Published OWOD methods use different detectors and training recipes. This
module intentionally exposes only the behavior implemented in this repository
and must not be presented as ORE, OW-DETR, PROB, or another external method.
"""

from __future__ import annotations

from typing import Mapping

import torch


DETECTOR_PROFILE = {
    "name": "deformable_detr_control",
    "detector": "Deformable DETR",
    "unknown_score": "one_minus_max_known_probability",
    "paper_baseline": False,
}


def detector_profile_dict() -> dict[str, object]:
    return dict(DETECTOR_PROFILE)


def unknown_score(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the local control's per-query unknown score in [0, 1]."""
    known_probability = outputs["pred_logits"].sigmoid().amax(dim=-1)
    return (1.0 - known_probability).clamp(0.0, 1.0)
