"""Gradient-subspace protection for graph-local low-rank updates."""

from __future__ import annotations

from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


def build_off_neighborhood_basis(sketches: Mapping[int, torch.Tensor],
                                 excluded_classes: Iterable[int],
                                 max_rank: int = 8,
                                 eps: float = 1e-8) -> torch.Tensor:
    """Build an orthonormal basis without an SVD over detector-size vectors."""
    if not sketches:
        raise ValueError("At least one sketch is required")
    excluded = {int(class_id) for class_id in excluded_classes}
    selected = [sketches[class_id].float().cpu().flatten()
                for class_id in sorted(sketches) if int(class_id) not in excluded]
    feature_dim = next(iter(sketches.values())).numel()
    if not selected or max_rank <= 0:
        return torch.empty(feature_dim, 0)
    rows = F.normalize(torch.stack(selected), dim=1, eps=eps)
    gram = rows @ rows.t()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    keep = torch.nonzero(eigenvalues > eps, as_tuple=False).flatten()
    if keep.numel() == 0:
        return torch.empty(feature_dim, 0)
    keep = keep[-min(int(max_rank), keep.numel()):]
    basis = rows.t() @ (eigenvectors[:, keep] /
                        eigenvalues[keep].sqrt().unsqueeze(0))
    return basis.contiguous()
