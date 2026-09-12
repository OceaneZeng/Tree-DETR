"""Objectness heads used by the local PROB and OWOBJ reimplementations.

Both heads operate on decoder query features and deliberately have no
dependency on the external baseline checkouts.  The scalar energy convention
matches the papers: low energy is object-like and high energy is background-
like.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _BatchNormEnergy(nn.Module):
    """Squared norm after non-affine batch normalization.

    The original PROB implementation uses BatchNorm1d without affine
    parameters.  For a one-query/one-image smoke batch, regular BatchNorm is
    undefined, so the running statistics are used until two samples are
    available.  This keeps the real distributed recipe unchanged while making
    local tests deterministic and finite.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shape = features.shape
        flat = features.reshape(-1, shape[-1])
        if self.training and flat.shape[0] > 1:
            normalized = self.norm(flat)
        else:
            normalized = F.batch_norm(
                flat, self.norm.running_mean, self.norm.running_var,
                weight=None, bias=None, training=False,
                momentum=0.0, eps=self.norm.eps)
        return normalized.reshape(*shape[:-1], shape[-1]).square().sum(-1)


class ProbObjectnessHead(_BatchNormEnergy):
    """PROB's probabilistic objectness energy."""

    pass


class SketchObjectnessHead(nn.Module):
    """OWOBJ sketch objectness with a stochastic Gaussian feature view."""

    def __init__(self, hidden_dim: int, sigma: float = 1.0):
        super().__init__()
        self.sigma = float(sigma)
        self.clean = _BatchNormEnergy(hidden_dim)
        self.sketch = _BatchNormEnergy(hidden_dim)

    def forward(self, features: torch.Tensor):
        clean_energy = self.clean(features)
        if self.training:
            noisy = features * torch.randn_like(features) + self.sigma * torch.randn_like(features)
            sketch_energy = self.sketch(noisy)
        else:
            sketch_energy = self.clean(features)
        return clean_energy, sketch_energy
