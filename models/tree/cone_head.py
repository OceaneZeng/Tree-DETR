# ------------------------------------------------------------------------
# Tree-DETR : cone embedding + objectness heads  (Eq A1, Eq B1)
# ------------------------------------------------------------------------
# These are the only two learnable heads that read the raw decoder query
# feature h in R^256.  Everything downstream (cascade, cones, losses) works on
# the unit-sphere embedding z, or on the single scalar ||h||.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn

from .config import TreeConfig, DEFAULT
from .geometry import project_to_sphere


class ConeEmbedHead(nn.Module):
    """Eq A1:  z = W_p * LN(h) / || W_p * LN(h) ||   in  S^{m-1}.

    A LayerNorm on h, a linear projection W_p in R^{m x d}, then L2 normalise.
    No bias on W_p (a pure directional embedding).
    """

    def __init__(self, cfg: TreeConfig = DEFAULT):
        super().__init__()
        self.cfg = cfg
        self.ln = nn.LayerNorm(cfg.d_model)
        self.w_p = nn.Linear(cfg.d_model, cfg.m, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return project_to_sphere(self.w_p(self.ln(h)), self.cfg.eps)


class ObjectnessHead(nn.Module):
    """Eq B1: depth-0 gate on the *scalar* feature norm.

        z_obj = f_obj( ||h|| / tau0 ),   claim "thing" iff sigma(z_obj) >= t0

    f_obj is a 2-layer MLP on a single scalar, so it has almost no capacity to
    re-encode class identity - magnitude is permitted here and *only* here
    (the invariance split of Module D).  tau0 is a buffer so it can be
    recalibrated per target domain by entropy minimisation (Eq D3) without
    touching any weights.
    """

    def __init__(self, cfg: TreeConfig = DEFAULT, hidden: int = 16):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("tau0", torch.tensor(float(cfg.tau0_init)))
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def logit(self, h: torch.Tensor) -> torch.Tensor:
        """Return z_obj (pre-sigmoid).  h: (..., d) -> (...,)."""
        norm = h.norm(dim=-1, keepdim=True) / self.tau0.clamp_min(self.cfg.eps)
        return self.mlp(norm).squeeze(-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return sigma(z_obj) in (0,1) - the probability that h is a thing."""
        return torch.sigmoid(self.logit(h))

    def is_thing(self, h: torch.Tensor) -> torch.Tensor:
        """Boolean depth-0 decision at threshold t0 (default 0.5)."""
        return self.forward(h) >= self.cfg.t0

    @torch.no_grad()
    def set_tau0(self, value: float) -> None:
        self.tau0.fill_(float(value))
