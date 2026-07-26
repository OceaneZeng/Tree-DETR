# ------------------------------------------------------------------------
# Tree-DETR : cone geometry primitives  (foundation for Modules B/C/D/E)
# ------------------------------------------------------------------------
# All regions in this method are *cosine cones* on the unit sphere S^{m-1}:
#     R_n = { z in S^{m-1} : angle(z, mu_n) <= theta_n }
# so every quantity here is angle-only and hence scale-invariant, which is
# the invariance requirement of Module D (angles transfer unchanged across
# domains; only the depth-0 magnitude scalar is recalibrated).
#
# Numerical note (from "Fixed design E"): grad arccos diverges at +/-1.  We
# therefore compute angles by the stable half-vector form
#     angle(a, b) = 2 * atan2( ||a - b||, ||a + b|| )
# which is smooth for unit vectors, and only fall back to clamped arccos for
# non-unit inputs.
# ------------------------------------------------------------------------
from typing import Optional
import torch
import torch.nn.functional as F

EPS = 1e-6


def project_to_sphere(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """L2-normalise the last dimension so rows live on S^{m-1}."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def stable_angle(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Angle (radians) between the last-dim vectors of ``a`` and ``b``.

    Uses the half-vector identity  angle = 2*atan2(||a-b||, ||a+b||)  on the
    *normalised* vectors, which has a finite gradient everywhere including the
    antipodal / coincident poles where arccos blows up.  ``a`` and ``b``
    broadcast against each other.
    """
    a = project_to_sphere(a, eps)
    b = project_to_sphere(b, eps)
    diff = (a - b).norm(dim=-1)
    summ = (a + b).norm(dim=-1)
    return 2.0 * torch.atan2(diff, summ)


def angle_arccos(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Reference clamped-arccos angle (notation table).  Kept for tests /
    cross-checks; ``stable_angle`` is preferred in gradient paths."""
    a = project_to_sphere(a, eps)
    b = project_to_sphere(b, eps)
    dot = (a * b).sum(dim=-1)
    return torch.arccos(dot.clamp(-1.0 + eps, 1.0 - eps))


def cone_contains(z: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor,
                  eps: float = EPS) -> torch.Tensor:
    """Boolean: does z fall inside cone R = {angle(., mu) <= theta}?"""
    return stable_angle(z, mu, eps) <= theta


def angular_margin_ratio(z: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor,
                         eps: float = EPS) -> torch.Tensor:
    """r = angle(z, mu) / theta   (Eq B2, the scale-invariant gate score).

    r <= 1  <=>  z is inside the cone.  ``theta`` is clamped away from 0 so the
    ratio is finite for a degenerate (zero-radius) cone.
    """
    return stable_angle(z, mu, eps) / theta.clamp_min(eps)


def cone_gap(theta_n: torch.Tensor,
             mu_n: torch.Tensor,
             mu_children: torch.Tensor,
             theta_children: torch.Tensor,
             eps: float = EPS) -> torch.Tensor:
    """Reserved-gap of a node (Eq E2):

        gap(n) = min_{c in C(n)} [ theta_n - angle(mu_n, mu_c) - theta_c ]

    The ``min`` over children handles the non-convex *union* of child cones
    exactly - the tightest-fitting child is the binding constraint - with no
    sampling and O(|C(n)|) cost.

    Shapes
    ------
    mu_n           : (m,)          axis of the parent
    theta_n        : scalar        radius of the parent
    mu_children    : (k, m)        axes of the k children
    theta_children : (k,)          radii of the k children

    Returns a scalar.  A *positive* gap means a non-empty annulus exists
    between the parent cone and the union of its children - which is where a
    "kind of n but none of n's children" object lands (halt-at-n novelty).
    Returns +inf when the node has no children (vacuously satisfied).
    """
    if mu_children.numel() == 0:
        return torch.tensor(float("inf"), device=mu_n.device, dtype=mu_n.dtype)
    ang = stable_angle(mu_n.unsqueeze(0), mu_children, eps)      # (k,)
    per_child = theta_n - ang - theta_children                   # (k,)
    return per_child.min()


def theta_from_logit(a: torch.Tensor, half_pi: float) -> torch.Tensor:
    """Bounded cone radius parameterisation theta = (pi/2) * sigmoid(a)  in
    (0, pi/2)  (Eq E1 / D2).  Keeps radii strictly inside a hemisphere."""
    return half_pi * torch.sigmoid(a)
