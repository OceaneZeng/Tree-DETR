# ------------------------------------------------------------------------
# Tree-DETR : Module D - path calibration and invariance
# ------------------------------------------------------------------------
# The invariance split (the one design decision that is a claim, not a
# convention):
#   depth 0 (objectness)   : magnitude ||h||        - recalibrated per domain (tau0)
#   depth >= 1 (all gates) : angle only, via r_c    - transfers unchanged
# Because exactly one scalar is domain-dependent, a domain shift costs ONE
# recalibration instead of D compounding ones.
#
# Equations:
#   D2  theta_c(s) = (pi/2) sigmoid(a_c + w_c . e(s)),  e(s) one-hot {S,M,L}, w_c[L]=0
#   D1  per-node temperature tau_n by NLL, fit once then frozen
#   D3  test-time tau0 by entropy minimisation over unlabelled target norms
#   D4  TTG (threshold-transfer gap) and ECE_d (per-depth calibration error)
# ------------------------------------------------------------------------
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TreeConfig, DEFAULT

# Scale-band ids
S_BAND, M_BAND, L_BAND = 0, 1, 2


def band(area, cfg: TreeConfig = DEFAULT):
    """COCO scale band of a box area (notation table): S < 32^2 <= M < 96^2 <= L.
    Accepts a python scalar or a tensor; returns the same kind (ids 0/1/2)."""
    if torch.is_tensor(area):
        b = torch.full_like(area, M_BAND, dtype=torch.long)
        b[area < cfg.area_small] = S_BAND
        b[area >= cfg.area_large] = L_BAND
        return b
    if area < cfg.area_small:
        return S_BAND
    if area >= cfg.area_large:
        return L_BAND
    return M_BAND


class ScaleConditionedRadius(nn.Module):
    """Eq D2: scale-conditioned cone radius.

        theta_c(s) = (pi/2) sigmoid( a_c + w_c . e(s) )

    This module owns ONLY the scale weights w (num_nodes, 3); the base radius
    logit ``a_c`` is supplied by the ConeField (so theta_c(s) reduces exactly to
    the Module-E radius theta_n = (pi/2)sigmoid(a_n) when w = 0).  The L column
    of w is pinned to 0 for identifiability, so a small blurry *known* object
    (band S) can only *widen* its admissible cone relative to L, never the
    reverse being ambiguous.
    """

    def __init__(self, num_nodes: int, cfg: TreeConfig = DEFAULT):
        super().__init__()
        self.cfg = cfg
        self.num_nodes = num_nodes
        self.w = nn.Parameter(torch.zeros(num_nodes, 3))
        # Pin the L column (index 2) to 0: mask it in every forward + zero grad.
        mask = torch.ones(num_nodes, 3)
        mask[:, L_BAND] = 0.0
        self.register_buffer("w_mask", mask)

    def _w(self) -> torch.Tensor:
        return self.w * self.w_mask

    def theta(self, a_base: torch.Tensor, node_ids: torch.Tensor,
              bands: torch.Tensor) -> torch.Tensor:
        """theta_c(s) for a batch of (node, band) pairs.

        a_base   : (num_nodes,) base radius logits from the ConeField
        node_ids : (P,) long
        bands    : (P,) long in {0,1,2}
        """
        w = self._w()[node_ids]                              # (P,3)
        e = F.one_hot(bands, num_classes=3).to(w.dtype)      # (P,3)
        logit = a_base[node_ids] + (w * e).sum(dim=-1)       # (P,)
        return self.cfg.half_pi * torch.sigmoid(logit)

    def theta_all(self, a_base: torch.Tensor, band_id: int) -> torch.Tensor:
        """theta_c(s) for all nodes at a single fixed band (S/M/L)."""
        w = self._w()[:, band_id]                            # (num_nodes,)
        return self.cfg.half_pi * torch.sigmoid(a_base + w)


# ========================================================================
# D1 : per-node temperature (fit once, then frozen -> exemplar-free)
# ========================================================================
def fit_node_temperature(margins: torch.Tensor, descends: torch.Tensor,
                         iters: int = 200, lr: float = 0.05) -> float:
    """Eq D1.  q_n(z;tau) = sigmoid( margin / tau ), margin = 1 - r_{c*}.

    Minimise the node's own NLL:
        - sum [y descends] log q  -  sum [y halts] log(1-q)
    over tau > 0, on the validation data available when the node is created.
    Returns the fitted scalar tau (then frozen forever).
    """
    margins = margins.detach().float()
    descends = descends.detach().float()
    log_tau = torch.zeros((), requires_grad=True)            # tau = exp(0) = 1
    opt = torch.optim.Adam([log_tau], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        tau = log_tau.exp().clamp_min(1e-3)
        logit = margins / tau
        loss = F.binary_cross_entropy_with_logits(logit, descends)
        loss.backward()
        opt.step()
    return float(log_tau.exp().clamp_min(1e-3).item())


# ========================================================================
# D3 : test-time recalibration of the single magnitude scalar tau0
# ========================================================================
def recalibrate_tau0(obj_head, norms: torch.Tensor,
                     iters: int = 200, lr: float = 0.05) -> float:
    """Eq D3.  tau0_tgt = argmin_tau0  E_{h~tgt}[ H( sigma(f_obj(||h||/tau0)) ) ]

    Entropy minimisation over the depth-0 decision on unlabelled target norms.
    ``obj_head`` is an ObjectnessHead; its 2-layer MLP stays frozen and only
    the scalar tau0 moves.  The fitted value is written back into the head.
    """
    norms = norms.detach().float().view(-1, 1)
    for p in obj_head.mlp.parameters():
        p.requires_grad_(False)
    log_tau = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([log_tau], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        tau0 = log_tau.exp().clamp_min(1e-3)
        p = torch.sigmoid(obj_head.mlp(norms / tau0)).clamp(1e-6, 1 - 1e-6)
        ent = -(p * p.log() + (1 - p) * (1 - p).log())
        loss = ent.mean()
        loss.backward()
        opt.step()
    val = float(log_tau.exp().clamp_min(1e-3).item())
    obj_head.set_tau0(val)
    return val


# ========================================================================
# D4 : metrics
# ========================================================================
def best_f1_threshold(scores: np.ndarray, labels: np.ndarray):
    """Threshold maximising F1 of the positive (unknown) class.
    Returns (threshold, f1)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    P = max(1, int(y.sum()))
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / P
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    k = int(np.argmax(f1))
    return float(s[k]), float(f1[k])


def _f1_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    pred = (np.asarray(scores) >= thr).astype(np.int64)
    y = np.asarray(labels)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def threshold_transfer_gap(src_scores, src_labels, tgt_scores, tgt_labels) -> float:
    """Eq D4: TTG = F1_unk(theta*_tgt) - F1_unk(theta*_src), evaluated on target.

    The headline invariance measurement: how much unknown-detection F1 is lost
    when the source-optimal threshold is transferred to the target domain.  A
    ratio/angle-based cascade should keep |TTG| small; a magnitude-based one
    should not.
    """
    thr_src, _ = best_f1_threshold(np.asarray(src_scores), np.asarray(src_labels))
    thr_tgt, f1_tgt_opt = best_f1_threshold(np.asarray(tgt_scores), np.asarray(tgt_labels))
    f1_tgt_transferred = _f1_at(np.asarray(tgt_scores), np.asarray(tgt_labels), thr_src)
    return f1_tgt_opt - f1_tgt_transferred


def ece_per_depth(confidences: Sequence[float], correct: Sequence[int],
                  depths: Sequence[int], cfg: TreeConfig = DEFAULT) -> Dict[int, float]:
    """Eq D4: ECE_d per depth, 15 equal-width bins.

    Verifies (B4)'s depth-normalised geometric mean actually removed the depth
    penalty: if ECE_4 >> ECE_1, deep paths are still incomparable.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    dep = np.asarray(depths, dtype=np.int64)
    out: Dict[int, float] = {}
    bins = np.linspace(0.0, 1.0, cfg.ece_bins + 1)
    for d in np.unique(dep):
        m = dep == d
        cd, rd = conf[m], corr[m]
        n = len(cd)
        if n == 0:
            continue
        ece = 0.0
        for b in range(cfg.ece_bins):
            lo, hi = bins[b], bins[b + 1]
            sel = (cd > lo) & (cd <= hi) if b > 0 else (cd >= lo) & (cd <= hi)
            if sel.sum() == 0:
                continue
            acc = rd[sel].mean()
            avg_conf = cd[sel].mean()
            ece += (sel.sum() / n) * abs(acc - avg_conf)
        out[int(d)] = float(ece)
    return out
