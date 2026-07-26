# ------------------------------------------------------------------------
# Tree-DETR : Module E - reserved-gap loss (the loss-level contribution)
# ------------------------------------------------------------------------
# Modules B and C both assume something the architecture cannot provide on its
# own: that there *is* somewhere for a "kind of n but none of n's children"
# object to sit, and that a newly inserted child takes its territory from that
# empty space rather than from its siblings.  This loss guarantees it by
# keeping a positive, unoccupied *annulus* between every parent cone and the
# union of its children.
#
# Regions are cosine cones on S^{m-1} (Eq E1):
#     R_n = { z : angle(z, mu_n) <= theta_n },   theta_n = (pi/2) sigmoid(a_n)
#
# The four terms (Eqs E3-E6) and total (E7):
#     L = L_task + a*L_contain + b*L_nest + eta*L_gap + nu*L_sib
# with the open-space budget gamma_n = rho * theta_n (Eq E8, rho=0.15 global).
#
# Numerical note: angles use the stable atan2 form (see geometry.stable_angle);
# grad arccos would diverge at the cone poles.
# ------------------------------------------------------------------------
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TreeConfig, DEFAULT
from .geometry import stable_angle, theta_from_logit, project_to_sphere, EPS
from .topology import TreeTopology


class ConeField(nn.Module):
    """Learnable cosine cones, one per tree node (Eq E1).

    Holds a raw axis matrix ``mu_raw`` (normalised on access -> mu in S^{m-1})
    and radius logits ``a`` (theta = (pi/2) sigmoid(a) in (0, pi/2)).  Axes and
    radii are ordinary parameters here; Module C replaces the trainable/frozen
    partition at insertion time (Eqs C3/C4) by freezing selected rows and
    swapping in the shrink-only radius reparameterisation.
    """

    def __init__(self, num_nodes: int, cfg: TreeConfig = DEFAULT):
        super().__init__()
        self.cfg = cfg
        self.num_nodes = num_nodes
        # Small random axes; they get located by L_contain/L_nest during warmup.
        self.mu_raw = nn.Parameter(torch.randn(num_nodes, cfg.m))
        # a = 0  ->  theta = (pi/2) * 0.5 = pi/4, a sensible mid-size cone.
        self.a = nn.Parameter(torch.zeros(num_nodes))

    @property
    def mu(self) -> torch.Tensor:
        return project_to_sphere(self.mu_raw)

    @property
    def theta(self) -> torch.Tensor:
        return theta_from_logit(self.a, self.cfg.half_pi)


class ReservedGapLoss(nn.Module):
    """The full Module-E loss (Eqs E3-E8).

    ``forward`` returns the four terms *unweighted* and named so they slot into
    the DETR ``weight_dict`` idiom (weights alpha/beta/eta/nu live in cfg and in
    the criterion's weight_dict).  ``weighted_total`` is provided for the
    standalone / test setting.

    Schedule (from "Fixed design E"): for the first ``warm_epochs`` epochs only
    L_contain and L_nest are active; L_gap and L_sib switch on afterwards, so
    the annulus is not enforced before the regions have located themselves.
    """

    def __init__(self, cfg: TreeConfig = DEFAULT):
        super().__init__()
        self.cfg = cfg

    # -- Eq E3 : containment (each z inside every cone on its own path) -------
    def loss_contain(self, cones: ConeField, topo: TreeTopology,
                     z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, theta = cones.mu, cones.theta
        # Flatten (sample, path-node) pairs so paths of unequal length vectorise.
        samp_idx: List[int] = []
        node_idx: List[int] = []
        y_list = y.tolist()
        for i, cls in enumerate(y_list):
            path = topo.leaf_paths.get(int(cls))
            if not path:
                continue
            for n in path:
                samp_idx.append(i)
                node_idx.append(n)
        if not node_idx:
            return z.new_zeros(())
        s = torch.as_tensor(samp_idx, device=z.device, dtype=torch.long)
        n = torch.as_tensor(node_idx, device=z.device, dtype=torch.long)
        ang = stable_angle(z[s], mu[n], self.cfg.eps)          # (P,)
        viol = F.softplus(ang - theta[n])                      # push z inside
        # normalise by number of supervised boxes (DETR convention)
        denom = max(1, z.shape[0])
        return viol.sum() / denom

    # -- Eqs E4/E5/E6 : structural terms over internal nodes ------------------
    def _structural(self, cones: ConeField, topo: TreeTopology, enable_gap_sib: bool):
        mu, theta = cones.mu, cones.theta
        cfg = self.cfg
        nest_terms: List[torch.Tensor] = []
        gap_terms: List[torch.Tensor] = []
        sib_terms: List[torch.Tensor] = []
        internal = topo.internal_nodes()
        for n in internal:
            ch = topo.children[n]
            ch_idx = torch.as_tensor(ch, device=mu.device, dtype=torch.long)
            mu_c = mu[ch_idx]                                   # (k, m)
            th_c = theta[ch_idx]                                # (k,)
            ang_nc = stable_angle(mu[n].unsqueeze(0), mu_c, cfg.eps)   # (k,) angle(mu_n, mu_c)

            # E4 L_nest: child cone contained in parent -> angle + theta_c <= theta_n
            nest_terms.append(F.softplus(ang_nc + th_c - theta[n]).sum())

            if enable_gap_sib:
                # E5 L_gap: keep a positive annulus of width gamma_n = rho*theta_n
                gap_n = (theta[n] - ang_nc - th_c).min()       # Eq E2
                gamma_n = cfg.rho * theta[n]                    # Eq E8
                gap_terms.append(F.relu(gamma_n - gap_n))

                # E6 L_sib: sibling cones mutually disjoint by margin lambda
                if len(ch) >= 2:
                    ang_cc = stable_angle(mu_c.unsqueeze(1), mu_c.unsqueeze(0), cfg.eps)  # (k,k)
                    sep = ang_cc - th_c.unsqueeze(1) - th_c.unsqueeze(0)                  # (k,k)
                    iu = torch.triu_indices(len(ch), len(ch), offset=1, device=mu.device)
                    pair = sep[iu[0], iu[1]]
                    sib_terms.append(F.softplus(cfg.lam - pair).sum())

        n_int = max(1, len(internal))
        z0 = mu.new_zeros(())
        loss_nest = torch.stack(nest_terms).sum() / n_int if nest_terms else z0
        loss_gap = torch.stack(gap_terms).sum() / n_int if gap_terms else z0
        loss_sib = torch.stack(sib_terms).sum() / n_int if sib_terms else z0
        return loss_nest, loss_gap, loss_sib

    def forward(self, cones: ConeField, topo: TreeTopology,
                z: torch.Tensor, y: torch.Tensor,
                epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Return the four unweighted loss terms.

        ``z`` : (N, m) matched-query cone embeddings, ``y`` : (N,) leaf-class
        labels.  ``epoch`` gates the warmup schedule; if None, all terms are
        active.
        """
        z = project_to_sphere(z, self.cfg.eps)
        enable_gap_sib = (epoch is None) or (epoch >= self.cfg.warm_epochs)
        loss_contain = self.loss_contain(cones, topo, z, y)
        loss_nest, loss_gap, loss_sib = self._structural(cones, topo, enable_gap_sib)
        return {
            "loss_contain": loss_contain,
            "loss_nest": loss_nest,
            "loss_gap": loss_gap,
            "loss_sib": loss_sib,
        }

    def weighted_total(self, terms: Dict[str, torch.Tensor]) -> torch.Tensor:
        """alpha*contain + beta*nest + eta*gap + nu*sib  (Eq E7, minus L_task)."""
        c = self.cfg
        return (c.alpha * terms["loss_contain"] + c.beta * terms["loss_nest"]
                + c.eta * terms["loss_gap"] + c.nu * terms["loss_sib"])

    def weight_dict(self, prefix: str = "") -> Dict[str, float]:
        """The alpha/beta/eta/nu coefficients keyed for a DETR weight_dict."""
        c = self.cfg
        return {f"{prefix}loss_contain": c.alpha, f"{prefix}loss_nest": c.beta,
                f"{prefix}loss_gap": c.eta, f"{prefix}loss_sib": c.nu}


@torch.no_grad()
def node_gaps(cones: ConeField, topo: TreeTopology) -> Dict[int, float]:
    """Diagnostic G1: per-node gap(n) (Eq E2).  A positive value means the
    annulus is non-empty at that node; plotted across tasks it is the
    capacity-erosion trace the note asks Module E to produce."""
    from .geometry import cone_gap
    mu, theta = cones.mu, cones.theta
    out: Dict[int, float] = {}
    for n in topo.internal_nodes():
        ch = torch.as_tensor(topo.children[n], device=mu.device, dtype=torch.long)
        out[n] = float(cone_gap(theta[n], mu[n], mu[ch], theta[ch]))
    return out
