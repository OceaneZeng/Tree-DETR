# ------------------------------------------------------------------------
# Tree-DETR : Module B - novelty as halt depth (the signal)
# ------------------------------------------------------------------------
# Descend the confusability tree by cosine-cone gates until no child claims the
# object.  The *halt depth* is the output:
#     halt at depth 0  -> background (not even a thing)
#     halt at depth d>=1 -> unknown object, novelty degree d
#     reach a leaf       -> known class
#
# This is a hierarchically-localised residual: unknown is never a positive
# target, it is "no child claims it" evaluated locally at each node.  The
# depth-0 objectness gate (magnitude) *corroborates* rather than replaces the
# angular tests - hard background must fail one magnitude test AND every angular
# sibling test, while a real novel object typically survives at least one level.
#
# Equations:
#   B2  r_c(z,s) = angle(z, mu_c) / theta_c(s) <= 1       (the gate; ratio, not angle)
#   B3  beam B=2 descent; halt when every beam node has min_c r_c > 1
#   B4  q_n = sigmoid((1 - r_{c*}) / tau_n);  S(path) = (prod q_n)^{1/|path|}
#   B5  output map + detection score = sigma(z_obj) * S(path)
#
# Cost is ~ B * E[depth] gates per object - MEASURED (see ``stats``), not an
# asymptotic log|C| claim (the tree is deliberately unbalanced).
# ------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .config import TreeConfig, DEFAULT
from .geometry import stable_angle
from .topology import TreeTopology
from .calibration import band as band_of, M_BAND


@dataclass
class HaltResult:
    halt_depth: int                 # 0 = background, d>=1 = unknown degree d, leaf-depth = known
    node: int                       # node the descent ended on
    is_known: bool                  # True iff a leaf was reached
    leaf_class: Optional[int]       # class label if known, else None
    path: List[int]                 # node ids visited (root .. end)
    path_score: float               # S(path), Eq B4
    gates: int                      # number of gate evaluations (cost measurement)


class Cascade(nn.Module):
    """Beam-search descent over the cone tree (Module B).

    Reads cone axes/base-radii from a ConeField and scale-conditioned radii
    from a ScaleConditionedRadius (Module D), plus a per-node temperature tau_n
    (fit by D1, frozen).  Holds no cone parameters of its own.
    """

    def __init__(self, topo: TreeTopology, cones, radius, cfg: TreeConfig = DEFAULT):
        super().__init__()
        self.cfg = cfg
        self.topo = topo
        self.cones = cones          # ConeField : .mu (N,m), .a (N,) base radius logits
        self.radius = radius        # ScaleConditionedRadius
        # per-node temperature; 1.0 until fit by calibration.fit_node_temperature.
        # A Parameter (frozen by default) so Module C (Eq C3) can unfreeze tau_n
        # at insertion time while D1 can also write it directly via set_tau.
        self.tau = nn.Parameter(torch.ones(topo.num_nodes), requires_grad=False)

    @torch.no_grad()
    def set_tau(self, node: int, value: float) -> None:
        self.tau[node] = float(value)

    # -- Eq B2 : gate ratio for the children of one node ----------------------
    def child_ratios(self, z: torch.Tensor, child_ids: List[int], band_id: int) -> torch.Tensor:
        """r_c = angle(z, mu_c) / theta_c(s) for every child c of a node.
        ``z`` is a single (m,) embedding; returns (len(child_ids),)."""
        idx = torch.as_tensor(child_ids, device=z.device, dtype=torch.long)
        mu_c = self.cones.mu[idx]                                   # (k, m)
        ang = stable_angle(z.unsqueeze(0), mu_c, self.cfg.eps)      # (k,)
        bands = torch.full((len(child_ids),), band_id, device=z.device, dtype=torch.long)
        theta_c = self.radius.theta(self.cones.a, idx, bands)       # (k,)
        return ang / theta_c.clamp_min(self.cfg.eps)

    # -- Eq B3/B4/B5 : descend one embedding ----------------------------------
    @torch.no_grad()
    def descend_one(self, z: torch.Tensor, box_area: Optional[float] = None) -> HaltResult:
        cfg = self.cfg
        band_id = M_BAND if box_area is None else int(band_of(torch.tensor(float(box_area)), cfg))
        B = cfg.beam

        # beam item = (node, path list, list of (parent_node, r_taken))
        beam = [(self.topo.root, [self.topo.root], [])]
        depth = 0
        gates = 0

        while True:
            claims = []           # (r, child, item)
            any_claim = False
            all_leaf = True
            for item in beam:
                node = item[0]
                ch = self.topo.children[node]
                if not ch:
                    continue      # leaf beam element - no expansion
                all_leaf = False
                rs = self.child_ratios(z, ch, band_id)
                gates += len(ch)
                min_r = float(rs.min())
                if min_r <= cfg.gate_threshold:
                    any_claim = True
                for c, r in zip(ch, rs.tolist()):
                    claims.append((r, c, item))

            if all_leaf or not any_claim or not claims:
                break             # halt at current depth

            claims.sort(key=lambda x: x[0])
            beam = []
            for r, c, item in claims[:B]:
                beam.append((c, item[1] + [c], item[2] + [(item[0], r)]))
            depth += 1

        # choose the best beam element: prefer a reached leaf, then best S(path)
        def score_item(item):
            return self._path_score(item[2])
        leaves = [it for it in beam if not self.topo.children[it[0]]]
        pool = leaves if leaves else beam
        best = max(pool, key=score_item)
        node = best[0]
        is_known = not self.topo.children[node]
        leaf_class = None
        if is_known:
            # map leaf node -> class via reverse of leaf_paths
            leaf_class = self._leaf_class(node)
        return HaltResult(
            halt_depth=depth,
            node=node,
            is_known=is_known,
            leaf_class=leaf_class,
            path=best[1],
            path_score=score_item(best),
            gates=gates,
        )

    def _path_score(self, decisions) -> float:
        """Eq B4: S(path) = (prod_n sigmoid((1 - r_{c*})/tau_n))^{1/|path|}.

        ``decisions`` = list of (parent_node n, ratio r taken at n).  The
        geometric mean removes the systematic penalty on deep paths.
        """
        if not decisions:
            return 1.0
        logs = 0.0
        for parent_node, r in decisions:
            tau = float(self.tau[parent_node].clamp_min(1e-3))
            q = torch.sigmoid(torch.tensor((1.0 - r) / tau)).item()
            q = min(max(q, 1e-12), 1.0)
            logs += torch.log(torch.tensor(q)).item()
        return float(torch.exp(torch.tensor(logs / len(decisions))).item())

    def _leaf_class(self, node: int) -> Optional[int]:
        for cls, path in self.topo.leaf_paths.items():
            if path and path[-1] == node:
                return cls
        return None

    # -- batch convenience ----------------------------------------------------
    @torch.no_grad()
    def descend(self, zs: torch.Tensor, box_areas: Optional[List[float]] = None) -> List[HaltResult]:
        n = zs.shape[0]
        areas = box_areas if box_areas is not None else [None] * n
        return [self.descend_one(zs[i], areas[i]) for i in range(n)]

    # -- Eq B5 : detection score ---------------------------------------------
    @staticmethod
    def detection_score(obj_prob: float, path_score: float) -> float:
        """score = sigma(z_obj) * S(path)  (for mAP / WI / A-OSE ranking)."""
        return float(obj_prob) * float(path_score)

    def stats(self, results: List[HaltResult]) -> Dict[str, float]:
        """Measured cost / halt statistics (the honest cost claim ~ B*E[depth])."""
        if not results:
            return {}
        depths = [r.halt_depth for r in results]
        gates = [r.gates for r in results]
        n_bg = sum(1 for r in results if r.halt_depth == 0)
        n_unk = sum(1 for r in results if r.halt_depth >= 1 and not r.is_known)
        n_known = sum(1 for r in results if r.is_known)
        return {
            "mean_depth": sum(depths) / len(depths),
            "mean_gates": sum(gates) / len(gates),   # the cost claim, measured
            "frac_background": n_bg / len(results),
            "frac_unknown": n_unk / len(results),
            "frac_known": n_known / len(results),
        }
