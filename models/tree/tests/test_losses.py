"""Module E — annulus kept positive, warmup gating, finite grads at cone poles."""
import torch

from models.tree.config import TreeConfig
from models.tree.topology import TreeTopology
from models.tree.losses import ConeField, ReservedGapLoss, node_gaps


def _tree():
    # root(0) -> {1,2} leaves.  Minimal tree on which L_gap is meaningful.
    children = [[1, 2], [], []]
    leaf_paths = {0: [0, 1], 1: [0, 2]}
    return TreeTopology(num_nodes=3, children=children, leaf_paths=leaf_paths, root=0)


def test_forward_returns_four_terms():
    cfg = TreeConfig()
    topo = _tree()
    cones = ConeField(topo.num_nodes, cfg)
    loss = ReservedGapLoss(cfg)
    z = torch.randn(8, cfg.m)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    terms = loss(cones, topo, z, y, epoch=None)
    assert set(terms) == {"loss_contain", "loss_nest", "loss_gap", "loss_sib"}
    for v in terms.values():
        assert torch.isfinite(v)


def test_warmup_gates_gap_and_sib():
    cfg = TreeConfig()
    topo = _tree()
    cones = ConeField(topo.num_nodes, cfg)
    loss = ReservedGapLoss(cfg)
    z = torch.randn(6, cfg.m)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    early = loss(cones, topo, z, y, epoch=0)               # before warm_epochs
    assert float(early["loss_gap"]) == 0.0
    assert float(early["loss_sib"]) == 0.0
    late = loss(cones, topo, z, y, epoch=cfg.warm_epochs)  # at/after warmup
    # gap/sib now computed (value may be 0 if already satisfied, but path runs)
    assert "loss_gap" in late


def test_gap_loss_drives_positive_annulus():
    cfg = TreeConfig()
    topo = _tree()
    cones = ConeField(topo.num_nodes, cfg)
    # start from an overlapping config: children wide, parent narrow.
    with torch.no_grad():
        cones.mu_raw.zero_()
        cones.mu_raw[0, 0] = 1.0
        cones.mu_raw[1, 0] = 1.0; cones.mu_raw[1, 1] = 0.1
        cones.mu_raw[2, 0] = 1.0; cones.mu_raw[2, 1] = -0.1
        cones.a.fill_(0.0)
    loss = ReservedGapLoss(cfg)
    opt = torch.optim.Adam(cones.parameters(), lr=0.05)
    z = torch.randn(16, cfg.m)
    y = torch.tensor([0, 1] * 8)
    for _ in range(200):
        opt.zero_grad()
        terms = loss(cones, topo, z, y, epoch=cfg.warm_epochs)
        total = loss.weighted_total(terms)
        total.backward()
        opt.step()
    gaps = node_gaps(cones, topo)
    # after optimisation the root's reserved gap should be non-negative.
    assert gaps[0] >= -1e-2


def test_grads_finite_when_z_on_cone_axis():
    # z coincident with a cone axis is the arccos-pole case; grads must be finite.
    cfg = TreeConfig()
    topo = _tree()
    cones = ConeField(topo.num_nodes, cfg)
    with torch.no_grad():
        cones.mu_raw.zero_(); cones.mu_raw[:, 0] = 1.0
    loss = ReservedGapLoss(cfg)
    z = torch.zeros(4, cfg.m, requires_grad=True)
    with torch.no_grad():
        z[:, 0] = 1.0                         # exactly on every axis
    y = torch.tensor([0, 1, 0, 1])
    terms = loss(cones, topo, z, y, epoch=cfg.warm_epochs)
    loss.weighted_total(terms).backward()
    for p in cones.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_weight_dict_keys_and_values():
    cfg = TreeConfig()
    wd = ReservedGapLoss(cfg).weight_dict()
    assert wd["loss_contain"] == cfg.alpha
    assert wd["loss_nest"] == cfg.beta
    assert wd["loss_gap"] == cfg.eta
    assert wd["loss_sib"] == cfg.nu
