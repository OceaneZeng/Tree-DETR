"""Module B — halt depth (bg / unknown / leaf), beam=2, depth-normalised score."""
import math
import torch

from models.tree.config import TreeConfig
from models.tree.topology import TreeTopology
from models.tree.losses import ConeField
from models.tree.calibration import ScaleConditionedRadius
from models.tree.cascade import Cascade


def _two_level_tree():
    # root(0) -> {1, 2}; node 1 -> {3, 4} leaves; node 2 -> {5, 6} leaves.
    children = [[1, 2], [3, 4], [5, 6], [], [], [], []]
    leaf_paths = {
        0: [0, 1, 3], 1: [0, 1, 4], 2: [0, 2, 5], 3: [0, 2, 6],
    }
    return TreeTopology(num_nodes=7, children=children, leaf_paths=leaf_paths, root=0)


def _planted_cascade(topo, cfg):
    """Cones whose axes are distinct one-hot directions so descent is decidable."""
    cones = ConeField(topo.num_nodes, cfg)
    m = cfg.m
    axes = torch.zeros(topo.num_nodes, m)
    # give every node a distinct axis; children near their parent's region
    dirs = {
        0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 2, 6: 4,   # 3 sits with 1, 5 sits with 2
    }
    for n, d in dirs.items():
        axes[n, d % m] = 1.0
    with torch.no_grad():
        cones.mu_raw.copy_(axes)
        cones.a.fill_(2.0)                     # wide cones so gates open
    radius = ScaleConditionedRadius(topo.num_nodes, cfg)
    return Cascade(topo, cones, radius, cfg)


def test_leaf_reached_is_known():
    cfg = TreeConfig()
    topo = _two_level_tree()
    casc = _planted_cascade(topo, cfg)
    # z aligned with node 3's axis (dim 1) -> should descend 0->1->3 (a leaf).
    z = torch.zeros(cfg.m); z[1] = 1.0
    res = casc.descend_one(z)
    assert res.is_known
    assert res.leaf_class is not None
    assert res.halt_depth >= 1


def test_background_halts_at_zero():
    cfg = TreeConfig()
    topo = _two_level_tree()
    cones = ConeField(topo.num_nodes, cfg)
    with torch.no_grad():
        # all child axes point to dim 0; make them tiny cones far from z.
        cones.mu_raw.zero_(); cones.mu_raw[:, 0] = 1.0
        cones.a.fill_(-8.0)                    # theta ~ 0 -> nothing claims
    radius = ScaleConditionedRadius(topo.num_nodes, cfg)
    casc = Cascade(topo, cones, radius, cfg)
    z = torch.zeros(cfg.m); z[5] = 1.0         # orthogonal to every axis
    res = casc.descend_one(z)
    assert res.halt_depth == 0
    assert not res.is_known


def test_unknown_halts_midtree():
    cfg = TreeConfig()
    topo = _two_level_tree()
    cones = ConeField(topo.num_nodes, cfg)
    with torch.no_grad():
        axes = torch.zeros(topo.num_nodes, cfg.m)
        axes[0, 0] = 1.0
        axes[1, 1] = 1.0; axes[2, 2] = 1.0     # depth-1 nodes
        axes[3, 5] = 1.0; axes[4, 6] = 1.0     # leaves under 1 -> far from z
        axes[5, 7] = 1.0; axes[6, 8] = 1.0     # leaves under 2 -> far from z
        cones.mu_raw.copy_(axes)
        cones.a.fill_(0.0)                      # theta = pi/4
    radius = ScaleConditionedRadius(topo.num_nodes, cfg)
    casc = Cascade(topo, cones, radius, cfg)
    # z near node 1's axis (dim 1): passes depth-1 gate, but no leaf claims it.
    z = torch.zeros(cfg.m); z[1] = 1.0
    res = casc.descend_one(z)
    assert res.halt_depth >= 1
    assert not res.is_known                     # unknown: descended but no leaf


def test_beam_width_respected():
    cfg = TreeConfig()
    assert cfg.beam == 2
    topo = _two_level_tree()
    casc = _planted_cascade(topo, cfg)
    z = torch.zeros(cfg.m); z[1] = 1.0
    res = casc.descend_one(z)
    # gates counted = children examined; with beam=2 over a 2-level tree it is
    # bounded and positive (measured cost, not asymptotic).
    assert res.gates > 0


def test_path_score_depth_normalised_in_unit_interval():
    cfg = TreeConfig()
    topo = _two_level_tree()
    casc = _planted_cascade(topo, cfg)
    z = torch.zeros(cfg.m); z[1] = 1.0
    res = casc.descend_one(z)
    assert 0.0 <= res.path_score <= 1.0


def test_detection_score_product():
    assert abs(Cascade.detection_score(0.5, 0.4) - 0.2) < 1e-9


def test_stats_fractions_sum_to_one():
    cfg = TreeConfig()
    topo = _two_level_tree()
    casc = _planted_cascade(topo, cfg)
    zs = torch.randn(16, cfg.m)
    results = casc.descend(zs)
    st = casc.stats(results)
    total = st["frac_background"] + st["frac_unknown"] + st["frac_known"]
    assert abs(total - 1.0) < 1e-9
    assert st["mean_gates"] > 0
