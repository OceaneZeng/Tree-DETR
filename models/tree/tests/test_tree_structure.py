"""Module A — affinity symmetry, UPGMA + n-ary collapse, insertion diagnostics."""
import numpy as np

from models.tree.config import TreeConfig
from models.tree.tree_structure import (
    confusion_to_rates, build_affinity, induce_tree, insert_class,
    offdiagonal_confusion_mass, ConfusabilityTree, _upgma, _collapse, _tree_stats,
)


def test_confusion_to_rates_rows_sum_to_one():
    counts = np.array([[10, 5, 0], [1, 8, 1], [0, 0, 4]], dtype=float)
    C = confusion_to_rates(counts)
    assert np.allclose(C.sum(axis=1), 1.0)


def test_confusion_to_rates_zero_row_safe():
    C = confusion_to_rates(np.zeros((3, 3)))
    assert np.isfinite(C).all()


def test_build_affinity_symmetric_zero_diag():
    C = np.array([[0.0, 0.4, 0.1], [0.2, 0.0, 0.3], [0.05, 0.25, 0.0]])
    A, D = build_affinity(C)
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 0.0)
    assert np.allclose(np.diag(D), 0.0)
    # D is a proper distance: high affinity -> small distance
    assert D[0, 1] < D[0, 2]


def test_upgma_two_tight_clusters():
    # classes 0,1 confuse each other; 2,3 confuse each other; blocks disjoint.
    A = np.array([
        [0.0, 0.9, 0.0, 0.0],
        [0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9],
        [0.0, 0.0, 0.9, 0.0],
    ])
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    children, height, leafclass, root = _upgma(D)
    # 4 leaves + 3 internal merges = 7 nodes, root is last
    assert root == 6
    # the two lowest merges join {0,1} and {2,3}
    merges = sorted((height[n], tuple(sorted(children[n]))) for n in children if children[n])
    assert merges[0][1] == (0, 1)
    assert merges[1][1] == (2, 3)


def test_induce_tree_respects_constraints():
    rng = np.random.RandomState(0)
    n = 12
    A = rng.rand(n, n) * 0.2
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    # two strongly-confused blocks
    for i, j in [(0, 1), (1, 2), (3, 4), (5, 6)]:
        A[i, j] = A[j, i] = 0.95
    D = 1.0 - A / A.max()
    np.fill_diagonal(D, 0.0)
    cfg = TreeConfig()
    tree = induce_tree(D, num_classes=n, cfg=cfg)
    assert tree.root == 0
    # every class present as a leaf
    assert sorted(tree.leaf_class.values()) == list(range(n))
    mc, dp = _tree_stats(tree.children, tree.root)
    assert mc <= cfg.max_children
    assert dp <= cfg.d_max(n)


def test_topology_roundtrip_and_validate():
    A = np.array([
        [0.0, 0.9, 0.0, 0.0],
        [0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9],
        [0.0, 0.0, 0.9, 0.0],
    ])
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    tree = induce_tree(D, num_classes=4)
    topo = tree.topology()
    topo.validate()
    # json round trip
    tree2 = ConfusabilityTree.from_json(tree.to_json())
    assert tree2.children == tree.children
    assert tree2.leaf_class == tree.leaf_class


def test_insert_class_vote_vs_affinity_disagreement():
    # flat tree over 4 classes: nodes 1..4 are leaves of root 0.
    children = {0: [1, 2, 3, 4], 1: [], 2: [], 3: [], 4: []}
    leaf_class = {1: 0, 2: 1, 3: 2, 4: 3}
    tree = ConfusabilityTree(children, leaf_class, root=0)
    # unknowns all halted at root (vote = 0) but affinity points at class 2's leaf.
    halts = [0, 0, 0]
    affinity_row = {0: 0.01, 1: 0.01, 2: 0.99, 3: 0.01}
    info = insert_class(tree, cls=4, unknown_halts=halts,
                        affinity_row=affinity_row, cfg=TreeConfig())
    assert info["n_vote"] == 0
    assert info["disagreement"] == 1                    # vote != aff (load-bearing signal)
    assert 4 in tree.class_leaf                          # new class inserted


def test_offdiagonal_confusion_mass_low_for_block_tree():
    A = np.array([
        [0.0, 0.9, 0.02, 0.02],
        [0.9, 0.0, 0.02, 0.02],
        [0.02, 0.02, 0.0, 0.9],
        [0.02, 0.02, 0.9, 0.0],
    ])
    D = 1.0 - A / A.max()
    np.fill_diagonal(D, 0.0)
    tree = induce_tree(D, num_classes=4)
    frac = offdiagonal_confusion_mass(A, tree)
    assert 0.0 <= frac <= 1.0
    assert frac < 0.40                                   # tree assumption holds
