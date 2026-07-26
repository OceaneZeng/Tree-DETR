# ------------------------------------------------------------------------
# Tree-DETR : Module A - confusability tree (the structure)
# ------------------------------------------------------------------------
# The tree over known classes where *siblings are classes that are hard to
# tell apart in feature space*, not classes that are semantically related.
# Induced by agglomerative average-linkage (UPGMA) merging of class prototypes
# under a confusion-based affinity, then collapsed from binary to n-ary.
#
# Equations implemented:
#   A2  C_ij = fraction of class-i boxes predicted as j ; A = 1/2 (C+C^T)(1-I)
#   A3  d_ij = 1 - A_ij / max_{k!=l} A_kl
#   A4  average linkage; binary->n-ary collapse h_u - h_v < delta_h * H ; repair
#   A5  insertion of a new class: n_vote (halt mode) vs n_aff (affinity argmax)
#   F1b off-diagonal confusion mass (is the relation even tree-shaped?)
#
# Pure numpy + a self-contained UPGMA so the module is testable without scipy;
# an optional scipy fast path is used when available.
# ------------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from collections import Counter
import json
import numpy as np

from .config import TreeConfig, DEFAULT
from .topology import TreeTopology


# ========================================================================
# Affinity  (Eqs A2, A3)
# ========================================================================
def confusion_to_rates(counts: np.ndarray) -> np.ndarray:
    """Row-normalise a raw confusion-count matrix into rates C_ij (Eq A2 inner
    term): C_ij = (# class-i boxes argmax-predicted as j) / |G_i|."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return counts / totals


def build_affinity(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """From a confusion-rate matrix C build the symmetric affinity A (Eq A2)
    and the linkage distance D (Eq A3).

    Returns (A, D), both (n, n) with zero diagonal on A.
    """
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    A = 0.5 * (C + C.T)
    np.fill_diagonal(A, 0.0)              # (1 - I) mask
    off = A[~np.eye(n, dtype=bool)]
    amax = off.max() if off.size and off.max() > 0 else 1.0
    D = 1.0 - A / amax                    # Eq A3
    np.fill_diagonal(D, 0.0)
    return A, D


# ========================================================================
# Self-contained UPGMA (average linkage) via Lance-Williams
# ========================================================================
def _upgma(D: np.ndarray):
    """Average-linkage agglomerative clustering.

    Returns dicts (children, height, leafclass) keyed by node id and the root
    id.  Leaf ids are 0..n-1 (== class index); internal ids follow.  Merge
    height = inter-cluster average distance (scipy 'average' convention).
    """
    n = D.shape[0]
    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    height: Dict[int, float] = {i: 0.0 for i in range(n)}
    leafclass: Dict[int, Optional[int]] = {i: i for i in range(n)}
    size: Dict[int, int] = {i: 1 for i in range(n)}
    active = set(range(n))

    dist: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = float(D[i, j])

    def getd(i: int, j: int) -> float:
        return dist[(i, j)] if i < j else dist[(j, i)]

    next_id = n
    while len(active) > 1:
        best = None
        for (i, j), d in dist.items():
            if i in active and j in active and (best is None or d < best[0]):
                best = (d, i, j)
        d, a, b = best
        new = next_id
        next_id += 1
        for k in active:
            if k == a or k == b:
                continue
            dnew = (size[a] * getd(a, k) + size[b] * getd(b, k)) / (size[a] + size[b])
            lo, hi = (new, k) if new < k else (k, new)
            dist[(lo, hi)] = dnew
        size[new] = size[a] + size[b]
        children[new] = [a, b]
        height[new] = d
        leafclass[new] = None
        active.discard(a)
        active.discard(b)
        active.add(new)
    root = next_id - 1
    return children, height, leafclass, root


def _upgma_scipy(D: np.ndarray):
    """Optional fast path; identical output contract to ``_upgma``."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    n = D.shape[0]
    Z = linkage(squareform(D, checks=False), method="average")
    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    height: Dict[int, float] = {i: 0.0 for i in range(n)}
    leafclass: Dict[int, Optional[int]] = {i: i for i in range(n)}
    for step, (l, r, h, _sz) in enumerate(Z):
        node = n + step
        children[node] = [int(l), int(r)]
        height[node] = float(h)
        leafclass[node] = None
    root = n + len(Z) - 1
    return children, height, leafclass, root


# ========================================================================
# Binary -> n-ary collapse  (Eq A4)
# ========================================================================
def _collapse(children, height, root, delta_h: float):
    """Collapse internal node v into its parent u iff h_u - h_v < delta_h * H.
    Returns a new children map (n-ary)."""
    H = height[root] or 1.0
    parent: Dict[int, int] = {}
    for p, chs in children.items():
        for c in chs:
            parent[c] = p

    def is_internal(x: int) -> bool:
        return len(children.get(x, [])) > 0

    transparent = set()
    for v in children:
        if v == root or not is_internal(v):
            continue
        if height[parent[v]] - height[v] < delta_h * H:
            transparent.add(v)

    def resolve(x: int) -> List[int]:
        out: List[int] = []
        for c in children.get(x, []):
            if c in transparent and is_internal(c):
                out.extend(resolve(c))
            else:
                out.append(c)
        return out

    new_children: Dict[int, List[int]] = {}

    def rec(x: int):
        if not is_internal(x):
            new_children[x] = []
            return
        chs = resolve(x)
        new_children[x] = chs
        for c in chs:
            rec(c)

    rec(root)
    return new_children


def _tree_stats(new_children, root):
    """(max branching factor, max leaf depth) with root at depth 0."""
    max_children = max((len(v) for v in new_children.values()), default=0)
    depth = {root: 0}
    stack = [root]
    max_depth = 0
    while stack:
        x = stack.pop()
        for c in new_children.get(x, []):
            depth[c] = depth[x] + 1
            max_depth = max(max_depth, depth[c])
            stack.append(c)
    return max_children, max_depth


# ========================================================================
# ConfusabilityTree
# ========================================================================
class ConfusabilityTree:
    """A confusability tree with contiguous node ids (root == 0).

    Leaves carry a ``class label``.  Internal nodes hold the sibling-local
    gates.  Convertible to a ``TreeTopology`` for the loss/cascade, and to/from
    JSON for the per-run logging the note requires.
    """

    def __init__(self, children: Dict[int, List[int]],
                 leaf_class: Dict[int, int], root: int = 0):
        self.children = {int(k): [int(c) for c in v] for k, v in children.items()}
        self.leaf_class = {int(k): int(v) for k, v in leaf_class.items()}
        self.root = int(root)
        self.parent: Dict[int, int] = {}
        for p, chs in self.children.items():
            for c in chs:
                self.parent[c] = p
        self.class_leaf = {v: k for k, v in self.leaf_class.items()}

    # --- structure queries -------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.children)

    def is_leaf(self, n: int) -> bool:
        return len(self.children.get(n, [])) == 0

    def leaves(self, n: Optional[int] = None) -> List[int]:
        """Leaf descendants L(n) (all leaves if n is None)."""
        if n is None:
            return [k for k in self.children if self.is_leaf(k)]
        out: List[int] = []
        stack = [n]
        while stack:
            x = stack.pop()
            if self.is_leaf(x):
                out.append(x)
            else:
                stack.extend(self.children[x])
        return out

    def leaf_classes(self, n: int) -> List[int]:
        return [self.leaf_class[l] for l in self.leaves(n)]

    def siblings(self, n: int) -> List[int]:
        """sib(n): the other children of n's parent."""
        p = self.parent.get(n)
        if p is None:
            return []
        return [c for c in self.children[p] if c != n]

    def path(self, leaf: int) -> List[int]:
        """Node ids from the root down to ``leaf`` inclusive."""
        out = [leaf]
        cur = leaf
        while cur in self.parent:
            cur = self.parent[cur]
            out.append(cur)
        out.reverse()
        return out

    def path_of_class(self, cls: int) -> List[int]:
        return self.path(self.class_leaf[cls])

    def depth_of(self, n: int) -> int:
        d = 0
        cur = n
        while cur in self.parent:
            cur = self.parent[cur]
            d += 1
        return d

    def max_depth(self) -> int:
        return max(self.depth_of(l) for l in self.leaves())

    # --- conversions -------------------------------------------------------
    def topology(self) -> TreeTopology:
        children = [self.children.get(n, []) for n in range(self.num_nodes)]
        leaf_paths = {self.leaf_class[l]: self.path(l) for l in self.leaves()}
        topo = TreeTopology(num_nodes=self.num_nodes, children=children,
                            leaf_paths=leaf_paths, root=self.root)
        return topo

    def to_json(self) -> str:
        return json.dumps({
            "children": {str(k): v for k, v in self.children.items()},
            "leaf_class": {str(k): v for k, v in self.leaf_class.items()},
            "root": self.root,
        }, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ConfusabilityTree":
        d = json.loads(s)
        children = {int(k): v for k, v in d["children"].items()}
        leaf_class = {int(k): v for k, v in d["leaf_class"].items()}
        return cls(children, leaf_class, d.get("root", 0))

    # --- mutation ----------------------------------------------------------
    def add_leaf(self, parent: int, cls: int) -> int:
        """Insert a new leaf for class ``cls`` under ``parent``; return its id."""
        new_id = self.num_nodes
        self.children[new_id] = []
        self.children[parent].append(new_id)
        self.parent[new_id] = parent
        self.leaf_class[new_id] = cls
        self.class_leaf[cls] = new_id
        return new_id


def _relabel_contiguous(children, leafclass, root) -> ConfusabilityTree:
    """BFS-relabel arbitrary ids to 0..N-1 with root -> 0, drop unary chains."""
    order: List[int] = []
    seen = set()
    queue = [root]
    while queue:
        x = queue.pop(0)
        if x in seen:
            continue
        seen.add(x)
        order.append(x)
        queue.extend(children.get(x, []))
    remap = {old: new for new, old in enumerate(order)}
    new_children = {remap[o]: [remap[c] for c in children.get(o, [])] for o in order}
    new_leafclass = {remap[o]: leafclass[o] for o in order if leafclass.get(o) is not None}
    return ConfusabilityTree(new_children, new_leafclass, root=remap[root])


# ========================================================================
# Induction driver  (Eq A4 with repair loop)
# ========================================================================
def induce_tree(D: np.ndarray, num_classes: Optional[int] = None,
                cfg: TreeConfig = DEFAULT, use_scipy: bool = False) -> ConfusabilityTree:
    """Induce the confusability tree from a distance matrix D (Eq A3).

    Runs UPGMA once, then collapses binary->n-ary at a collapse fraction delta_h
    (Eq A4) chosen to satisfy both the branching (|C(n)| <= max_children) and
    depth (<= D_max) constraints.

    Note on the repair search: the note sketches a halve/double adjustment of
    delta_h, but the feasible window can be narrow (increasing delta reduces
    depth while *increasing* branching), so a naive binary step oscillates past
    it.  We instead sweep a deterministic grid of delta values and select the
    best tree lexicographically -- depth feasibility first (it is the hard
    architectural limit on cascade length), then branching, then the smallest
    delta (retain the most hierarchy).  cfg.delta_h is always included in the
    grid so the note's nominal value is honoured when it is already feasible.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if num_classes is None:
        num_classes = n
    d_max = cfg.d_max(num_classes)

    if use_scipy:
        children0, height, leafclass, root0 = _upgma_scipy(D)
    else:
        children0, height, leafclass, root0 = _upgma(D)

    # Deterministic delta grid: the nominal delta_h plus a geometric sweep of
    # the (0, 1) collapse range.  Geometric spacing is dense at the small-delta
    # end, where the feasible window between "too deep" and "too bushy" is
    # narrowest (a uniform grid can step straight over it).
    grid = sorted({cfg.delta_h, *[round(float(x), 5)
                                  for x in np.geomspace(0.002, 0.98, 400)]})

    def penalty(mc: int, dp: int, delta: float):
        # lexicographic: depth overflow, then branching overflow, then -delta
        # is NOT used; smaller delta wins on ties (keeps more structure).
        return (max(0, dp - d_max), max(0, mc - cfg.max_children), delta)

    best = None                            # (penalty_key, nc)
    for delta in grid:
        nc = _collapse(children0, height, root0, delta)
        mc, dp = _tree_stats(nc, root0)
        key = penalty(mc, dp, delta)
        if best is None or key < best[0]:
            best = (key, nc)
            if key[0] == 0 and key[1] == 0:
                # fully feasible at the smallest such delta (grid ascending): done
                break

    return _relabel_contiguous(best[1], leafclass, root0)


# ========================================================================
# Insertion of a new class  (Eq A5)
# ========================================================================
def insert_class(tree: ConfusabilityTree, cls: int,
                 unknown_halts: List[int], affinity_row: Dict[int, float],
                 cfg: TreeConfig = DEFAULT) -> Dict[str, object]:
    """Insert class ``cls`` with two independent parent estimates (Eq A5).

    unknown_halts : node ids where the discovered unknown instances of this
                    class halted (their U_k). n_vote = mode of these.
    affinity_row  : class-label -> A(cls, label), affinity of the new class to
                    each existing known class.

    n_aff = argmax_n  mean_{l in L(n)} A(cls, l).  Insert under n_vote; override
    with n_aff only if its affinity margin over n_vote exceeds cfg margin (0.10).
    Reports the n_vote != n_aff disagreement - the direct measurement of whether
    "the halt site *is* the insertion site".
    """
    # n_vote : most common halt node
    if unknown_halts:
        n_vote = Counter(int(h) for h in unknown_halts).most_common(1)[0][0]
    else:
        n_vote = tree.root

    # n_aff : node whose leaf-descendant classes have the highest mean affinity
    def node_affinity(n: int) -> float:
        classes = tree.leaf_classes(n)
        if not classes:
            return -1.0
        return float(np.mean([affinity_row.get(c, 0.0) for c in classes]))

    candidates = [n for n in range(tree.num_nodes)]
    aff_scores = {n: node_affinity(n) for n in candidates}
    n_aff = max(candidates, key=lambda n: aff_scores[n])

    margin = aff_scores[n_aff] - aff_scores.get(n_vote, -1.0)
    parent = n_aff if margin > cfg.insert_aff_margin else n_vote
    # A new class is inserted as a sibling of the chosen node's children, i.e.
    # as a new child of that node (a leaf under it).
    new_leaf = tree.add_leaf(parent, cls)

    return {
        "class": cls,
        "parent": parent,
        "new_leaf": new_leaf,
        "n_vote": n_vote,
        "n_aff": n_aff,
        "affinity_margin": margin,
        "disagreement": int(n_vote != n_aff),
    }


# ========================================================================
# F1b : off-diagonal confusion mass (is the relation tree-shaped?)
# ========================================================================
def offdiagonal_confusion_mass(A: np.ndarray, tree: ConfusabilityTree) -> float:
    """Fraction of confusion mass on non-sibling class pairs (F1b):

        sum_{i<j} A_ij 1[i,j not siblings]  /  sum_{i<j} A_ij

    Two classes are 'siblings' iff their leaves share a parent.  Above ~0.40
    the tree assumption is wrong and a DAG is required (Module A redesign).
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    # class -> parent-of-its-leaf
    leaf_parent = {}
    for cls, leaf in tree.class_leaf.items():
        leaf_parent[cls] = tree.parent.get(leaf, tree.root)
    total = 0.0
    off = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = A[i, j]
            total += w
            same_parent = leaf_parent.get(i) == leaf_parent.get(j)
            if not same_parent:
                off += w
    return off / total if total > 0 else 0.0
