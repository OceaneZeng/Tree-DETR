# ------------------------------------------------------------------------
# Tree-DETR : lightweight tree-topology container
# ------------------------------------------------------------------------
# A plain, framework-free description of the tree's *shape*, decoupled from
# both the geometry (cone parameters) and the rich ConfusabilityTree object.
# Both losses.py (Module E) and tree_structure.py (Module A) speak this type,
# so the loss can be unit-tested on a hand-built 2-level tree (arm EE-0)
# without any induction machinery.
#
# Node ids are contiguous ints [0 .. num_nodes).  Node 0 is the root
# (depth 0 = the objectness gate "is this a thing at all?").
# ------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TreeTopology:
    num_nodes: int
    children: List[List[int]]          # children[n] = child node ids of n ([] for a leaf)
    leaf_paths: Dict[int, List[int]]   # leaf-class label -> node ids on the root->leaf path
    root: int = 0

    def internal_nodes(self) -> List[int]:
        """Nodes that have at least one child (i.e. hold a discrimination gate)."""
        return [n for n in range(self.num_nodes) if self.children[n]]

    def leaves(self) -> List[int]:
        return [n for n in range(self.num_nodes) if not self.children[n]]

    def depth_of(self, node: int) -> int:
        """Root has depth 0.  Computed by walking parent links derived from
        ``children``.  O(num_nodes)."""
        parent = self._parent_map()
        d = 0
        cur = node
        while cur != self.root and cur in parent:
            cur = parent[cur]
            d += 1
        return d

    def _parent_map(self) -> Dict[int, int]:
        parent: Dict[int, int] = {}
        for n in range(self.num_nodes):
            for c in self.children[n]:
                parent[c] = n
        return parent

    def validate(self) -> None:
        """Cheap structural sanity checks - each non-root node has exactly one
        parent, and every leaf_path is a real root->leaf descent."""
        parent = self._parent_map()
        for n in range(self.num_nodes):
            if n != self.root:
                assert n in parent, f"node {n} has no parent"
        for cls, path in self.leaf_paths.items():
            assert path, f"empty path for class {cls}"
            assert not self.children[path[-1]], f"path for class {cls} does not end at a leaf"
