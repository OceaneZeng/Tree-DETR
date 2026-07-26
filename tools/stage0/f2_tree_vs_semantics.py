#!/usr/bin/env python
# ------------------------------------------------------------------------
# Stage-0 / F2 : is the confusability tree DIFFERENT from a semantic taxonomy?
# ------------------------------------------------------------------------
# Run this FIRST - it is the cheapest and the most likely to refute the method.
# If the confusability tree merely reproduces the semantic supercategories, the
# whole "confusability != semantics" premise is dead and nothing new is gained.
#
# Metric: overlap = fraction of confusable sibling pairs that ALSO share a
# semantic supercategory.  Low overlap = the structures are genuinely different.
# Gate (note): overlap <= 0.60.
# ------------------------------------------------------------------------
import argparse
import itertools
import numpy as np

import common
from models.tree import build_affinity, confusion_to_rates, induce_tree


def sibling_pairs(tree):
    """Set of unordered known-class pairs whose leaves share a parent."""
    pairs = set()
    for node, ch in tree.children.items():
        leafcls = [tree.leaf_class[c] for c in ch if tree.is_leaf(c) and c in tree.leaf_class]
        for a, b in itertools.combinations(sorted(leafcls), 2):
            pairs.add((a, b))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="F2: confusability tree vs semantic taxonomy")
    common.add_common_args(ap)
    args = ap.parse_args()
    feats = common.load_or_synth(args)

    C = confusion_to_rates(common.confusion_counts(feats))
    A, D = build_affinity(C)
    tree = induce_tree(D, num_classes=feats.num_known)

    conf_pairs = sibling_pairs(tree)
    if not conf_pairs:
        return common.verdict("F2 tree/semantics overlap", 0.0, True,
                              "overlap <= 0.60 (no confusable siblings)")

    # semantic sibling pairs: classes sharing a supercategory
    super_id = feats.class_super
    if super_id is None:
        print("[SKIP] F2: no semantic supercategory labels in features")
        return 0
    sem_pairs = set()
    for a, b in itertools.combinations(range(feats.num_known), 2):
        if super_id[a] == super_id[b]:
            sem_pairs.add((a, b))

    overlap = len(conf_pairs & sem_pairs) / len(conf_pairs)
    return common.verdict("F2 tree/semantics overlap", overlap,
                          overlap <= 0.60, "overlap <= 0.60")


if __name__ == "__main__":
    raise SystemExit(main())
