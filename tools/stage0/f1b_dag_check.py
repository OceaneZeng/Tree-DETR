#!/usr/bin/env python
# ------------------------------------------------------------------------
# Stage-0 / F1b : is the confusability relation even TREE-shaped?
# ------------------------------------------------------------------------
# A tree can only capture confusability if most confusion mass falls between
# siblings.  If a lot of mass is on non-sibling pairs, the relation is a DAG and
# Module A (a pure tree) is the wrong structure - a redesign, not a tweak.
#
# Metric: off-sibling confusion mass = sum of affinity on non-sibling class
# pairs / total affinity  (models.tree.offdiagonal_confusion_mass).
# Gate (note): off-sibling mass <= 0.40.
# ------------------------------------------------------------------------
import argparse

import common
from models.tree import (build_affinity, confusion_to_rates, induce_tree,
                         offdiagonal_confusion_mass)


def main():
    ap = argparse.ArgumentParser(description="F1b: off-sibling confusion mass (tree vs DAG)")
    common.add_common_args(ap)
    args = ap.parse_args()
    feats = common.load_or_synth(args)

    C = confusion_to_rates(common.confusion_counts(feats))
    A, D = build_affinity(C)
    tree = induce_tree(D, num_classes=feats.num_known)
    mass = offdiagonal_confusion_mass(A, tree)
    return common.verdict("F1b off-sibling confusion mass", mass,
                          mass <= 0.40, "off-sibling mass <= 0.40")


if __name__ == "__main__":
    raise SystemExit(main())
