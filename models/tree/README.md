# Tree-DETR — recognition cascade with sibling-local adapters

A self-contained, opt-in package implementing the method in
`recognition-cascade-with-sibling-local-adapters.md`. The confusability hierarchy
is reused as a **parameter-allocation structure**, not an output vocabulary:
an object is recognised by *descending* a tree of cosine-cone gates, novelty is
the *halt depth*, and a newly discovered class is learned by inserting one
**sibling-local adapter** at its halt site — leaving every other branch frozen.

Nothing here rewrites the base Deformable-DETR. `build_tree(args)` calls the
stock `build()` and then attaches the tree head; the vanilla pipeline
(`from models import build_model`) is untouched.

## File → module → equation map

| File | Module | Equations | What it holds |
|------|--------|-----------|---------------|
| `config.py` | — | fixed-design tables | `TreeConfig`: every hyperparameter, single source of truth |
| `geometry.py` | foundation | E1, E2, B2 | cosine-cone primitives: `stable_angle`, `cone_gap`, `angular_margin_ratio`, `theta_from_logit` |
| `topology.py` | — | — | `TreeTopology`: framework-free tree shape shared by A and E |
| `cone_head.py` | A, B | A1, B1 | `ConeEmbedHead` (h→z on S^{m-1}), `ObjectnessHead` (depth-0 gate on ‖h‖) |
| `tree_structure.py` | A | A2, A3, A4, A5, F1b | affinity, UPGMA induction, n-ary collapse + repair, class insertion, `ConfusabilityTree` |
| `cascade.py` | B | B2, B3, B4, B5 | `Cascade`: beam descent, halt depth, depth-normalised path score |
| `calibration.py` | D | D1, D2, D3, D4 | `ScaleConditionedRadius`, node-temperature fit, `tau0` recalibration, TTG / ECE_d |
| `adapter.py` | C | C1, C2, C3, C4 | `ParallelFFNAdapter` (zero-init), width rule, insertion freeze/mask partition |
| `losses.py` | E | E1, E3–E8 | `ConeField`, `ReservedGapLoss` (contain/nest/gap/sib), `node_gaps` diagnostic |
| `tree_detr.py` | integration | — | `TreeHead`, `TreeCriterion`, `attach_tree_head`, `build_tree` |

## Data flow

```
h = hs[-1][matched]            # R^256 matched decoder query feature (base model)
    │
    ├─ ObjectnessHead ──▶ σ(f(‖h‖/τ0))         depth-0 "is it a thing?" gate  (B1)
    └─ ConeEmbedHead  ──▶ z ∈ S^{m-1}          directional embedding          (A1)
                          │
                          ├─ training:  ReservedGapLoss(cones, topo, z, y)     (E3–E6)
                          └─ inference: Cascade.descend(z, box_area)           (B2–B5)
                                          └▶ halt_depth, leaf_class, S(path)
```

At training time only Module E and the objectness BCE are added to the base loss
dict; the cascade is inference-only. Insertion of a new class (Module C) is a
separate incremental step driven by `configure_insertion`.

## Two deliberately weakened claims (carried from the note's verdict)

The docstrings do **not** overclaim, matching the note's own hedges:

1. **Depth *corroborates*, it does not *replace* the learned boundary.** The
   depth-0 objectness magnitude gate (`ObjectnessHead`, B1) and the angular
   sibling gates are *both* required: hard background must fail the magnitude
   test **and** every angular test. See `cascade.py` header and `TreeHead.infer`.
2. **Cost is *measured*, not asymptotic.** The per-object gate count is
   `≈ B · E[depth]`, reported empirically by `Cascade.stats` (`mean_gates`), not
   claimed as `log|C|` — the induced tree is deliberately unbalanced.

## Usage

```python
from models.tree import build_tree, ConfusabilityTree

# 1. vanilla detector + attached tree head (flat 2-level tree by default)
model, criterion, postprocessors, head = build_tree(args)

# 2. or attach to an already-built model with an induced tree
from models.tree import attach_tree_head, induce_tree, build_affinity
A, D = build_affinity(confusion_counts)      # A2, A3
tree = induce_tree(D, num_classes=80)        # A4
head = attach_tree_head(model, tree)

# 3. inference on a matched query feature
res = head.infer(h, box_area=48*48)          # HaltResult: halt_depth / leaf_class / path_score
```

## Tests

Runnable with a torch+numpy environment (no scipy needed — induction ships a
self-contained UPGMA):

```
python models/tree/tests/run_tests.py
```

Covers: angle stability at the poles, gap sign / containment, affinity symmetry,
UPGMA + n-ary collapse respecting `D_max`/max-children, insertion vote/affinity
disagreement, cascade halt depths (0 / unknown / leaf) with beam=2, gap loss
keeping a positive annulus + warmup gating + finite grads at cone poles,
zero-init adapter identity + width clip bounds, scale bands + temperature NLL.

## Stage-0 falsification suite (`tools/stage0/`)

Runs on a *trained* detector before any tree training; each script prints its
metric **and** the note's pass/fail gate. See `tools/stage0/README` for order
(cheapest first: `f2` → `f1`/`f1b` → `f3`/`f4` → `n1` → `g1`).
