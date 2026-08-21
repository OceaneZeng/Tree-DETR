# Risk-Conditioned Graph Consolidation

## 1. Research question

Uniform replay and uniform distillation protect every old class equally. This
spends a fixed continual-learning budget on classes that may not be affected by
the new task, while under-protecting classes whose detector gradients conflict
with the new update.

The new method, **Risk-Conditioned Graph Consolidation (RCGC)**, tests one
coherent hypothesis:

> A class-conditioned gradient-conflict graph can estimate old-class forgetting
> risk, and the same risk signal can allocate a fixed replay budget and weight
> teacher consolidation more efficiently than uniform or random allocation.

This is an incremental object detection method first. It is not called OWOD
until an explicit unknown-object branch and OWOD metrics are added.

## 2. Three tightly coupled modules

### M1. Conflict-risk estimator

For each old class `c`, compute a normalized class-conditioned gradient sketch
`g_c` on the frozen base detector. For the new increment, compute `g_new` and
form positive conflict edges:

`A[new,c] = max(0, cosine(g_new, g_c)) / sqrt(F_new F_c)`

where `F` is a class-conditioned gradient-energy normalization. The old-class
risk is the row sum `r_c = sum_new A[new,c]`, calibrated on the base training
split only. Frequency and object-size normalization are mandatory controls so
that the graph is not just a class-frequency or scale graph.

M1 has one job: predict which old classes will lose performance after the new
update. It does not itself change the detector.

### M2. Risk-budgeted replay allocator

Given a fixed memory budget `B`, allocate class quotas as:

`B_c = round_with_fixed_total(B * (epsilon + r_c) / sum_j(epsilon + r_j))`.

Within each class, select exemplars by spatial and appearance coverage, not by
the future validation labels. Uniform replay, random quotas, and global replay
are matched-budget controls. M2 has one job: spend the same replay budget where
M1 predicts the greatest risk.

### M3. Risk-weighted teacher consolidation

The frozen base teacher supplies old-object pseudo labels on increment images,
preventing unlabeled old objects from being treated as background. Distillation
is weighted by the same risk vector:

`L_consolidate = sum_c (epsilon + lambda*r_c) L_teacher,c`.

High-risk classes receive stronger output/box consolidation; low-risk classes
are not forced to consume the same update capacity. M3 has one job: preserve
old predictions during the new-class update. The pseudo-label completion is a
sub-part of M3, not a separate claimed contribution.

The core method therefore has one signal and two actions. LoRA, local margin,
and off-neighborhood projection are removed from the primary method: LoRA is a
parameter-efficient control, and projection overlaps prior gradient-subspace
methods such as InfLoRA.

## 3. Falsifiable predictions

1. M1 top-risk classes have larger post-update AP loss than low-risk classes.
2. M2 beats uniform replay and matched random quotas at identical memory size.
3. M3 beats uniform distillation at identical teacher/data access.
4. RCGC beats both controls on old-class AP and forgetting without reducing new
   class AP by more than the pre-registered tolerance.

If M1 does not predict harm, the method is not justified even if replay happens
to improve AP. If RCGC does not beat matched random allocation, the graph is
not useful and the central claim is rejected.

## 4. Benchmark and baselines

### Primary: standard COCO incremental detection

Use the disjoint-image COCO 2017 protocols from CL-DETR:

- `40+20x2` and `40+10x4` as the main multi-step settings;
- `70+10` as a two-phase sanity check;
- three category/data orders and three seeds;
- fixed 10% exemplar memory.

Required baselines:

1. vanilla Deformable DETR fine-tuning;
2. uniform replay / balanced fine-tuning;
3. uniform teacher distillation plus replay;
4. matched random-risk replay plus risk-weighted distillation;
5. global replay and consolidation;
6. RCGC full method;
7. oracle post-hoc risk analysis (diagnostic only, never a deployable method).

Report AP, AP50, AP75, APs, APm, APl, old AP, new AP, all AP, and forgetting
percentage points (FPP). Report mean, standard deviation, and paired results
over the same category/data orders.

There is no arbitrary `AP50 >= 0.40` gate. The base detector must reproduce the
official Deformable DETR reference within a documented tolerance; otherwise the
run is an implementation failure, not evidence about RCGC.

### Secondary: OWOD extension

Only after Phase A succeeds, add a class-agnostic unknown/objectness branch and
evaluate on S-OWODB first, then M-OWODB for historical comparison. Compare
against vanilla D-DETR, ORE*, OW-DETR, PROB, and an Oracle. Report known mAP,
U-Recall, A-OSE/WI, and UDR/UDP where supported. Until this extension exists,
the method must be described as IOD rather than OWOD.

## 5. Experiment order

1. Implement the official COCO IOD split and verify class/image disjointness.
2. Reproduce the vanilla D-DETR and uniform replay baselines.
3. Run the frozen-model M1 risk prediction diagnostic.
4. Run matched M2/M3 controls under the same budget.
5. Run RCGC and all component ablations.
6. Repeat the strongest comparison over three seeds and report confidence
   intervals.
7. Only then implement the OWOD unknown branch.

Oxford-Pet remains a small engineering smoke test only; its current results are
not used to support the RCGC or OWOD claim.
