# OWOD experiment policy (user decision, 2026-10-08)

Use exactly two author-defined protocols: M-OWODB from Joseph et al.,
Towards Open World Object Detection (CVPR 2021), and S-OWODB from Gupta et al.,
OW-DETR (CVPR 2022), Appendix A.2/Table 7. S was explicitly authorized as a
supplement to the ORE source; it is not defined in ORE.

Use configs/owod and docs/owod-benchmarks.md as the active specification.
Use tools/owod/manage_data.py for data plan/check/prepare/verify. Raw sources
are configured independently of generated data. Do not recreate deleted pilot
launchers or old setting documentation; tools/analysis is for historical logs only.
Keep the released image identities and class orders. Never create random,
heuristic, 20x4-for-S, or DEUS-derived replacement splits. Cross-task image
overlap in official lists is permitted; duplicate annotation records and future
class supervision are forbidden. Increment training and balanced finetuning are
separate phases. Do not replace the released memory by the historical custom
replay exposure/budget settings.

Require schema-3 manifests, verify input hashes, and continue checkpoints only
within the same manifest and the correct phase/task sequence. Do not use
allow-unverified or restart old pilot launchers. Historical outputs remain
historical and cannot initialize or substantiate official benchmark results.

Report author-style AP50 for Previous/Current/Known, WI at recall 0.8, A-OSE,
and U-Recall where unknown classes remain. H-Score is an additional project
metric, not an ORE metric. Keep original author evaluator conventions explicit;
do not substitute COCO's 101-point AP50. Task 4 omits unknown metrics.

Dataset protocol reproduction does not establish reproduction of ORE or
OW-DETR model results. Keep method architecture, optimizer, pretraining,
finetuning schedule, and unknown supervision separately attributed. Never
invent an unstated paper hyperparameter. Preserve historical logs/checkpoints
and unrelated user changes when retiring old experiment settings.

Git artifact policy (user decision, 2026-10-08): keep source code, runtime
configuration, pinned author ImageSets, necessary Markdown documentation and
licenses. Generated documents, plots, archives, weights and logs are ignored.
Do not force-add those artifacts or recreate removed paper exports. Preserve
raw datasets, weights, metric logs and external source revisions locally.
