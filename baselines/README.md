# External baselines

This directory contains pinned upstream implementations used for reproduction:

| Directory | Method | Upstream |
| --- | --- | --- |
| `prob/` | PROB | `orrzohar/PROB` |
| `owobj/` | OWOBJ | `AI4Math-ShanZhang/OWOBJ` |
| `ow-detr/` | OW-DETR | `akshitac8/OW-DETR` |

Initialize them after cloning Tree-DETR:

```bash
git submodule update --init --recursive
```

External source stays here. Shared datasets belong under `data/`; checkpoints,
metrics, and logs belong under the project-level `exps/`. Do not write experiment
outputs inside these checkouts.

Experiment data is managed by `tools/owod/manage_data.py` using the pinned
author M/S-OWODB releases. Historical shared-M adapters and launch scripts
have been removed. Do not change upstream submodule revisions when cleaning
generated artifacts in the Tree-DETR repository.
