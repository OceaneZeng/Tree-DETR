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

`tools/owod/prepare_shared_mowodb.py` creates the two untracked
`configs/*_SHARED.sh` launchers in `prob/` and `owobj/` after validating the
official manifest. They are generated launch files, not additional model
implementations; regenerate them instead of hand-editing upstream configs.
