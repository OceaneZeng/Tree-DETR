# Project layout

The repository uses one owner for each kind of artifact:

| Path | Contents |
| --- | --- |
| `baselines/` | Pinned external method source checkouts only |
| `configs/` | Tree-DETR detector configuration |
| `data/` | Original local datasets plus reproducible derived views; ignored by Git |
| `datasets/` | Tree-DETR dataset loader code |
| `docs/` | Current runbooks and historical experiment notes |
| `exps/` | All checkpoints, metrics, manifests, and run logs; ignored by Git |
| `models/` | Tree-DETR and controlled internal model implementations |
| `pretrained/` | Local initialization weights; weight files are ignored by Git |
| `tools/` | Dataset preparation, launchers, checks, and result aggregation |
| `util/` | Shared runtime utilities |

The canonical official M-OWODB experiment tree is:

```text
exps/owod/m-owodb/order0/official/
|-- baselines/
|   |-- prob/
|   `-- owobj/
`-- tree_detr/
    `-- <idea-or-ablation-name>/
```

Temporary downloads, copied repositories, package archives, Python caches, and
ad-hoc manifests do not belong in the repository. Use an operating-system temp
directory for disposable files and keep durable run metadata beside its result.

`data/coco/` and the official stage JSON are the M-OWODB source of truth.
`data/derived/m-owodb-voc/` is a rebuildable serialization cache for upstream
PROB/OWOBJ loaders that cannot read the COCO JSON layout; it does not define a
second split and its `JPEGImages/` entries should be symlinks to COCO images.
