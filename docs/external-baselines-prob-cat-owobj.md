# PROB, CAT and OWOBJ reproduction status

The repository now provides `tools/owod/prepare_external_baselines.py` to
create pinned, independently auditable checkouts and a command manifest. It
does not rename the local detector as an external method.

PROB (`orrzohar/PROB`) and OWOBJ (`AI4Math-ShanZhang/OWOBJ`) each ship their
own Deformable-DETR fork, VOC-style OWOD data view, training schedule, replay
files and evaluator. Their official M-OWODB recipes therefore cannot be
called a same-setting run on Tree-DETR's internal COCO manifest. A controlled
COCO transfer would require a separate port of the model, data loader and
evaluator, and its numbers would be marked `paper_comparable=false`.

CAT is the CVPR 2023 paper *CAT: LoCalization and IdentificAtion Cascade
Detection Transformer for Open-World Object Detection*. The paper's listed
URL, `https://github.com/xiaomabufei/CAT`, currently returns 404. No CAT
training command is emitted until a verified author checkout or archive is
available.

Prepare the available official repositories on the server:

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
python tools/owod/prepare_external_baselines.py \
  --methods prob owobj cat \
  --repo-root "$PWD/exps/external_baselines/repos" \
  --gpus 0,1
```

Inspect the generated `exps/external_baselines/external_baseline_manifest.json`.
It records each commit and the exact official config. Run an official recipe
only after installing that repository's documented environment and placing
its required VOC/TOWOD data under that repository:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('exps/external_baselines/external_baseline_manifest.json')
d = json.loads(p.read_text())
for method, command in d['commands'].items():
    print(f'{method}: {command or "BLOCKED: source/config unavailable"}')
PY
```

The scripts use the authors' epoch counts and replay rules. Do not pass the
Tree-DETR D2/D4 checkpoint to them: the repositories expect their own DINO
ResNet-50 initialization and checkpoint format. Report these as official
recipe runs separately from Tree-DETR's internal controlled baselines.
