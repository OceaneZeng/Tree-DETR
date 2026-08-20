# Windows RTX 5060 experiment setup

This repository has been configured and checked on Windows with an NVIDIA RTX
5060 Laptop GPU (8 GB), driver 573.24, CUDA driver API 12.8, and the separate
Conda environment `tree-detr`.

## Installed runtime

| Component | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| CUDA Toolkit | 12.8.2 |
| GPU architecture compiled for | sm_120 |

The compiled operator is `models/ops/MultiScaleDeformableAttention.cp310-win_amd64.pyd`.
Its numerical tests have passed. Rebuild it after changing PyTorch, CUDA, or
the extension source:

```powershell
Set-Location C:\programlearning\Tree-DETR
.\tools\windows\build_extension.ps1
```

## Dataset layout

Pass the directory that contains this COCO 2017 layout to the training script:

```text
<coco_path>/
  train2017/
  val2017/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
```

## Reproducible single-GPU controls

Run matching baseline and tree runs with the same seed, dataset split, epoch
count, batch size, and learning rate. The default learning rate, `1.25e-5`, is
the reference `2e-4` linearly scaled from effective batch 16 to this single-GPU
batch-1 configuration. It is a resource-constrained setting and must not be
compared directly with published multi-GPU numbers.

```powershell
# Baseline
.\tools\windows\train_single_gpu.ps1 -CocoPath D:\datasets\coco

# EE-0 tree ablation
.\tools\windows\train_single_gpu.ps1 -CocoPath D:\datasets\coco -WithTree
```

`-WithTree` enables the currently integrated **EE-0 flat two-level tree**:
one root and one leaf per COCO class, the cone/objectness losses, and the last
two decoder FFN adapter insertion points. It is a loss-level and integration
validation, not yet a full induced-confusability-tree or incremental OWOD
experiment. The adapters are initially zero-output identities; the current
EE-0 run has no inserted unknown class adapter.

To evaluate a checkpoint:

```powershell
.\tools\windows\train_single_gpu.ps1 -CocoPath D:\datasets\coco -Resume C:\path\checkpoint.pth -Eval
```

## Lightweight natural-image feasibility dataset

Download Oxford-IIIT Pet and prepare the fixed class-held-out split without
starting training:

```powershell
Set-Location C:\programlearning\Tree-DETR
.\tools\windows\download_oxford_pet.ps1
```

The Oxford mirror does not support HTTP Range, so the script downloads each
archive as a single atomic `curl.exe` transfer and verifies its size. The result
contains 29 known breeds and 8 completely held-out unknown breeds under
`C:\programlearning\datasets\oxford-pet-tree-detr\coco_pet_mini`.

After preparation, start the baseline and real Stage-0 gates with:

```powershell
.\tools\windows\run_pet_feasibility.ps1
```

This first run stops after the falsification gates. It does not train the tree
when a gate fails. Add `-RunTreeAfterPass` only after reviewing a passing
Stage-0 report.
