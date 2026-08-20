# Linux 2-GPU experiment setup

The target server can run the experiment on two RTX 3090 cards. The default
configuration uses GPUs `0,1`, one image per GPU, and the standard COCO-
pretrained Deformable-DETR R50 checkpoint. The checkpoint classifier is
filtered by shape when the six-class Pet model is initialized.

From the repository root on Linux:

```bash
bash tools/setup_linux.sh
conda activate tree-detr

bash tools/download_experiment_assets.sh \
  --data-root "$PWD/data/oxford-pet-tree-detr" \
  --checkpoint-dir "$PWD/pretrained"

GPU_LIST=0,1 bash tools/run_pet_coco_pretrained_ddp.sh
```

The asset script downloads the Oxford-IIIT Pet archives, creates the fixed
six-known-class COCO split (`coco_pet_pretrained_small6`), and saves the
official multi-scale checkpoint as `pretrained/r50_deformable_detr.pth`.
The runner uses `torchrun` and treats `BATCH_SIZE` as the batch size per GPU:

```bash
GPU_LIST=0,1 BATCH_SIZE=1 EPOCHS=50 \
  bash tools/run_pet_coco_pretrained_ddp.sh
```

Set `DATA_ROOT`, `CHECKPOINT`, or `OUTPUT_DIR` to use another location. The
runner refuses to mix with a completed output directory. A failed partial run
should use a new `OUTPUT_DIR` unless it is intentionally resumed with the
existing checkpoint.

`setup_linux.sh` compiles the custom CUDA operator for `sm_86` by default,
which matches the RTX 3090 cards. The visible RTX 5090 is not part of this
default run; to compile for it, set `TORCH_CUDA_ARCH_LIST=12.0` and use a
PyTorch/CUDA stack that supports Blackwell before changing `GPU_LIST`.
