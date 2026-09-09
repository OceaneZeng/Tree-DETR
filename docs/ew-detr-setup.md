# EW-DETR 实验配置与运行

本文说明如何在当前 Tree-DETR 仓库中配置并运行 EW-DETR。实验通过 Git 同步到服务器，使用仓库内的 `tools/owod/run_paper_baselines.py` 自动生成并按顺序执行 Stage 0 到 Stage 3。

当前配置是将 EW-DETR 适配到项目内部 COCO OWOD 四阶段协议的受控实验，不是 EW-DETR 原论文数据集、训练轮数或分数的精确复现。

## 1. 实验配置

| 项目 | 当前设置 |
| --- | --- |
| 数据集 | COCO2017 |
| 类别协议 | 4 个互不重叠的 20 类增量阶段 |
| 数据划分 | `data/coco-owod/m-owodb/order0/split_manifest_deus.invalid.json` |
| 检测器 | Deformable DETR |
| Backbone | ResNet-50，完全冻结 |
| Encoder/Decoder | 6 层 / 6 层 |
| Query 数量 | 300 |
| Stage 0 | 50 epochs，`lr_drop=40` |
| Stage 1-3 | 每阶段 20 epochs，沿用 D4 的 `lr_drop` |
| Batch | 每卡 2 张图，使用 2 张 GPU |
| Seed | 42 |
| 回放 | EW-DETR 不使用旧类回放，保持 exemplar-free |
| 初始化 | Stage 0 独立初始化，不加载 D2 或 D4 的训练权重 |
| 增量权重 | 每阶段使用上一阶段的 `checkpoint_consolidated.pth` |

EW-DETR 的基础检测器参数和 aggregate adapters 冻结；task adapters、检测头和校准相关模块参与训练。每个阶段结束后执行 consolidation，并将 consolidated checkpoint 传给下一阶段。

## 2. 环境要求

服务器上应已经存在：

- 仓库：`~/disks/new-hdd/zhy/Tree-DETR`
- Conda 环境：`/home/top/disks/new-hdd/conda_envs/tree-detr`
- COCO 图片：`data/coco/train2017` 和 `data/coco/val2017`
- OWOD manifest 及各阶段标注：`data/coco-owod/m-owodb/order0/`
- D4 Stage 1 已完成，并存在有效 checkpoint
- GPU `0,1` 可用于两卡训练

不要删除已有 D4 或 baseline 输出。启动器会检查 D4 完成状态、输入文件、代码哈希和 GPU 占用。

## 3. Git 同步

### 3.1 新服务器准备数据集

仓库中的 `tools/download_data.sh` 只负责 COCO2017 和预训练权重，不负责 OWOD 四阶段标注。建议使用封装脚本完成 COCO 下载或旧服务器迁移，并在最后统一校验：

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr

# 方案 A：新服务器联网，下载完整 COCO2017（约 18 GB train + 1 GB val）
bash tools/prepare_ew_dataset.sh --download-coco

# 方案 B：已有旧服务器数据。把 HOST、USER 和 PROJECT 替换为实际值
bash tools/prepare_ew_dataset.sh \
  --sync-from <old-user>@<old-server>:<old-project>

# 只检查已经准备好的数据，不做复制或下载
bash tools/prepare_ew_dataset.sh --validate-only
```

脚本要求以下文件：

```text
data/coco/train2017/*.jpg
data/coco/val2017/*.jpg
data/coco/annotations/instances_train2017.json
data/coco/annotations/instances_val2017.json
data/coco-owod/m-owodb/order0/split_manifest_deus.invalid.json
data/coco-owod/m-owodb/order0/stage_{0,1,2,3}/
```

每个 `stage_N` 必须有 `instances_increment_train2017.json`（或
`instances_increment_only_train2017.json`）、`instances_train2017.json`、
`instances_val2017.json` 和 `instances_val2017_full.json`。脚本会把迁移来的 manifest 路径改成本机路径，并保留 `.before_path_rewrite` 备份；不会修改标注内容或重新生成类别划分。当前 `split_manifest_deus.invalid.json` 是内部未验证 pilot，结果不能当作论文官方 OWOD 划分。

在本地仓库提交并推送代码：

```powershell
Set-Location C:\programlearning\Tree-DETR
git status --short --branch
git add tools/owod/run_paper_baselines.py tools/owod/smoke_paper_baselines.py tools/owod/verify_paper_checkpoint.py models/paper_baselines docs/ew-detr-setup.md
git commit -m "Document EW-DETR experiment setup"
git push origin main
```

如果这些文件已经包含在当前提交中，`git commit` 会提示没有新的改动，此时只需继续执行服务器端的 `git pull`。

服务器拉取代码：

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
git pull --ff-only origin main
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
```

## 4. 运行前检查

先运行单元测试和 baseline smoke check：

```bash
python -m unittest tools.owod.tests.test_paper_baselines tools.owod.tests.test_stage1_diagnostics
python tools/owod/smoke_paper_baselines.py
```

检查 D4、已有队列和 GPU：

```bash
python tools/owod/run_stage1_diagnostics.py \
  --output-dir "$PWD/exps/owod/m-owodb/order0/pilot_unverified/stage1_diagnostics_v2" \
  --summarize
cat exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/queue_status.json
nvidia-smi
```

如果已有 `paper_baselines_ew_v1` 队列正在运行，不要重复启动。确认旧队列已结束且 GPU 空闲后再执行下一步。

## 5. Dry-run

默认输出目录：

```text
exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/
```

运行 dry-run：

```bash
python tools/owod/run_paper_baselines.py \
  --methods ew-detr \
  --gpus 0,1 \
  --master-port 29579 \
  --dry-run
```

dry-run 会检查 D4 配置和 checkpoint、四个阶段的类别和标注、COCO 图片路径、EW-DETR 的零回放配置、两张 GPU 配置以及输出目录冲突。dry-run 不启动训练，也不创建训练 checkpoint。

## 6. 启动 EW-DETR 队列

当前默认只运行 EW-DETR：

```text
EW-DETR Stage 0-3
```

OW-DETR 不会被默认启动。只有显式加入 `--methods ow-detr ew-detr` 时，才会运行历史联合队列。

启动命令：

```bash
nohup setsid python -u tools/owod/run_paper_baselines.py \
  --methods ew-detr \
  --gpus 0,1 \
  --master-port 29579 \
  > paper_baselines_launcher.log 2>&1 < /dev/null &
echo "queue PID: $!"
```

启动器会等待 D4 完成和 GPU 空闲，然后按阶段顺序训练。某一个阶段完成，不代表整个队列已经结束。

## 7. 监控进度

```bash
tail -n 60 paper_baselines_launcher.log
cat exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/queue_status.json
python tools/owod/run_paper_baselines.py --summarize
nvidia-smi
```

EW-DETR 的结果目录：

```text
exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/ew-detr/stage_0/
exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/ew-detr/stage_1/
exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/ew-detr/stage_2/
exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/ew-detr/stage_3/
```

每个阶段应包含 `console.log`、`metrics.jsonl`、`run_config.json`、`checkpoint.pth`、`checkpoint_consolidated.pth`、`consolidated_metrics.json` 和 `training_complete.json`。

汇总时必须分别保留 EW-DETR 的 `task` 和 `consolidated` 两类结果，不能只挑选数值较高的一行。

## 8. 中断后恢复

确认中断原因已处理、没有残留训练进程后，使用原输出目录恢复：

```bash
nohup setsid python -u tools/owod/run_paper_baselines.py \
  --methods ew-detr \
  --gpus 0,1 \
  --master-port 29579 \
  --resume \
  >> paper_baselines_launcher.log 2>&1 < /dev/null &
echo "resume PID: $!"
```

恢复会跳过已验证完成的阶段，并从中断阶段自己的 `checkpoint.pth` 继续。不要替换已经开始运行的代码或配置；代码、输入和计划发生变化时，应指定新的 `--output-dir`，避免混合不同实验配方。

## 9. 结果记录

完成后保存以下命令输出：

```bash
python tools/owod/run_paper_baselines.py --methods ew-detr --summarize | tee ew_detr_summary.txt
cat exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/queue_status.json
```

比较时记录每个阶段、每个 epoch 和两种权重状态的 Previous AP50、Current AP50、Known AP50、U-Recall、H。Stage 3 已没有未知类 ground truth，未知类指标和 H 不再具有与 Stage 1/2 相同的开放世界含义。

## 10. 推荐执行顺序

下面是一套从服务器检查到正式训练的完整命令。每条命令前的注释说明它的作用。

```bash
# 进入仓库并加载本项目 Conda 环境
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr

# 确认服务器已经拿到本地推送的代码；不要使用 ZIP 或手工复制文件
git pull --ff-only origin main

# 验证 EW-DETR 模块、checkpoint consolidation 和 baseline 启动器
python -m unittest tools.owod.tests.test_paper_baselines tools.owod.tests.test_stage1_diagnostics
python tools/owod/smoke_paper_baselines.py

# 查看已有 EW 队列；status=running 时不要再次启动
cat exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/queue_status.json

# 确认两张 GPU 没有被其他训练任务占用
nvidia-smi

# 只检查配置、数据路径和 D4 依赖，不会训练
python tools/owod/run_paper_baselines.py --methods ew-detr --gpus 0,1 --master-port 29579 --dry-run

# dry-run 通过后，后台启动完整 baseline 队列
nohup setsid python -u tools/owod/run_paper_baselines.py \
  --methods ew-detr \
  --gpus 0,1 \
  --master-port 29579 \
  > paper_baselines_launcher.log 2>&1 < /dev/null &
echo "queue PID: $!"
```

默认启动器只处理 EW-DETR。旧的 OW-DETR 结果不会被删除，也不会因为默认启动命令而重新训练。

## 11. 如何判断运行完成

```bash
# 查看队列最终状态；应为 status=complete
cat exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/queue_status.json

# 生成所有阶段的 task/consolidated 汇总
python tools/owod/run_paper_baselines.py --methods ew-detr --summarize | tee ew_detr_summary.txt

# 检查 EW 每个阶段最后一个 checkpoint 和 consolidation 文件
for stage in 0 1 2 3; do
  test -f "exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/ew-detr/stage_${stage}/checkpoint.pth" && echo "stage_${stage}: checkpoint OK"
  test -f "exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_ew_v1/ew-detr/stage_${stage}/checkpoint_consolidated.pth" && echo "stage_${stage}: consolidated OK"
done
```

一个阶段只有同时满足 `training_complete.json`、最终 `metrics.jsonl`、`checkpoint.pth` 和 EW 的 consolidated 文件存在时，才算完成。不能只根据日志中出现 `Epoch: [19]` 判断完成。

## 12. 指标总结方法

`--summarize` 中的数值已经转换为百分比。每个 EW 阶段通常有两行：

- `task`：当前阶段 task adapter 尚未 consolidation 的结果，反映当前任务适配状态。
- `consolidated`：完成 consolidation 后的结果，也是传给下一阶段的权重状态。连续学习主表优先使用这一行，同时保留 `task` 行作为过程记录。

对 Stage 1-3，按下面的含义总结：

| 指标 | 总结含义 |
| --- | --- |
| Previous AP50 | 旧类保持能力；下降说明遗忘更严重 |
| Current AP50 | 当前新类学习能力 |
| Known AP50 | Previous 与 Current 合并后的已知类总体能力 |
| U-Recall | 未知目标被未知预测召回的比例 |
| H | Known AP50 与 U-Recall 的调和平均；两者有一个很低时 H 也会低 |

每个阶段至少记录最终 epoch 的 consolidated 行，并比较以下变化：

1. Current AP50 是否提高，确认新类学到了多少。
2. Previous AP50 是否大幅下降，判断旧类遗忘。
3. Known AP50 是否同时保持，避免只提升新类而损害整体检测。
4. U-Recall 和 H 是否保持，判断开放世界未知发现能力。
5. task 与 consolidated 的差异，确认 consolidation 是否造成明显性能变化。

不要把不同阶段、不同权重状态或不同 epoch 的最优数值拼成一行。建议使用以下记录格式：

```text
method=ew-detr, weights=consolidated, stage=1, epoch=20,
previous=..., current=..., known=..., u_recall=..., h=...
```

Stage 0 没有 Previous 类和未知类评估，重点记录 Current/Known AP50。Stage 3 没有未知类 ground truth，U-Recall 和 H 不应作为与 Stage 1/2 等价的开放世界结论。

## 13. 中断和失败处理

```bash
# 先确认当前没有残留的 torch.distributed 训练进程
nvidia-smi

# 查看最近的失败原因
tail -n 100 paper_baselines_launcher.log

# 环境问题修复后，沿用同一输出目录恢复；已完成阶段会跳过
nohup setsid python -u tools/owod/run_paper_baselines.py \
  --methods ew-detr \
  --gpus 0,1 \
  --master-port 29579 \
  --resume \
  >> paper_baselines_launcher.log 2>&1 < /dev/null &
```

如果代码、数据、配置或输出计划发生变化，不要在旧目录强行恢复。使用新的 `--output-dir` 建立独立实验，并重新执行 dry-run。
