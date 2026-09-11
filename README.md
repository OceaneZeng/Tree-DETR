# Tree-DETR：基于 GNN 回放的开放世界持续目标检测

更新日期：**2026-09-09**。本文记录项目配置、已经取得的实验结果和后续工作。

当前研究主线是 **D2：Deformable DETR 常规微调 + GNN 指导的旧类回放 + Stage 0 教师补标**。允许更新 ResNet backbone 的 `layer2/3/4`，浅层冻结；Encoder、Decoder、投影层和检测头正常训练。不再把 LoRA 或完全冻结 backbone 作为主方案。

结果来源是截至上述日期用户提供的服务器汇总，以及仓库已有的历史日志分析。本文不是实时服务器监控：`incomplete` 表示最后一次收到的记录未完成，不等于此刻仍在训练。新脚本已经准备好，不代表对应训练已经执行。

## 1. 如何配置项目

### 1.1 现有服务器：优先复用已经能够训练的环境

服务器项目和 Conda 环境路径：

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr

nvidia-smi
python -c "import sys, torch, torchvision; print(sys.version); print('torch:', torch.__version__, 'torchvision:', torchvision.__version__); print('CUDA runtime:', torch.version.cuda, 'available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
```

现有实验采用两张 GPU，每卡 batch size 为 2。README 中的实验启动命令默认使用 GPU `0,1`。正在运行实验时，不更新环境或重编译算子。

### 1.2 新 Linux 环境

需要 Conda、NVIDIA 驱动、兼容的 C/C++ 编译器，以及编译 CUDA 扩展所需的 `nvcc`。仓库的 [environment.yml](environment.yml) 提供面向 Linux / RTX 3090 的参考环境：

| 组件 | 配置文件中的版本 |
| --- | --- |
| Python | 3.10 |
| PyTorch / torchvision | 2.4.1 / 0.19.1 |
| CUDA runtime / toolkit | 12.1 / 12.1 |
| NumPy | < 2 |
| 其他依赖 | scipy、cython、pycocotools、tqdm、gdown |

这是环境文件的配置，不是对当前服务器实际安装版本的重新验收；复现实验时应保留实际环境版本。`nvidia-smi` 显示的 CUDA 版本也不等于 PyTorch 使用的 CUDA runtime 版本。

在尚无同名环境的新机器上，从仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate tree-detr
python -m pip install -r requirements.txt

export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=2
nvcc --version

cd models/ops
python setup.py build_ext --inplace
python test.py
cd ../..
```

`8.6` 对应 RTX 3090；其他 GPU 按实际架构和兼容的 PyTorch/CUDA 版本配置。Blackwell 显卡不能直接照用这套 CUDA 12.1 环境。扩展必须在当前 Python、PyTorch 和 CUDA 环境中编译，不能拷贝另一台机器的二进制文件代替编译。

从根目录直接调用训练入口时设置：

```bash
export PYTHONPATH="$PWD/models/ops:$PWD:${PYTHONPATH:-}"
TORCH_LIB="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
python -c "import MultiScaleDeformableAttention as m; print(m.__file__)"
```

D1–D4 和 D2 后续实验启动器会为训练子进程设置相关环境变量，以及当前双卡配置所用的 `NCCL_IB_DISABLE=1`、`NCCL_P2P_DISABLE=1`。

Windows 本地开发入口为 [build_extension.ps1](tools/windows/build_extension.ps1) 和 [train_single_gpu.ps1](tools/windows/train_single_gpu.ps1)。前者当前固定编译 `sm_120`，依赖 Visual Studio 2022 Build Tools 和对应 CUDA 环境，不是通用的 RTX 3090 安装脚本；本文结果来自 Linux 服务器实验。

### 1.3 数据和协议

COCO 2017 图像与原始标注的目录结构：

```text
data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

OWOD 实验还需要 `stage_0` 到 `stage_3` 的分阶段标注和 manifest。阶段编号从 0 开始，依次有 20/40/60/80 个已知类别；`stage_1` 对应论文常见的 Task 2。当前 pilot 使用：

```text
data/coco-owod/m-owodb/order0/split_manifest_deus.invalid.json
```

**当前记录属于内部、未验证的 COCO 20×4 划分，`paper_comparable=false`。目录中出现 `m-owodb` 不代表已经通过官方 M-OWODB 协议验收。** 训练、验证、类别顺序及源文件哈希必须与原实验一致。

如需正式协议，应先取得具有明确来源的分阶段标注，再使用 [prepare_protocol.py](tools/owod/prepare_protocol.py) 注册；不能把当前 manifest 改名或改标记当作完成验收。标注文件要求和注册说明见 [OWOD 工具文档](tools/owod/README.md)。该文档中的三模块/LoRA 训练方案属于历史阶段；当前方法选择以本 README 为准。

### 1.4 继续 D2 实验需要哪些已有产物

后续启动器依赖已经完成的 D2，不能在仅克隆代码、没有实验数据和权重的新目录直接启动。

```text
exps/owod/m-owodb/order0/
├── vanilla_d_detr/stage_0_50ep/checkpoint.pth
└── pilot_unverified/
    ├── full_gnn_calibration_stage0_v2/gnn_stage0.pt
    ├── full_three_module_stage1_v1/
    │   ├── run_config.json
    │   ├── graph.json
    │   ├── annotations/...
    │   └── graph/...
    └── stage1_diagnostics_v2/
        └── d2_full_finetune_no_extra_losses/
            ├── diagnostic_plan.json
            └── graph/
                ├── run_config.json
                ├── metrics.jsonl
                ├── training_complete.json
                └── checkpoint.pth
```

同时保留 D2 计划中引用的原始训练标注、验证标注和 manifest。以上是现有实验的目录约定，实际文件路径由原 `diagnostic_plan.json` / `run_config.json` 确定。不要用 D2 最终 checkpoint 替换 Stage 0 初始化和教师。

从零建立一条新实验链时，需要先完成数据协议、Stage 0 检测器训练和 GNN 校准；新链的分数不能冒充下面这条历史链的复现结果。

## 2. 已经跑了哪些实验

### 2.1 指标口径

- Previous / Current / Known 分别表示旧类、新类和全部已知类的 AP50，下面均以百分数展示。
- U-Recall 是当前实现下的未知类召回率；`H = 2 × Known AP50 × U-Recall / (Known AP50 + U-Recall)`，两项使用相同单位。
- 轮数从 1 开始，原始 `metrics.jsonl` 中的 `epoch` 从 0 开始。
- 汇总中的 `complete` 是整个实验的完成状态，不表示每一行都是训练终点。
- 当前未知评分、候选去重、匹配和简化 UDP/A-OSE/WI 尚需统一审计，不能直接视为论文官方 evaluator 的指标。

### 2.2 Stage 1：D0–D4 设置与最终已知结果

这些组复用原 Stage 0 初始化、GNN 邻域、训练标注、教师补标和回放曝光；训练计划为 20 轮，主学习率 1e-4，第 15 轮后降学习率，seed 42，两卡每卡 batch 2。回放采用旧类基础配额 10 张、GNN 选中的 5 类额外 40 张，已分析记录为 398 张回放图片、约 10% 的采样曝光。

| 实验 | 参数更新方式 | 附加损失 | 状态 / 报告轮数 | 旧类 AP50 | 新类 AP50 | Known AP50 | H |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| D0 | 末两层 Decoder FFN rank-8 LoRA + 新类分类器行 | margin 0.5、projection 0.1 | 完成 / 20 | 50.401 | 14.317 | 32.359 | 31.011 |
| D1 | 同 D0 | 两项关闭 | 完成 / 20 | 50.438 | 14.152 | 32.295 | 30.920 |
| **D2，当前主线** | 常规微调；backbone layer2/3/4 可训练 | 两项关闭，无蒸馏 | **完成 / 20** | **53.441** | **56.969** | **55.205** | **41.076** |
| D3 | 全 6 层 Decoder FFN rank-8 LoRA + 完整分类/定位头；其余冻结 | 两项关闭 | 未完成记录 / 5 | 50.988 | 13.680 | 32.334 | 31.604 |
| D4 | 常规微调，但整个 backbone 冻结 | 两项关闭，无蒸馏 | 完成 / 20 | 52.890 | 52.776 | 52.833 | 41.736 |

D2 相比 D4 的旧类、新类和 Known AP50 分别高 **0.551、4.193、2.372 个百分点**；H 低 0.660。当前选择 D2 继续优化，D4 保留为冻结 backbone 消融。D3 只有第 5 轮结果，不能当作完整 20 轮实验。D0/D1 的结果支持受限更新方案存在瓶颈，但这些比较尚未单独证明 GNN 的收益。

<details>
<summary>D0–D4 已收到的完整评估曲线</summary>

| 实验 | 轮数 | 旧类 AP50 | 新类 AP50 | Known AP50 | H |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 | 5 | 50.226 | 12.171 | 31.199 | 30.698 |
| D0 | 10 | 50.478 | 13.645 | 32.062 | 30.750 |
| D0 | 15 | 50.270 | 14.283 | 32.276 | 31.025 |
| D0 | 20 | 50.401 | 14.317 | 32.359 | 31.011 |
| D1 | 5 | 50.048 | 12.590 | 31.319 | 30.463 |
| D1 | 10 | 50.280 | 13.415 | 31.848 | 30.594 |
| D1 | 15 | 50.491 | 14.144 | 32.318 | 31.141 |
| D1 | 20 | 50.438 | 14.152 | 32.295 | 30.920 |
| D2 | 5 | 48.096 | 49.038 | 48.567 | 39.499 |
| D2 | 10 | 48.596 | 51.474 | 50.035 | 38.427 |
| D2 | 15 | 47.752 | 51.875 | 49.813 | 41.614 |
| D2 | 20 | 53.441 | 56.969 | 55.205 | 41.076 |
| D3 | 5 | 50.988 | 13.680 | 32.334 | 31.604 |
| D4 | 5 | 50.884 | 45.916 | 48.400 | 40.547 |
| D4 | 10 | 51.123 | 49.001 | 50.062 | 40.393 |
| D4 | 15 | 51.008 | 49.847 | 50.428 | 40.851 |
| D4 | 20 | 52.890 | 52.776 | 52.833 | 41.736 |

</details>

诊断结果在服务器重新汇总：

```bash
python tools/owod/run_stage1_diagnostics.py \
  --output-dir "$PWD/exps/owod/m-owodb/order0/pilot_unverified/stage1_diagnostics_v2" \
  --summarize
```

### 2.3 OW-DETR 适配版：Stage 0 完成，Stage 1 收到第 15 轮

以下是用户提供的 `run_paper_baselines.py --summarize` 输出，权重类型均为 `task`。

| Stage | 轮数 | 旧类 AP50 | 新类 AP50 | Known AP50 | U-Recall | H | 阶段状态 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 5 | NA | 20.650 | 20.650 | 0.000 | 0.000 | 完成 |
| 0 | 10 | NA | 24.958 | 24.958 | 0.187 | 0.371 | 完成 |
| 0 | 15 | NA | 29.644 | 29.644 | 0.370 | 0.731 | 完成 |
| 0 | 20 | NA | 31.002 | 31.002 | 0.194 | 0.385 | 完成 |
| 0 | 25 | NA | 32.610 | 32.610 | 1.003 | 1.946 | 完成 |
| 0 | 30 | NA | 35.042 | 35.042 | 0.588 | 1.157 | 完成 |
| 0 | 35 | NA | 37.910 | 37.910 | 0.688 | 1.352 | 完成 |
| 0 | 40 | NA | 38.784 | 38.784 | 0.771 | 1.513 | 完成 |
| 0 | 45 | NA | 42.150 | 42.150 | 1.090 | 2.124 | 完成 |
| 0 | 50 | NA | 42.488 | 42.488 | 1.052 | 2.052 | 完成 |
| 1 | 5 | 21.380 | 43.596 | 32.488 | 0.000 | 0.000 | 未完成记录 |
| 1 | 10 | 16.394 | 45.372 | 30.883 | 0.363 | 0.717 | 未完成记录 |
| 1 | 15 | 15.019 | 47.114 | 31.067 | 0.767 | 1.497 | 未完成记录 |

**尚未收到 OW-DETR Stage 1 第 20 轮、Stage 2/3 或 EW-DETR 的训练结果。** EW-DETR 已实现并进行过本地功能测试，不代表完成了 COCO 精度实验。

当前 baseline 队列仍采用此前的 D4 配方：包括 Stage 0 在内完全冻结 backbone，各方法独立训练 Stage 0。D2/D4 则继承历史上允许部分 backbone 更新的 Stage 0。OW-DETR 的预训练、训练安排、回放和教师补标也与 D2 存在差异。因此这批结果属于内部适配实验，既不是作者论文分数的精确复现，也不是严格隔离方法差异的 D2 对照。

```bash
python tools/owod/run_paper_baselines.py --summarize
cat exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_v1/queue_status.json
```

队列会按 OW-DETR Stage 0→3、EW-DETR Stage 0→3 自动推进。某一阶段完成不等于整个队列停止。EW 的 `task` 和 `consolidated` 结果要分别报告。Stage 3 已没有未知 GT，未知指标和 H 不再具有相同的开放世界解释。

### 2.4 历史实验和校准记录

| 实验 | 已有记录 | 解释 |
| --- | --- | --- |
| 历史 Stage 0 检测器 | 已作为 D0–D4 的起点；日志摘录 AP50 约 50.2% | 未在本次统一验证文件上重评，不作为精确遗忘基线 |
| `stage_1_cosine_k5_v2`，20 轮 | 旧类 50.8270、新类 55.4180、Known 53.1225、U-Recall 32.2649、H 40.1462（%） | H 由相同公式重算；学习率、回放来源/预算/曝光不同，仅作历史参考 |
| 更早的 `stage_1_cosine` | known-only AP50 约 53.6%；full AP50 约 26.8% | 类别平均口径不同，存在续跑，不能混入上面的同口径表 |
| 历史 GNN 校准 | 20 类、380 条有向边、138 条正边；留出 4 个源类的平均 Pearson 0.3343、Top-5 overlap 0.50 | 表明有一定排序信号，尚不能证明检测收益或多阶段泛化 |

来源与限制详见 [2026-09-06 实验诊断](docs/experiment-diagnostics-2026-09-06.md)。目录中其他 smoke、旧本地 baseline 或重复评估目录，没有足够的统一指标记录时，不列为新增的正式实验结果。上游 Deformable DETR 论文分数也不计入本项目已跑结果。

## 3. 还需要跑哪些实验

### 3.1 当前主线：D2 优化与必要对照

| 优先级 | 实验 | 要回答的问题 | 当前状态 |
| --- | --- | --- | --- |
| 本轮 | **D2-long：30 轮** | D2 在第 15 轮降学习率后，继续训练是否还能改善新旧类 AP？ | 启动器已准备；未收到训练结果 |
| 本轮对照 | **D2 配方的 Random-K：20 轮** | 在相同微调方式、教师和回放预算下，GNN 选择是否优于随机选择？ | 启动器已准备；未收到训练结果 |
| 后续必要 | 关键方案至少 3 个 seed | 提升是否稳定，均值和标准差如何？ | 未完成；先根据本轮结果确定组合 |
| 后续必要 | 旧类逐类 AP / 遗忘分析 | GNN 选择的高风险类别是否得到保护？ | 需汇总逐类结果，并统一评估 Stage 0 参考权重 |
| 后续必要 | 未知评分和 evaluator 审计 | U-Recall/H 的变化来自真实检测能力还是阈值、候选和误报差异？ | 尚未完成统一评估 |
| 后续必要 | D2 主方法完整 Stage 0→3 | 方法能否在持续多阶段学习中保持收益？ | 目前仅有历史 Stage 0 和本次 Stage 1 结果 |
| 正式比较前 | 协议验收、baseline 训练政策与初始化对齐 | 是否可以形成可比较的正式论文结果？ | 尚未完成 |

D2 第 15→20 轮旧类 AP 从 47.752 提高到 53.441，新类从 51.875 提高到 56.969，这是安排 D2-long 的依据。D2-long 从原 Stage 0 独立初始化、沿用原 GNN 训练标注，保持学习率与 `lr_drop=15`；它是增加训练预算的实验，不能当作等预算优势。

Random-K 只用于对照，不替代 D2 的 GNN 主方案。它必须匹配 D2 实际回放图片数、总数据长度、每轮训练步数和回放曝光；启动器会记录去重后的补齐/裁减情况。

**Cosine 不是当前主方法，也不在默认待跑队列。** 脚本保留 `--experiments cosine` 作为之后可单独决定的消融入口；当前不自动启动。D3/LoRA 扩展也不在当前优先队列。

GNN 的 node encoder、message passing、ranking loss 组件消融，以及教师补标消融，在确认主方案和预算后再安排；不要一次同时修改回放比例、学习率和多个损失。

### 3.2 已有 baseline 队列的剩余记录

OW-DETR Stage 1 最终结果、Stage 2/3，以及 EW-DETR Stage 0–3 尚需收集或运行。先核对现有 `queue_status.json` 和进程，避免重复启动。现有队列仍是历史 D4 设置；不会因为主线改为 D2 就自动变成 D2 对齐版本。是否重跑，应在核对未知评估和正式比较配置后决定，不能混合两套设置填写同一结果表。

## 4. 如何启动当前 D2 后续实验

使用 [run_d2_followups.py](tools/owod/run_d2_followups.py)。默认输出根目录为：

```text
exps/owod/m-owodb/order0/pilot_unverified/d2_followups_v1/
```

先确认原 baseline 队列已结束调度，GPU 0/1 空闲。下面两种启动方式任选一种，不同时执行。

### 4.1 只优化 D2：运行 D2-long

```bash
if python tools/owod/run_d2_followups.py --experiments long --dry-run
then
  nohup setsid python -u tools/owod/run_d2_followups.py --experiments long \
    > d2_long_launcher.log 2>&1 < /dev/null &
  echo "launcher PID: $!"
fi
```

### 4.2 同时安排 Random-K 对照与 D2-long

默认队列顺序为 Random-K → D2-long；显式指定可避免误启动旧版包中的 Cosine 默认队列：

```bash
if python tools/owod/run_d2_followups.py --experiments random long --dry-run
then
  nohup setsid python -u tools/owod/run_d2_followups.py --experiments random long \
    > d2_followups_launcher.log 2>&1 < /dev/null &
  echo "launcher PID: $!"
fi
```

dry-run 会检查 D2 配置、输入哈希、标注重建和图像路径，不启动训练。训练前检查 GPU 占用；忙碌则退出。一组训练失败，后续组不会继续启动。

### 4.3 结果、续跑和文件记录

```bash
python tools/owod/run_d2_followups.py --summarize
```

各组目录内保存 `plan.json` 和 `graph/train.log`、`graph/metrics.jsonl`、`graph/run_config.json`、`graph/checkpoint.pth`、`graph/training_complete.json`。汇总中的 delta 对比原 D2 的相同轮数；第 25/30 轮没有原 D2 同轮参考，显示 NA。

中断后确认原进程退出，且该组已有自己的 checkpoint，再在原命令上增加 `--resume`。例如只恢复 D2-long：

```bash
nohup setsid python -u tools/owod/run_d2_followups.py --experiments long --resume \
  >> d2_long_launcher.log 2>&1 < /dev/null &
```

续跑恢复该组的模型和优化器，教师保持原 Stage 0。修改代码、输入或计划会触发一致性检查；不要覆盖正在运行的代码或删除历史结果来绕过检查。原训练器未保存完整随机数状态，中断续跑不承诺逐步复现未中断轨迹。

详细预算规则、单组启动和兼容说明见 [D2 后续实验操作说明](docs/d2-followups-runbook.md)。

## 5. 代码入口和验证记录

| 文件 / 目录 | 用途 |
| --- | --- |
| [main.py](main.py)、[engine.py](engine.py) | 检测器训练和评估 |
| [models/graph_local](models/graph_local) | GNN、回放、教师补标和历史 LoRA 模块 |
| [calibrate_interference_gnn.py](tools/owod/calibrate_interference_gnn.py) | 以实测干扰校准 GNN |
| [run_graph_local_increment.py](tools/owod/run_graph_local_increment.py) | 构造图邻域与回放、启动增量学习 |
| [run_stage1_diagnostics.py](tools/owod/run_stage1_diagnostics.py) | 历史 D1–D4 启动与 D0–D4 汇总 |
| [run_d2_followups.py](tools/owod/run_d2_followups.py) | 当前 D2-long / Random-K 实验 |
| [run_paper_baselines.py](tools/owod/run_paper_baselines.py) | 历史 D4 配置的 OW-DETR / EW-DETR 队列 |
| [models/owod_metrics.py](models/owod_metrics.py) | 当前内部 OWOD 指标 |
| [baselines](baselines) | 固定版本的外部复现源码（不存放结果） |
| [project-layout.md](docs/project-layout.md) | 数据、源码和实验输出的目录边界 |
| [stage1-diagnostics-runbook.md](docs/stage1-diagnostics-runbook.md) | D0–D4 设置演变与操作记录 |
| [paper-baselines.md](docs/paper-baselines.md) | 外部方法适配来源、设置及限制 |

2026-09-09，本地 D2 后续启动器及原 Stage 1 启动器的 **26 项测试通过**，覆盖配置继承、回放预算、只读预检、队列顺序、GPU 占用、失败停止、结果汇总与 CPU Cosine 评分。它们是本地功能验证，不是服务器 COCO 训练结果或性能保证。

在相同兼容环境中可运行：

```bash
python -m unittest tools.owod.tests.test_d2_followups tools.owod.tests.test_stage1_diagnostics
```

每次新增结果，应同时更新本 README 的轮数、指标、完成状态和记录日期；仅有启动命令、目录或测试通过，不算完成训练。

## 6. 上游来源和许可证

本项目基于 [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) 开展持续检测研究，并非新的 Deformable DETR 官方发布仓库。上游论文为 [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)，作者 Xizhou Zhu、Weijie Su、Lewei Lu、Bin Li、Xiaogang Wang、Jifeng Dai。

保留 [Apache 2.0 LICENSE](LICENSE) 和源文件中的上游版权说明。本文实验分数与上游论文公开分数分开记录。
