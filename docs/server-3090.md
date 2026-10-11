# 双 RTX 3090 服务器实验

目标服务器：`~/disks/new-hdd/zhy/Tree-DETR`，物理 GPU **0、1**。
入口默认只生成并核验计划，`--execute` 才执行。其余 GPU 不参与。
默认每卡 batch 2，全局 batch 4，workers 每卡 4；这是项目硬件设置，
不是声称复现 OW-DETR 论文的多 GPU 配置，不自动缩放论文 recipe 学习率。

## 环境

`nvidia-smi` 的 CUDA 13.2 表示驱动支持能力，不代表已安装的 nvcc 版本。
建议采用与本机验证一致的 PyTorch 2.7.1 / torchvision 0.22.1 / CUDA 12.8
runtime，并安装匹配的 CUDA 12.8 编译工具链来构建项目扩展。

```bash
conda create -n tree-detr python=3.10 -y
conda activate tree-detr
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install ninja pytest
# 如果服务器没有 CUDA 12.8 toolkit，可在此环境中安装：
conda install -c nvidia cuda-toolkit=12.8 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
python -m pip install --no-build-isolation ./models/ops
export PYTHONPATH="$PWD/models/ops:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

若使用系统已安装的 toolkit，将 CUDA_HOME 指向实际 CUDA 12.8 安装目录，
不必再次安装。Linux 需可用的 C++ 编译器。不要复制 Windows 的 `.pyd` 扩展。

已有 PyTorch 2.4.1 / CUDA runtime 12.1 环境时，可以先保留该环境并执行以下
自检；torchvision 对应版本为 0.19.1。编译扩展需 CUDA 12.1 toolkit，不能
直接照上面的 12.8 toolkit 命令安装。若当前没有匹配的 nvcc，可用
`conda install -c nvidia cuda-toolkit=12.1` 安装，再将 CUDA_HOME 指向实际
toolkit 目录。报告生成的 checkpoint 元数据读取已兼容缺少 safe_globals 的
PyTorch 2.4；该兼容测试不代表已在服务器完整训练通过。

双卡自检（真实 CUDA 扩展前后向、两轮 DDP、EMA、原型同步及受限定位更新）：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/owod/server_cuda_check.py
```

此自检使用随机小输入，仅证明计算与通信路径，不是 AP 验证。执行长实验时
入口会先自动运行该检查，失败则停止。本地只有一张卡，未声称已经通过服务器
双卡 NCCL 实测。

## 数据迁移

Git 不包含原始数据、DINO 权重或诊断输出；单独复制原始 COCO 数据与 DINO
权重至 `data/raw/coco`、`pretrained/dino_resnet50.pth`。不要直接复用记录 Windows
绝对路径的生成 manifest。在服务器的新输出目录重新准备并核验：

```bash
python tools/owod/manage_data.py prepare --benchmark s-owodb --coco-root data/raw/coco --output-root data/owod
python tools/owod/manage_data.py verify --benchmark s-owodb --coco-root data/raw/coco --output-root data/owod --decode-images
```

如已复制旧的生成目录，使用另一个新的 `--output-root`，并在训练时指定新
manifest；不要覆盖或手工改写已有实验的 lineage。M 需要完整的 ORE 原始图像
和 XML，目前本地 VOC 部分完成不代表 M 已就绪。M/S 划分均由原作者固定清单
生成，不因迁移改变。M 准备命令见 [数据指南](data-preparation.md)。

## 完整训练与消融

```bash
# 先检查数据及三组四任务计划，不执行训练
python tools/owod/run_server_experiment.py
# 在 tmux 等持久终端内执行，依次占用 GPU 0、1
python tools/owod/run_server_experiment.py --execute
# 中断后，配置与输入不变时续跑
python tools/owod/run_server_experiment.py --execute --resume
```

三组顺序为：hard（原评分、硬路由、冻结定位头）、soft（只改软路由）、
soft-localization（软路由基础上只开放 bbox 最后一层坐标投影）。
默认模型共享该头，因此定位更新额外开放 4×256+4=1,028 个参数。
更前面的 bbox MLP、backbone 和 decoder 基础权重仍按增量冻结策略处理；
已知类别分类行、adapter、发现路径和旧类蒸馏按原方法训练。
Task 1 三组都进行完整 dense bootstrap；定位开关从 Task 2 起生效。
定位开关绑定 checkpoint，禁止在已有链上临时更换。

所有组都采用 hierarchical 原评分、相同 DINO、seed、batch、作者图片清单、
50 epoch 增量与 20 epoch 独立 balanced FT（OW-DETR recipe），并保留正式
interference GNN 校准、匿名继承和旧类蒸馏。四任务共七个训练阶段，校准作业
不算新增任务。不用本机短程诊断权重初始化。这是项目方法的完整协议实验，
不是 ORE/OW-DETR 模型结果复现。

只运行某一组可用 `--variant hard|soft|soft-localization`；更换 seed 同时指定
新输出目录，如 `--seed 43 --output output/server-3090-seed43`。
M/S 分别运行，用 `--manifest` 指向对应 schema-3 manifest。

输出位于 `output/server-3090/<variant>/<protocol>/seed_42/`；三组完整结束后
自动生成 routing/localization 配对报告。重点看 Known/Current/Previous AP50、
U-Recall、项目 H-Score，以及 WI/A-OSE 是否恶化；Task 4 省略未知指标。
定位更新是新消融，尚无 AP 增益证据。软路由的本机单种子微小增益也不能替代
完整日程和多种子验证，详见 [已有实验结果](owod-discovery-transition-design.md)。
