# Discovery v5 运行优化 v2：服务器短测与采用范围

日期：2026-10-28（Asia/Shanghai）。基于本地和服务器源码 `f52ea01`。
这是运行实现版本 2，**方法版本仍为 discovery v5**。
此前运行实现版本 1 的修复和验证见 [运行优化记录](owod-runtime-optimization.md)。

## 测试范围与边界

- 使用原服务器 `tree-detr` 环境，PyTorch 2.4.1，空闲物理 GPU 6（RTX 3090）。
  保持 `CUDA_DEVICE_ORDER=PCI_BUS_ID`；未使用正在训练的 GPU 1、2。
- 候选函数和最终本地源码通过 SSH 标准输入在独立 Python 进程内加载。
  未修改服务器源码、部署补丁、停止或重启训练，也未向正式输出目录保存模型。
  设置 `PYTHONDONTWRITEBYTECODE=1`，测试日志在本地保存。
- 只读编号权重 `stage_0/increment/checkpoint0014.pth`：保存 epoch=14，
  即已完成 15 轮。测试前后 SHA256 均为
  `ee081e1a1955a21286c7fe5605c5b2e7ee3ef586072538d7d02644f4cd5b7a38`。
- 严格载入学生和 EMA，并恢复 optimizer；在临时内存中分别模拟 warmup
  和 warmup 后的一步。该诊断不产生正式训练成绩，不把旧 checkpoint 改写成新阶段。
- GPU 性能测试为单卡，未验证双 GPU NCCL 完整训练。
  服务器源码提交号不能证明已运行进程加载了相同源码。

## 性能结果与决定

| 候选 | 原实现 | 候选实现 | 决定 |
| --- | --- | --- | --- |
| warmup 完整训练一步 | 391.892 ms | 309.138 ms | 采用：耗时减少 21.12% |
| 21 个类别分支、2 层的活动标志屏蔽 | 0.693 ms | 0.311 ms | 采用 |
| 41 个类别分支、2 层的活动标志屏蔽 | 1.288 ms | 0.538 ms | 采用 |
| 61 个类别分支、2 层的活动标志屏蔽 | 1.987 ms | 0.789 ms | 采用 |
| 图像与 mask 的纯 H2D 传输 | 2.168 ms | 2.079 ms | 仅局部小收益，不据此采用 |
| 真实数据加载与预取，4 workers | 88.588 ms/batch | 93.668 ms/batch | 未证实稳定收益，pinning 不修改 |

warmup 使用真实 v5 模型、batch 2、实际训练图像的确定性验证变换，padding 后
形状为 `2×3×800×1204`。执行真实 `train_one_epoch` 的学生前向、criterion、反向、
梯度裁剪、optimizer、原型更新和 EMA；预先将固定 batch 放在 GPU，计时不含数据加载。
模型配置、activation checkpointing 和损失不变。两个实现均预热 3 步，交替顺序测
7 组、每组每个实现 5 步，表中为各组每步均值的中位数。
每组开始前恢复相同模型、EMA、optimizer 和随机种子；恢复过程不计入耗时。
峰值已分配显存约从 2906 MiB 降至 2755 MiB，计数含短测驻留的状态副本，
不代表正式训练的显存峰值，也未覆盖全部随机增强尺寸。

活动标志测试构建真实 `RiskRoutedStage`，覆盖每三分支激活一个的情形。
交替测 7 组，每组 100 次。计时含单卡活动标志计算和屏蔽，不含跨卡 all-reduce，
不能将其百分比当成完整训练的加速比。另做了已有 AdamW 动量和 weight decay 下的
一步状态对照，梯度是否为 None 及参数更新逐元素一致。

纯传输测试形状为 `2×3×800×1344`，不含提前 pinning 的时间。
完整加载测试用 64 张真实训练图像、确定性验证变换、batch 2、4 个 spawn worker、
真实 prefetcher 和 GPU 数据读取；5 组交替顺序，前 8 个 batch 后计时剩余 24 个。
图片 ID、尺寸、图像求和与 mask 求和检查一致；该汇总检查不是逐像素等价证明。
结果波动较大，pinning 没有显示稳定收益，因此保留当前 `NestedTensor` 实现。

## 数值与行为检查

- warmup epoch=0 和边界 epoch=1 分别进行旧/旧/新三次完整一步对照。
- 学生最大绝对差 `1.1921e-7`，与旧实现自身重复运行差异相同；EMA 最大绝对差
  分别为 `1.4901e-8`、`2.9802e-8`，也与相应旧/旧差异相同。
- 所记录指标最大绝对差 `1.5259e-5`；optimizer 状态最大绝对差不超过 `4.65e-9`。
  CPU/CUDA RNG 状态一致。CUDA 一步结果并非逐位一致，不据此承诺全量 AP 相同。
- 最终本地代码在服务器内存加载后通过 8 项检查：warmup 前后教师调用、EMA 更新、
  原型观察、全空/全激活/混合分支，以及全局归约后的活动标志屏蔽。
- 另以两个 CPU/Gloo 进程验证：不同 rank 分别激活不同分支，MAX 归约后两者都保留
  梯度，第三分支保持 None，原有动量和 decay 不修改该分支，两 rank 更新后一致。
  这不是双 GPU NCCL 测试。
- 服务器 `engine.py`、`main.py`、`models/graph_local/routed_adapter.py` 测试前后哈希一致。

## 本地改动与续训

采用的运行源码只有：

- `engine.py`：warmup 内跳过 criterion 不使用的 EMA 教师前向；每步原型更新、
  EMA 更新及 `ema_update_count` 保持原位置和次数，warmup 后照常使用教师。
- `models/graph_local/routed_adapter.py`：在原有全局 MAX 归约完成后一次读取活动标志，
  保留全局未激活分支的 `grad=None` 及 optimizer 行为。
- `main.py`：`training_runtime.version=2`，记录上述实现选择。

不改方法版本、state_dict 键、optimizer 分组、manifest、运行计划、卡数、batch、
学习率、阈值、epoch、评估频率和作者指标。现有同一 v5 链的完整 checkpoint 可以
经原校验续训，无需改写 checkpoint 或 `benchmark_plan.json`。
旧 checkpoint 没有完整 RNG 状态，仍不承诺与从未中断的运行逐步复现。

当前 Task 1 已过 warmup，且尚无 `RiskRoutedStage`；两项对该阶段剩余轮次没有
直接加速收益。主要受益范围是后续阶段 warmup 和带类别 adapter 的训练步骤。
不能外推为整个实验快 21%，也不能作为现在中断训练的理由。

用户通过 Git 同步服务器后，沿用原启动器 `--resume`、原输出和日志；本次未代为
部署或重启。不要使用本次临时诊断权重初始化任何正式实验。
作者评测改为仅 rank 0 计算尚无实际双卡耗时/完整指标等价验证，本次不修改。

## 本地验证与日志

Windows PowerShell，仓库根目录、已有 `tree-detr` 环境：

```powershell
python -m pytest tools/owod/tests/test_runtime_forward_and_activity.py tools/owod/tests/test_training_runtime.py tools/owod/tests/test_strict_setting.py -q
```

18 项通过。没有启动全量训练或全测试集 AP 评估。

本地保留的测试脚本和原始日志位于忽略目录 `tmp/`：

- `benchmark_runtime_candidates.py`、`runtime-candidates-server.log`
- `benchmark_pinning_pipeline.py`、`runtime-pin-pipeline-server.log`
- `server_validate_runtime_changes.py`、`runtime-candidate-validation-server.log`

临时脚本含本次服务器路径和编号 checkpoint，不作为正式启动器或 Git 交付文件。
