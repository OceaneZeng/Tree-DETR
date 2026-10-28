# Discovery v5 运行效率与续训可靠性（2026-10-28）

本文记录运行实现版本 1。后续版本 2 的服务器对照测量、采用范围与续训边界见
[运行优化 v2](owod-runtime-optimization-v2.md)。

适用于正在运行的 `output/tree-detr-s-v5-full/s-owodb/seed_42/stage_0/increment/`。
本次代码只在本地修改；服务器测试使用通过 SSH 标准输入传入的内存模块，
未部署源文件、停止训练、写入实验目录或覆盖 checkpoint。

## 改动与实验边界

- evidence 从 Python 对象收集改为可变长张量收集。保留 rank 顺序、各 rank 内行顺序、
  标签 dtype；支持个别 rank 无 evidence，以及所有 rank 为空。
- 原型更新一次性读取标签和有限值/权重判据，减少逐个已知目标的 CUDA 同步。
  匿名分配和原型更新仍逐条进行，保留阈值、顺序与浮点更新表达式。
- EMA 使用分开的 foreach 乘法、加法和 buffer 拷贝；不合并成不同的浮点公式，
  不增加必须随 checkpoint 保存的 Python 计数缓存。
- DataLoader 使用 `spawn`，避免每轮创建 worker 时 fork 已初始化 CUDA 的父进程。
  保留每轮重新创建 worker 的行为，未启用 persistent workers；零 worker 路径也可运行。
- checkpoint 先写同目录临时文件，flush/fsync 完成后原子替换。写入失败保留旧断点，
  清理本次临时文件；可捕获失败会清理，强制杀进程可能留下孤立临时文件。
  新日志增加 `Checkpoint saved: epoch=... completed_epochs=...`。
- 恢复优化器前只保存 LR 元数据，不再 deepcopy 所有 GPU 参数。
  正式续训要求完整模型、optimizer、scheduler、epoch；discovery 还必须有 EMA。
  discovery 模型按严格键匹配恢复。
- `run_config.json`、后续 checkpoint 的 args 增加 `training_runtime` 实现版本记录。
  模型 state_dict、优化器分组和正式 manifest/阶段校验不变。
- 启动脚本保留用户在服务器采用的 GPU `1,2`，两进程、每卡 batch 2、4 workers。

未修改 v5 架构、损失、未知监督、原型阈值、EMA 日程、训练轮数、学习率、
数据/增强、有效 batch、activation checkpointing、评估频率和作者指标定义。
DDP 默认 buffer 广播和更新后的显式广播也暂时保留，未在缺乏双卡验证时关闭它们。

## 已完成验证

本地环境：PyTorch 2.7.1+cu128。以下 31 项相关测试通过：

```powershell
# C:\programlearning\Tree-DETR，tree-detr Python 环境
python -m pytest tools/owod/tests/test_training_runtime.py tools/owod/tests/test_runtime_distributed.py tools/owod/tests/test_learning_diagnostics.py tools/owod/tests/test_strict_setting.py tools/owod/tests/test_logging.py tools/owod/tests/test_discovery_workflow.py -q
```

服务器环境：PyTorch 2.4.1。隔离 CUDA 测试仅使用空闲 GPU 6，不占用训练 GPU 1、2。
服务器原子保存测试仅在自动清理的临时目录创建一次性 checkpoint。

| 检查 | 结果 |
| --- | --- |
| 原子写入故障注入 | 模拟部分写入失败后旧 checkpoint 可读取，随后可正常替换 |
| EMA、原型更新与旧实现对照 | CPU/CUDA 单元测试逐元素一致；包含重复匿名候选、满容量、NaN 特征、非正权重、空 evidence |
| 两进程 Gloo | 不等长、单 rank 为空、全空，收集顺序和 dtype 与旧对象收集一致 |
| 单进程 NCCL | CUDA 张量通信及空 evidence 通过；不是双 GPU NCCL 验证 |
| 真实数据 spawn | 父进程已初始化 CUDA，用真实训练 dataset 连续创建两轮 worker，读取成功 |
| 实际 checkpoint | epoch=17（已完成第 18 轮）模型和 EMA 严格加载；optimizer、scheduler 恢复成功 |
| 真实 batch 完整一步 | 同样随机种子下所有 loss 完全一致；student 最大状态绝对差 1.19e-7，teacher 1.49e-8 |

短测计时（同一空闲 3090，不能外推整轮收益）：

| 项目 | 旧实现 | 新实现 |
| --- | --- | --- |
| 24 条 evidence 的原型更新 | 7.57 ms | 4.80 ms |
| 真实模型 EMA 更新 | 7.01 ms | 2.51 ms |
| 单卡真实 batch 完整步骤（各 10 步） | 283.78 ms | 283.08 ms |

完整步骤差异很小，短测不足以证明整体明显提速。局部计时含不同程度的同步，不能直接
把两个局部节省相加当作整步节省。尚未验证双卡 NCCL 完整训练、连续多轮稳定性、
所有大尺寸增强的显存峰值；没有进行全量训练或宣称根治 CUDA unknown error。
旧报错是轮次切换时首次 CPU→GPU 拷贝报告的 CUDA unknown error，不是明确的 OOM。

## 用户上传与切换

必须一起上传以下 6 个运行文件，保持相对目录：

```text
main.py
util/misc.py
util/training_runtime.py
models/graph_local/discovery.py
models/graph_local/discovery_memory.py
tools/owod/run_discovery_full.sh
```

建议同时上传本说明和两个新增测试文件：
`tools/owod/tests/test_training_runtime.py`、`tools/owod/tests/test_runtime_distributed.py`。
保留服务器其他未提交修改。不要覆盖数据、output、权重或原有 benchmark_plan.json。
不需要重建 manifest、更换输出目录、增加新实验或修改旧 checkpoint。

先让旧进程完成当前轮训练与评估。旧进程没有新增的保存提示，可在看到下一轮开始时，
由用户 Ctrl+C 结束整个启动进程组；这一刻上一轮 checkpoint 已保存，最多重跑新一轮
尚未保存的少量进度。确认两个旧 rank 及 worker 均退出，再上传并执行：

```bash
# 服务器：原 tree-detr 环境、原 tmux
cd /home/top/disks/new-hdd/zhy/Tree-DETR
python tools/owod/manage_data.py verify --benchmark s-owodb --output-root data/owod
# 上一步成功后再执行以下命令
python -m pytest tools/owod/tests/test_training_runtime.py tools/owod/tests/test_runtime_distributed.py -q
bash tools/owod/run_discovery_full.sh --resume --dry-run
bash tools/owod/run_discovery_full.sh --resume
```

`--dry-run` 仅显示计划，不能代替实际续训时的 manifest、完整状态与阶段校验。
续训仍先评估恢复的 checkpoint，随后从 `epoch + 1` 开始，不是从头训练。
查看新的 `training_runtime` 记录、恢复 epoch、optimizer 分组及后续保存提示。
启动脚本保持原完整多阶段计划；本次只对 Task 1 做了真实 checkpoint 步骤验证。

checkpoint 仍按整轮保存，未增加 batch 中途恢复。原 checkpoint 没有完整随机数状态，
因此可延续已有训练状态，但不承诺与从未中断运行逐步数值一致；本次没有伪造缺失状态。
若遇到缺失状态或严格校验失败，应保留报错与旧文件，不能放宽校验或仅载入模型权重。
