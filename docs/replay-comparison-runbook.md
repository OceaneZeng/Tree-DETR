# GLCA-DETR 与 Uniform Replay 的同设置对比

本实验对应当前 D2 主线：Deformable DETR 常规微调，backbone layer2/3/4 可训练，GNN 指导回放，上一阶段冻结教师补标。这里的 GLCA-DETR 不包含历史 LoRA、local margin、off-neighborhood projection 或蒸馏损失。

## 1. 实验设计与结论边界

| 组别 | 回放图片选择 | Teacher supervision | 可回答的问题 |
| --- | --- | --- | --- |
| GLCA-DETR | 原 GLCA 的 GNN 回放集合 | 开启 | 主方法 |
| Uniform Replay | 从旧类候选图片中等概率、无放回抽样 | 关闭 | 普通图片级回放基线 |
| Uniform Replay + teacher supervision | 与上一组完全相同的回放图片及标注 | 开启 | 隔离教师补标收益；与 GLCA 隔离回放选择收益 |

Teacher supervision 复用 `main.py` / `engine.py` 已有的 `--teacher-completion`，由冻结上一阶段模型给遗漏的旧类目标补伪标签；不额外引入 logits/box distillation。纯 Uniform 只关闭这一开关。教师阈值、去重规则和每图上限均继承 GLCA 的实际配置。

**同一 Task 内三组从相同的上一阶段 checkpoint 独立初始化，使用新的优化器，均重新训练。** GLCA 不直接引用旧日志当作本次实验结果。这样得到的是逐 Task 的条件受控实验。Task 3/4 的 Uniform 也从该 Task 的 GLCA 参考初始化出发，并非从自己的上一 Task 输出继续训练，因此不能将图中的跨 Task 连线解释为三条独立持续学习轨迹。后者需要另外为每条方法链重建各阶段 GNN/教师/回放，并单独报告。

默认先做已有的 Stage 1 / Task 2；传入多个实际存在的 GLCA 来源目录可做 Task 2、3、4。Stage 0 / Task 1 没有旧类回放，三种干预在此没有区别，不重复训练或伪造三个独立结果。Task 1 基础模型评估可另列为共享起点。

启动器从真实 `run_config.json` 继承模型、增广、batch、GPU 数量、学习率及调度、训练轮数、seed、各检测损失系数、评估频率、类别集合与未知阈值。历史配置没有记录而当前入口新增的参数，会使用当前默认值并列在 `defaults_missing_from_source`；三组都显式固定这些参数。来源必须具有 GNN provenance，不能拿 Random-K/Cosine 目录冒充 GLCA。

## 2. 回放预算如何对齐

- Uniform 从 manifest 的上一阶段 `train` 标注中抽取**唯一旧类图片**，不是按类别均衡抽样，也不是 Random-K。
- 已在本阶段 increment 中出现的图片不再计入可抽样回放池。
- 图片预算严格等于来源 GLCA 标注中 `owod_replay=true` 的去重图片数；两组 Uniform 共用同一个抽样结果。
- 当前图片、其 GT、整个数据集长度、每 rank 样本数、每 epoch 优化器步数、`replay_sampling_fraction` 与 GLCA 一致。训练器继续使用 `ReplayBalancedSampler`。配额记录是 `drop_last` 前的曝光；三组使用相同索引布局、seed 和 batch，截断机制相同。
- 旧协议若在当前图片上已有额外旧类 GT，三组固定保留同样的 GT。数量记录在 `fixed_old_annotations_on_current_images`，避免抽样时悄悄改变当前图的监督。参考数据缺少当前 GT、回放标注与旧池不一致、未来类别泄露或图像 ID 冲突会使预检失败。
- 等图片预算不等于等类别/框数量；类别覆盖差异正是被比较的选择策略的一部分。`plan.json` 保存图片 ID、实际每类回放框数和输入哈希。
- 现有 GLCA 的各阶段预算可能增长；只保证同一 Task 三组预算相同，不声称固定跨阶段总内存。

## 3. 服务器启动

复用已可训练的 Linux 环境、COCO 图像、manifest、GLCA 来源配置/标注、上一阶段初始化权重及 GNN 来源记录。本地仓库不包含这些训练产物。先确认目标 GPU 空闲；启动器检测到 compute 进程时会退出，不自动抢占。

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
python -m pip install 'matplotlib>=3.7'

# 默认来源：stage1_diagnostics_v2/d2_full_finetune_no_extra_losses/graph
# 默认三组全部重跑；队列顺序 GLCA -> Uniform -> Uniform + teacher。
python tools/owod/run_replay_comparison.py --dry-run

# dry-run 成功后启动；不要在旧队列仍占用 GPU 时启动。
nohup setsid python -u tools/owod/run_replay_comparison.py \
  > replay_comparison_launcher.log 2>&1 < /dev/null &
```

`--dry-run` 校验来源、输入哈希、类别、实际回放预算、图像路径和已存在输出，不加载 CUDA 模型、不创建输出、不启动训练。默认采用来源 seed（历史 D2 为 42）和 GPU `0,1`；GPU 数量必须与来源 `world_size` 相同，旧配置没有该字段时默认按两卡检查，可用 `--reference-world-size` 明确其真实数量。

只有已准备好 Stage 2/3 的 GLCA 来源时才使用以下多 Task 命令；阶段来源缺失时不会自动拿另一阶段权重代替：

```bash
PILOT="$PWD/exps/owod/m-owodb/order0/pilot_unverified"
python tools/owod/run_replay_comparison.py \
  --source-dirs \
    "$PILOT/stage1_diagnostics_v2/d2_full_finetune_no_extra_losses/graph" \
    "$PILOT/d2_continual_v1/stage_2/graph" \
    "$PILOT/d2_continual_v1/stage_3/graph" \
  --output-dir "$PILOT/replay_comparison_tasks234_v1" \
  --dry-run
```

确认预检通过后，以同样参数去掉 `--dry-run` 启动。三阶段使用相同 manifest。需要多 seed 可加 `--seeds 42 43 44`，并使用新的输出目录；多 Task、多 seed、三组均进入同一顺序队列。

多 seed 时，GLCA 复用来源的 GNN 选择/回放图片，Uniform 按每个 seed 抽样，同一 seed 的两种 Uniform 共用图片。它衡量固定 GLCA 记忆下的训练随机性与 Uniform 抽样变化，**不是 GNN 重新校准、重新选图的端到端多 seed 复现**。

## 4. 输出、恢复和绘图

```text
replay_comparison_v1/
├── comparison.json
├── queue_status.json
├── stage_1/seed_42/
│   ├── glca/
│   ├── uniform/
│   └── uniform_teacher/
│       ├── plan.json
│       ├── train.json
│       └── graph/  # run_config、metrics.jsonl、checkpoint、完成标记、train.log
└── comparison_report/
    ├── report.md
    ├── metrics.csv
    ├── final_summary.csv
    ├── run_status.csv
    ├── task_metrics.png / .svg
    ├── previous_current.png / .svg
    └── task_2_curves.png / .svg  # 每个请求的 Task 一张
```

所有训练完成后自动生成报告，也可随时单独执行：

```bash
python tools/owod/run_replay_comparison.py --summarize

# 自定义实验根目录 / 图表目录：
python tools/owod/plot_replay_comparison.py \
  --output-dir exps/owod/m-owodb/order0/pilot_unverified/replay_comparison_tasks234_v1
```

指标口径：

| 指标 | 来源与展示 |
| --- | --- |
| mAP | `test_owod_known_ap50`，图上明确标为 Known mAP@50；不是全 80 类 AP，也不是 COCO AP@[.50:.95] |
| U-Rec | `test_owod_u_recall`，沿用同一 unknown threshold 和内部 evaluator |
| H-score | `2 × Known AP50 × U-Rec / (Known AP50 + U-Rec)`，重算并检查日志一致性 |
| Previous / Current | 对应旧类/本阶段新增类 AP50，Task 2–4 单独画柱状图和训练曲线 |

数值从日志的 0–1 统一转为百分数。没有未知 GT 的阶段（通常 Task 4）将 U-Rec/H 记为 NA，不填 0。没有训练数据或没有完成的组不会生成虚构结果；空图位置显示 NA。

最终柱状图只使用**三组都完成预定 epoch 的共同 seed**，按 final epoch 汇总，不按各方法最优轮数挑选结果。折线图只比较同一 Task 三组均已评估的同 epoch/seed；未完成的部分保留在明细 CSV 中。多 seed 显示均值和样本标准差，单 seed 不显示误差线；柱状图的 `n` 为完整配对 seed 数。跨 epoch 的可用 seed 集合可能随训练进度变化，完整进度以 CSV 为准。

中断后确认原进程退出，使用**原命令和相同参数加 `--resume`**。当前组从自己的 checkpoint 恢复优化器，但初始化/教师仍引用原来的上一阶段权重；已完成组跳过，尚未开始的组正常启动。有日志但无 checkpoint 的失败组不能无声覆盖，应另选输出目录。输入、代码、标注或计划改变会拒绝混用旧结果。一组失败立即停止后续训练。

仅绘图不加载 checkpoint，可将 `comparison.json`、每组 `plan.json`/`train.json`、`graph/run_config.json`/`metrics.jsonl`/`training_complete.json` 按原相对结构拷回本地报告；计划中的原始绝对路径只用于核对训练记录。

## 5. 验证与当前状态

```bash
python -m unittest tools.owod.tests.test_replay_comparison -v
```

测试覆盖三组参数/预算一致性、图片级确定性抽样、当前/旧图重叠、只读预检、GPU 数量不匹配、续跑防篡改、失败停止队列、H-score 校验、无未知 GT、未配对 seed 排除和 PNG/SVG 实际渲染。测试指标为临时目录中的合成测试数据，不是检测器实验结果。

新增真实精度结果仍需在具有数据和来源权重的服务器上训练后产生。现有 D2/Random-K/论文 baseline 历史结果不被填入这组三方法对比表。
