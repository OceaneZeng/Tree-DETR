# D2 后续实验：Random-K 对照和 30 轮训练

日期：2026-09-09。用户决定以 D2 为主线，允许更新 backbone 的 layer2/3/4；D4 保留为冻结 backbone 消融。默认只运行 Random-K 对照和 D2-long；Cosine 保留为可选消融，不进入本轮默认队列。

## 本轮队列

| 顺序 | 目录 | 参数更新和监督 | 唯一有意变化 |
| --- | --- | --- | --- |
| 参考 | 原 D2 | 常规微调、GNN 回放、教师补标 | 已完成 20 轮 |
| 1 | `d2_random_k/graph` | 沿用 D2 | 随机选 K 个旧类获得额外回放 |
| 2 | `d2_gnn_30ep/graph` | 沿用 D2，并直接复用其训练标注 | 总轮数 20→30 |
| 可选，默认不运行 | `d2_cosine_k/graph` | 沿用 D2 | 用 Stage 0 分类器行余弦相似度选 K 个旧类 |

主学习率 1e-4，backbone 学习率从 D2 实际配置读取（必须大于零），batch 每卡 2、两卡、seed 42、StepLR 周期 15。全部从原 Stage 0 权重初始化，教师也使用原 Stage 0。D2-long 不加载 D2 的第 20 轮权重，不是原 D2 的续跑。30 轮属于额外训练预算；StepLR 在第 30 轮结束后再次降学习率，不影响这 30 轮内的训练。

不启用 LoRA、margin、projection 或蒸馏。GNN 不重新训练、不重新选邻域；当前只验证既有选择策略。最终关键对照的多 seed 重复在本轮结果后确定，不自动增加训练。

## 预算和对照范围

1. 读取 D2 的 `diagnostic_plan.json` 和 `graph/run_config.json`，核对原始输入哈希、完成标记、最终指标及 checkpoint 是否存在。
2. 从 manifest 的 Stage 1 新类标注和 Stage 0 旧类标注重建原 GNN 回放。图像列表、标注列表、类别列表必须与 D2 原训练文件逐项一致；不一致就停止，不悄悄改用另一份数据。
3. Random/Cosine 沿用全部旧类基础配额与 K 个高风险类额外配额、图像采样种子和原选择算法。K、配额均从原实验读取。
4. 回放总量对齐 D2 **实际带 `owod_replay=true` 的唯一图片数**，而不是配额之和。去重后不足时按风险类优先轮转补齐，超出时确定性裁减并保护基础样本；`plan.json` 记录补齐/裁减数、最终图像 ID 和逐类标注量。这是同预算选择对照的一部分，不能隐去预算修正。
5. 和当前图像重合的回放图像保持 current 标记，按原实现合并已有旧类标注。选择变化可能改变这些图像上的旧类监督；它属于回放选择策略的效果，不是图像总量增加。计划记录重合数量。
6. 训练数据集长度、每卡样本数、回放比例和优化器每轮步数与 D2 对齐。教师、增强、优化器及模型参数使用 D2 记录的 argparse 配置显式传入，避免新默认值漂移。

Cosine 读取原 Stage 0 最后一个 decoder 分类头的行向量，计算新→旧的余弦相似度，采用与原 GNN 相同的 `max` 或 `top_mean` 聚合（本轮预计 top-3 mean）。新类分类器行在 Stage 0 尚未经过新类监督，因此它只是明确规定的简单对照，不能解读为语义先验。它也不是历史 cosine 完整训练配方的复刻。

现有未知评分和评估保持不变。H/UDP 等仍是内部实现的诊断指标；manifest 的 unverified 标记不因本轮实验而改变。

## 同步

服务器需同步当前仓库中的新增启动器、测试和本文。项目不再保存发布 ZIP；按 Git
提交或对应源码路径同步文件，不替换模型、训练入口或已有 baseline 启动器。现有服务器
需要保留之前已同步的 D2/D4 和 paper baseline 工具。

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
python -m unittest tools.owod.tests.test_d2_followups tools.owod.tests.test_stage1_diagnostics
```

## 检查与启动

先检查现有 baseline 队列状态和 GPU。旧队列会自动从一个阶段进入下一个阶段，仅看到某阶段完成或短暂无 GPU 进程，不代表整个队列已经退出。本启动器不会自动停止它。应等原队列结束，或由操作者在明确的阶段边界管理原队列，然后启动新队列。

```bash
cat exps/owod/m-owodb/order0/pilot_unverified/paper_baselines_v1/queue_status.json
nvidia-smi
python tools/owod/run_d2_followups.py --dry-run
```

dry-run 会核对输入权重哈希、标注、配置、代码哈希并检查图像路径，不训练、不创建实验输出；只有显式指定 Cosine 时，才额外在 CPU 加载权重计算余弦评分。确认启动条件后：

```bash
nohup setsid python -u tools/owod/run_d2_followups.py \
  > d2_followups_launcher.log 2>&1 < /dev/null &
echo "launcher PID: $!"
```

默认 GPU `0,1`，master port `29583`。Random-K、D2-long 两组顺序运行，不同时占卡。每组启动前检查所选 GPU 的计算进程；忙碌则退出。输出目录锁防止同一新队列重复启动，但不能锁住其他工具或管理员的调度，操作者仍需确认原队列已停止调度。

也可以显式指定本轮两组（效果与默认命令相同，旧版包也支持此用法）：

```bash
python tools/owod/run_d2_followups.py --experiments random long --dry-run
nohup setsid python -u tools/owod/run_d2_followups.py --experiments random long \
  > d2_followups_launcher.log 2>&1 < /dev/null &
```

仅在之后决定开展 Cosine 消融时，才使用 `--experiments cosine`。重跑默认队列会跳过已完成且计划一致的组，不会自动加入 Cosine。不要同时执行上述两个启动方案。

如果旧版实验包已经启动，新版文件不会改变正在运行的队列。也不要覆盖正在运行或需要续跑的启动器：计划包含代码哈希，替换后旧计划会拒绝续跑。尚未启动时可直接使用新版；已有实验时保留原代码，后续启动显式传入 `--experiments random long`。

## 进度与续跑

```bash
tail -n 60 d2_followups_launcher.log
python tools/owod/run_d2_followups.py --summarize
```

默认输出根目录是 `exps/owod/m-owodb/order0/pilot_unverified/d2_followups_v1`。训练细节在各组 `graph/train.log`、`graph/metrics.jsonl`，计划和预算审计在各组 `plan.json`。汇总对比同轮 D2；第 25/30 轮的 delta 为 NA，避免错配到 D2 第 20 轮。

训练中断后确认旧进程退出，且中断组已有自己的 `graph/checkpoint.pth`，再执行：

```bash
nohup setsid python -u tools/owod/run_d2_followups.py --resume \
  >> d2_followups_launcher.log 2>&1 < /dev/null &
```

恢复仅使用中断组自身 checkpoint，教师保持原 Stage 0。代码/输入/计划变化会拒绝续跑；需另选 `--output-dir`。若在首个 checkpoint 前失败，同样使用新输出目录。原训练器不保存完整随机数状态，续跑不承诺逐步等同于未中断轨迹，应记录中断位置。

## 结果判断

- Random/Cosine 与 D2：同轮数对比旧类、新类、Known AP50 及逐类遗忘；暂不把单 seed 微小波动当显著收益。
- D2-long：先比较其第 20 轮与原 D2，确认重复运行的差异，再看第 25/30 轮变化。额外 10 轮是否有收益需要实测。
- 正式结论需补关键组合多 seed 和统一未知类评估。不得按每个方法最有利的 epoch 或阈值拼接结果。

本地测试覆盖预算去重修正、重合标注合并、配置继承、Stage 0 初始化、源文件变更拒绝、只读 dry-run、续跑保护、GPU 忙碌与失败停止、同轮汇总以及真实 CPU torch checkpoint 的 Cosine 评分。服务器完整 COCO 训练仍需运行后验证。
