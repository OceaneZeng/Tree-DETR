# 从同一个 Task 1 起点重做 Task 2 对照

2026-10-16：首次 8k 筛选中，三组 Task 1 的 AP50 已有明显差异，不能把
Task 2 最终差距全部归因于路由/定位策略。新入口只重做 Task 2，不重训 Task 1。

## 固定内容

- 默认共同起点：`output/screening-s-8k/screen/hard/s-owodb/seed_42/stage_0/increment/checkpoint.pth`。
- 共同 GNN：同一 hard 组的 `gnn/stage_0/gnn_stage0.pt`，不重新校准。
- 原诊断数据：`output/screening-s-8k/data/screening.json`，不重新抽样。
- 从旧 `screen_plan.json` 继承种子、GPU、batch、workers、训练轮数和评估频率。
  当前实验对应 seed=42，8,000 张 Task 2 增量图、1,000 张验证图，12 轮增量＋6 轮完整均衡微调。
- 三组只改变 hard/frozen、soft/frozen、soft/bbox-last；使用相同 Task 1 检测器和校准来源。

固定选原 hard 基线为起点，不按 Task 1 验证成绩挑最高的一组。
如确有需要，可显式 `--anchor-variant soft`，但必须使用另一个新输出目录。
这控制了起点差异，仍不能保证 CUDA 自定义算子的训练轨迹逐位可重复，
也不能把单种子、已知类验证结果解释为完整四任务或未知类效果最优。

## 运行

本次只修改本地源代码。用户通过 Git 同步后，在服务器仓库目录、已激活的
tree-detr 环境中，建议在 tmux 内执行：

```bash
# 读取并验证共同检查点、GNN、数据身份，打印三组启动命令
python tools/owod/run_task2_comparison.py

# 正式执行这次 Task 2 对照；不会启动全量实验
python tools/owod/run_task2_comparison.py --execute
```

新输出目录为 `output/task2-shared-s-8k/`；旧日志、权重及数据不覆盖。
三组各自只运行 Task 2 increment → finetune，预计训练/评估合计约 **8 小时**
（按上一轮各组 Task 2 实测约 2.6 小时估算），默认运行窗口 **12 小时**。
估算不保证完成时间，资源竞争会影响速度。

```bash
tail -f output/task2-shared-s-8k/hard.log

# 中断后从最近完整 epoch 检查点恢复，追加 12 小时运行窗口
python tools/owod/run_task2_comparison.py --execute --resume --budget-hours 12
```

`comparison_plan.json` 固定共同检查点、GNN、父 manifest、预训练和诊断描述的
SHA-256。每个子计划也绑定这些输入；恢复时任何身份变化都会被拒绝。
GNN 元数据必须证明它来自该 Task 1 检查点和对应训练数据。

检查点策略只对 **诊断 Task 1 完成模型 → Task 2 increment 初始化** 放开
尚未生效的 routing/localization 配置变化；不改写旧检查点。
Task 2 内续训及 increment → finetune 仍严格保持方法和共同父节点身份。
正式全量运行、其他任务、未完成 Task 1、不同数据及不同 GNN 都不能借此绕过验证。

## 读结果

三组全部完成后输出 `selection.json`，包含最终 Previous/Current/Known AP50、
hard→soft 与 soft→soft-localization 差分、Task 2 耗时、增量/微调评估曲线，以及
共同起点的哈希。AP 值为 0–1，乘 100 得到百分数。

没有自动启动全量的选项。先检查共同起点下的排名、后期趋势和新类学习能力，
再决定正式全量配置。诊断检查点仍禁止初始化正式 benchmark。
