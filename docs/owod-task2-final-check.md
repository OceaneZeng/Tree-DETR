# 最后一轮 Task 2 验证

目的：仅训练 hard/frozen、关闭旧类输出蒸馏的一组，完成增量和微调；
与已有 hard 基线在同一评分设置下比较。不启动全量，不重跑 Task 1、GNN 或 soft。

继承 `output/screening-s-8k/screen_plan.json` 的数据、seed、训练预算、GPU 和 batch。
当前这份计划为 8k 增量图、1000 留出图、12 增量 epoch、6 微调 epoch、seed 42。
微调仍使用已发布的平衡记忆。两阶段均关闭旧类输出蒸馏；EMA 一致性和
其他 discovery 机制保留。训练仍使用原 hierarchical/Top-50 评估配置，
避免改变训练配置和阶段兼容性；最终独立评估输出四种评分。

先在本地提交并推送代码，服务器拉取后，在已激活 tree-detr 环境的 tmux 中：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
git pull
python tools/owod/run_task2_final_check.py
python -u tools/owod/run_task2_final_check.py --execute
```

第一条 Python 命令只校验输入并打印计划。执行命令默认有12小时上限，
这是运行上限，不是耗时预测。输出在 `output/task2-final-check/`，自动保存日志。
中断或达到上限后：

```bash
python -u tools/owod/run_task2_final_check.py --execute --resume
```

续训必须保持数据、来源权重和配置一致；不允许在同一 Task 2 链中切换蒸馏开关。
中断的评分输出保留，重试使用新目录。不要改名/删除历史基线或共享 Task 1/GNN。

训练后自动评估 baseline/no_distill × increment/finetune 四份权重，每份输出
hierarchical/focal × Top-50/Top-300。共享每份模型的前向输出，无需四次训练。
`final_report.json` 生成表示整个流程完成，包含百分数单位的 AP 和候选占比。
训练中显示的 Top-50 指标不能直接与此前 Top-300 结果比较。

判断以同一评分、同一阶段的 baseline/no_distill 为单位，优先看最终微调
Previous/Current/Known AP50，并查看增量→微调变化。focal/Top-300 是优先候选，
hierarchical/Top-300 用于检查新类收益与旧类保持的权衡，不自动挑最高分并启动全量。
若新类仍在约1%水平，停止扩大训练预算，转为实现排错。这个单种子已知类留出集
不能验证未知类指标或稳健优势，也不属于正式基准结果。诊断权重不能初始化正式训练。
