# 冻结参考分支与新类分支的隔离检查

这是诊断原型，不是正式模型架构。上一轮开放 decoder/bbox 提高了32张训练图
拟合，却改变了旧类的共享特征和框；本轮检查能否通过独立输出来源隔离这类退化。
不恢复 soft-localization，不重跑完整 Task 2，不替换官方数据或正式评估约定。

参考分支：原 hard/no-distill Task 2 微调完成权重，完整冻结、eval模式。
新类分支：从相同权重重新加载，按上一轮 decoder_bbox_open 范围训练，
相同32张图、300步、seed42、batch2、lr2e-4；旧类输出蒸馏关闭，EMA一致性保留，
两条分支关闭dropout。各自独立运行完整模型，当前是两次前向的成本，不声称高效。

评估时，从参考分支取旧类和 unknown 的分数与框，从新类分支取新类分数与框。
两边评分均使用完整已知类集合（hierarchical 的 softmax 分母不缩为各自类别），
然后统一执行一次全局 Top-50/300。没有GT路由、分支配额或评分调参。
旧类排序前输出不变，但新类候选可能把旧类挤出 Top-K，所以不保证旧类 AP 不降。
unknown 输出固定也不等于未知检测性能得到验证。

本地提交推送后，在服务器已激活环境的 tmux 运行：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
git pull
mkdir -p output/learning-diagnostics/logs
set -o pipefail
python -u tools/owod/diagnose_learning.py isolation \
  --comparison-root output/task2-final-check --score-phase finetune \
  --images 32 --steps 300 --capacity-lr 0.0002 \
  --gpu 0 --batch-size 2 --workers 4 \
  --output output/learning-diagnostics/isolation-final \
  2>&1 | tee output/learning-diagnostics/logs/isolation-final.log
```

仅训练一个新类分支。输出根目录 complete.json 表示全部结束。
decoder_bbox_open/ 下的主要文件：

- heldout_before.json：原模型起点。
- isolated_heldout_before.json：同权重双分支的起点，指标应与原起点一致（允许浮点差异）。
- heldout_after.json：训练后新分支独立输出，复核上一轮遗忘现象。
- isolated_heldout_after.json：冻结参考输出与新类输出合并后的结果。
- isolation_scope.json：分支来源、候选限制与诊断范围。
- parameter_scope.json、gradient_paths.json、updates.jsonl、probe_before/after.json：沿用容量检查日志。

判断：同评分下比较起点、单分支训练后、隔离后新旧/已知类 AP。若隔离改善旧类，
但新类未超过起点，只能支持隔离机制有效，不能开始全量或宣称解决泛化问题。
如果旧类仍下降，要结合共享 Top-K 的候选竞争分析，不继续盲目扩大训练预算。
隔离报告中的 raw_* 几何覆盖仅来自训练分支，不是合并预测的覆盖率。
所有数据均为已知类诊断，不能替代官方 unknown 指标或正式benchmark结论。

上轮短拟合未导出权重，因此本轮需要重新执行300步；不重跑冻结训练组。
本轮也不导出权重，不修改主训练策略或checkpoint续训规则；输出不覆盖、不支持resume。
