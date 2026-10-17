# 新类监督与评分链路诊断（不训练）

本次代码检查：COCO 标签保留 source category ID，matcher 按这些 ID 取分类列。
DiscoveryCriterion 将匹配 GT 设为 known/objectness 正例、novelty 负例；
mine_candidates 排除匹配 GT 的 query 和与可见 GT IoU > 0.3 的 query。
基础分类损失会恢复每个辅助层自身匹配 GT 的权重为1。
这些规则未显示直接的“匹配 GT 同时作为伪 unknown”冲突；它们不保证训练收敛。
训练用 sigmoid focal，hierarchical 推理用 softmax 和 novelty 门控，确有评分形式差异，
但差异本身不是已经证实的实现错误。

使用上一轮表现最好的 no-distill 微调结束权重，在原增量训练数据中确定性选取
160张图（与此前拟合采用同一规则），保留新类可见标注。固定评估变换，无反向传播。
输出每个 GT 的框覆盖、类别 argmax、Hungarian 匹配 query 的分数和 novelty，
以及正确类别且 IoU > 0.5 的 query/class 对能否进入实际全局 Top-50/300。
额外移除 hierarchical 的 (1-novelty) 乘数，但保持 unknown 分数不变，以隔离门控影响。
该反事实评分不是新默认设置。

本地提交推送，服务器 git pull 后，在已激活环境的 tmux 中：

```bash
mkdir -p output/learning-diagnostics/logs
set -o pipefail
python -u tools/owod/diagnose_learning.py supervision \
  --comparison-root output/task2-final-check --score-phase finetune \
  --gpu 0 --images 160 --batch-size 2 --workers 4 \
  --output output/learning-diagnostics/supervision-final \
  2>&1 | tee output/learning-diagnostics/logs/supervision-final.log
```

生成 `complete.json` 为完成；`supervision.json` 是总体/逐类统计，
`objects.jsonl` 是逐目标记录，`data/` 保存选样及哈希。
输出拒绝覆盖非空目录。没有权重导出，不修改任何训练默认值或旧输出。

读取顺序：any_box_iou50 → correct_argmax_box_iou50 → 各评分 top50/top300。
前两者检查几何覆盖与已知类 argmax；后者允许非 argmax 的正确类别候选，因为
实际后处理保留 query/class 对。它们不是严格嵌套的漏斗，不应直接相减归因。
matched_means 的 novelty 和 GT 分类分数是对 Hungarian 匹配 query 的均值，
不是仅对 IoU合格框计算。低IoU匹配、高novelty和分类错误应结合逐目标数据分析。

这些覆盖率不是 AP 或 U-Recall，同一个 query 可以覆盖多个 GT。只使用新类
训练标注，不能推断旧类/未知类性能。这里只读当前模型，不重建历史 EMA 教师、
训练时 dropout 或伪标签，因此不能当作历史每步监督的回放。发现具体问题后再
设计短修复验证，不自动开启下一轮长训练。
