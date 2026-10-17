# 固定小集容量检查：两组短拟合

仅用于排查新类定位/分类学习不足，不是第四种正式方法，不恢复 soft-localization。
当前 hard/frozen 冻结基础 decoder 和 bbox 头，但 bbox 损失仍可经特征到达适配器；
hard 路由仅让选中的类别分支参与更新，路由监督使用 detach 后的输入。
这些机制限制可训练范围，但不是已证实的错误。

两组从同一个已完成 hard/no-distill Task 2 微调权重重新加载，使用相同 seed、
确定性选择的32张增量训练图（覆盖新类）、相同 minibatch 序列、batch2、300步。
固定 resize/normalize，禁用两组 dropout，无旧类输出蒸馏；保留 discovery
损失、EMA 一致性、memory 更新和 hard 路由。新 AdamW/EMA，恒定 lr=2e-4，
保留原分类器学习率倍率和裁剪。这是显式诊断学习率，不是正式微调续训。

- frozen：当前新类分类行、类别适配器及 discovery 模块。
- decoder_bbox_open：以上参数，加基础 decoder 和完整 bbox MLP；backbone、
  encoder、旧类分类行和 old-evidence 分支维持冻结。不会解开旧分支的冻结策略。

同时放开 decoder/bbox 用于检验联合容量限制，不能单独归因到其中一个模块。
两组都是新类小集拟合，不能直接用作正式增量/平衡微调方案。

本地提交推送后，在服务器已激活环境的 tmux 中运行：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
git pull
mkdir -p output/learning-diagnostics/logs
set -o pipefail
python -u tools/owod/diagnose_learning.py capacity \
  --comparison-root output/task2-final-check --score-phase finetune \
  --images 32 --steps 300 --capacity-lr 0.0002 \
  --gpu 0 --batch-size 2 --workers 4 \
  --output output/learning-diagnostics/capacity-final \
  2>&1 | tee output/learning-diagnostics/logs/capacity-final.log
```

输出根目录 `complete.json` 表示两组及所有评估结束。每组包含：

- parameter_scope.json：实际可训练张量清单、分类行范围及学习率。
- gradient_paths.json：第一个 batch 的分类/定位分项损失流向各参数组的梯度平方范数，裁剪前。
- updates.jsonl：训练总损失、分项损失和各参数组梯度及累计变化。
- probe_before/after.json：训练图的框覆盖、分类正确率、novelty 和候选统计。
- before/after.json：训练新类 AP，四种评分设置。
- heldout_before/after.json：原1000张留出图的新旧/已知类 AP，同四种评分。

保持两组起点报告一致；首先检查打开的 decoder/bbox 是否收到梯度并更新，再看
训练框覆盖、分类和 AP 是否明显改善；留出旧类用于暴露遗忘风险。若开放组改善，
支持当前可训练范围受限，但不证明全量泛化或该方案适合主实验。若两组都失败，
不能直接判定没有容量问题，需结合损失、匹配和优化记录解释。
不会保存任何训练权重、修改主实验默认值或自动续入全量。输出拒绝覆盖，不支持resume。
