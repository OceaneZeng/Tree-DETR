# Task 2 学习能力诊断：先评估，再做极小集拟合

只读使用已有诊断检查点；不修改正式训练代码、原始数据、旧输出或 checkpoint。
所有新增训练是一次性诊断，不导出权重，不允许续入正式实验。后续只保留
hard/frozen 与 soft/frozen，soft-localization 的历史结果留存，不再扩展训练。

用户通过 Git 同步本地改动后，在服务器仓库中激活 tree-detr 环境，建议在 tmux
运行以下命令。默认使用 PCI 顺序 GPU 0，一张 3090 即可；与此前两卡筛选的
有效 batch 不同，因此不能把这里的训练数值当成正式方法对照。

## 1. 不训练：检查评分和候选截断

```bash
python tools/owod/diagnose_learning.py scores --gpu 0 \
  --output output/learning-diagnostics/scores
```

从 `output/task2-shared-s-8k/comparison_plan.json` 找到共同 Task 1 检查点，
核验 manifest、筛选数据和 checkpoint 哈希。使用原来的 1,000 张留出图，
只读取 Task 1 已知类标注；同一次前向输出计算四种评分：

1. Task 1 的 19 类，hierarchical，top-50。
2. 扩展至 Task 2 的 40 类，hierarchical，top-50。
3. 同样 40 类，focal，top-50。
4. 同样 40 类，hierarchical，top-300。

结果为 `scores.json`。主要比较 `metrics.previous_ap50`，以及 `composition`
中的 old/current/unknown 候选数量和平均分数。这里没有 Task 2 GT，因此只输出
Previous AP，不输出容易误读的扩展 Current/Known AP；文件中也注明这一限制。

`raw_old_box_recall_iou50` 是所有 query 框对旧类 GT 的几何覆盖率，不依赖
分类和 top-k，不是正式 U-Recall。若同一权重仅扩展类别范围就导致旧类 AP
大幅下降，说明评分/候选竞争有贡献；若 top-300 或 focal 恢复部分 AP，
可进一步定位评分机制。它们只是读出诊断，不能直接据此更改正式实验设置。

## 2. 固定 160 张训练图，检查能不能继续学

```bash
python tools/owod/diagnose_learning.py fit --gpu 0 \
  --images 160 --steps 600 --batch-size 2 --workers 4 \
  --output output/learning-diagnostics/fit
```

起点为共享起点对照中 **hard 的已完成 Task 2 increment checkpoint**，不是
Task 1，也不是 finetune checkpoint。这是在实际低 AP 模型上检查可学习性，
不是重做从 Task 1 到 Task 2 的完整实验。

数据管理函数 `tools/owod/manage_data.py:prepare_fit_data` 从原来的 8k Task 2
训练数据中确定性选择覆盖全部新类的 160 张图；保留原阶段标签可见性，
不读取验证标签来选样本。选择清单和输入/输出 SHA-256 写入 `data/`。

固定 resize/normalize，无随机裁剪和翻转增强；保留模型训练时的 dropout，
hard/frozen 参数范围、损失系数和学习率。重新建立 AdamW 和 EMA，无 LR
衰减，默认 600 optimizer steps，等于对 160 张图约 7.5 次遍历。
这不是严格续训，也不承诺一定拟合成功。EMA 教师的翻转视图一致性仍保留。

结果在 `distill_on/`：

- `before.json`、`after.json`：这 160 张**训练图**上的 Current AP50、原始新类框覆盖率。
- `heldout_before.json`、`heldout_after.json`：原 1,000 张留出图的 Previous/Current/Known AP50。
- `updates.jsonl`：每 50 步记录分项加权损失、裁剪前梯度范数、新/旧分类行及
  适配器的梯度平方范数和相对初始权重的变化平方范数。第一步和最后一步也记录。

判断顺序：新类分类行/适配器是否有梯度和参数变化 → 训练 Current AP 是否
上升 → 原始框覆盖率是否足够 → 留出集是否改善。旧分类行在 increment 中
应保持冻结；若其参数出现变化，需要优先排查。
不能用训练 AP 改善代替泛化结果，也不凭一次 600 步失败断言方法不可用。

## 3. 根据前两步结果，再决定做蒸馏对照

```bash
python tools/owod/diagnose_learning.py distill --gpu 0 \
  --images 160 --steps 600 --batch-size 2 --workers 4 \
  --output output/learning-diagnostics/distill
```

依次运行 `distill_on/` 与 `distill_off/`，两组从同一个 hard Task 2 checkpoint
重新加载，具有相同图像、seed、batch、优化器和步数。仅关闭旧类输出蒸馏，
**不关闭 discovery 的 EMA 一致性损失**。旧类教师为原共同 Task 1 checkpoint。
单 GPU CUDA 算子仍可能存在非确定性；这是低成本诊断，不是显著性结论。

若关闭蒸馏明显提高训练新类 AP，同时损害留出旧类 AP，说明有可塑性与保持
之间的权衡，后续应测试蒸馏权重，而不是直接取消保护。如果两组都学不动，
优先关注参数更新、框质量和基础检测能力，不扩大 hard/soft 比较预算。

所有模式拒绝覆盖非空输出目录；没有 resume。中断后换一个输出目录重新做
短诊断即可。不会生成 `.pth`，不会启动全量训练。
运行时间取决于数据读取和 GPU 占用；先做 scores，再做 fit，不一次性排长队。
