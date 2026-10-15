# 小数据方法筛选 → 全量训练

2026-10-15 用户授权的小规模诊断流程。只支持已核验的 schema-3 M/S
manifest，保持原类别顺序与阶段标签可见性。筛选数据不构成第三种 benchmark，
不修改作者 ImageSets 或正式 manifest，也不能作为论文正式结果。

## 为什么要跑到 Task 2

当前实现的 hard/soft 路由和 frozen/bbox-last 定位策略从 Task 2 的
类别适配器阶段才生效。只跑 Task 1 无法比较它们。

三组都按相同 seed=42、数据与预算完成：

| 阶段 | 默认训练图像 | Epochs |
| --- | --- | --- |
| Task 1 increment | 2,000 | 8 |
| Task 2 increment | 2,000 | 8 |
| Task 2 finetune | 完整作者均衡清单（S 为 1,679 张） | 4 |

三组分别为 hard/frozen、soft/frozen、soft/bbox-last。保留模型规模、DINO
预训练、增强、优化器和学习率；8/4 epochs 是项目诊断预算，非论文 recipe。
每 4 轮及最后一轮评估。每组单独训练 Task 1，避免改写检查点配置或放宽
方法链验证；三个 Task 1 本身不是路由效果对照。

## 数据隔离与选择标准

通过 `tools/owod/manage_data.py prepare-screen` 派生独立诊断数据。
从前两个任务的官方 **训练** 标注合并已发布的已知类记录，在类别覆盖约束下
按 `SHA256(seed:source_id)` 固定排序选择 400 张留出图。未来类别、正式测试集
标签不参与选择。留出图优先排除 Task 2 的完整均衡微调清单，随后从所有增量
训练、GNN 校准、梯度探测和原型初始化来源中排除；因此作者均衡清单保持完整。
原始对象身份保留、重复对象记录合并，训练标签始终按阶段过滤。

每个阶段重新验证父 manifest、输入文件哈希、数据成员与标签、留出隔离；
诊断描述文件及其 SHA-256 写入计划和检查点。诊断与正式检查点相互拒绝混用。
生成的 JSON、日志和权重仅保存在被 Git 忽略的 output 下。

按 **Task 2 微调最后一轮的留出 Known AP50** 选优，保留父协议的作者 AP
计算约定，并同时报告 Previous/Current AP50、两组差分和训练/评估时间。
AP 数值范围为 0–1。相同分数按 hard、soft、soft-localization 顺序取先者。
`selection.json` 保存全部结果和第一、二名差距。

留出数据没有未来类标签，因此不计算 WI、A-OSE、U-Recall 或 H-Score。
不能把这个选择解释为未知类最优或完整四任务最优；单种子、短训练的排名具有
不确定性。全量实验仍报告正式协议所要求的所有适用指标。

## 服务器运行

本次改动只提交本地源代码，由用户自行通过 Git 同步。进入服务器仓库并激活
`/home/top/disks/new-hdd/conda_envs/tree-detr` 环境后，建议在 tmux 中运行：

```bash
python tools/owod/run_screening.py --execute
```

默认使用 PCI 顺序的 GPU 0、1，预检要求两张 RTX 3090；每 GPU batch=2、
workers=4、activation checkpointing。默认路径：

- 父 manifest：`data/owod/s-owodb/split_manifest.json`
- 权重：`pretrained/dino_resnet50.pth`
- 输出：`output/screening-s-seed42/`

不加 `--execute` 只打印流程，不准备数据、不启动训练。
`--screen-only` 可以仅筛选；默认三组全部完成后自动以优胜方法从 DINO
重新开始正式全量四任务训练（50 increment / 20 finetune epochs），
输出到 `full/<winner>/`，绝不从小数据检查点继续。

筛选默认有 **12 小时运行上限**，包含数据准备和 GNN 校准，不承诺 12 小时
必定完成。超时终止该流程自己的进程组，保留已有检查点；任何失败、不完整
结果或三组 Known AP50 均为零都不启动全量训练。全量训练不受该上限限制。

```bash
# 查看某组的实时输出
tail -f output/screening-s-seed42/hard.log

# 超时/中断后，保持数据和训练配置不变，追加至多 4 小时的运行窗口
python tools/owod/run_screening.py --execute --resume --budget-hours 4

# 新一轮预算或数据量必须用新的输出目录；例如增加训练轮数
python tools/owod/run_screening.py --execute --output output/screening-s-longer \
  --increment-epochs 12 --finetune-epochs 6 --budget-hours 16
```

恢复窗口是本次调用的时间上限；更改数据量、seed、epochs、batch 或 GPU 配置
不能复用原输出目录。若全量阶段中断，同一 `--resume` 命令验证筛选结果后恢复
独立全量链。已完成训练阶段会跳过；GNN 校准沿用原有内容寻址缓存验证机制。

单独准备/核验诊断数据：

```bash
python tools/owod/manage_data.py prepare-screen \
  --manifest data/owod/s-owodb/split_manifest.json \
  --screen-output output/screening-s-seed42/data --train-images 2000 --val-images 400 --seed 42
python tools/owod/manage_data.py verify-screen \
  --manifest data/owod/s-owodb/split_manifest.json \
  --screen-output output/screening-s-seed42/data
```

独立准备数据时，请为后续完整自动流程指定另一个空输出目录；自动流程自己
管理 `screen_plan.json` 和数据准备，拒绝覆盖不明来源的非空目录。
