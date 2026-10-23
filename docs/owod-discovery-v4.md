# Discovery v4：恢复已知类别的完整监督

2026-10-23，服务器 discovery v3 的 Task 1 完成第5轮时，Known AP50为12.78%，
同轮control为45.78%。两者manifest、DINO权重、batch、学习率和seed相同。
正式评估中的unknown输出183503/247600（74.1%），unknown precision为1.53%。
这些观测说明存在明显退化，不证明单一根因，也不用于选择测试集上的最佳评分参数。

对已保存的checkpoint0004.pth进行了只读推理：训练集按dataset顺序取前32张，
使用eval transforms；159个可见GT中，所有原始框IoU>0.5覆盖104个，
hierarchical Top50正确类别候选覆盖63个，Top300覆盖102个。
matched query的objectness均值0.993、novelty均值0.233。
3200个query中有956个未匹配query与已知GT的IoU>0.3，它们被排除出未知/背景挖掘。
无EMA时仅85个query被标为背景；这个数是两视图可信背景数的上界，不能充当训练时的实际计数。
这只是训练图像机制诊断，不是新划分、AP、U-Recall或代表性泛化估计。

## 修改

v3将unknown/background挖掘的可信权重同时用于已知类别focal loss。
第5轮训练日志中93.2%的query没有可信发现标签，因而也失去了已知类负样本梯度。
尤其是未被Hungarian匹配选中的重复已知框，不应因此逃避标准DETR分类约束。

v4恢复可见已知类上的标准一对一focal监督，包括全部query以及辅助decoder层。
未匹配query是已知类别的负例，不等于其区域被标成无物体；未知物体存在性仍是另一项任务。
objectness、novelty、router和一致性继续使用原可信掩码，模糊query不会被新增成背景标签。
未来类别不参与分类监督，原型更新和GT过滤不变。

不修改hierarchical评分、Top50、warmup、阈值和蒸馏参数；没有利用测试结果挑选TopK。
因此输出竞争风险仍需在新训练过程中观察，本次修正不承诺AP恢复或增量性能提升。
新增日志：discovery_known_supervised_fraction应为1；discovery_background_fraction、
discovery_positive_fraction记录发现监督来源。discovery_ignored_fraction从v4起仅表示发现分支的忽略比例。
同时修正run_config中错误的unknown公式说明，使其与实际objectness * novelty一致。

DISCOVERY_VERSION升级为4，现有链路校验禁止从v3继续训练或跨版本进入Task2。
v3权重可做只读诊断，不能作为本次v4正式链的起点。control及历史文件保留。

## 服务器启动

先在运行旧实验的tmux中用Ctrl+C结束旧任务并确认两张卡释放，再拉取本地已提交推送的代码。
不要在旧进程运行时更新代码或让它自动进入后续任务。

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
git pull
python tools/owod/manage_data.py verify --benchmark s-owodb --output-root data/owod
# 仅在校验成功后执行：
bash tools/owod/run_discovery_full.sh --dry-run
bash tools/owod/run_discovery_full.sh
```

独立目录output/tree-detr-s-v4-full；日志output/tree-detr-s-v4-logs/console.log。
从DINO骨干预训练重新开始Task1；完整执行Task1至Task4及官方平衡微调。
每个increment为50轮，Task2至Task4的finetune为20轮；阶段内不降学习率。
两卡0/1、每卡batch2、seed42；配方名称不表示复现OW-DETR模型结果。
同一v4实验中断后使用bash tools/owod/run_discovery_full.sh --resume。
启动器不执行已有v3训练的停止、删除、迁移或自动恢复。

## 重复只读诊断

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python tools/owod/diagnose_discovery_task1.py \
  --checkpoint output/tree-detr-s-full/s-owodb/seed_42/stage_0/increment/checkpoint0004.pth \
  --manifest data/owod/s-owodb/split_manifest.json --images 32
```

示例使用空闲3090；先确认卡的可用性。工具检查manifest与annotation哈希以及checkpoint前后哈希，
不优化、不保存权重、不改数据、不做测试集AP选择。只读加载v3模型不需要解除正式续训的版本检查。
覆盖统计可能一个query覆盖多个GT，不是AP匹配。
