# 从头运行正式全量 Task 1 基础检测控制组

目的：确认基础检测训练能力，再决定项目增量方法的正式配置。不再扩大历史小集诊断。
使用 S-OWODB 作者 Task 1 清单与19类标签；schema-3 manifest及输入哈希校验。
不加载旧Task1、筛选、拟合或隔离实验权重。DINO ResNet-50仅初始化骨干；
检测头与其余未预训练组件从头训练。DINO加载器检查骨干缺失键，不能静默部分加载。

方法是本地 Deformable DETR control，不含 discovery、路由适配器和旧类蒸馏，
不是 ORE/OW-DETR 模型复现。沿用仓库已记录的 ow-detr-paper 训练参数：
50 epochs、AdamW、LR2e-4、骨干LR2e-5、weight decay1e-4；现启动器lr_drop=50，
因此本次50轮内无学习率下降。这是当前实现选择，不能宣称未核实的作者LR下降日程。
两张3090（PCI顺序0,1），每卡batch2，有效batch4，seed42，activation checkpointing开启。
官方评估每轮执行，保留Top-50和作者AP50约定，报告适用的unknown指标；
control未知分数是项目基线机制，不能当作有监督未知发现方法。

本地提交推送后，在服务器已激活tree-detr环境的tmux中：

```bash
cd /home/top/disks/new-hdd/zhy/Tree-DETR
git pull
python tools/owod/manage_data.py verify --benchmark s-owodb --output-root data/owod
bash tools/owod/run_task1_control.sh --dry-run
bash tools/owod/run_task1_control.sh
```

校验失败先修复，不跳过哈希。启动器只生成Task1一阶段，结束后不进入Task2或GNN。
输出：`output/task1-control-full/s-owodb/seed_42/stage_0/increment/`，含
run_config.json、metrics.jsonl、checkpoint.pth、training_complete.json。
完整控制台记录在 `output/task1-control-logs/console.log`，pipefail传播训练失败。
保留已有输出，不覆盖旧NaN模型；新输出目录重复执行需使用严格resume。

```bash
bash tools/owod/run_task1_control.sh --resume
```

resume只恢复当前新实验的同manifest/配置检查点；若出现NaN，先查日志，不盲目重启。
总loss非有限或梯度范数非有限立即终止，记录epoch与本rank图像ID；
不会跳过坏batch、把NaN替换为0或在异常梯度后继续optimizer.step。
这不是永不NaN的保证，matcher/前向计算更早报错仍以原始异常为准。

观察节点固定为首个epoch、epoch5/10、最终epoch50。前几个节点用于检查加载、
耗时、数值稳定性及明显实现异常，不利用正式测试集挑阈值、Top-K、最佳epoch或调参。
结果按最终epoch报告；仅因指标短期波动不追加实验组合。不要在没有首轮耗时前承诺总时长。
若基础训练仍明显异常，排查基础实现；若正常，冻结方法方案后开始正式增量实验。
control权重不能直接冒充discovery Task1权重，架构和方法链校验仍有效。
