# 官方 OWOD baseline 方案

更新日期：2026-09-06。

## 实验定位

论文主结果表限定为 DETR 系列检测框架下的 OWOD 方法，优先选择 Deformable DETR 实现。DETR 是检测框架；backbone 指 ResNet、Swin、ViT 等特征提取网络，仍需单独对齐或披露。Cosine、Random-K、Global Replay、普通 Deformable DETR control 和 D1/D2 只属于内部对照/诊断；完整 GNN 三模块是待评估方法本身。

按当前选型决定，ORE、OrthogonalDet 和 DEUS 不进入正式 baseline 名单或复现队列。历史文献表与协议来源记录保留，不代表这些方法仍是实验目标。结论范围相应限定为所比较的 DETR 系列方法，不能据此宣称超过所有 OWOD 方法。

不能通过在本项目中添加简化 objectness、能量阈值或损失分支，再把实验命名为 PROB、OW-DETR、ORE 来代替官方实现。旧 `prob`、`ow_detr`、`ore_star` 目录仅视为历史本地实验，除非能核实作者代码、核心方法、完整配置、数据和结果来源。

## 已核实的作者仓库

2026-09-06 通过 GitHub API 读取了以下仓库元信息和 README，核实的是代码入口与公开说明，尚未安装或运行这些外部实现，也未验证链接中的权重是否可下载。

| 优先级 | 方法 | 官方实现 | 选择原因与已核实条件 |
| --- | --- | --- | --- |
| 1 | PROB，CVPR 2023 | https://github.com/orrzohar/PROB | README 明确为官方实现，基于 Deformable DETR，具有概率 objectness；提供 M/S-OWODB 训练/评估配置以及预训练权重链接。适合先打通官方评估，再做训练复现。 |
| 2 | OW-DETR，CVPR 2022 | https://github.com/akshitac8/OW-DETR | README 明确为官方实现，具有 attention-driven pseudo-labeling、novelty classification、objectness scoring；与当前检测器同属 Deformable DETR 系列。 |

ORE 作者仓库 https://github.com/JosephKJ/OWOD 曾核实，仅作历史记录，已退出当前复现队列。

## 扩展候选与准入条件

| 方法 | 当前定位 | 加入正式实验前需要完成的核实 |
| --- | --- | --- |
| CAT | DETR 系列候选 | 核实作者实现、检测器配置、backbone、预训练及 M/S-OWODB 协议。 |
| OWOBJ | 较新方法候选 | 核实作者实现的 DETR 版本、权重、完整增量训练配置及评估口径。 |
| O1O | 较新方法候选 | 核实作者实现是否满足当前 DETR 范围、训练流程及修正后的标注版本。 |
| EW-DETR（D-DETR 版本） | 低秩增量学习专项候选 | 用户提供原文确认了 Deformable DETR 版本；官方代码仍待核实，M/S-OWODB 适配结果不能称为原论文协议复现。 |

这是六个方法的候选池，不是六个已完成复现的 baseline。近期候选通过准入检查后再确定主表，不能仅按数量或缩写纳入。

EW-DETR 原文研究无旧样本回放、类别与域共同变化的 EWOD，使用 Pascal 跨域和天气变化序列。其论文 AP/FOGS 不能直接并入 M/S-OWODB 主表；若适配到相同数据协议，需要披露零回放与本方法回放预算的差异。只考虑其 D-DETR 版本作为当前专项对照，RF-DETR 版本暂不纳入。

## 复现顺序

1. 确定最终 M-OWODB/S-OWODB 版本、任务类别顺序和图像/标注文件。记录每个文件的 SHA-256，核对重复图像问题。当前 `unverified_pilot` 不作为论文正式分数。
2. 单独建立 PROB 官方代码目录和 conda 环境，记录所用 commit、实际依赖与必要兼容补丁。不得替换当前正在使用的 tree-detr 环境或覆盖现有实验。
3. 优先运行作者提供 checkpoint 的 Task 1 评估，核对数据和评估定义；成功后运行其官方 Task 1 训练及 Task 1 到 Task 2 的增量流程。checkpoint 评估复现与从训练开始的复现要分别标记。
4. 用相同基准版本复现 OW-DETR。每个方法保留自身定义、推荐训练配方和自己的前一任务 checkpoint，不能统一改成当前 20 轮 pilot，也不能给所有方法套用我们训练出的 Stage 0 模型。
5. 核实 CAT、OWOBJ、O1O，加入符合 DETR 范围和同一基准协议的方法；单独评估 EW-DETR（D-DETR）的适配可行性。对主方法开展完整四任务评估。Task 1 对应本项目的 stage 0，Task 2 对应 stage 1，禁止错位比较。

官方原配方复现与预算匹配实验分别报告。主比较需要透明记录预训练、backbone、外部数据、训练轮数、有效 batch、回放预算及评估设定；如果为公平控制修改某方法配方，明确标为受控重跑，不能宣称精确复现其论文数字。

## 实现条件与风险

- PROB README 使用 Python 3.10.4、PyTorch 1.12 / CUDA 11.3，并提供 3090 的用户配置记录。这只是兼容性起点，不能保证本服务器直接达到论文精度。
- OW-DETR README 的原环境为 Python 3.7、PyTorch 1.8、CUDA 10.2，原训练脚本为 8 卡；服务器使用 3090 时需要核实 Ampere/CUDA 与自定义算子兼容性，保留所有兼容补丁记录。
- 两者 README 都使用 DINO 自监督 ResNet-50 backbone。当前本地 pilot 的初始化不同，需明确预训练来源；方法效果和预训练收益不能混为一谈。
- OW-DETR 和 PROB 的数据加载/评估采用 VOC XML 与 ImageSets 格式。数据可进行等价格式转换或建立独立数据视图，但不能更改图像划分、类别映射、标注内容、困难样本/crowd 处理而不记录。
- PROB README 列出的历史 M-OWODB 分数与本仓库转录的 DEUS Table 1 修正版分数不同。修正过重复标注的数据重跑不能直接以旧 README 分数作为必须达到的阈值，必须注明所用版本。
- 当前项目的简化 UDP/A-OSE/WI 和 COCO-style AP50 不能直接假定与作者 VOC evaluator 等价。先核对 AP 插值、max detections、unknown 匹配、置信度过滤和 ignore 规则，再比较数值。

## 表格与来源标记

论文主表至少记录：方法、来源（作者 checkpoint 评估/训练复现/文献引用）、代码 commit、数据版本、backbone/预训练、Task、Previous/Current/Known AP50、U-Recall、H-Score，以及标准实现下的 WI/A-OSE（若报告）。H-Score 可在一致的 Known AP50 和 U-Recall 上重算。

文献引用分数必须带来源，不得放进“本机复现”一栏。内部消融另表记录 Full、w/o GNN、w/o LoRA、w/o graph-conditioned objective；cosine/random/global 和 D1/D2 根据诊断用途单独标明。

## 当前进度

已完成：将正式比较限定为 DETR 系列；核实 PROB、OW-DETR 作者仓库；阅读 EW-DETR 原文并确认 D-DETR 版本与协议差异；制定 PROB 优先的复现顺序。

未完成：扩展候选作者实现核实、官方数据版本验收、外部仓库固定版本安装、官方权重下载与评估、外部方法训练复现。因此目前不存在可以宣称已经跑出的官方 baseline 分数。

现有 D1/D2 仍可用于查找我们方法的新类学习瓶颈，但它们不能支撑“优于已有 OWOD 方法”的结论。本次方案调整不自动停止服务器正在运行的诊断实验。
