# 实验数据准备

数据管理入口为 `tools/owod/manage_data.py`。它只按 `configs/owod/` 的
作者 ImageSets 选择数据，不生成随机划分，不根据类别 ID 大小重排任务。
两套基准分别准备、分别训练和报告。

## 1. 准备原始数据

| 基准 | 原始输入 | 原始输入目录 |
| --- | --- | --- |
| M-OWODB | ORE 作者发布的 VOC2007/2012 + COCO 合并版 | `Annotations/*.xml`、`JPEGImages/*.jpg` |
| S-OWODB | 完整 COCO2017 | `train2017/`、`val2017/`、`annotations/instances_train2017.json`、`annotations/instances_val2017.json` |

M 使用 [ORE 作者数据入口](https://drive.google.com/drive/folders/1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k)。
需要作者合并后的完整标注和图像；COCO2017 单独不足以构造 M 的 VOC Task 1。

S 从 COCO 官方获取
[train2017](https://images.cocodataset.org/zips/train2017.zip)、
[val2017](https://images.cocodataset.org/zips/val2017.zip)、
[2017 标注](https://images.cocodataset.org/annotations/annotations_trainval2017.zip)。
解压后保留官方目录结构。

默认布局如下；原始数据也可以保存在另一块磁盘上，修改配置中的路径即可。

```text
data/
  raw/
    ore/
      Annotations/
      JPEGImages/
    coco/
      annotations/
      train2017/
      val2017/
  owod/                    # 脚本生成的标注与 manifest
    m-owodb/
    s-owodb/
```

`configs/data/owod.example.json` 是路径模板。可以复制为本地配置后修改
`ore_root`、`coco_root`、`output_root`；相对路径按项目根目录解析，Windows
绝对路径建议写成 `D:/datasets/coco`。本地配置中的数据必须是真实原始数据。

## 2. 查看和生成划分

在项目根目录运行：

```bash
python tools/owod/manage_data.py plan
python tools/owod/manage_data.py check --config configs/data/owod.example.json
python tools/owod/manage_data.py prepare --config configs/data/owod.example.json
python tools/owod/manage_data.py verify --config configs/data/owod.example.json --decode-images
```

- `plan`：核验发布清单哈希，列出每任务的新类、已知类和图片数量，无需原始数据。
- `check`：检查两套基准需要的全部原始图像和标注路径，报告缺失数量及样例。
- `prepare`：转换标注、验证四任务协议后发布数据。转换失败不留下半成品任务目录。
- `verify`：检查全部阶段文件哈希、标签可见性、图片存在性及原始标注来源哈希。
  `--decode-images` 进一步验证图片解码。

只操作一套基准时加 `--benchmark m-owodb` 或 `--benchmark s-owodb`；默认两套。
也可以直接覆盖路径：

```bash
python tools/owod/manage_data.py prepare --benchmark m-owodb --ore-root D:/datasets/ORE --output-root data/owod
python tools/owod/manage_data.py prepare --benchmark s-owodb --coco-root D:/datasets/coco --output-root data/owod
```

所有命令可加 `--report 路径.json` 保存检查报告。缺数据、哈希变化或协议错误时
退出码为 1。再次 `prepare` 会验证并复用相同数据，不会重新抽样；旧协议或半成品
非空目录会被拒绝。原始图像不复制、不移动，两套基准引用配置中的原始图像。

## 3. 每个任务实际使用什么

| 项目 | M | S |
| --- | --- | --- |
| 每任务新增类别 | 20 / 20 / 20 / 20 | 19 / 21 / 20 / 20 |
| 增量训练清单图片数 | 16551 / 45520 / 39402 / 40260 | 89490 / 55870 / 39402 / 38903 |
| Task 2–4 均衡微调清单图片数 | 1743 / 2361 / 2749 | 1679 / 2345 / 2664 |
| 发布测试清单 | 10246 行，9119 张唯一图片 | 4952 张图片 |

每个基准生成 `stage_0` 至 `stage_3`，对应 Task 1–4。每阶段文件为：

| 文件 | 用途与可见标签 |
| --- | --- |
| `instances_increment_train2017.json` | 作者本任务训练图片，仅本任务新类标签 |
| `instances_train2017.json` | Task 1 为首任务训练集；Task 2–4 为作者均衡微调集，含全部已知类标签 |
| `instances_val2017.json` | 同一作者测试图片，仅已知类标注的视图 |
| `instances_val2017_full.json` | 同一作者测试图片、完整 80 类标注，用于开放世界评估 |
| `split_manifest.json`（基准根目录） | 协议、类别顺序、作者测试序列、阶段路径、数据来源与哈希 |

任务间的作者图片重叠保留。M 的作者测试清单重复行也保留于 manifest，评估按
原始次数处理，JSON 图片记录本身保持唯一。未知类标签不进入增量/均衡训练，
测试时所有未来类合并为 unknown；Task 4 不报告 unknown 指标。

文件名里的 `val` 是适配现有 loader 的命名，指作者测试集，不能用来训练、
拟合未知阈值或选择超参数。ORE 的 1000 张 held-out 验证及 EBUI 属于方法复现；
没有作者清单时，本工具不会自行添加这种划分。

## 4. 开始实验

训练使用 `data/owod/<benchmark>/split_manifest.json`。标准训练入口
`tools/owod/run_benchmark.py` 执行 Task 1 增量训练，然后依次执行 Task 2–4
的增量训练和均衡微调；方法 recipe、DINO 预训练和评估约定见
[协议文档](owod-benchmarks.md)。

当前完成的是数据协议与本地 detector 运行支持；没有真实数据准备/训练记录时，
不得报告已复现 ORE 或 OW-DETR 论文数值。
