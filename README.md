# Tree-DETR

当前 OWOD 实验只采用两套作者协议：**M-OWODB（ORE，CVPR 2021）**和
**S-OWODB（OW-DETR，CVPR 2022 附录 A.2）**。更新于 2026-10-08。

[协议来源与评估定义](docs/owod-benchmarks.md) ·
[数据准备操作指南](docs/data-preparation.md) · [项目目录](docs/project-layout.md)。

- M 的新增类别数为 20/20/20/20；S 为 19/21/20/20。
- 固定作者类别顺序和 ImageSets；禁止自定义或随机重划分。
- 当前增量训练后单独进行作者清单上的均衡微调；禁止未来类别标签泄漏。
- 使用作者 VOC 评测约定计算 AP50、U-Recall、WI@0.8、A-OSE。
- 仅接受经过哈希和清单验证的 schema-3 manifest；旧 pilot 和 unverified 入口停用。

```bash
python tools/owod/manage_data.py plan
python tools/owod/manage_data.py check --config configs/data/owod.example.json
python tools/owod/manage_data.py prepare --config configs/data/owod.example.json
python tools/owod/manage_data.py verify --config configs/data/owod.example.json
python tools/owod/run_benchmark.py --help
```

数据准备入口不会改动原始图像，训练入口先完整检查数据与 checkpoint 来源。
原始数据路径在配置中指定；`check` 未通过时不得开始准备或训练。

本项目 detector 与研究方法仍属于本地实现；采用作者 benchmark 不代表复现了
ORE 或 OW-DETR 方法的论文数值。方法训练 recipe 必须明确记录。

旧 setting 的启动器、配置及说明文件已删除，清理记录见
[project-cleanup.json](docs/project-cleanup.json)。已有日志、checkpoint 和外部源码保留。
已有绘图代码位于 `tools/analysis/`，仅分析历史结果。

Git 只保存源码、运行配置、作者 ImageSets、必要说明和许可证。
PDF、图片、论文文稿、压缩包、权重、日志与临时输出均忽略；不要使用
`git add -f` 将这些生成产物加入仓库。
