# PROB / CAT / OWOBJ 与 Tree-DETR 的统一 M-OWODB 对比

目标是让 PROB、CAT、OWOBJ 和你的 Tree-DETR/idea 在同一份官方 M-OWODB
四阶段标注、同一图像集合、同一 20/20/20/20 类别顺序和同一指标定义下比较。
PROB/OWOBJ 的官方代码仍保留为独立 checkout；这里只增加数据适配层，不把
它们改名成 Tree-DETR 模型。当前仓库内的 `pilot_unverified` manifest 不能用于
论文比较，必须先拿到官方标注并注册成 `official_annotations=true`。

## 已确认源码

| 方法 | 源码 | 当前 commit | 状态 |
| --- | --- | --- | --- |
| PROB | `https://github.com/orrzohar/PROB.git` | `b1d7fe68f5d55ffe8bc996ab15d7a260fee1e30a` | 可运行 |
| OWOBJ | `https://github.com/AI4Math-ShanZhang/OWOBJ.git` | `f5c583e39593168e2313c149ba69801d79619f42` | 可运行 |
| CAT | `https://github.com/xiaomabufei/CAT` | 无法获取，GitHub 返回 404 | 暂停 |

生成/核对源码 manifest（本地已有源码时不会改动源码目录）：

```bash
cd /path/to/Tree-DETR
python tools/owod/prepare_external_baselines.py \
  --methods prob owobj cat \
  --source-root baselines \
  --gpus 0,1 \
  --output exps/owod/m-owodb/order0/official/baselines/run_manifest.json
```

外部源码统一位于 `baselines/`。克隆主项目后先执行
`git submodule update --init --recursive`，不要在 `data/` 或 `exps/` 复制源码。

## 1. 注册官方 M-OWODB 数据

官方目录需要包含完整 COCO 训练/验证 JSON、图像和四个阶段文件。先在
Tree-DETR 中注册：

```bash
python tools/owod/prepare_protocol.py \
  --annotation-root "$PWD/data/coco-owod/m-owodb/official" \
  --protocol m-owodb \
  --source-reference "<official-release-url-or-commit>" \
  --output "$PWD/data/coco-owod/m-owodb/split_manifest.json"
```

转换器会拒绝 pilot/未验证 manifest，并从这份 manifest 的四个
`instances_train2017.json`/`instances_val2017_full.json` 读取图像选择；它不会
重新抽样、重排类别或生成新的划分。PROB/OWOBJ 使用 VOC XML 是序列化格式变化，
XML 中的框和类别来自同一份官方 COCO 标注。

```bash
python tools/owod/prepare_shared_mowodb.py \
  --manifest "$PWD/data/coco-owod/m-owodb/split_manifest.json" \
  --train-coco "$PWD/data/coco/annotations/instances_train2017.json" \
  --val-coco "$PWD/data/coco/annotations/instances_val2017.json" \
  --image-root "$PWD/data/coco/train2017" \
  --image-root "$PWD/data/coco/val2017" \
  --image-mode symlink \
  --output "$PWD/data/derived/m-owodb-voc" \
  --baseline-root "$PWD/baselines" \
  --clean
```

Windows 没有创建 symlink 的权限时，把 `--image-mode symlink` 改成
`--image-mode copy`。转换结果只生成一份，PROB 和 OWOBJ 都通过
`MOWODB_DATA_ROOT` 读取该目录。

## 2. 环境和数据

PROB/OWOBJ README 要求 Python 3.10.4、PyTorch 1.12.0+cu113，并需要编译
`models/ops`。建议每个方法使用独立环境：

```bash
conda create -n prob python=3.10.4 -y
conda create -n owobj python=3.10.4 -y

conda activate prob
cd /path/to/Tree-DETR/baselines/prob
pip install -r requirements.txt
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113
cd models/ops && sh make.sh

conda activate owobj
cd /path/to/Tree-DETR/baselines/owobj
pip install -r requirements.txt
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113
cd models/ops && sh make.sh
```

两个仓库都需要 DINO ResNet-50 权重 `dino_resnet50_pretrain.pth` 放到各自
`models/`。上一步生成的 `data/derived/m-owodb-voc/` 包含 `Annotations/`、`ImageSets/TOWOD/`
和 `ImageSets/OWDETR/`；两套目录写入相同 split，只是同时保留了 `owod_t1_*` 和
`t1_*` 两种作者命名。

OWOBJ 原始 commit 把 split、COCO JSON 和 HDF5 图像硬编码到作者服务器路径，
直接传 `--data_root` 不会使用共享数据。对已核验的 OWOBJ commit 应用项目中的
数据适配补丁（只替换数据后端，模型和损失不变）：

```bash
cd /path/to/Tree-DETR/baselines/owobj
git apply /path/to/Tree-DETR/baselines/patches/owobj_shared_mowodb.patch
```

补丁应用后应检查 `git diff --check`，并记录补丁文件的 SHA-256；重新 checkout
或换机器时必须重新应用同一补丁。

## 3. 训练和评估命令（tmux，GPU 0/1）

下面是作者提供的完整 M-OWODB 四阶段 recipe。配置脚本内部会依次运行
Task 1 到 Task 4，并包含 replay/fine-tuning 阶段。

```bash
# 先生成 shared 配置和命令 manifest
cd /path/to/Tree-DETR
python tools/owod/prepare_external_baselines.py \
  --methods prob owobj cat \
  --source-root baselines \
  --shared-mowodb \
  --gpus 0,1 \
  --output exps/owod/m-owodb/order0/official/baselines/run_manifest.json

# PROB：配置内部依次运行 t1、t2、t2_ft、t3、t3_ft、t4、t4_ft
tmux new -s mowodb-prob
conda activate prob
PROJECT_ROOT=/home/top/disks/new-hdd/zhy/Tree-DETR
export MOWODB_DATA_ROOT="$PROJECT_ROOT/data/derived/m-owodb-voc"
export MOWODB_OUTPUT_ROOT="$PROJECT_ROOT/exps/owod/m-owodb/order0/official/baselines"
mkdir -p "$MOWODB_OUTPUT_ROOT/prob/logs"
cd "$PROJECT_ROOT/baselines/prob"
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 \
  ./tools/run_dist_launch.sh 2 configs/M_OWOD_BENCHMARK_SHARED.sh \
  2>&1 | tee "$MOWODB_OUTPUT_ROOT/prob/logs/train.log"
# 按 Ctrl-b d 脱离，不要按 Ctrl-c

# 训练结束后重新连接同一个 session 执行评估
tmux attach -t mowodb-prob
CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 \
  ./tools/run_dist_launch.sh 2 configs/EVAL_M_OWOD_BENCHMARK_SHARED.sh \
  2>&1 | tee "$MOWODB_OUTPUT_ROOT/prob/logs/eval.log"

# OWOBJ：先应用上面的 shared adapter，再运行同样的四阶段 recipe
tmux new -s mowodb-owobj
conda activate owobj
PROJECT_ROOT=/home/top/disks/new-hdd/zhy/Tree-DETR
export MOWODB_DATA_ROOT="$PROJECT_ROOT/data/derived/m-owodb-voc"
export MOWODB_OUTPUT_ROOT="$PROJECT_ROOT/exps/owod/m-owodb/order0/official/baselines"
mkdir -p "$MOWODB_OUTPUT_ROOT/owobj/logs"
cd "$PROJECT_ROOT/baselines/owobj"
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 \
  ./tools/run_dist_launch.sh 2 configs/M_OWOD_BENCHMARK_SHARED.sh \
  2>&1 | tee "$MOWODB_OUTPUT_ROOT/owobj/logs/train.log"
# 按 Ctrl-b d 脱离

tmux attach -t mowodb-owobj
CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 \
  ./tools/run_dist_launch.sh 2 configs/EVAL_M_OWOD_BENCHMARK_SHARED.sh \
  2>&1 | tee "$MOWODB_OUTPUT_ROOT/owobj/logs/eval.log"
```

查看运行状态：

```bash
tmux ls
tmux attach -t mowodb-prob
tmux attach -t mowodb-owobj
```

不要同时启动 PROB 和 OWOBJ；两者都使用 GPU 0/1，应先完成一个方法再运行另一个。

本服务器按当前约定固定使用 GPU `0,1`；因此 `CUDA_VISIBLE_DEVICES=0,1`、
`GPUS_PER_NODE=2` 和 launcher 参数 `2` 必须保持一致，不要改成 `4`。

输出目录统一为根目录 `exps/owod/m-owodb/order0/official/baselines/`：PROB
位于 `prob/`，OWOBJ 位于 `owobj/`。checkpoint、指标和 `logs/` 不再写入外部
源码 checkout。

Tree-DETR/你的 idea 也必须使用同一 `split_manifest.json`、同一类别顺序和
`instances_val2017_full.json` 全类验证。运行你自己的四阶段训练入口时，固定
随机种子、GPU 数量、输入图像和输出的 Previous/Current/Known mAP、U-Rec、
H-Score；不要把 PROB/OWOBJ 的作者表格数字直接当作本地复现结果。

## CAT

CAT 的论文代码链接当前无法访问，因此 `prepare_external_baselines.py` 仍将官方
源码状态记录为 `pending_source`。项目另提供严格区分来源的论文级重实现，包含
共享 decoder 级联、自适应伪标签和 selective search 输入先验。实现假设、正式
M-OWODB 预处理以及 tmux 命令见 `docs/cat-ew-detr-mowodb.md`。结果必须标记为
`CAT (paper reimplementation)`，不能标记为作者官方实现结果。
