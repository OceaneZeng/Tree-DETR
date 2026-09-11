# Stage 1 D1 / D2 / D3 / D4 实验操作

## 2026-09-10：后续采用 D2

用户提供的第 20 轮结果：D2 Previous/Current/Known/H 为
53.441/56.969/55.205/41.076，D4 为 52.890/52.776/52.833/41.736。
D2 的 Known AP50 高 2.372 个百分点，D4 的 H 高 0.660；选择 D2
作为已知类检测优先的工作配方，不能表述为所有指标都更好。

当前决策取代此前 backbone 全冻结的限制：沿用 D2 的部分 backbone
更新策略（ResNet layer2/3/4）、其余检测器常规微调，保留 GNN 回放和
teacher completion，关闭 LoRA、margin、projection、distillation。
不重新训练已完成的 D2 Stage 1；Stage 2 从其最终 checkpoint 开始，
Stage 3 从这次 Stage 2 的最终 checkpoint 开始。GNN 使用原有校准权重，
每阶段从当时的检测器重新提取训练集梯度并重新选择旧类邻域。

新增 `tools/owod/run_d2_continuation.py` 从服务器 D2 和原 D0 pilot
的记录读取参数、GNN 路径及回放配额，检查前一阶段完成后再启动。
预检不提取梯度、不启动训练、不创建输出。同步代码后，确认 EW 队列
和训练已停止、GPU 0/1 可用，再执行：

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
if python tools/owod/run_d2_continuation.py --stage 2 --dry-run
then
  nohup python -u tools/owod/run_d2_continuation.py --stage 2 \
    > d2_stage2_launcher.log 2>&1 < /dev/null &
  echo "launcher PID: $!"
fi
```

```bash
tail -n 60 d2_stage2_launcher.log
python tools/owod/run_d2_continuation.py --summarize
```

Stage 2 完成并检查结果后，同样启动 Stage 3：

```bash
if python tools/owod/run_d2_continuation.py --stage 3 --dry-run
then
  nohup python -u tools/owod/run_d2_continuation.py --stage 3 \
    > d2_stage3_launcher.log 2>&1 < /dev/null &
  echo "launcher PID: $!"
fi
```

输出为 `exps/owod/m-owodb/order0/pilot_unverified/d2_continual_v1/stage_2/graph`
和 `stage_3/graph`，训练日志为 `train.log`，指标为 `metrics.jsonl`。
中断后使用对应命令加 `--resume`，必须有当前阶段 checkpoint、原始图和标注；
已完成的阶段不会重跑。不要同时重复启动同一个阶段。

保持原有每类配额 10、风险类额外 40、K=5 时，名义图像配额总和为
Stage 1/2/3 的 400/600/800，去重后实际图像数可能更少。
这不是固定总 memory 的实验。现有 runner 从 manifest 的前一阶段
train 文件选择回放并计算旧类梯度，并非严格只访问上一阶段实际保留
的有限 memory。继续这条链用于内部验证；论文比较前必须统一并审计
候选池、存储预算和训练时可访问的旧图像，不能直接与在线固定 398 张
memory 的 OW-DETR 适配结果作公平性结论。

实验顺序：先完成 D2 的 Stage 2/3；再在共同 Stage 0 初始化、同一
训练预算下做 Stage 1 的 GNN/随机/均匀分配对照与 teacher 去除对照；
严格匹配总回放数量及采样比例，不同时改学习率或增加损失。最终候选
至少补 3 个种子，报告平均值和波动。已完成的 OW-DETR 使用 D4 冻结
设置，保留为旧设置结果；它不能直接证明采用 D2 更新策略的方法更优。

## 本轮目的

先测试附加损失和参数更新范围。D1 保留 rank-8 末两层 LoRA，把 local margin 和 off projection 系数设为零。D2 同样关闭两个损失，使用常规检测器微调。D2 也会开放旧分类器行和 box head，不是只改变矩阵秩。

两组直接复用 D0 已生成的训练标注和 GNN 邻域；起始模型和教师仍为 Stage 0。无需重新校准 GNN，也不会重新选回放图像。学习率 1e-4、20 轮、lr-drop 15、seed 42、两卡每卡 batch 2、10% 回放均沿用 D0。

## 文件与环境

服务器需同步 `tools/owod/run_stage1_diagnostics.py`、其测试和本文对应的当前版本。
项目不再保存发布 ZIP，也不包含模型代码或训练数据的复制包。

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
python tools/owod/run_stage1_diagnostics.py --dry-run
```

默认读取 `exps/owod/m-owodb/order0/pilot_unverified/full_three_module_stage1_v1`。检查原实验的配置、完成标记、训练标注、验证标注、manifest、Stage 0 权重必须仍存在。dry-run 会读取和计算输入哈希，但不会训练或创建实验输出。

## 启动

确认 GPU 0、1 可用后，一次启动两组，脚本按 D1、D2 顺序运行，不会同时抢占两张卡。

```bash
nohup setsid python -u tools/owod/run_stage1_diagnostics.py --experiment both \
  > stage1_diagnostics_launcher.log 2>&1 < /dev/null &
echo "launcher PID: $!"
```

想先只跑 D1，则将 `--experiment both` 改为 `--experiment d1`。D1 成功后再次运行 both 会核对计划并跳过已完成的 D1。不要在原进程仍运行时再次启动或使用 resume。

## 查看进度与比较结果

```bash
tail -n 40 stage1_diagnostics_launcher.log
python tools/owod/run_stage1_diagnostics.py --summarize
```

汇总随时可以运行。第 5 轮评估完成之前，D1/D2 暂无 AP 行是正常现象。输出含轮数、Previous/Current/Known AP50、H-Score，以及同轮数相对 D0 的 AP 差值，均以百分点展示。

默认输出目录为 `exps/owod/m-owodb/order0/pilot_unverified/stage1_diagnostics_v1/`，两个子目录分别是 `d1_lora_no_extra_losses` 和 `d2_full_finetune_no_extra_losses`。每组包含：

- `diagnostic_plan.json`：启动命令、输入 SHA-256、复用的 GNN 邻居。
- `graph/train.log`：训练与评估输出，以及分布式进程启动错误。
- `graph/metrics.jsonl`：每轮完整指标。
- `graph/run_config.json`：实际配置。
- `graph/checkpoint.pth`：该组自身的续跑权重。
- `graph/training_complete.json`：正常完成标记。

## 中断续跑

确认旧进程已经结束后执行：

```bash
nohup setsid python -u tools/owod/run_stage1_diagnostics.py --experiment both --resume \
  >> stage1_diagnostics_launcher.log 2>&1 < /dev/null &
```

脚本跳过已完成的实验，恢复中断组自己的 checkpoint；教师不会变成中断组的模型。输入内容或计划改变时会拒绝续跑。若在产生第一个 checkpoint 前启动失败，应排除错误，再使用新的 `--output-dir`；不要把 D0 权重当作 D1/D2 的 resume 权重。

如果日志出现 `Received Signals.SIGHUP`，表示进程收到外部挂断信号，具体来源可能是终端/SSH 会话结束或外部发信号，日志本身不能唯一确定原因。Torchrun 会安装自己的信号处理器，单独 nohup 并不总能隔离终端会话的影响。上面的 setsid 为整个启动器建立独立会话；新版脚本也为训练子进程创建独立 POSIX 会话。这不屏蔽管理员主动终止进程或系统资源策略。

每个完整 epoch 结束后保存 checkpoint，轮内中断需要重跑当前轮。原有 checkpoint 不记录所有随机数状态，续跑不会逐步复现一条未中断轨迹；实验记录应注明中断轮数。若运行在非默认目录（例如 `stage1_diagnostics_v2`），启动、续跑和汇总必须都传入同一个 `--output-dir`。

## 把结果发回来

优先粘贴 `--summarize` 输出。第 5/10 轮就可以做第一轮判断，但不要因为一两次小幅波动得出最终方法结论。

需要完整分析时，打包新实验的轻量日志：

```bash
python - <<'PY'
from pathlib import Path
import zipfile
root = Path('exps/owod/m-owodb/order0/pilot_unverified/stage1_diagnostics_v1')
names = {'diagnostic_plan.json', 'metrics.jsonl', 'run_config.json', 'training_complete.json'}
with zipfile.ZipFile('stage1_diagnostics_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name in names and 'run_history' not in p.parts:
            z.write(p, p.relative_to(root))
print('stage1_diagnostics_results.zip')
PY
```

如果进程报错，再附 `graph/train.log` 或 launcher 日志最后 80 行；上述结果包不包含文字错误输出。

## 决策依据

D0 第 5/10 轮新类 AP50 为 12.17%/13.64%，旧类为 50.23%/50.48%。

- D1 明显改善：分别加回 margin、projection，定位具体损失。
- D1 仍低、D2 明显改善：优先调整 LoRA 适配范围，先 rank 16，再末两层与末四层。
- 两组都低：检查标注、伪标签与新类初始化，之后单独比较 lr 1e-4/2e-4。

确认适配方案后，再开展同更新范围、同预算、同曝光下的 GNN/cosine/random 对照。历史 cosine 的 55.42% 新类 AP 仅作参考，不能代替这组受控实验。

## 2026-09-07 更新：D3 扩大 LoRA 范围并开放检测头

根据 D2 最终结果和当前研究约束，下一轮保留 GNN 和 LoRA，backbone 全冻结，不增加新的 adapter 结构。此前建议的 rank-16 或新 adapter 位置实验暂不执行。D3 是新的训练组，不是从已完成的 D1/D2 继续训练。

| 设置 | D1 | D3 |
| --- | --- | --- |
| LoRA 位置 | Decoder 第 5、6 层 FFN 的 linear1/linear2 | Decoder 全部 6 层 FFN 的 linear1/linear2 |
| LoRA rank | 8 | 8 |
| 分类头 | 仅当前新类行 | 完整分类头可训练，包括旧类行 |
| 定位头 | 冻结 | 可训练 |
| Backbone / Encoder / 原 attention / 原 FFN | 冻结 | 冻结 |
| Margin / projection | 0 / 0 | 0 / 0 |
| 分类头 weight decay | 0 | 0，保持与 D1 一致 |

D3 复用 D0 的原始训练标注、GNN 邻域、Stage 0 初始化及教师、回放曝光和 20 轮计划。不会因 LoRA 覆盖层数变化而重新计算 GNN 或保护基；这是为了固定选择策略，只测试适配范围。确定最终适配策略后，再研究与之对应的 GNN 校准。

本地按当前 6 层、hidden_dim=256、91 槽位、共享检测头的真实模型核对：LoRA 可训练参数 122,880，分类头 23,387，定位头 132,612，总计 278,879。服务器日志以实际参数为准。启动时应看到 `decoder_layers: 6`、`wrapped_linears: 12`、`classifier_update_scope: all_rows`、`box_head_trainable: true`，以及完整可训练参数名列表；其中不应有 backbone、Encoder 或原 attention/FFN 权重。

D3 同时扩大 LoRA 深度和开放检测头，是适配方案的整体实验。若改善，不能把全部提升归因于 LoRA 层数；若需要论文中的独立贡献，再分别拆分。GNN、回放与教师补标仍保留，当前组不验证 GNN 的独立收益。

### 同步与启动

本次不只修改了启动器。服务器需要同步 `main.py`、`models/graph_local/lora.py`、
`tools/owod/run_stage1_diagnostics.py` 及相关测试。项目不再保存发布 ZIP；使用
Git 同步当前仓库，或按这些路径逐个同步文件，不要把实验输出打包进源码目录。

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr

OUT="$PWD/exps/owod/m-owodb/order0/pilot_unverified/stage1_diagnostics_v2"
if python tools/owod/run_stage1_diagnostics.py \
  --output-dir "$OUT" --experiment d3 --dry-run
then
  nohup setsid python -u tools/owod/run_stage1_diagnostics.py \
    --output-dir "$OUT" --experiment d3 \
    > stage1_d3_launcher.log 2>&1 < /dev/null &
  echo "launcher PID: $!"
fi
```

继续使用 v2 根目录，使汇总可以同时读取已有 D1/D2。D3 写入新的 `d3_lora_all_decoder_train_detection_heads/graph` 子目录，不覆盖 D1/D2。`--experiment both` 仍然只运行 D1/D2，不会自动启动 D3。默认 GPU 仍为 0、1，两卡每卡 batch 2。

```bash
tail -n 40 stage1_d3_launcher.log
python tools/owod/run_stage1_diagnostics.py \
  --output-dir "$PWD/exps/owod/m-owodb/order0/pilot_unverified/stage1_diagnostics_v2" \
  --summarize
```

中断后先确认原进程结束；只有 D3 已保存自己的 checkpoint 时，才以相同命令加 `--resume` 续跑，并将日志重定向改为 `>> stage1_d3_launcher.log`。

第 20 轮参考结果（AP50 / H，单位 %）：

| 组 | Previous | Current | Known | H |
| --- | ---: | ---: | ---: | ---: |
| D0 | 50.401 | 14.317 | 32.359 | 31.011 |
| D1 | 50.438 | 14.152 | 32.295 | 30.920 |
| D2 | 53.441 | 56.969 | 55.205 | 41.076 |

先关注 D3 能否明显改善 D1 的新类 AP，以及距 D2 还有多少差距。第 5/10 轮仅作进度诊断，最终判断完成 20 轮后再做；本次 D2 在降学习率后的最后五轮仍有明显改善。

本地验证命令：

```bash
python -m unittest tools.owod.tests.test_stage1_diagnostics models.graph_local.tests.test_lora_heads
```

## D4：停止扩展 LoRA，采用冻结 backbone 的常规微调

D3 第 5 轮结果为 Previous 50.988、Current 13.680、Known 32.334、H 31.604（%）。用户据此决定后续方法保留 GNN，舍弃 LoRA，backbone 全部冻结，其余检测器参数直接微调。D3 仅有已收到的第 5 轮结果，不标记为完整训练完成；保留其输出和权重。

D4 相比 D2 仅将 `--lr_backbone` 设为 `0`，并清理无效的 LoRA 配置项。现有 `build_backbone` 在此设置下会将所有 backbone 参数的 `requires_grad` 设为 `False`；不仅是把该参数组学习率设成零。ResNet 使用 FrozenBatchNorm，统计量也保持固定。Encoder、Decoder（含 attention 和 FFN）、input projection、query embedding、分类头和定位头均按原常规微调路径训练，不添加 adapter、不施加分类行掩码。

| 设置 | D2 | D4 |
| --- | --- | --- |
| Backbone | 默认训练 layer2/3/4，浅层冻结 | 全部冻结 |
| 其余检测器可训练参数 | 常规微调 | 常规微调 |
| LoRA | 无 | 无 |
| GNN 邻域 / 训练标注 / 回放 / 教师 | 复用 D0 | 同左 |
| 主学习率 / 分类头 weight decay | 1e-4 / 1e-4 | 同左 |
| 训练计划 | 20 轮，第 15 轮后降学习率 | 同左 |
| Margin / projection / distillation | 全部关闭 | 同左 |

保留 D0/D1/D2/D3 作为历史结果；新方法中的第二部分可描述为 `Graph-Guided Detector Adaptation`，而非 LoRA Adapter。GNN 通过回放和训练监督影响适配，不在前向过程中路由参数，也没有在线参与检测器优化。常规微调本身不应表述为新的独立算法，方法贡献需围绕 GNN 及其对持续学习的作用验证。原 LoRA 参数投影项在 D4 中不适用。

### 同步与启动 D4

服务器需先同步当前仓库中的 `main.py`、诊断启动器、已有 LoRA 依赖和相关测试。
项目不再保存发布 ZIP，也不把实验数据或权重复制到源码目录。D3 若仍占用 GPU 0、1，
应先等待其退出或单独结束 D3，再启动 D4；当前助手没有远程执行停止或启动操作。

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr

OUT="$PWD/exps/owod/m-owodb/order0/pilot_unverified/stage1_diagnostics_v2"
if python tools/owod/run_stage1_diagnostics.py \
  --output-dir "$OUT" --experiment d4 --dry-run
then
  nohup setsid python -u tools/owod/run_stage1_diagnostics.py \
    --output-dir "$OUT" --experiment d4 \
    > stage1_d4_launcher.log 2>&1 < /dev/null &
  echo "launcher PID: $!"
fi
```

D4 从原 Stage 0 初始化，不从 D3 checkpoint 接着训练；输出为 v2 下的 `d4_frozen_backbone_finetune/graph/`，与所有历史组隔离。`--experiment both` 仍然只指 D1/D2，不会启动 D3/D4。

```bash
tail -n 40 stage1_d4_launcher.log
python tools/owod/run_stage1_diagnostics.py \
  --output-dir "$PWD/exps/owod/m-owodb/order0/pilot_unverified/stage1_diagnostics_v2" \
  --summarize
```

启动日志应显示 `Frozen-backbone update policy` 中的 `backbone_trainable_parameters: 0` 和 `lora_enabled: false`。当前模型本地实测非 backbone 可训练参数为 16,614,753，其中 Encoder 4,541,184，Decoder 6,123,264，input projection 5,639,168，query embedding 153,600，分类头 23,387，定位头 132,612，reference points 投影 514。参数审计确认不存在 LoRA，所有非 backbone 参数均可训练；一次优化器更新后，全部 backbone 权重保持逐元素相同。尚未在服务器执行 D4 训练。

训练中断后，确认原进程已结束且 D4 存在自己的 checkpoint，再使用同一 D4 命令增加 `--resume`；日志重定向改为追加 `>>`。

D4 第 5 轮先与 D2 的 48.096 / 49.038 / 48.567（Previous / Current / Known，%）比较。主判断是：仅冻结 backbone 后，能保留多少 D2 的新类学习能力和旧类表现。不能在 D4 尚无结果时声称其优于 D2。
