# Stage 1 D1 / D2 / D3 实验操作

## 本轮目的

先测试附加损失和参数更新范围。D1 保留 rank-8 末两层 LoRA，把 local margin 和 off projection 系数设为零。D2 同样关闭两个损失，使用常规检测器微调。D2 也会开放旧分类器行和 box head，不是只改变矩阵秩。

两组直接复用 D0 已生成的训练标注和 GNN 邻域；起始模型和教师仍为 Stage 0。无需重新校准 GNN，也不会重新选回放图像。学习率 1e-4、20 轮、lr-drop 15、seed 42、两卡每卡 batch 2、10% 回放均沿用 D0。

## 文件与环境

将 `tools/owod/run_stage1_diagnostics.py` 同步到服务器同名目录。提供的 ZIP 中只包含这个新增脚本、其测试和本文，不包含模型代码或训练数据。ZIP 应解压在 Tree-DETR 根目录。

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
python -m zipfile -e stage1_diagnostics_tools.zip .
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

本次不只修改了启动器。服务器需要同步 `main.py`、`models/graph_local/lora.py`、`tools/owod/run_stage1_diagnostics.py`。本地 `exps/diagnostic_tools/stage1_d3_tools.zip` 包含上述三份代码、本操作说明和两份测试文件。将包传到服务器仓库根目录后执行以下命令；解压会更新包内列出的同名文件。

```bash
cd ~/disks/new-hdd/zhy/Tree-DETR
conda activate /home/top/disks/new-hdd/conda_envs/tree-detr
python -m zipfile -e stage1_d3_tools.zip .

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
