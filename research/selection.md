# Tree-DETR 历史对话蒸馏与规则取舍

整理日期：2026-10-28（Asia/Shanghai）。整理时仓库为 `main@f52ea01`。本文件保存证据和阶段状态，不作为新的实验配置；运行状态不能从这里推定。只修改本项目 `AGENTS.md` 和本文件，不改业务代码、不启动训练、不创建子代理。

## 读取方法与范围

先读取 `C:/Users/23642/.codex/AGENTS.md`、项目根 `AGENTS.md`，检查 `C:/AGENTS.md`、`C:/programlearning/AGENTS.md`、用户目录规则和项目隐藏目录中的规则。祖先路径未发现额外规则，项目未发现子目录 `AGENTS.md`，因此无子目录冲突需要修改。全局已有中文沟通、完整命令、讨论/实施边界、有限检查、保留用户改动等通用要求，本次不重复写入或改动全局文件。

历史读取不是按标题或摘要推断：

- 使用 `list_projects` 确认当前项目 ID：`82892def-9c85-4b02-ae09-613b46dba161`；使用 `list_threads` 辅助定位，列表最多返回 50 条，未将其当成全量范围。
- 以只读 SQLite 连接读取 `C:/Users/23642/.codex/state_5.sqlite` 的线程归属，以及 `thread_history_1.sqlite` 的 `thread_items.item_json`。从实际 `userMessage` 提取原文，按候选事项回读 `agentMessage`，区分建议、接受、纠正和后续决定。数据库初始含 223 条线程元数据、285 个有聊天条目的线程上下文。
- 扫描本地 `sessions/` 的全部 93 个 JSONL 及 `archived_sessions/`；后者无现存 JSONL。读取当前项目可用 JSONL，补充数据库没有的新用户消息；不能把缺失 JSONL 误报为整段历史不可读。
- 候选范围含 84 个有当前项目 cwd 的非子代理上下文、52 个通过用户原文提及项目而找到的关联/继承上下文，以及用户明确引用的 `01a01d69-f22c-7640-924e-2232c91ea179`，共 **137 个上下文**，含本次请求。原始用户记录共 **1,309 条**：初次提取 1,169 条，分页补回 139 条，引用链补回 1 条。这是未跨继承分支和多源去重的记录数，不是独立对话/证据数量；相同消息、相同文本和继承副本不重复支持一条规则。
- `01a14ee2-bccc-79b0-85bd-099a8fa76540` 和 `01a06b9e-7e73-7622-b436-f1a8bd40dc13` 初次数据库提取无用户正文；实际调用 `read_thread` 分别读完 4 页/41 条、10 页/98 条用户消息，均到 `hasMore=false`。明确引用的 `01a01d69-…` 也用 `read_thread` 读到末页。
- 读取用户粘贴消息指向的 **90 份可访问 `pasted-text.txt`**（0 份缺失），用于补全被移入附件的日志和文本；工具输出、论文引文及图像 prompt 不自动视为用户长期要求。截图、PDF、压缩包本体未逐一重审，未从其不可见内容提取规则。
- 排除元数据明确标记为 subagent/guardian 的 16 个当前项目上下文；未注册的继承片段只辅助回溯，不能因同一话在多个片段出现而增加证据次数。

归档与覆盖限制：

- `list_archived_threads(source=codex)` 返回 `threads=[]`、无下一页；进一步核对本地状态库发现唯一归档 `019eb173-ac04-7e93-88b7-e0888f374cef`。读取数据库中全部 3 条原始用户消息，内容为 Song 的增量检测论文、ETM/EAM 及创新性分析，无当前项目关联，排除；其 JSONL 已缺失。没有发现可读取的 Tree-DETR 归档正文。
- `list_archived_threads(source=chatgpt)` 返回当前账户不可用，无法覆盖该账户之外、云端 ChatGPT 或其他机器上的历史。未恢复归档、未向其他聊天发消息。
- 全量 JSONL 元数据扫描额外找到 `01a1340c-55f6-7ec0-9bf4-eff82f0862b2`：起初 cwd 为 Tree-DETR，用户原文实际是史莱姆/骷髅兵像素画，随后 cwd 转到 `C:/game/demo`；读过原文后排除，未据目录或标题强行归入本项目。
- 用户原文涉及服务器、AutoDL 和 Obsidian 时，只抽取与 Tree-DETR 明确相关的决定；不将其他库、论文写作任务和临时授权扩展成全局约定。本次没有连接训练服务器，所以不能确认远端当前进程、权重、成绩和运行配置。
- 本地主要项目历史覆盖 2026-07-27 至 2026-10-28；更早被删除、未同步、未缓存、无权限的正文无法恢复。读取到的历史不代表覆盖所有曾发生过的聊天。

## 原始证据与规则候选

定位方式：`线程 ID @ordinal` 对应 SQLite 的 `rollout_ordinal`；`JSONL L数字` 对应该线程索引所指 JSONL 的实际行。消息 ID 在需要排除继承重复时使用。置信度表示“用户确实表达了该要求”，不是实验效果可信度。P0 指本次用户显式提供且原文件已存在的 2026-10-08 长期项目政策；其中技术细节另以当前协议文档核对。

| 编号 / 候选 | 原始用户证据（短摘录与定位） | 类型、重复/长期依据 | 取舍与置信度 |
| --- | --- | --- | --- |
| E01 作者 M/S 协议取代自定义 setting | `01a11a16-b107-7212-9e12-ff40ad7dd34e @51`：“以后的实验严格按照这两个标准执行”；`@123`：“清楚以前的旧实验setting”；`@171` 用户答案：“两个基准都保留，S-OWODB 补充采用 OW-DETR 原始定义” | 明确长期决定；后续答案修正“ORE 同时定义 M/S”的误解 | 保留并中文压缩，S 的来源不再归给 ORE；高 |
| E02 数据、manifest、阶段与评测技术边界 | P0；E01 的长期标准要求；`01a11a16-… @1148` 对“四任务/七训练阶段”提出核对 | 现存显式长期政策，不把助手推演伪装成新用户引文 | 保留 schema-3、哈希、发布记忆、无未来监督、作者 AP、Task 4 边界；细节链接协议说明；高 |
| E03 每次实验持久化日志 | `01a0322f-fd33-77f2-93d0-f4ad6a20b3a6 @366`：“后面的每一个实验都会这样要求”；`01a0334a-e5de-7ae0-87ae-b3266af219e4 @1179`：“不管是跑baseline还是自己idea的实验，都需要记录log文件！”；`01a15760-808c-79c3-b6e8-4801b8a9d451 @280`：“log不会写入console.log了吗” | 多次且明确长期 | 新增，覆盖 baseline/主方法/诊断，沿用本次启动器路径；高 |
| E04 本地改代码，用户 Git 同步服务器 | `01a02cac-74db-78e0-82eb-25e0b9d865c9 @649`：“在本地仓库修改吧，我自己同步到远程”；`01a13f90-108c-73d0-9e44-44745244c8a9 @163`：“修改本地代码，我自己通过git上传到服务器”；`01a18282-1902-7ef0-9f42-e3dbd30009fe @101`：“可以在服务器上进行相关测试，但是不能直接修改服务器上的代码” | 跨阶段重复，最新明确实施边界 | 新增。测试许可属于具体请求，不扩成永久远程执行授权；高 |
| E05 Git 迁移，避免实验包分发 | `01a08481-cbff-7191-a885-55c6c93da12d @271`：“我的实验都是通过git迁移到服务器训练，不要给我下载实验包之类的指令了”；结合 E04 | 明确以后工作方式，并被后续 Git 同步要求重复支持 | 新增，不把“不要实验包”扩大成禁止下载原始数据；高 |
| E06 用已有 Conda 环境，区分本地/远端与 shell | `01a02cac-… @202`：“我连接是服务器，你不要在本地环境里面找了”；`01a0747d-af2e-7143-8db3-3c85f7ef1e2f @357`：“我用conda了，重新给我指令”；`01a0936e-b921-7f60-8306-0ca6679fca41 @405`：“我有自己的conda环境”；早期 `019fa362-78af-7653-91da-4326c96692a7 @107` 展示 Bash 错用 PowerShell 命令后的报错 | 多次纠正；具体实现用当前环境文件核对 | 新增项目环境差异，全局“完整可复制命令”不重复；高 |
| E07 tmux 与查看日志 | `01a0908f-60f0-7921-b74a-a96e426d3b45 @881`：“训练用tmux吧”；`01a06fef-acd5-7392-9397-fcbe94936511 @6724`：“我想要使用tmux训练”；恢复分页中的 `01a1455a-2899-74d0-ac2b-dcae449b7a91`：“我用tmux跑，你能读取实验日志吗” | 多次使用与明确选择 | 新增长训练使用 tmux，具体 session 名不固定；高 |
| E08 保留有效 checkpoint，先解释性能修改影响 | `01a15760-… @200`：“启动脚本还是要--resume”；`01a18282-… @11`：“不浪费现有实验结果，在现有checkpoint上续训”；`01a17c37-bb82-72a1-9c56-fc18aa35e75e` 用户问优化是否影响 AP | 多次续训诉求；旧的“全部重训”针对具体坏权重 | 加入续训边界与影响说明，不能为了保成果接不兼容版本；高 |
| E09 模块、图、论文和代码一致 | `01a06fef-acd5-7392-9397-fcbe94936511 @5960` 追问为何删改 adapter/联合目标，`@6005` 追问为何实现只剩 GNN；`01a15402-d104-7d23-b2ff-b2adc637eca7 @21`：“你自己看代码，第一个模块有没有使用GNN”；`@75` 追问消息传递是否符合当前代码 | 持续纠正模式 | 新增对应关系约束；不冻结历代三模块名称或禁止一切模块改造；高 |
| E10 Deformable DETR 与 OWOD 目标 | `01a084bd-da09-7d01-849d-1b871b1fc932 @172` 指出本方法是 DETR、非 OrthogonalDet；`@638/@691` 纠正只强调 CIL/抗遗忘；`01a11a16-… @1217` 要求“真正实现未知物体发现和未知→已知的增量学习” | 跨对话重复 | 新增架构与研究范围；不把尚未验证的能力写成成绩；高 |
| E11 外部对照是真实 OWOD 方法、统一数据/指标 | `01a0334a-… @602`：“以OWOD的baseline来做比较”；`01a0908f-… @301`：“同样的MOWODB实验协议，数据也用一样的”；`01a09f11-9050-72e3-87a4-f04b36ac3589 @157`：“用tree-detr的metric和data” | 多次明确比较要求 | 新增比较与归因约定；不恢复当时不合规的旧协议/脚本，不强行把所有方法 recipe 改成相同；高 |
| E12 看实际 AP，不用机制通过代替效果 | `01a06fef-acd5-7392-9397-fcbe94936511 @6751`：“肯定还是ap指标啊”；`01a11a16-… @1266` 要求验证可行之后验证 AP 提升；`01a1642a-4c9b-7e62-a427-1631c4127094 JSONL L23/L232/L362` 分别质疑弱于 control、逐 epoch AP 大幅变化和 null 输出 | 多次验收纠正 | 加入效果证据边界；P0 的 AP50/U-Recall 等指标继续适用；高 |
| E13 Git 只留必要源码和文档 | P0；`01a11a16-… @812`：“太多pdf png等奇怪的文件…只保留代码文件”；`@895` 不提交无用 txt | 明确长期政策且重复 | 保留必要 Markdown/配置/许可证/作者 ImageSets 的既有例外；不把清理当成删除原始数据授权；高 |
| E14 论文材料不要混入实验仓库 | `01a0a2ff-bc40-7e63-a02d-69bce0f6adc7 @111`：“别把我的实验弄乱了”；`01a0b29f-3b3b-7a22-8111-4fd0f6c181d8 @124` 要删除解压进项目的论文文件 | 两次明确纠正 | 新增材料边界；不据此恢复或移动现有论文文件；高 |
| E15 科研严谨、创新要有依据 | `01a0149f-f671-7dd3-b537-7d87576fc276 @55`：“不要附和我”；`01a02900-3a2c-7303-9275-f1d3b8dc2485 @9` 要求文献核验且不能简单拼凑；引用线程 `01a01d69-… @9` 同类要求 | 重复用户偏好 | 用户证据可信；项目内只落实可执行的归因、实现/成绩边界，不再添加泛化口号或全局新流程；高 |

`…` 只在同表内缩写已给出全 ID 的线程，不是另一个来源。E07 的 `@6724`、E09 的 `@5960/@6005`、E12 的 `@6751` 位于续接上下文 `01a06fef-acd5-7392-9397-fcbe94936511`，按该完整线程 ID 检索；它们不能与父链继承副本算作多次独立证据。

## 去重、替代与未采纳内容

| 历史要求或建议 | 后续证据 / 当前文件 | 处理 |
| --- | --- | --- |
| DEUS、COCO-only、随机 20×4、unverified 可用于本地一致性比较 | E01 / P0 在 2026-10-08 明确改为作者 M/S，旧入口停用 | 过时，不恢复、不写新启动指令 |
| 根目录 `log/` 统一保存所有日志；之后要求放 `exps/` | `01a0322f-… @503`、`01a0908f-… @930`；2026-10-20 用户明确关心 `console.log`，当前 v5 启动器用 `output/` | 保留“每次必须留日志”，不新增固定 `log/` 规则，不移动既有结果。`exps/` 与 `output/` 按当前启动器分工 |
| 不使用 LoRA / 改用常规微调；后来要求 GNN score 选择性更新 adapter | 2026-09-08/09-12 相关纠正早于 E01 所在线程 `@964` 的方法改造；当前文档含低秩分支 | 记录演进，不写“永远禁止 LoRA”或“只能常规微调” |
| “三个模块固定为 CIGNN/GGDA/GCCO” | 旧 `01a08481-… @1241`；后有 2026-10-08 discovery 改造及 10-19 affinity/interference 讨论 | 只保留图文代码一致；具体模块由当前方法文档说明 |
| 12 小时 / 24 小时三组筛选，或舍弃第三组 | `01a13f90-… @181/@406` 与后续筛选对话；后来用户多次询问何时开始主实验 | 阶段预算，不让每次任务强制重新跑筛选；不新增无止境诊断验收门槛 |
| GPU 0/1、后来 1/2、甚至加第三卡；50/20 epochs、hard 路由、frozen 定位 | 用户 `01a15760-… @38` 明确改卡；`01a18282-… @129` 提供服务器改动；当前脚本已核对 | 属于当前/历史运行配置，写阶段记录并引用文档，不锁死为所有实验政策 |
| “全部重新训练、不用任何预训练”或“不能 resume” | 早期 scratch/损坏权重纠正；最新 E08 要求续训，当前 v5 允许 `--resume` 并用 DINO 骨干 | 不写永久重训/禁预训练规则，不允许因此丢弃有效成果 |
| Class-Affinity GNN 命名、“未知类别”取代“匿名类别” | `01a15402-… @43` 是询问；`01a15d39-3d43-7bb2-a483-130ff484dbc4 @1404` 是单次术语纠正 | 不把问题或单次具体措辞提升为长期规则；代码里的匿名原型也不擅自改名 |
| 图的字体、图例、尺寸、是否 2×4、无框公式 | 属于论文/图的具体交付且前后反复变动 | 不进项目 AGENTS；通用延续参考风格已在全局 |
| Google Scholar / Zotero、CCF A 或 A/B、保存到 OWOD 分组 | 研究任务中的具体检索/外部写入要求，后有“给我 title、我自己检索” | 不固化自动外部写入权限，不将文献检索插入每次代码任务 |
| 为突出方法调整对照绘图数值、补造表格数值 | 发生于旧实验表格/图修改；这些不构成新作者协议的实验记录 | 不采纳为项目规则，不把历史改写表格作为正式实验成绩或来源 |
| 助手建议的 Top-K、关闭蒸馏、分支隔离、停止/重启等 | 未见统一长期接受；部分只是诊断方案 | 不替用户决定；只记录明确采纳且仍适用的部分 |

原项目 AGENTS 没有需要整条废除的有效协议政策。本次是中文重组与去重，保留全部实质边界，并将易变细节下沉到已有文档；没有声称删除了原文件里并不存在的旧命令。全局无需修改，子目录无规则需要修正。

## 官方材料审查

用户指定来源：[OpenAI：Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。已读取 `C:/Users/23642/.codex/skills/.system/openai-docs/SKILL.md`，使用 OpenAI Docs 技能尝试官方域搜索和直接打开；网页工具未返回可读取正文。再以 Python `urllib.request.urlopen` 直连同一 URL，返回 **HTTP 403 Forbidden**。

因此本次**未能按该文章正文逐项核验，也不引用其观点**。本次采用用户当前请求明确给出的取舍：保持简洁、通用偏好不重复、项目规则写可执行边界、详细流程按需链接已有文档、不为文档整理增加业务测试。未用第三方转载或模型记忆冒充官方材料。

## 当前阶段、验证与待确认事项

- 阶段状态以“用户报告/本地文件/实际远端验证”分开：2026-10-28 用户在 `01a18282-… @57` 报告远端运行 `tree-detr-s-v5-full`，`@101` 已允许相关测试但要求本地改代码、自己上传。此为另一项实验优化任务的授权与状态，不是本次文档任务的远程操作要求。
- 当前本地 `tools/owod/run_discovery_full.sh` 已静态核对：v5、S-OWODB、`ow-detr-paper`、DINO 骨干、hard/frozen/hierarchical、GPU 1/2、进程数 2、每卡 batch 2、workers 4、seed 42、activation checkpointing。只接受 `--resume` / `--dry-run`，输出 `output/tree-detr-s-v5-full/`，控制台追加至 `output/tree-detr-s-v5-logs/console.log`。这是文件快照，不能证明远端正在用同一份脚本。
- 当前环境文件核对：Linux `environment.yml` 为 Python 3.10、torch 2.4.1、torchvision 0.19.1、CUDA runtime/toolkit 12.1，`setup_linux.sh` 默认 CUDA arch 8.6；Windows `build_extension.ps1` 目标 arch 12.0，默认解释器 `C:/Users/23642/miniconda3/envs/tree-detr/python.exe`。这些用于说明环境差异，不是未经测量的当前硬件探测结果。
- 命令逐项对照当前文件而非抄历史。实际运行 `manage_data.py --help`、`run_benchmark.py --help` 和 `manage_data.py plan`，均成功；plan 只读固定 ImageSets，核对两协议的任务顺序与类别数量。未运行 prepare/verify、训练 dry-run、CUDA 编译、pytest、AP 评估或远程命令。
- 上述 CLI 核验使用 Codex bundled Python：`C:/Users/23642/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`；裸 `python` 在当前 shell 命中 Microsoft Store 别名不可用。AGENTS 中 `python ...` 的前提明确为激活项目 Python 环境，未修改用户 PATH 或 Conda 环境。
- 静态核对 `main.py`、`util/experiment_log.py`：运行配置保存于阶段目录，结构化指标为 `metrics.jsonl`，训练日志为 `train.log`，resume 对应追加；外层启动器另有 `console.log`。不把文件名存在等同于日志当前完整或训练已完成。
- 文档验证：核对 Markdown 相对链接、代码路径、UTF-8、命令参数、修改范围与 `git diff --check`；不强制增加与本次文档整理无关的测试。Git 的 LF→CRLF 提示属于工作区换行约定，检查无 whitespace error。
- 待确认但不阻塞本次交付：官方文章若以后可访问，可再做一次材料审查；云端/其他机器历史若另获访问，可补证。当前远端结果、完整四任务 AP 增益、最新训练进度均未在本次核验，不给新成绩或性能承诺。没有需要用户批准的新项目规则候选。

## 已读上下文索引

以下列出读取范围供复核；标题不作为证据，省略易包含整段用户正文的原始标题。表中用户数是本次各源提取数，含继承副本与 JSONL 补充，不能据此计独立出现次数。“关联片段”包括 Obsidian 中讨论项目的聊天和迁移后的续接上下文，不自动代表单独的人类对话。

| 上下文 ID | 用户记录数 | 范围 / 原文来源 |
| --- | ---: | --- |
| `019fa362-78af-7653-91da-4326c96692a7` | 3 | 项目 cwd；SQLite |
| `019ffef1-0d9a-7190-b041-4d7863adbed6` | 5 | 关联片段；SQLite |
| `01a0149f-f671-7dd3-b537-7d87576fc276` | 7 | 关联片段；SQLite |
| `01a01854-8c7a-7253-bd3a-0d4fdad099a5` | 37 | 关联片段；SQLite |
| `01a018b4-bebb-7ea3-978b-85740e7cbaf7` | 3 | 关联片段；SQLite |
| `01a019e9-61d9-7211-b7fd-f121d36909f9` | 43 | 关联片段；SQLite |
| `01a019ea-9525-7de2-bab3-7e6ba5775c1d` | 24 | 关联片段；SQLite |
| `01a01d4b-1f13-70e3-a6f7-47cee17d27a1` | 8 | 项目 cwd；SQLite |
| `01a01d71-2b90-7230-b0c3-5909c5677a70` | 12 | 项目 cwd；SQLite |
| `01a01e56-1945-7792-8a49-7eaae9b42104` | 49 | 项目 cwd；SQLite |
| `01a022ac-766b-7102-bf66-8558f66cecb3` | 57 | 项目 cwd；SQLite |
| `01a02900-3a2c-7303-9275-f1d3b8dc2485` | 15 | 关联片段；SQLite + JSONL |
| `01a02998-7c84-7c53-8026-2ffa7ad3e548` | 6 | 项目 cwd；SQLite |
| `01a02cac-74db-78e0-82eb-25e0b9d865c9` | 13 | 项目 cwd；SQLite |
| `01a02cfd-3072-7170-aeeb-1f869cff2380` | 9 | 项目 cwd；SQLite |
| `01a02e2a-408e-7dc2-a1ec-f54045739d8c` | 2 | 项目 cwd；SQLite |
| `01a02e32-ea7d-7a61-bbac-6bcc558c7c01` | 3 | 关联片段；SQLite |
| `01a0322f-6d75-7d41-b57a-0d6b9b9d2d6b` | 1 | 项目 cwd；SQLite |
| `01a0322f-fd33-77f2-93d0-f4ad6a20b3a6` | 12 | 项目 cwd；SQLite |
| `01a032ab-408b-74b3-8c5e-4a8ba19ba912` | 10 | 关联片段；SQLite + JSONL |
| `01a0334a-e5de-7ae0-87ae-b3266af219e4` | 12 | 项目 cwd；SQLite |
| `01a03dd8-5448-70c3-a942-42bb502eaf78` | 8 | 关联片段；SQLite |
| `01a03df0-2125-7c32-ba59-f436433284f6` | 6 | 项目 cwd；SQLite |
| `01a04233-5d91-74c2-9693-f67987022f42` | 10 | 项目 cwd；SQLite |
| `01a04bb2-e425-74b1-b041-924c24735782` | 22 | 项目 cwd；SQLite |
| `01a05b83-0ea4-7cd2-af52-c535060bba18` | 20 | 项目 cwd；SQLite |
| `01a05b9c-c8a8-7f52-8a85-b395cb3982ab` | 1 | 项目 cwd；SQLite |
| `01a0661c-793a-7800-906e-7dc137426719` | 18 | 项目 cwd；SQLite |
| `01a06b0e-5e12-7f61-aaab-535cdcd8632c` | 19 | 关联片段；SQLite |
| `01a06b9e-7e73-7622-b436-f1a8bd40dc13` | 98 | 项目 cwd；read_thread 全部分页 |
| `01a06be9-ef46-7652-992d-074fbacefde5` | 1 | 项目 cwd；SQLite |
| `01a06bf6-0640-7031-a744-0e72a9e9bfdd` | 1 | 项目 cwd；SQLite |
| `01a06bfe-4868-7361-bc3f-b21c33e20f59` | 8 | 关联片段；SQLite |
| `01a06c19-3992-78f0-8cb6-56fc59e488f9` | 1 | 关联片段；SQLite |
| `01a06c1b-2bce-79c2-ab4b-04e8a8a788d8` | 3 | 关联片段；SQLite |
| `01a06fec-71ca-70a0-8439-60e5726a36b0` | 1 | 关联片段；SQLite |
| `01a06fef-acd5-7392-9397-fcbe94936511` | 15 | 关联片段；SQLite |
| `01a07059-292c-7e23-a8e7-fbb92e5f3e91` | 2 | 项目 cwd；SQLite |
| `01a0747d-af2e-7143-8db3-3c85f7ef1e2f` | 18 | 项目 cwd；SQLite |
| `01a075a9-6a1f-7381-be78-3605010f17d2` | 6 | 项目 cwd；SQLite |
| `01a075ec-10c0-7911-a878-a1265fef6815` | 1 | 项目 cwd；SQLite |
| `01a07a5c-6674-7a82-b4af-45deb5808270` | 19 | 项目 cwd；SQLite |
| `01a07bec-4c44-7cb0-a1f8-c5b220fe9699` | 7 | 关联片段；SQLite |
| `01a07f66-c1d2-7203-a94a-31d711f7431b` | 1 | 项目 cwd；SQLite |
| `01a080a5-6a25-72e3-b696-77b488a5d3ff` | 1 | 项目 cwd；SQLite |
| `01a080af-4d5a-79e3-a70e-e241743ed102` | 1 | 关联片段；SQLite |
| `01a080d4-b376-7363-9524-3c82c2278432` | 1 | 关联片段；SQLite |
| `01a080d6-47be-7c70-940a-bb7241773ef0` | 1 | 关联片段；SQLite |
| `01a0841e-dfcb-7292-be83-9dd28cc3002c` | 1 | 关联片段；SQLite |
| `01a08421-113c-7a71-8c84-271be32ab3b7` | 1 | 关联片段；SQLite |
| `01a08423-92bd-7661-acff-5c2399858461` | 1 | 关联片段；SQLite |
| `01a08425-5946-7f22-8a4d-6c345801cdc6` | 3 | 关联片段；SQLite |
| `01a08481-cbff-7191-a885-55c6c93da12d` | 21 | 项目 cwd；SQLite |
| `01a084bd-da09-7d01-849d-1b871b1fc932` | 13 | 关联片段；SQLite |
| `01a085f7-7f40-7d40-967d-9f4d96d06a52` | 5 | 项目 cwd；SQLite |
| `01a08947-6103-7b90-83a8-fd82a7ed65fd` | 1 | 关联片段；SQLite |
| `01a08947-ea78-7562-894d-57a2bcc14aae` | 8 | 关联片段；SQLite |
| `01a08a20-09bf-74d2-80ce-7c6abca2391a` | 9 | 项目 cwd；SQLite |
| `01a08a5c-df97-7510-a6f0-2492c13efae0` | 1 | 关联片段；SQLite |
| `01a08a63-446e-7fd1-b690-a83ceeda9b9f` | 1 | 关联片段；SQLite |
| `01a08a6d-3b11-7d71-8557-40bf674a1923` | 1 | 关联片段；SQLite |
| `01a08ab9-0a19-70d2-a5d2-adccdb6b6a21` | 2 | 关联片段；SQLite |
| `01a08ab9-1931-7340-92f6-45946d56534f` | 6 | 关联片段；SQLite |
| `01a08aea-0553-7de3-93bd-6406524afce3` | 7 | 项目 cwd；SQLite |
| `01a08b96-c80c-7002-b4c0-3e06becba2fa` | 1 | 关联片段；SQLite |
| `01a08e9c-915a-7562-8880-0f46ac16291f` | 15 | 项目 cwd；SQLite |
| `01a08ea0-1bc1-7762-a99f-1f0bb29f4a50` | 15 | 关联片段；SQLite |
| `01a08ed6-a9f6-7c60-910e-574b130b1f66` | 1 | 项目 cwd；SQLite |
| `01a08f42-ae9e-7261-ab33-ed9eb7316377` | 15 | 项目 cwd；SQLite |
| `01a09055-cee9-7611-bdb4-03c88553716c` | 6 | 关联片段；SQLite |
| `01a0908f-60f0-7921-b74a-a96e426d3b45` | 17 | 项目 cwd；SQLite |
| `01a090f8-9566-7020-881f-dc95a00d8bb2` | 13 | 关联片段；SQLite |
| `01a0936e-b921-7f60-8306-0ca6679fca41` | 39 | 项目 cwd；SQLite |
| `01a09396-ef5d-7fb2-8cfd-a352a8a42019` | 7 | 项目 cwd；SQLite |
| `01a093c4-e060-7b51-925a-99eb0e9f4328` | 18 | 项目 cwd；SQLite |
| `01a0945b-cd56-75e0-93f3-88075bf7a005` | 12 | 项目 cwd；SQLite |
| `01a094bf-9be5-7a70-b210-2a1169388859` | 2 | 项目 cwd；SQLite |
| `01a09990-0e11-7872-866d-75302819a21b` | 17 | 项目 cwd；SQLite |
| `01a099f6-0a7b-7842-82d2-88b5ff0f4ed0` | 26 | 项目 cwd；SQLite |
| `01a09e99-8168-7311-bc55-8ecac8bbe30e` | 3 | 项目 cwd；SQLite |
| `01a09eed-e5cc-75c0-ae19-dc749000c538` | 7 | 关联片段；SQLite |
| `01a09f11-9050-72e3-87a4-f04b36ac3589` | 14 | 项目 cwd；SQLite |
| `01a0a2ff-bc40-7e63-a02d-69bce0f6adc7` | 4 | 项目 cwd；SQLite |
| `01a0a314-78b5-7430-a57e-7f68481653f6` | 1 | 关联片段；SQLite |
| `01a0a314-e964-7321-98a6-a0fcb702b04c` | 6 | 关联片段；SQLite |
| `01a0a315-3cfd-7ea3-94e6-37fcce016be7` | 4 | 项目 cwd；SQLite |
| `01a0a345-53e0-7492-99fa-82e929a824a5` | 9 | 关联片段；SQLite |
| `01a0a385-1f94-7a33-b378-b8528bd171e7` | 2 | 项目 cwd；SQLite |
| `01a0a4c4-55a9-7911-82c1-9fc4dc0aed3c` | 3 | 关联片段；SQLite |
| `01a0a7f3-4d54-7fb1-965a-7b0c6f0f3494` | 15 | 项目 cwd；SQLite |
| `01a0a808-78cd-7fd3-b41c-d2a23f4290d2` | 3 | 项目 cwd；SQLite |
| `01a0a85f-35a2-7d00-8927-9492945e9195` | 11 | 项目 cwd；SQLite |
| `01a0ae2a-a706-7cb0-ba3e-b1ac17d8936a` | 2 | 项目 cwd；SQLite |
| `01a0afa3-43f7-7b50-bcb5-8f4fafe48ac4` | 1 | 项目 cwd；SQLite |
| `01a0b270-141d-7ee2-a049-7e57e9e06149` | 1 | 项目 cwd；SQLite |
| `01a0b28f-f134-7e80-ad3f-776c950779b6` | 28 | 项目 cwd；SQLite |
| `01a0b292-c8e3-7f40-a7c8-e8f4b65a7a0f` | 2 | 项目 cwd；SQLite |
| `01a0b29f-3b3b-7a22-8111-4fd0f6c181d8` | 3 | 项目 cwd；SQLite |
| `01a0b34c-9c58-74b1-ac7e-136ea437dc6f` | 1 | 项目 cwd；SQLite |
| `01a0b34e-18f5-7ad3-87ce-1417ccb6882e` | 5 | 项目 cwd；SQLite |
| `01a0b376-ef8d-70c1-990a-cba92302c93c` | 11 | 项目 cwd；SQLite |
| `01a0b3dc-35c1-7ec2-a2ee-be476a1004f7` | 16 | 项目 cwd；SQLite |
| `01a0b830-86ac-7d43-988c-d804bfe1abd0` | 1 | 项目 cwd；SQLite |
| `01a0bec4-865a-7732-8dc5-f5014f543449` | 3 | 项目 cwd；SQLite |
| `01a0becc-bcf0-7032-a6b9-5b690b27882f` | 2 | 项目 cwd；SQLite |
| `01a0bee5-54e7-78e2-b4e1-7143584228d2` | 5 | 项目 cwd；SQLite |
| `01a11a16-b107-7212-9e12-ff40ad7dd34e` | 20 | 项目 cwd；SQLite |
| `01a11a8c-e9b4-7b61-9af0-f3827c9f1348` | 1 | 项目 cwd；SQLite |
| `01a11fb0-fad7-7020-b2c2-a0e767ec00b4` | 2 | 项目 cwd；SQLite |
| `01a1206d-326c-74f2-86a5-7196af56d352` | 4 | 项目 cwd；SQLite + JSONL |
| `01a12182-b5dc-78f1-9a69-ee76240dbb8f` | 13 | 关联片段；SQLite |
| `01a12944-9027-7a51-a0d1-a5b4082e0763` | 5 | 关联片段；SQLite |
| `01a1296f-4dc1-7df1-a824-186030cc1295` | 2 | 关联片段；SQLite |
| `01a1299c-4f6a-7770-bc29-92b6e5c61e99` | 2 | 关联片段；SQLite |
| `01a129b0-3800-7c03-9564-a2920b5a3422` | 16 | 项目 cwd；SQLite + JSONL |
| `01a129ca-412e-7c42-a7be-1cf438d23b3b` | 10 | 关联片段；SQLite |
| `01a13a20-6759-7552-a0fd-52a3ed77eaba` | 1 | 项目 cwd；SQLite + JSONL |
| `01a13a30-0a12-7173-bd4b-f933bd3f4da3` | 5 | 项目 cwd；SQLite + JSONL |
| `01a13f90-108c-73d0-9e44-44745244c8a9` | 15 | 项目 cwd；SQLite + JSONL |
| `01a13fbe-999e-78d0-b0f7-5e824895f9b4` | 12 | 关联片段；SQLite |
| `01a145a6-319c-7b52-bda4-93062228d328` | 7 | 关联片段；SQLite |
| `01a145b7-590a-73c2-97ab-d977e90bc2db` | 2 | 项目 cwd；SQLite + JSONL |
| `01a1481b-8fe0-7411-a8e0-80642033cfae` | 19 | 关联片段；SQLite |
| `01a14ee2-bccc-79b0-85bd-099a8fa76540` | 41 | 项目 cwd；read_thread 全部分页 |
| `01a15402-d104-7d23-b2ff-b2adc637eca7` | 7 | 项目 cwd；SQLite + JSONL |
| `01a15760-808c-79c3-b6e8-4801b8a9d451` | 13 | 项目 cwd；SQLite + JSONL |
| `01a15d2e-1dc9-76b2-9b7e-edc02445a854` | 6 | 关联片段；SQLite |
| `01a15d39-3d43-7bb2-a483-130ff484dbc4` | 5 | 项目 cwd；SQLite + JSONL |
| `01a161c3-ba49-7d60-891d-b8cd5abc3631` | 2 | 项目 cwd；SQLite + JSONL |
| `01a1642a-4c9b-7e62-a427-1631c4127094` | 10 | 项目 cwd；SQLite + JSONL |
| `01a1642a-f16d-7f62-8097-d5cb276363f6` | 9 | 关联片段；SQLite |
| `01a17275-482d-7d02-bd0b-f21cdf9e07d0` | 1 | 项目 cwd；SQLite + JSONL |
| `01a17276-1f7e-7443-b03d-90e9cd3399ae` | 1 | 项目 cwd；SQLite + JSONL |
| `01a17c37-bb82-72a1-9c56-fc18aa35e75e` | 3 | 关联片段；SQLite + JSONL |
| `01a18282-1902-7ef0-9f42-e3dbd30009fe` | 4 | 项目 cwd；SQLite + JSONL |
| `01a182a2-63e1-7990-bcba-17f15f42895f` | 1 | 项目 cwd；SQLite + JSONL |
| `01a01d69-f22c-7640-924e-2232c91ea179` | 1 | 用户明确引用；SQLite + read_thread 末页 |
| `01a1340c-55f6-7ec0-9bf4-eff82f0862b2` | 不计 | 额外读过 JSONL；像素画且迁出项目，排除 |
| `019eb173-ac04-7e93-88b7-e0888f374cef` | 不计 | 额外读过归档的 3 条用户原文；非项目，排除 |
