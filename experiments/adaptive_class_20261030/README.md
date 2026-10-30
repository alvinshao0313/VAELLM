# 整36gate同父动态阶数与Rice对照

本轮继承[10月29日已审设计](../adaptive_grouped_fit_20261029/README.md#再次决策不加旧g09-gate训练保存真实ab-raw-gate并补齐公用核心)，在预定完整九项选择出的B结构上执行。B为gate D64/K64、两阶、asymmetric H128/R0；并非根据本轮局部误差选结构。当前完整九项及总目标尚未完成；唯一执行计划仍在[项目计划](../../docs/experience/records/w25_transfer/current_plan.md)。本目录只管理同一配对的实现、运行入口和证据。

**2026-10-30 09:53（中国时间）暂停推进，准备迁移。** 用户要求本次安排完成后暂停；从此不启动新任务、不进入正式两臂恢复、不改活动源码、不继续轮询或清理。以下五个已启动任务保留自然运行，运行中目录仍会变化，不能把本快照当最终退出结果。后续只在用户恢复后先核真实终态，再决定接续。

## 固定合同

- 原始两阶来自`adaptive_native_model_20261029/cat_gate_prefixes/B`真实raw归档，UUID `7c0b1199-b9ff-46af-a1b7-c2f5b7dabf7f`。live目标与非gate补集来自共同`qwen_recompression_20261026/C/initial_after_o`，UUID `b63bfe79-3889-40c6-bf66-1d002893f874`。不换成HF或恢复后donor。
- 两臂共享实际compact两阶码、BF16 decoder、INT8保护、sign和旋转。第三阶残差为live旋转FP32目标减该compact实际顺序BF16前缀；稀疏残差为同一live目标减实际compact原坐标BF16权重。保护列先保持相同。
- 第三阶训练全部真实向量，前两阶冻结；每层10000更新、B4096、seed2031、AdamW(.9,.95)/WD0、LR.003线性降零且无warmup。每层MSE、commit、样本熵和K位边际熵独立，求和更新；分组/分卡不做DDP、不跨M统计边际。原CAT前两阶及live up/down继续原M36 pooled BSQ。
- `capture.py`只读取既定FineWeb train前64篇/15183有效tokens；真实after_o前向，去保护后input sign→forward Hadamard，按每输入坐标累计二阶矩并均值归一。只保存36份h及来源，不落全激活、不用后64篇排序。
- 控制固定abs005、INT8 blocked values与256×256 block、Rice7无损坐标压缩。复用原生选址/量化/解码和既有Rice7实现。length2隐含，不收费额外长度流。动态只同K保留1–3阶，第三阶全bank完成后一次路由，之后码与长度冻结。
- 每层cap取本轮实际`fixed2_rice7.pt`完整文件字节；动态同一`torch.save(BytesIO)`序列化器逐轮实包，包含三个decoder、保护/sign、码padding、zlib frame、metadata和容器，最多四轮取最后可行，不看输出择优。没有第四decoder保留费或旧16KiB头公式。
- 两臂各从自己的raw替换态完成原gate500恢复、自己的live up S2×10K/500恢复、自己的live down S2×10K/500恢复。恢复的五源train、B4/max1024、dynamic padding、rank12仅未来dense、decoder/LoRA LR及250/500保存保持原配方。每个未来VAE类别初始化前两臂显式共同seed31；shuffle仍500031/600031，恢复seed仍35/36/37。这是匹配fresh初始化，非原A/B exact续接。
- 最终库存216native＋36compact、pending_dense为空，实际完整logical Linear≤2.6另核；packed及完整目录bytes单列。超预算立即停止该成品依赖，不调密度/其它类别补账。完整同模型九项由[独立评测接入](../adaptive_class_evaluation_20261030/README.md)执行，原始替换误差不代替质量结论。

## 代码边界

`sources.py`绑定真实来源；`fit.py`复用已验证`IndependentGroupedVAE`和N原BSQ/融合，按实际D/K/H执行独立拟合；`sparse.py`只是现有abs005/Rice7到共享compact的薄消费者；`artifact.py`是共同实际完整文件序列化器；`continuation.py`复用N原fitter和mixed恢复下层；`run_pipeline.py`管理固定三类别，完整保存后用独立进程strict fresh与同输入BF16 logits核对。

复用包只作必要扩展：`adaptive_grouped_fit_20261029/grouped.py`从硬写两层D32扩成同规格任意非空组；`adaptive_native_model_20261029/checkpoint.py`识别Rice子类/计费；`recovery.py`按真实gate/up/down库存选择训练与融合，最后无未来dense时仅训练decoder。N snapshot/public源码均未修改。旧固定两阶CPU/GPU保存/回训/next-up合同继续复用，新增短验只覆盖实际三阶/Rice、完整36compact及最终无dense路径。

## 实际证据和运行位置

共同根为`/data/star/VAELLM/adaptive_class_20261030/`，每份`report.json`记录实际命令、PID、源码SHA、输入身份、真实步数和费用。当前已完成：

| 证据 | 实测范围 | 状态 |
| --- | --- | --- |
| `live_h/report.json` | 原after_o全部144native真实前向，36gate前64篇train h | 9.80秒、exit0 |
| `cpu01/report.json` | 实际L0/L18/L35 D64K64三层FP32 grouped损失/全部叶梯度/抽取；L0 Rice全部字段及完整文件fresh原坐标权重exact | 12.84秒、exit0；梯度relative-L2约4.3–5.0e−8 |
| `short01/report.json` | 真实L0–5，group6、B4096，两次原10K日程更新、全向量编码、实际cap和一次路由 | 34.99秒、exit0；Torch峰allocated1,600,864,256B |
| `short36/report.json` | 同一新增路径覆盖L6–35，其余五个group6 | 156.10秒、exit0 |
| `formal_gpu{1,2,3}/report.json` | 每卡12层/两组group6，每层10K | 第一组10K及其一次路由已完成；第二组快照约5300–5400/10K，任务仍运行 |
| `short_dynamic/report.json` | 全36动态compact的三类别短链，恢复每类2/500、原生up/down S2×2 | gate2/500、完整mixed保存回训/融合/独立fresh通过；live up S2×2完成，当前up恢复；尚未验证最终无dense |
| `short_rice/report.json` | 全36 Rice compact同一短链 | gate第1次真实更新及step1保存完成、继续训练中；尚未声明两更新或fresh通过 |

L0控制完整文件实测13,820,101 B，短验动态13,819,397 B；该动态为2更新机制输入，不能据此称10K算法或质量通过。每层重新计算自己的cap，不能复用L0数值。`short01`的source_variance用样本var，后继正式及其余短验沿既有约定用population var；同层各候选乘共同正比例，不改变训练，短验不用于质量比较。

调用者先激活`bitvae`，Python版本确认3.11.16；所有执行在tmux/nohup后台，单卡由主线程分配，不占卡等待。正式分组实际命令以各report为准，例如GPU1：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -B -u \
  -m experiments.adaptive_class_20261030.run_fit \
  --h /data/star/VAELLM/adaptive_class_20261030/live_h \
  --start-layer 0 --stop-layer 12 \
  --output /data/star/VAELLM/adaptive_class_20261030/formal_gpu1
```

恢复入口要求给齐覆盖36层的真实完成报告，拒绝缺层、重复层、SHA变化、训练预算不符或cap超限。short明确要求short来源，正式明确要求10K来源：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python -B -u \
  -m experiments.adaptive_class_20261030.run_pipeline --arm dynamic \
  --artifacts /data/star/VAELLM/adaptive_class_20261030/formal_gpu1 \
    /data/star/VAELLM/adaptive_class_20261030/formal_gpu2 \
    /data/star/VAELLM/adaptive_class_20261030/formal_gpu3 \
  --output /data/star/VAELLM/adaptive_class_20261030/dynamic
```

该正式恢复命令当前未执行；暂停期间不能自动接续。用户恢复后先核新增路径短链终态，补齐下述待办，再按届时资源使用新输出目录；Rice臂只改`--arm rice`与独立输出。正常启动后确认实际推进再交付，不新建监控或自动重试队列。

最终`final_model`是原v2混合格式，调用`adaptive_native_model_20261029.checkpoint.load()`读取自身单权重文件。分支`report.json`仅真正正式全链完成时置`completed_whole_class_pipeline`；short为`completed_short_whole_class_pipeline`。报告绑定原始归档manifest及SHA、arm、phase、实际/计划更新、最终UUID和`final_strict_reload_report`，不能将short成品传入正式九项。

本组尚无新增质量经验。收尾依[工作流程](../../docs/experience/WORKFLOW.md)先结果总结、再经验增量、导航及清理；活动输入/正式输出保留，结束后的short中点及Adam/FP32权重无后继用途即按既有授权删除，只保留最小配置、核心日志和原始证据。

## 暂停移交的运行身份

以下时间为2026-10-30中国时间，所有worker在调用者激活的`bitvae`中以`python -B -u -m ...`运行。完整参数在各`report.command`，报告中的模块真实文件名与下面等价的`-m`入口对应。`start_ticks`为`/proc/PID/stat`第22字段；本次09:53只读快照之后不再主动监控。

| GPU／范围 | worker PID／start_ticks | tmux（shell PID） | 实际启动时间 | 输出子目录 |
| --- | --- | --- | --- | --- |
| 1／L0–11 stage3 | 647008／16442986 | `adaptive_class_fit_GPU1_20261030`（647005） | 09:40:23 | `formal_gpu1` |
| 2／L12–23 stage3 | 647012／16442987 | `adaptive_class_fit_GPU2_20261030`（647009） | 09:40:23 | `formal_gpu2` |
| 3／L24–35 stage3 | 647017／16442988 | `adaptive_class_fit_GPU3_20261030`（647013） | 09:40:23 | `formal_gpu3` |
| 0／dynamic short | 653352／16470039 | `adaptive_class_dynamic_short_GPU0_20261030`（653350） | 09:44:54 | `short_dynamic` |
| 6／Rice short | 660830／16500395 | `adaptive_class_rice_short_GPU6_20261030`（660828） | 09:49:57 | `short_rice` |

五个输出均在共同根`/data/star/VAELLM/adaptive_class_20261030/`；对应日志为根目录`<输出子目录>.log`，外层shell在worker退出后写`<输出子目录>.exit`。09:53快照五个exit文件均尚不存在，五份report仍为running。Trainer DataLoader子进程归各short worker管理，strict fresh子进程也由原worker同步启动，不是新任务或额外调度器。

三份拟合入口为`-m experiments.adaptive_class_20261030.run_fit --h <根>/live_h --start-layer {0,12,24} --stop-layer {12,24,36} --output <根>/formal_gpu{1,2,3}`。两份short入口为`-m experiments.adaptive_class_20261030.run_pipeline --arm {dynamic,rice} --artifacts <根>/short01 <根>/short36 --short --output <根>/short_{dynamic,rice}`，分别只看到物理GPU0／6。

迁移时仍在写入的是上述五个输出子树、各自根日志与未来exit文件；不要改源码pins或删除它们。其输入`live_h`、`short01`、`short36`、真实B raw归档与共同after_o保持只读。共同根09:53实际约40GiB，后续short保存/原生替换会继续增长，不把此数字当迁移最终大小。GPU6仅临时short，结束后自然释放；没有安排它接正式长恢复。

动态已核证的可加载中间模型为`short_dynamic/04_gate_proj/after_recovery`，UUID `5a02cf39-561b-4f35-bbd8-c248da26dc16`，3063字段、同输入BF16 logits exact；独立报告`short_dynamic/04_gate_proj/strict_reload.json`。对应live up拟合目标digest为`aa405fc996fc636c9b4e639b3bfecd9c1730f49480bb35fbd704f1cd063165e0`，原M36、S2×2、shuffle500031，gate字段exact。它是有72dense的中间证据，不是最终预算或正式质量成品。

正式`dynamic/`与`rice/`恢复均未启动；完整九项也未启动。两臂最终216native＋36compact＋0dense、完整500/up/down正式10K链和相同最终模型九项仍未完成。共同future VAE seed31规则已实现；`initialization.py`已写只读optimizer创建时参数digest观测helper，但**尚未接入continuation或运行验证**，本次暂停不修改活动源码。用户恢复后需要接入并记录两臂实际各stage初始化digest，避免把“声明seed相同”当已证初始参数相同。文档索引/链接检查由主线程在合并本轮各代理文档后统一处理，本代理未执行或声称通过。
