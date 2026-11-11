# 整36gate同父动态阶数与Rice对照

本轮继承[10月29日已审设计](../adaptive_grouped_fit_20261029/README.md#再次决策不加旧g09-gate训练保存真实ab-raw-gate并补齐公用核心)，在预定完整九项选择出的B结构上执行。B为gate D64/K64、两阶、asymmetric H128/R0；并非根据本轮局部误差选结构。两臂正式训练与完整九项已于10月31日闭合，见下节；总目标未完成。唯一执行计划仍在[项目计划](../../docs/experience/records/w25_transfer/current_plan.md)。本目录只管理同一配对的实现、运行入口和证据。

## 完整结果与取舍

2026-10-31 10:45接续核实：原driver及两worker已退出，固定链`sequence.exit=0`、五个stage均exit0。dynamic、Rice各完成gate／up／down原500更新，自身live up／down各S2×10K；两臂四组实际optimizer初始化记录相同，各类冻结持久字段exact。各最终独立fresh为3927字段及同输入BF16 logits exact。每臂完整full8为64叶／24742样本、LongBench为21任务／4750样本且无待评目录，最终权重stat与评测绑定保持；acceptance exit1表示质量未达，不是运行失败。实际argv、来源、退出及原始全分项由[固定链终态](../../result/qwen3_8b_mmlu65_20261004/cmcc_sequence_20261030/status.json)和各自验收报告保存。

| 完整端点 | dynamic | Rice |
| --- | ---: | ---: |
| 九项均分 | 62.73443844063971 | 63.89663436633599 |
| 相对BF16下降，pp | 5.312706477268768 | 4.150510551572495 |
| 八项均分 | 65.2636904479388 | 66.63294207675278 |
| MMLU | 61.49408916108816 | 62.09229454493662 |
| LongBench21宏均值 | 42.500422382246974 | 42.00617268300166 |
| logical bpw | 2.5216497234676196 | 2.52164744294208 |
| packed bpw | 2.5350243837937065 | 2.5350221032681675 |
| 整模型tensor bytes | 4,690,897,095 | 4,690,895,115 |
| 保存时完整目录bytes | 4,694,076,387 | 4,694,046,846 |
| 正式后缀秒数，含导出 | 55,540.929815890006 | 53,925.44700039999 |
| 完整评测秒数 | 8,959.170812129974 | 12,297.107557535172 |

dynamic UUID为`f434fcae-a1ee-4c10-aaad-1404637375f9`，Rice为`81442c7d-37c8-4107-ab6f-e338f05e8e04`。原始未舍入分项、同协议BF16、完整费用组成及严格阈值分别见[dynamic验收](../../result/qwen3_8b_mmlu65_20261004/adaptive_class_evaluation_20261030/cmcc_sequence_20261030_dynamic/longbench/merged/nine_task_acceptance.json)和[Rice验收](../../result/qwen3_8b_mmlu65_20261004/adaptive_class_evaluation_20261030/cmcc_sequence_20261030_rice/longbench/merged/nine_task_acceptance.json)。logical包含主码、全部decoder／norm、INT8保护值／scale／索引、旋转及完整route或Rice载荷；packed另含码padding，整目录含非投影及序列化。dynamic额外第三阶拟合／路由成本见原`formal_gpu1/2/3/report.json`，不把上表后缀时间当完整候选生成成本；评测分别用3卡和2卡，其墙钟差不是推理速度比较。

**决定：** Rice九项比dynamic高1.162195925696281pp，八个短任务均高、LongBench低0.4942496992453158pp。在本共同raw2／after_o、实际近同cap、同恢复和匹配后缀初始化下，选择Rice为下一批唯一fresh C1表示候选；结束本dynamic固定配置的追加训练预算，不否定其它阶码长、评分或训练方案。该结果是单seed完整配方比较，不单独识别路由因果。Rice比旧结构B九项高0.11264353738658pp，但旧B的native消费者和连续RNG后缀与本配对不同，不能据此认定稀疏补偿优于普通两阶。

下一项有决定价值的控制是从新共同after_o／raw2重建Rice载荷，完成fresh C0与C1以及各自同规则恢复；不等待C0单独分数来消除本来就存在的父源混杂，也不另加旧父fixed2长链。新C1仍需隔离来源入口，旧Rice成品和36层载荷不冒充新父产物。经验已归并[向量级位宽分配](../../docs/experience/lessons/compression_choices.md#向量级位宽分配的待闭环问题2026-10-29)。

**结束产物收尾：** 新C1实现已确认不读取旧两臂权重／patch；两臂没有安排续训或复评。核实进程、后继用途和活动C0／Muon pins后，仅删除`cmcc_sequence_20261030_dynamic`及`cmcc_sequence_20261030_rice`内132个无用途生成文件：12份中间／最终模型、训练master／optimizer／scheduler／RNG及重复tokenizer。实际按单链接文件分配块释放94,188,847,104 B（87.72 GiB），不是共享磁盘空闲量差。400份配置、strict／训练审计、same-input证据、原sequence记录和两臂原评测文件聚合SHA前后相同；活动1003／1021项pins逐文件SHA均未变。原始raw／after_o、正式36层码流及所有其它目录继续保护。精确路径、大小、引用核查和两项系统会话读取限制见[唯一清理记录](../../result/qwen3_8b_mmlu65_20261004/cmcc_sequence_20261030/cleanup.json)。旧两臂现在只留证据、不可加载模型；上表存储数字及原报告保持保存时历史实测，不回写其SHA。

**2026-10-30 09:53（中国时间）曾暂停推进，准备迁移。** 当时五个已启动任务保留自然运行。用户现已在新机恢复；09:53快照仅保留历史运行身份，当前接续和真实终态见文末。

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

下表为09:53迁移前历史快照，共同根为`/data/star/VAELLM/adaptive_class_20261030/`；每份`report.json`记录实际命令、PID、源码SHA、输入身份、真实步数和费用。新机完整短链终态见文末，不沿用下表的在途状态：

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

该旧机正式恢复命令在09:53暂停时未执行；新机已由文末的一次性执行链接管，在独立输出下接正式恢复。Rice臂只改`--arm rice`与独立输出，不另建监控或自动重试队列。

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

09:53旧机暂停时，正式`dynamic/`与`rice/`恢复和完整九项均未启动，初始化观测helper也尚未接入。该历史待办已由下文新机真实两臂short和初始化配对证据替代；完整500/up/down正式10K链及同模型九项仍须按各自正式终态验收。

## 10月30日 cmcc 新机接续

已核实迁入的`formal_gpu1/2/3`均为`completed_formal_stage3_and_once_routes`／exit0：全部36层第三阶10K、全bank及一次路由已结束，**不重训第三阶**。旧`short_dynamic`／`short_rice`均在down部署logits的合并断言失败；原表达式先检查BF16，短路时没有执行finite检查，不能将日志解释为已证NaN。CPU真实Accelerate机制复现显示，`prepare_model_export`后FP32输出wrapper仍存在，显式`unwrap_model(..., keep_fp32_wrapper=False)`后恢复BF16且有限、原字段及同输入BF16结果不变。机制证据在`result/migration_audit_20261030/adaptive_dtype_diagnosis/cpu_mechanism.json`；下文真实Qwen两臂短链已补足末端保存／fresh GPU证据。

最小修复在`recovery.finalize`的decoder／PEFT结构收束之前移除训练包装；`run_pipeline.logits`分别检查dtype和finite，原BF16要求及同输入strict fresh不变。`continuation.fit_next`已接只读`initialization.observe`，记录实际两阶optimizer创建时的参数digest、形状、dtype和LR；不改变参数或RNG，不以声明seed相同替代实际摘要比较。

新机入口为`experiments.cmcc_runtime_20261030.adaptive_pipeline`，只重绑已确认旧根与当前实体路径；历史JSON、metadata、日志和归档runtime原样保留，新recipe及runtime另存。当前CPU准备在tmux`adaptive_cmcc_prepare2_20261030`完成／exit0，证据为`/data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/prepared_v2/prepared.json`及同级外侧`prepare_v2.log`：short／formal、dynamic／Rice四种来源均完整36层，真实状态、训练步数、文件SHA和逐层cap通过；runtime算法identity保持，恢复500/B4/1024，CUDA未初始化。N snapshot的五源train实际解析另由`result/migration_audit_20261030/snapshot_data_validation.json`核证。

Qwen完整BF16的五分片已按固定revision及原HF LFS SHA通过，证据`result/prepare_eval_20261030/qwen_verified.json`。14:06:50在GPU0/1分别实际启动新输出`short_dynamic`／`short_rice`，从真实raw/live和`short01+short36`重放gate/up/down，每类恢复2/500、live up/down原M36 S2×2。tmux分别为`adaptive_cmcc_dynamic_short_gpu0_20261030`／`adaptive_cmcc_rice_short_gpu1_20261030`，worker82760／82766，start_ticks7779854／7779855；完整argv／env／日志／exit及Qwen验证SHA在同根`short_launch.json`。

这次两臂均于首次恢复更新前exit1：`N train_utils.lora_training`所需`trl`未安装，真实`CanonicalCatSFTTrainer`继承`trl.SFTTrainer`，不能旁路guard。`short_dynamic`／`short_rice`的原失败report／metadata／日志保留，二者仅完成真实raw替换和`04_gate_proj/round_base`保存，不能当216native＋36compact＋0dense末态；无后继用途权重按文末收尾。旧down可选大权重未迁入，不声称exact resume。

用户随后明确授权仅安装仓库／N snapshot固定的`trl==0.12.0`。实际`--no-deps`安装exit0，未增加其它包、升级或删除现有依赖；`result/migration_audit_20261030/trl_install_verification.json`保留安装前后证据。新干净解释器的N真实`ensure_lora_training_stack_available`及`SFTTrainer`／`DataCollatorForCompletionOnlyLM`导入成功。14:24:09在GPU0/1各重新启动一次新独立输出`short_dynamic_trl01`／`short_rice_trl01`；完整CLI／环境／日志／exit／PID与start_ticks在同根`short_trl01_launch.json`，旧失败输出未覆盖。

| 臂 | tmux | worker PID／start_ticks | shell PID／start_ticks |
| --- | --- | --- | --- |
| dynamic／GPU0 | `adaptive_cmcc_dynamic_short_trl01_gpu0_20261030` | 88009／7883713 | 88006／7883713 |
| Rice／GPU1 | `adaptive_cmcc_rice_short_trl01_gpu1_20261030` | 88013／7883714 | 88011／7883714 |

两臂真实`short_dynamic_trl01`／`short_rice_trl01`均已自然exit0，`report.status=completed_short_whole_class_pipeline`；[一次性配对核证](../../result/qwen3_8b_mmlu65_20261004/cmcc_sequence_20261030/short_pair_verification.json)为`verified_actual_short_pair`／exit0，并绑定两份原报告及全部strict报告SHA。gate／up／down各实际2/500更新，第2步均有FP32 master变化，冻结持久字段exact；step1保存、回训至step2、FP32 master／Adam对象与值／scheduler／RNG保存不变均通过。每臂的三个独立fresh均为`strict_full_state_and_BF16_logits_verified`，同输入BF16 logits exact且有限；最终3927字段、216native＋36compact＋0dense，down实际仅训练decoder且无dense融合。up／down均从各自fresh边界完成原M36 pooled BSQ两阶、每阶2/10K更新；四组实际初始化digest、形状、BF16 dtype、LR.003和seed31逐项配对相同。

| 短验臂 | 最终UUID | logical Linear bpw | packed Linear bpw | 保存时完整目录bytes |
| --- | --- | --- | --- | --- |
| dynamic | `5052dadd-f12c-425a-b1a6-f02666a6cd1a` | 2.5216302376438455 | 2.5350048979699324 | 4,694,059,662 |
| Rice | `58dbf07c-b43c-4a6f-90d4-9e7638eb2a8c` | 2.52164744294208 | 2.5350221032681675 | 4,694,046,828 |

上述logical账本包含全部投影码、decoder／norm、INT8保护值／scale／索引、rotation及完整route／Rice字段；packed另含码padding，完整目录另含非投影、序列化和metadata／config，不能混称同一位宽。两臂均≤2.6；只是修复／接口与初始化合同通过，不证明动态质量收益、10K拟合效果、完整500恢复或exact resume。可复用边界已更新[checkpoint经验](../../docs/experience/lessons/checkpoint_lifecycle.md)。

一次性runner已据此进入两臂独立正式恢复，实际命令和输出由[固定plan](../../result/qwen3_8b_mmlu65_20261004/cmcc_sequence_20261030/plan.json)管理；正式恢复仍500、各自live up/down仍S2×10K，正式第三阶码流复用。后续读取各臂自身正式成品，短验不接正式九项。本次收尾不监控或重复启动后台任务。

```bash
CUDA_VISIBLE_DEVICES=0 python -B -u -m experiments.cmcc_runtime_20261030.adaptive_pipeline \
  --prepared /data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/prepared_v2 \
  --arm dynamic --short \
  --artifacts /data1/cmcc/VAELLM/artifacts/adaptive_class_20261030/short01 \
    /data1/cmcc/VAELLM/artifacts/adaptive_class_20261030/short36 \
  --output /data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/short_dynamic
```

上例为首次失败short的实际命令，不能原目录重复执行；Rice使用GPU1、`--arm rice`及独立`short_rice`。后续成功short使用上表的`*_trl01`独立输出；正式运行以固定plan为准。新机完整九项配置接入见[评测说明](../adaptive_class_evaluation_20261030/README.md)，未核完整末态时不声明完整质量通过。

本次短验收尾仅覆盖绝对根`/data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/`下四个已结束输出。删除前已核固定plan的全部312项SHA匹配、无候选文件被pin；当前runtime、后续formal／evaluation命令及`prepared_v2`／`evaluation_support`均无短验payload输入。runner已生成实际pair并进入formal，后续仅消费各臂正式输出；原Rice round_base的有限CPU／GPU消费者也已exit0。一次性检查全部可访问的相关进程cmdline／fd／mmap无候选引用；无关系统服务／内核的权限受限项不作已查声明。

| 已结束输出子目录 | 无用途payload清理范围 | 文件数 | 逻辑bytes | 已分配bytes |
| --- | --- | --- | --- | --- |
| `short_dynamic` | `04_gate_proj/round_base/pytorch_model.bin` | 1 | 10,758,363,065 | 10,758,369,280 |
| `short_rice` | `04_gate_proj/round_base/pytorch_model.bin` | 1 | 10,758,376,505 | 10,758,381,568 |
| `short_dynamic_trl01` | 下述6份模型及48份训练／tokenizer副本 | 54 | 47,116,675,464 | 47,116,804,096 |
| `short_rice_trl01` | 下述6份模型及48份训练／tokenizer副本 | 54 | 47,071,585,432 | 47,071,731,712 |

每份成功short的6份模型精确为`04_gate_proj/{round_base,after_recovery}/pytorch_model.bin`、`05_up_proj/{round_base,after_recovery}/pytorch_model.bin`、`06_down_proj/round_base/pytorch_model.bin`及`final_model/pytorch_model.bin`；48份副本精确为三个类别目录各`trainer_state/checkpoint-{1,2}/`中的`training_model_state.pt`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth`、`tokenizer.json`、`vocab.json`、`merges.txt`和`added_tokens.json`。其用途均已由原始审计、strict及pair闭合，无已安排续训／复评；删除前每个解析后绝对路径和大小逐一核对、均为普通文件且nlink=1，不跟随链接，不清理其他目录。

以上110份已逐个删除并复核不存在，移除115,705,000,466逻辑B，实际释放115,705,286,656已分配B（107.7589454651 GiB，仅计本次nlink=1文件块，不计共享盘空闲量变化）。清理后全部312项pins再次匹配，pair绑定的8份report／strict的SHA保持；四个输出分别保留4／4／58／58文件，全部report／strict／metadata／config、训练参数／审计／history、核心日志和exit、六份`same_input_deployment.pt`均在。输入raw／after_o、short01／short36、正式36层码流、支持配置以及全部后台输出不动。short目录现在只留证据、不可再加载模型；原报告保存时bytes仍是历史实测，不改写原JSON或其SHA。
