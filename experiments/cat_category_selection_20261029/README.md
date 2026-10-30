# A控制分支的gate250／500只读节点比较

本任务对应[唯一执行计划](../../docs/experience/records/w25_transfer/current_plan.md)的类别恢复节点候选。事前固定A结构，不从A/B质量结果选择结构；原A/B固定500及其它主线继续。只比较原500调度中的250／500两个实际快照，按未舍入完整八项均分选择一次；250必须严格更高才胜，同分取500。**固定A250→A500完整八项链已在GPU6实际推进**，启动核证含180／180预热与A250 MMLU2049／56168请求；尚无已验证配对结果、选择或后缀。本轮不运行后续up/down，不称九项最优或盲测；实际运行状态以各`evaluation.json`和`full8_pair_launch.json`为准。

源为[已保全的小端点和真实raw36gate](../adaptive_native_model_20261029/README.md)。A raw UUID为`94c8c25b-19dd-4c6d-852c-19f5b12c4d6b`，A250训练节点`6b99b752-2ac9-49ac-8a56-cdae212890d8`，A500训练节点`ffef68e6-14af-4afd-ab8c-c69cb23ac85c`；共同after_o UUID`b63bfe79-3889-40c6-bf66-1d002893f874`。原raw大目录可能已被CAT正常prune，入口只实际读取保留的A归档＋after_o，不解析搬家后的`../../round_base`。归档原metadata字节和相对引用均不改。

`sources.py`在隐藏CUDA的bitvae进程内，从自身config构造完整Qwen3架构、按原N specs重建native模块，恢复全部2559个原raw字段并核原始完整digest。构造不读取HF dense权重，保留FA2部署policy及FP32 RoPE；没有用临时可见GPU绕过旧`from_pretrained`的FA2前提。`export.py`根据实际immutable合同及mutable manifest重建180个prefix decoder和72个future dense rank12 PEFT，原生strict恢复全部1080 D＋144 LoRA的FP32字段，再调用未改N的decoder finalize、LoRA merge及BF16 export。没有Trainer、optimizer step或训练重放。

两节点的完整原immutable合同须strict相同；另重解析原A配方、按原数据构建前的规则调用`data.validate()`，核data／loss／optimization／aux／runtime／precision和实际LoRA拓扑。teacher和dataset identity保留原声明及文件绑定，不为只读导出新加载teacher或训练数据，不声称exact-resume。源合同作为来源记录，部署root不带旧training_step immutable合同或外置LoRA。

A250保存到新独立`/data/star/VAELLM/cat_category_selection_20261029/A250/model`，由原v6 saver生成新UUID，明确actual250／planned500及selection pending。保留前四条真实q/k/v/o历史，追加实际gate250记录；`completed_categories`在这个假设finalize边界包含gate，表示该可加载边界而非已选中或已跑满500。带原A raw runtime侧车，因原`channel_refresh_after_category=false`可复用其统计；fresh须核`lora_round_idx=5`及下一恢复seed36。up拟合shuffle500031仅是sampler seed，未来若选择250，初始化、重seed及数据／统计后缀规则仍须先固定；本任务不宣称与原连续500后缀随机轨迹相等。

A500复用实际控制`/data/star/VAELLM/cat_structure_full_20261029/A/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/after_gate_proj`，UUID`49fd6f6c-9e5b-4f09-b668-e32096f956c2`，这是本轮明确的后续评测消费者保护源。CPU完整重建／恢复／融合500节点后，与该GPU原生控制逐项比较：**2487个非dense-fusion字段exact**，含全部1080 decoder及其余冻结码／保护／骨干；72个BF16 dense权重中48字段共84个元素有差，maxabs **0.000244140625**，逐差字段relative-L2范围 **1.193025129e-9～1.339654333e-6**。原值按字段保存在报告，没有补偿或任意阈值门控，不能称跨设备融合tensor exact或已证质量等价。后续full8使用这个**原A500控制**，未保存CPU500重复大模型；两节点消费路径均为相同原native FA2/BF16/group8。

两个边界仍含72个真实dense投影，账本计入全部native码／decoder／保护／旋转、dense、其它字段、padding及完整文件字节；`final_budget_applicable=false`。仅最终无future dense的模型适用≤2.6验收，不能用上层终态门控拒绝这次正常中间节点比较。评测两者使用同一种原native FA2/BF16消费者和group8预热。

`prepare.py`执行真实CPU导出及独立fresh，然后生成完整八项的`evaluation_plan.json`；CPU原始状态与边界以`/data/star/VAELLM/cat_category_selection_20261029/report.json`为准。`evaluate.py`按root已分配的GPU6及固定顺序执行计划，调用原`lm_eval_sample_artifacts_20261010/cat_eval.py`和成熟full8汇总：0shot、MMLU batch64／七项batch32、64叶／24742例、无limit、保留全部原始样本。没有新增proxy、LongBench或后缀。

CPU有效运行tmux`cat_A_gate250_500_export_CPU02_20261029`，shell574811／worker574813；独立fresh250／500 PID575529／576505，**159.9166秒／exit0**。首次17.9841秒的CPU尝试在保存前发现`CatTrainHFTrainingArguments`没有`tf32`字段；修为按原`_build_training_arguments`的实际来源核precision：stage的bf16/fp16、cfg.opt的checkpointing、未显式传入的`TrainingArguments.tf32`类默认None，不放宽合同。失败源pins／原异常／退出已归并主报告，原独立失败散目录已清，无失败权重。

两节点各恢复1224个FP32字段、finalize180个decoder并融合72份dense LoRA；1407个持久冻结字段exact，全部1080个decoder导出与实际FP32→BF16 cast exact。各自2559字段保存／独立fresh全值／shape／dtype exact，全部参数冻结；FA2 policy及FP32 RoPE保留，history5、lora_round_idx5／下一恢复seed36正确。

| 实际评测输入 | UUID | 权重文件B | 包含全部文件的目录B |
| --- | --- | ---: | ---: |
| 新A250 | `289d1cbc-1cec-4e73-a629-c39b0f02928e` | 10,737,298,413 | 10,776,965,016 |
| 原A500 | `49fd6f6c-9e5b-4f09-b668-e32096f956c2` | 10,737,298,477 | 10,776,787,016 |

两者实际logical **9.484542183254076**／packed **9.497916843580162 bpw**，denominator6,945,767,424，padding92,897,280 bits；dense及全部部署字段均已计入。目录表另包含真实22,729,698 B CAT runtime、tokenizer和metadata，不能把这些目录字节混作投影logical bpw。新A250实际权重SHA`de6243c7b2f67f6bb292249d98f54b45ded7ada3a74a65f2ea710f0164fe9ca1`，原A500为`e00389922de0e346950b8b31258e5711afe2e7f6763c11101945cfda673d8a66`。

Astra窄审后仅补两项：新A250 metadata的`extra.stage`由误继承的`round_base`明确修为`after_category`，并对新runtime侧车序列化读回与原归档逐字段／张量核exact，实际1520个tensor／22,282,240 B及461个标量。修订前meta SHA、修订源码及scope都保留在`CPU_review_closure`／`metadata_review_revision`；没有重建或重存10.7GB权重，原full-state fresh证据仍保留，metadata读回及history重新核过。最终A250 meta SHA为`775baf24ea039ea48a90008bf49773298c0da61603e8f223b6a25a90deece446`。完整八项计划已重新绑定此SHA；旧运行源码pins与本次修订pins分别记录，不伪称先前代码已含修复。

本次实际CPU命令如下，调用者激活bitvae；核心日志`/data/star/VAELLM/cat_category_selection_20261029.log`，报告`/data/star/VAELLM/cat_category_selection_20261029/report.json`。入口绑定本次唯一新目录并拒绝覆盖；目录已有本轮配对消费者需要的成品，当前不重复执行，不自动续跑。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -B -u \
  experiments/cat_category_selection_20261029/prepare.py
```

完整八项准备已完成，`evaluation_plan.json`记录两份真实UUID／metadata SHA／权重stat／全部调用源码、两条原0shot命令及64叶24742例合同。其`CPU_prepared_GPU_not_started`是**准备时的绑定状态**，现在已有GPU实际运行，不能改已绑定plan；实际状态读各`evaluation.json`及`full8_pair_launch.json`。调用者CUDA隐藏，仅子评测进程使用root明确分配的物理GPU。CPU交付时未执行GPU推理或八项的历史状态由下节实际启动记录接续；尚无配对选择。A250新成品和原A500控制为已安排配对评测保留，原A/B小归档、runtime和after_o继续保护。没有改活动CAT、public/N源码、mixed原型或其它实验pins，没有安装或Git写操作。复用了既有CPU构造／精度／冻结经验，本轮无新的质量或优化算法经验。

## 10月29日：GPU6完整八项配对已实际启动

root分配GPU6前，已核原scale LongBench第二片worker472548／start9954922自然退出：native为`partition_complete`、外层exit0，12任务2650条预测完整，实际UUID仍为scale成品`6a446110-a963-45a5-9847-8458f5d88c40`。这只证明该分片完成，不把整体scale九项写成完成。随后实核GPU6为0MiB且无compute进程，未抢占其它任务。

在调用者已激活`bitvae`、项目根cwd、CPU协调器隐藏CUDA的条件下，唯一tmux `CAT_A250_A500_full8_GPU6_20261029`顺序执行以下两个原入口。固定有限shell仅在A250实际执行exit0后接A500，任一执行失败就结束；没有质量筛选、等待资源、重试或新队列。原0shot、MMLU B64／七项B32、64叶24742例及无limit合同保持；72个future dense仍计入账本，不新增final2.6门槛。

```bash
CUDA_VISIBLE_DEVICES= python -B -u experiments/cat_category_selection_20261029/evaluate.py --step 250 --gpu 6
CUDA_VISIBLE_DEVICES= python -B -u experiments/cat_category_selection_20261029/evaluate.py --step 500 --gpu 6
```

源plan SHA为`dcdcb85d65436a94cf95ffd622766de3f6cbf98329b96ba29cca726bb8415383`，两份UUID／metadata SHA／实际weight stat与plan一致。shell584652／start10873042；A250协调器584655／10873043，MMLU worker584910／10873561。子评测实际`CUDA_VISIBLE_DEVICES=6`，仍由原`environment(6)`生成其余环境，不改源码或做重复GPU短验／CPU模型重建。

唯一启动记录为`/data/star/VAELLM/cat_category_selection_20261029/full8_pair_launch.json`，包含准确命令、环境、原scale分片退出证据、GPU UUID、进程身份及结果路径。两次stdout／stderr分别落`A250_full8.log`／`A500_full8.log`，各外退出为`A250_full8.exit`／`A500_full8.exit`，配对外退出为`pair_full8.exit`，自然结束才写相应文件。各自完整结果在`evaluation/A250/full8/`与`evaluation/A500/full8/`，子阶段日志／命令／实际PID和退出由其`evaluation.json`保存。

有限核实确认A250原生group8预热180／180、skipped／failed0（2.02秒），实际MMLU loglikelihood连续推进至2049／56168请求。在该启动核证时A250运行中、A500尚未执行，启动核证不代表质量结论；后续实际进度以各`evaluation.json`为准。尚无已验证的完整配对结果、所选节点或后缀。已在实际计算确认后停止主动监控，不等待两份结束，不自动训练后缀；原A/B固定500及其它主线均继续。

## 两份完整结果齐备后的只读选点入口

新增独立`summarize_selection.py`，没有修改已绑定的evaluate／prepare／sources／export代码、`evaluation_plan.json`或活动评测输出。它只读取这份冻结计划、两份`evaluation.json`、各`full8_summary.json`及其绑定的两个小result JSON；不读大模型，不重扫逐题预测，不启动GPU或后缀训练。

必须两份都为`complete_original_full8`／exit0、64叶／24742例，实际stage命令／UUID／目录／source-plan SHA均与原计划一致，summary的UUID／8任务及metric key／输入结果SHA／版本／未舍入均分绑定全部成立，才产生一次`/data/star/VAELLM/cat_category_selection_20261029/selection.json`，拒绝覆盖。复用每臂完成时已执行的raw artifact核证；正常summary算术沿原`rel_tol=0/abs_tol=1e-12`检查，选点本身用`Fraction(str(task_metric))`精确比较八项已记录分数的等权均值：250严格更高才选250，否则选500，exact tie取500，没有新增质量门槛。

输出记录两UUID、完整八项、分项及均分差的精确分数／百分点值、所有小结果SHA、固定规则及`selected_candidate`，明确开发选点次数1、不证明九项或后缀。root在真实两份完成后，激活bitvae并通过tmux／nohup调用：

```bash
CUDA_VISIBLE_DEVICES= python -B -u experiments/cat_category_selection_20261029/summarize_selection.py
```

准备时仅完成AST／import及**真实缺结果拒绝**最小检查：tmux`cat_A_selection_summary_check_CPU_20261029`，PID589370，bitvae／CUDA隐藏，检查exit0；当时A250未满足完整终态，入口明确拒绝，没有生成`selection.json`，没有伪造结果测试。冻结plan SHA仍为`dcdcb85d65436a94cf95ffd622766de3f6cbf98329b96ba29cca726bb8415383`，检查前后相同。检查日志为`/data/star/VAELLM/cat_category_selection_summary_check.log`；上述是入口准备证据，正向选择的真实结果现由下节接续。

## 10月29日：完整配对选A500及A250产物收尾

两节点均已真实`complete_original_full8`／exit0，64叶／24742题；A250未舍入八项为68.72134111415737，原A500为68.72428636310474。独立`selection.json`已完整产生，按事前精确规则选择 **A500**，A250−A500为−0.00294524894735625pp；这是一次开发选点，不证明九项最优或后缀收益，不因近分结果追加频率或训练250后缀。原A/B固定500主线继续。

已确认独立A250没有后继续训、复评或交付用途；实际命令引用、打开FD和内存映射均未发现消费者。夜间只把原节点链作为GPU6释放的资源依赖，读取`pair_full8.exit`，不消费A250权重；这个exit0文件明确保留，不改变夜间任务／manifest或停止任何进程。

本次仅从`/data/star/VAELLM/cat_category_selection_20261029/A250/model/`删除8份无用生成副本：`pytorch_model.bin`、`cat_runtime_state.pt`和6份重复tokenizer文件。原始权重SHA已实际复核为`de6243c7b2f67f6bb292249d98f54b45ded7ada3a74a65f2ea710f0164fe9ca1`后删除；该目录只保留`config.json`和`checkpoint_meta.json`，**A250现不可再直接加载**。原完整CPU重建／fresh证明、源码身份和历史源路径保持，不将其写成现存可加载成品。

另将`A250_full8.exit`／`A500_full8.exit`／`selection.exit`三个自有exit0归入唯一`report.json.cleanup`后删除散文件。合计11文件、逻辑10,775,910,539 B、实际allocated释放 **10,775,945,216 B**；逐文件精确路径、原大小／blocks／inode／SHA、原因及删除核验均在该cleanup，不另建清单。

删除后157份受保护文件SHA保持，包括两份完整评测、全部逐题预测、summary、selection、冻结plan、源config／metadata、启动记录及核心日志；原A500大权重stat也保持。所有原A/B初态、raw与小节点归档、runtime原件、after_o、G03/G09及其它当前源未触碰。现夜间资源依赖与保留结果仍可用，本次不新增质量、因果或容量结论。
