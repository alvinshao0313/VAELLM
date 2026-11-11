# C1 二阶段异源科学题单臂

本轮只检验一个数据问题：原C1 raw fresh、32次原配方Muon/在线OPD恢复中，将每组M32的后16题换成固定SciQ train，是否比指定原C1 Muon32同时提高完整九项与MMLU。新一阶段重新压缩基线优先，本臂只用空余GPU4；不占用GPU0/2、GPU1的E256或GPU6/7线程，不等待或停止其它任务。

共同父为 `744e3b98-8a09-4a1d-8b30-013db5d73d44`，控制为原C1 Muon32 `3ebcc40c-e572-41db-96c9-1dc58be564a9`，九项64.89598790075989、MMLU61.57954707306651、LongBench43.91154232360084。复用有效控制，不重训、不从full_teacher或任何恢复终点接续。

唯一变量是M数据：原32组各保留前16个auxiliary_train（共512），后16个槽顺次填入SciQ512。原QA每组8例、身份、顺序、权重保持，独立Qasper32/Hotpot96、呈现64/192。M和QA域各`.5`，M保持原条件四项`.5 teacher + .5 gold`一次混合；保留aux512直接复用原mixed缓存。新SciQ教师使用相同原始HF BF16、FA2、FP32 RoPE、逐例B1、最后query四选条件softmax，再由原helper混合一次。QA仍为当前学生greedy OPD和原HF全词表KL，生成轨迹和实际监督tokens允许反馈变化，不能称逐值相同。

全部252投影D、R0、Muon NS5/`.95` Nesterov/aux Adam、首个真实联合梯度按原规则校准κ、joint clip1.5、seed137、原500 linear前32更新、1×LR保持。训练、保存和fresh直接调用原 `rice_muon_cmcc_20261031`，只在新进程绑定独立数据/profile；不修改既有源码、prepared或pins。部署的码/Rice/保护/旋转/非D冻结和logical/packed/整模型真实存储契约保持，不能将理论位宽当实测。

数据来源是[allenai/sciq固定revision](https://huggingface.co/datasets/allenai/sciq/tree/2c94ad3e1aafab77146f384e23536f97a4849815)，`train`11679行，CC-BY-NC-3.0；本机[capture](../../data/allenai_sciq_2c94ad3e1aafab77146f384e23536f97a4849815/capture.json)记录固定版本、文件大小和官方LFS SHA。Parquet3993099B、SHA256 `19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167`。仅按既有`sha256('causal_output_fit/137/'+case_id)`排序前512，`case_id=allenai_sciq/train/{row}`，无筛题、无测试集选样。答案位置调用既有`e2e_common.data._stable_sciq_answer_index(stripped_question)`，correct插入原distractor1/2/3。每题实际使用原native无subject四选模板、PairTokenizer完整pair，**support不进入输入**。

CPU逐例核真实native token/四选gold/最后query、全部32组保留aux和QA逐值顺序权重、原父/参数/源码绑定。SciQ512共18977输入tokens，24–74/题；保留aux512为182935，被替换aux512原为181599，M输入总量364534→201912，1024答案位、32更新、最长958仍保持。这不是等token或等算力比较。实际QA prompt与OPD生成/监督tokens在数据证据和正式报告分别记录。

复用相同原C1/Rice/Muon两更新、切换、保存及strict fresh工程证据，不重做GPU short或exact-resume；只对新增cache做真实save/read、一次混合、归一化/有限性、教师不变及严格释放检查。额外读出事前固定first32所选SciQ和first32保留aux，在raw、指定原Muon32和新终态各64次B1前向，共192次；报告条件teacher KL/gold CE/准确率，仅作训练拟合诊断，迁移取舍以完整九项/MMLU为准。读出排在正式训练和完整九项之后，诊断失败单列，不阻塞可进行的正式部分。

预算：新教师cache15–30分钟预留，正式32更新上限4小时，fresh后GPU4完整九项约6小时（估计，沿成熟timeout）；读出最多192小前向。无retry、占卡、ratio/seed/step网格。正式后台确认真实M+QA联合更新、有限梯度、实际LR及FP32/BF16存活并进入下一组后交付停止主动监控。

只有完整九项和M均高于控制才支持进一步知识恢复；M涨而九项降不采用，九项涨M不涨仅作综合候选。无净收益结束本512/512配置，不据单seed或训练loss推广结论。最终跨模型、原始BF16整链和总目标验收保持未完成，等待独立证据。

经验依据：[KD与数据边界](../../docs/experience/lessons/kd_and_data.md)要求按同父、完整九项和真实曝光取舍；同源M4096完整净负和C1目标/LR仅小净差支持本次不同合法来源的单变量问题；[原C1控制与工程证据](../rice_muon_cmcc_20261031/README.md)复用；[C1目标/LR结果及BF16 teacher batch差](../c1_m_target_muon_cmcc_20261102/README.md)要求新SciQ严格保留原B1教师执行，不混入M32差异。跨模型、比例最优或数据瓶颈均属未验证假设。

复现顺序（调用者先激活`bitvae`，所有Python/测试/实验通过tmux/nohup后台运行）：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_sciq_recovery_cmcc_20261103.prepare
CUDA_VISIBLE_DEVICES=4 python -B -u -m experiments.c1_sciq_recovery_cmcc_20261103.cache
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_sciq_recovery_cmcc_20261103.prepare --finalize
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.c1_sciq_recovery_cmcc_20261103.sequence
```

独立artifact为 `/data1/cmcc/VAELLM/artifacts/c1_sciq_recovery_cmcc_20261103/`；结果、plan/run、core logs、退出及完整评测为[独立结果目录](../../result/qwen3_8b_mmlu65_20261004/c1_sciq_recovery_cmcc_20261103/)。`dataset.json`保存CPU证据、`teacher/cache.json`保存新增cache证明、`prepared.json`冻结源码和输入，`plan.json`/`run.json`记录真实命令、PID/start ticks、退出和比较。未提交源码按SHA绑定，不虚构commit ID。

CPU数据准备已exit0（57.39秒），payload15125898B、SHA256 `7288a3a6f667fea4806790db60f3b2b2f8cc69329085bbc7dd97c6a8ee879bb8`；source/reader CPU核证exit0。Astra/ultra核心窄审通过，放行新增cache→finalize→正式链。审查修复仅为cache尾`options` GPU张量及时释放；一次过早cache启动在CPU HF shard SHA读取时被中止（exit130），尚无teacher目录、模型加载、GPU前向或父/学生更新。旧新源码SHA与数据payload不变证明保存在`dataset.json/source_pin_revision`，重绑后CPU核证exit0。

新增cache实际exit0（加载/计算内计时59.40秒），512逐例B1、一次混合、归一化/有限性、save/read exact、教师不变及完全释放全部通过；GPU4峰allocated16397931520B、reserved16406020096B。CPU finalize主体完成保留512目标逐值和完整记录核证并写出prepared；其末尾重读source因默认大线程池在4核下过度调度而中断，实际栈在C1 source inspect的Rice CPU拓扑重建，exit130保留。只把未闭合的`read_prepared`在独立位置以OMP/MKL4和`torch.set_num_threads(4)`复核，28.83秒exit0，见`finalize_reader.json`；source/pins、dataset和GPU cache均未改变或重跑。

11月3日11:00（北京时间）正式tmux `C1_SciQ_g4_20261103` 已启动，controller PID2030823/start41223274、formal worker PID2031181/start41226883；真实身份、argv、源码SHA及阶段退出由`plan.json`/`run.json`记录。[启动核证](../../result/qwen3_8b_mmlu65_20261004/c1_sciq_recovery_cmcc_20261103/startup_verification.json)已通过：首组M32＋QA8真实联合更新，M loss `.1206929535`、QA `.2228287258`，clip前norm232.8102112、κ5.56793103，实际Muon/aux Adam LR `5.56793103e-6/1e-6`，576 Muon＋1152 Adam状态step1有限。FP32位移L2 `.0023064797`，BF16改变196685元素；首更新257.92秒，峰allocated35915792896B/reserved58307117056B。随后group1真实在线rollouts与F targets已完成（8文档、195生成tokens），原路径master/optimizer/κ连续检查通过，并已进入下一组反向。

正式32更新与训练导出已exit0，外层8093.94秒（内部训练/导出记录8056.67秒）；最终UUID `b34ce6be-6a3f-4e9e-ba3c-4b471529efd6`。全252 D、1728叶，原teacher保持、Rice冻结、保存decoder与最终FP32转换逐值一致；32组均完成，实际QA呈现为Qasper64次/32独立文档、HotpotQA192次/96独立文档，监督1288/1000 tokens，训练报告保留完整轨迹。独立fresh另79.32秒/exit0，3927字段strict/完整状态、同完整输入BF16 logits exact且有限，2199冻结字段与Rice载荷保持。原件为 [formal报告](../../artifacts/c1_sciq_recovery_cmcc_20261103/formal32/report.json) 和 [fresh审计](../../artifacts/c1_sciq_recovery_cmcc_20261103/formal32/fresh_audit.json)；正式报告按既有schema保留`fresh_pending`名称，fresh完成以独立审计和序列真实exit0为证，不回写历史来源。

fresh实测logical2.5216473392818286/packed2.5350219996079155，全部252投影压缩、0 pending dense；full-model tensor4,690,895,025B、完整部署目录4,694,097,147B，完整计入既有编码/decoder/保护/旋转/Rice路由及packed padding。13:17:34固定链已自动启动同UUID完整九项，controller2030823/start41223274，评测worker2117631/start42044249，GPU4；实际命令/退出/后继见 [唯一run](../../result/qwen3_8b_mmlu65_20261004/c1_sciq_recovery_cmcc_20261103/run.json)。13:22有限启动核已见MMLU worker2118754实际loglikelihood主循环11265/56168请求（约20%），原B64/14042题协议保持，GPU4正常计算，停止主动监控。以上为13:22启动节点；其后完整九项及事前192次读出均已完成，终态见下文。仍按指定原Muon32控制和原停止条件收束，不扩比例/seed/步数。

完整MMLU已exit0：57学科/14042题、原B64/0-shot/limitNone、同最终UUID，得分61.57954707306651（8647题），与指定原Muon32控制完全持平，差0pp；样本manifest共14042条。原件见 [MMLU结果](../../result/qwen3_8b_mmlu65_20261004/c1_sciq_recovery_cmcc_20261103/evaluation/full8/mmlu.json)。该分项当前无净提升。其余七项也已exit0，同UUID完整64叶/24742题八项均分67.26592438266026，原Muon32控制67.51904359790477，差−0.2531192152445101pp；原metric keys/版本一致，未舍入均分可由八项原值重算，见 [完整八项](../../result/qwen3_8b_mmlu65_20261004/c1_sciq_recovery_cmcc_20261103/evaluation/full8/full8_summary.json)。LongBench及完整九项于19:07:57自然结束，原评测worker2117631/start42044249及预测2125751/start42109782均退出；`evaluation.status=complete_full8_full21_current_goal`，full9、MMLU、seven、LB outer、bind及merge均exit0。最终同UUID的[九项验收](../../result/qwen3_8b_mmlu65_20261004/c1_sciq_recovery_cmcc_20261103/evaluation/longbench/merged/nine_task_acceptance.json)与[指定原C1 Muon32控制](../../result/qwen3_8b_mmlu65_20261004/rice_muon_cmcc_20261031/evaluation/longbench/merged/nine_task_acceptance.json)比较如下：

| 完整任务（分） | 原C1 Muon32 | SciQ512/aux512 | 本臂−控制（pp／分） |
| --- | ---: | ---: | ---: |
| boolq | 83.3944954128 | 83.4250764526 | +0.0305810398 |
| rte | 80.1444043321 | 79.4223826715 | -0.7220216606 |
| winogrande | 67.1665351223 | 66.4561957380 | -0.7103393844 |
| arc_easy | 79.6717171717 | 79.4191919192 | -0.2525252525 |
| arc_challenge | 52.6450511945 | 52.4744027304 | -0.1706484642 |
| openbookqa | 39.0000000000 | 38.8000000000 | -0.2000000000 |
| piqa | 76.5505984766 | 76.5505984766 | +0.0000000000 |
| mmlu | 61.5795470731 | 61.5795470731 | +0.0000000000 |
| longbench_v1 | 43.9115423236 | 43.8288064889 | -0.0827358347 |
| 九项综合 | 64.8959879008 | 64.6618001722 | -0.2341877285 |

主结果九项−0.23418772851206882pp、MMLU仍8647/14042持平、LongBench21任务宏平均−0.08273583465254575分，未满足“九项与M均高于控制”的预指定支持条件。主评测成功，质量`not_met`／验收码1只表示目标未达：对BF16九项68.04714491790848下降3.385344745660659分，超过严格<0.9界；实测预算通过。九项是混合官方任务指标的综合，不是统一accuracy。权威原控制LB为43.91154232360084，原文早期近似数字已按原件校正，不改原JSON或评分。

完整性核证沿已通过fresh：formal／fresh／full8／LB／九项均为`b34ce6be-6a3f-4e9e-ba3c-4b471529efd6`，最终metadata SHA与评测绑定一致。候选和控制各64叶／24742及LB21／4750的记录SHA、非空行数全部核对；下游doc_id/doc_hash/prompt_hash/target_hash逐条同身份，full8推理设置／B64-B32及metric keys同协议。LB逐条dataset／row_index／_id／答案／input_ids_sha256／输入截断／max_new_tokens／generation_options一致，官方协议及数据archive SHA相同，保持原0-shot、32K及21任务无权宏平均。仅CPU核原始记录，不重评分、重训或加载模型。

账本与独立fresh一致：logical2.5216473392818286／packed2.5350219996079155bpw；全tensor4,690,895,025B、序列化权重4,692,249,325B、完整目录4,694,097,147B（实核文件大小），含既有code、decoder、保护indices/values/scales、rotation、Rice索引/值/scale/zero/stream元数据及packed padding，投影之外embedding/head/norm另纳入全模型。没有通过只报理论码率隐藏Rice存储。

[预定train读出](../../result/qwen3_8b_mmlu65_20261004/c1_sciq_recovery_cmcc_20261103/readout.json)亦自然完成：schema=`C1_SciQ_fixed_train_readout_v1`、status=complete，192次B1前向，准确字段为`results.{raw,control,candidate}.summary.{sciq,aux}`。三份模型UUID分别为父744e3b98…、控制3ebcc40c…、候选b34ce6be…；每域固定first32选样、逐例case_id／input_signature相同，三个模型state_unchanged=true，summary由64行原读出复算相同。该样本来自本轮SciQ train和保留auxiliary_train，属于训练拟合诊断，不是held-out迁移证据。

| 固定train域／指标 | raw | 原Muon32控制 | SciQ候选 | 候选−控制 |
| --- | ---: | ---: | ---: | ---: |
| sciq／conditional_teacher_KL | 0.3308113886 | 0.2874755742 | 0.0608807914 | -0.2265947828 |
| sciq／conditional_gold_CE | 0.4133793567 | 0.3605085045 | 0.1005066797 | -0.2600018247 |
| sciq／accuracy_percent | 87.5000000000 | 84.3750000000 | 96.8750000000 | +12.5000000000 |
| aux／conditional_teacher_KL | 0.2506083090 | 0.1888533810 | 0.2041395366 | +0.0152861557 |
| aux／conditional_gold_CE | 0.2954236487 | 0.1366251756 | 0.1528339641 | +0.0162087885 |
| aux／accuracy_percent | 87.5000000000 | 100.0000000000 | 100.0000000000 | +0.0000000000 |

逐例按wrong→correct为修复、correct→wrong为损坏：候选对控制在SciQ修复4／损坏0（27→31/32），aux修复0／损坏0（32→32/32）；候选对raw分别修复3／4、均损坏0。SciQ训练域KL/CE明显改善，保留aux准确率虽持平100%，KL/CE仍分别恶化0.0152861557／0.0162087885；不能把训练命中修复换成完整九项收益，也不能因准确率饱和称保留域分布完全无损。

实际成本：正式32外层8093.9448秒（2.2483小时）、fresh79.3236秒、完整九项21022.6605秒（5.8396小时）；读出内部137.3698秒、外层176.4970秒。四阶段实际外层合计29372.4259秒（8.1590小时），不含先前CPU／新增teacher cache和固定前驱。19:11:03唯一run已`completed_C1_SciQ512_aux512_formal_fresh_full9`／exit0，controller2030823及readout2366803/start44146543均自然退出；阶段、读出和退出均真实完成，不将诊断状态混同九项状态。唯一run按原停止条件记录`end_this512_512_configuration`／knowledge_recovery_supported=false；root／Astra已审定结束此512/512固定配置，不采用、预算结束，原三阶段条件卡不触发保持或32→64。结论只限本次同父／预算配方，不外推异源数据或SciQ一般无效，不扩比例／seed／步数。192次读出已用完全部端点，candidate无具体后继／复评用途，按既有授权完成下方最小收尾；共同raw／原Muon32控制、全部原指标／预测／readout／退出绑定保护。未新增任何GPU任务，交付后停止监控。

收尾盘点：原数据下载/设计盘点已exit0，全部唯一信息并入原始`capture.json`、`design_inventory.json`及本文；无活动进程引用后清理`/tmp/vaellm_sciq_inventory_1103.py`（5242B）、同名前缀`.log`（1202B）与`.exit`（2B），合计6446B逻辑/16384B allocated。另19个已结束的一次性脚本、临时线程限制回执和重复验证日志已合并到`prepare.log`/`source_audit.json`/`finalize_reader.json`后精确清理，35474B逻辑/94208B allocated；后者保留精确清单、原因和前后保护文件SHA。最后6个启动临时核证脚本/重复日志同法收尾，9124B逻辑/28672B allocated，清单归`startup_verification.json`。三批共28文件、51044B逻辑/139264B（136KiB）allocated；两个自启且已空闲的CPU/cache验证tmux已关闭，正式tmux保留。官方原始数据、source/pins、有效cache及在途权重全部保护。主评测与预定train读出均已完成；本次新增证据限于该512/512固定配置未获完整收益，以及训练域拟合与保留域分布、下游收益不能互换的条件性边界。本固定点不采用、预算结束；正式生成权重无后续用途，精确收尾见下段。

最终收尾已完成：确认外层／评测／读出自然exit0、无FD/mapping或外部实验代码／夜间后继依赖后，精确删除本SciQ包生成的`formal32/final_model/pytorch_model.bin`、用完fresh夹具`formal32/same_input_output.pt`、本包tokenized混合载荷`dataset.pt`、本包SciQ教师载荷`teacher/targets.pt`，以及`fresh.log`／`full9.log`／`readout.log`三份已归并wrapper日志，共7文件，逻辑4,708,074,991B，按删除前st_blocks实释4,708,093,952B（4.38475418GiB allocated）。本包未另生成独立optimizer／scheduler／RNG checkpoint，未将不存在文件计入释放量，也不与前述136KiB历史清理重复计数。

精确路径／原因／字节、无reader与无悬空symlink检查、172份保护证据SHA、原raw／原Muon32控制weight stat和metadata SHA、4份官方SciQ来源文件SHA前后核查并入既有[启动核证](../../result/qwen3_8b_mmlu65_20261004/c1_sciq_recovery_cmcc_20261103/startup_verification.json)的`terminal_cleanup`，未新建清单体系。保留唯一实际prepared、所有配置／源码绑定、formal报告／execution／expected_state／fresh_audit／部署metadata、核心formal/cache/prepare/controller日志、全部九项原指标／预测／readout及退出绑定；`teacher/cache.json`和`dataset.json`保留选择、来源、目标hash与token证明。官方SciQ原始data、共同C1 raw／原Muon32控制、原aux/shared teacher cache和其它包均未触及，保护证据核查未变。

本包`final_model`现为metadata／配置证据集合，不能再次加载或resume；保留JSON中的载荷路径／SHA是原实验来源记录，不表示已删除数值载荷仍有效。正式模型与own teacher/tokenized payload无具体后继用途，按已确认结束决定释放；新夜间组合只读原C0 after_up，与这些SciQ载荷无依赖。固定点不采用、预算结束，训练诊断事实保留，不泛化异源数据无效。
