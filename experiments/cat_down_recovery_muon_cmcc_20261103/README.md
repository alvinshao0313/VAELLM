# 原C0末down类别500恢复：Muon与原AdamW

本包只回答一阶段**最后down类别恢复**的optimizer问题。父为原C0 down raw `ed507bcb-9b83-4e4c-9859-daaef44d80d2`；控制复用原AdamW500完整成品 `cbb15c9f-1c1f-4394-817c-65c226d76ce7` 的完整九项63.58416595020284，不重训控制，不借Muon32在线OPD目标。该末类配对不能称为全部七类别已验证。

唯一覆盖入口是新进程内 `MuonMixedCatTrainer.create_optimizer`，仍用原MixedCatTrainer／N canonical Trainer的数据、教师、损失、归约、BF16计算、FP32 all252 D master、R0、B4/max1024、clip1.5、stage_seed37/data_seed31、500 cosine/warmup.01及常规save250/500。HF构造配置保留合法`adamw_torch`作为原stage参照；新报告、训练保存manifest和部署provenance明确记录**effective Muon override**、κ、实际辅助AdamW和入口，不冒称optimizer exact或伪装原运行身份。

按真实native `_parallel_stage_layout`与compact逐stage decoder metadata拆504 decoder stages、1008独立矩阵（576实际参数叶）；1152个norm/bias叶保留原辅助AdamW(.9,.999)/eps1e-8/WD0／decoder baseLR1e-5，不按ndim猜矩阵。Muon复用成熟NS5、μ.95、Nesterov和`.2sqrt(maxshape)`数学。新adapter继承`torch.optim.Optimizer`，标准`state`/`param_groups`持有全部真实状态；辅助计算直接调用原torch AdamW，不能把旧非标准Mu32组合optimizer接给HF保存。

κ每类别一个、全矩阵共同固定：首真实已clip梯度，在克隆中以非零baseLR1e-5计算fresh原AdamW和baseMuon的未舍入FP32候选总Frobenius norm比，不改真实parameter/gradient/optimizer state/RNG。外部param_group lr、initial_lr和HF scheduler base_lrs始终原1e-5；矩阵step内部用`effective_lr=scheduled_lr*kappa`，aux用原scheduled_lr。真实step1保持warmup LR0，两个optimizer状态推进到1而master不动；step2原正LR开始真实变化。κ进入可序列化param_group，日志同时记录base/effective LR，不扫倍率。

新路径必要核证：CPU从真实原父核布局、矩阵独立性、全部FP32 D和aux、原真实saved TrainingArguments；短验用原500日程前2更新、完整252D真实B4/max1024，save1后回训step2，标准Mu/Adam状态与master/scheduler/RNG保存exact、冻结码/保护/rotation及独立BF16部署fresh。短验不作质量筛选，不做exact-resume。短验和正式各自从原raw fresh，不能接续短验权重或optimizer。

预算只一个1×正式点：短验＋fresh共享30分钟；核心窄审及必要证据通过后，原父fresh500／保存／fresh共享12小时，随后GPU3完整九项8小时。新入口用标准外层`timeout`执行这些限制；full9同时把同一绝对deadline传入所有子命令，进程内包原`make_plan`使脱离外层进程组的LB unit三处时间值同步：outer截至全链deadline前60秒，native再保留原180秒终止余量。原子任务/评测采样协议不变。无auto retry、占卡、LR/seed网格。技术失败停止依赖；质量无净收益结束该固定末类配置，不据此否定所有类别或VAE optimizer研究。结果比较同原AdamW500的完整九项、MMLU/LongBench及其余代价和实际bit/存储，训练loss与短验不能替代质量结论。若有净收益只形成本父末类候选，推广全部七类另需完整证据；BF16整链和跨模型目标仍未完成。

GPU3是本臂唯一候选资源，启动前实际核占用；不使用GPU0的D16、1的E256、2的VAE、4的SciQ或6/7任务，不改它们依赖的源码、prepared、pins、数据或产物。独立artifact为 `/data1/cmcc/VAELLM/artifacts/cat_down_recovery_muon_cmcc_20261103/`，结果/plan/run/真实PID/start ticks/命令/源码SHA/退出与完整评测在[本包结果目录](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/)。

经验依据：[阶段optimizer的证据边界](../../docs/experience/lessons/compression_choices.md#超参作用与故障诊断2026-10-29)明确已有Mu32不能替代VAE／类别恢复；[原C0流程与控制](../fresh_fixed2_cmcc_20261030/README.md)固定原父和完整质量；[成熟Muon真实分组](../muon_compact_cmcc_20261030/README.md)复用矩阵数学。本次不同的是类别目标、原AdamW beta2=.999/base1e-5与warmup LR0，故不能直接复制旧OPD κ／aux／日程。当前完整八项已完成并呈本固定点负差，LongBench／最终九项仍待；证据增量见下文，不推广全部七类别或其它Muon目标。

调用者先激活`bitvae`；以下Python均由tmux/nohup后台执行，激活不写入实验脚本：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_down_recovery_muon_cmcc_20261103.prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_down_recovery_muon_cmcc_20261103.sequence --phase short
# 必要新证据和Astra窄审通过后，从原raw独立fresh正式500。
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_down_recovery_muon_cmcc_20261103.sequence --phase formal
```

CPU当前源码核证已exit0（17.68秒），真实1728 FP32 D／5336064参数、576矩阵叶／1008独立矩阵、1152 aux、每stage独立布局和原实际saved AdamW args/N canonical来源通过，CUDA未初始化。最初stage比较只差四处tuple/list容器（data sources/weights、FP32 components、inactive_fields），逐字段证据确认JSON值全同后，双方JSON规范化且仅忽略`output_dir`，其余字段仍精确比较；失败日志与诊断另存，不跳过语义。

资源入口CPU审计也已exit0：真实原C0 merge_plan副本模拟已用3600秒后，每个LB unit的native/outer/argv为24960/25140/24960秒，指向同一deadline，保持60秒清理边界及原180秒native余量；还原三处时间值后完整原plan逐值相同。仅新evaluate/sequence两个相关pin重绑并保留旧新SHA，原布局/optimizer证明复用。Astra/ultra核心和入口窄审已通过，放行GPU3短验，全部必要短验通过后直接原父fresh正式500，不重复数值审批。

GPU3真实短链已全部exit0：[最小核证](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/short_verification.json)记录κ5.55604146，首步used LR0、state1且FP32/BF16均0变化；第2步原base LR2e-6、矩阵effective LR1.11120829e-5，FP32改变5324451元素，BF16改变383039元素。save1→回训2和save2的1728标准Mu/Adam状态、FP32 master、scheduler/RNG exact，冻结码/保护/rotation保持；3675字段独立fresh、同输入BF16 logits exact。训练117.25秒、训练导出168.09秒、fresh26.70秒、短链213.67秒，峰allocated35716548608B/reserved36207329280B；短验不证明质量或exact-resume。

原dynamic padding未改变：实际首批为`[4,456]`、cap1024。一次性汇总中固定`[4,1024]`的错误断言已更正，活动源码/pins和采样未改；原C0完整500/max1024前后向证据与本次新optimizer实际状态证据共同复用，未重跑GPU。长度上限与实际批形状的通用规则统一见[动态padding经验](../../docs/experience/lessons/memory_precision.md#动态padding的长度上限与实际批形状)。

按预定预算，短验通过后已启动GPU3正式tmux `C0_downMu_formal_g3_20261103`，独立从原raw fresh500→fresh→同UUID完整九项，不继承短状态。[正式启动核证](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/startup_verification.json)已exit0：controller2079726/start41654696，timeout launcher2080027/start41656313，实际worker2080028/start41656313。正式κ独立重算为5.55586025193；step1实际LR0、状态推进且master不动，step2实际base LR2e-6／矩阵effective LR1.11117205e-5，FP32改变5324570、BF16改变382925元素，576 Muon／1152 AdamW／1008矩阵状态step2有限，原500日程和save250/500保持。

11月3日18:01有限核证：[正式run](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/formal_run.json)已转为`full9_running`，formal及formal_fresh均exit0；原worker2080028／launcher2080027及fresh2315727／timeout2315726均自然退出。实际[formal报告](../../artifacts/cat_down_recovery_muon_cmcc_20261103/formal/report.json)为`completed_original_C0_down_Muon_formal`，actual_updates／actual_steps／schedule均500。`original_stage_reference`与prepared原stage逐值相同，contract亦相同；optimizer是明确记录的Muon override，未声称原AdamW optimizer exact。

正式κ=5.555860251931083由本次首真实梯度独立标定，576 Muon叶／1152 AdamW叶、1008独立矩阵全部推进到step500且FP32有限；首步LR0／master与BF16均不动，次步原baseLR2e-6／矩阵effectiveLR1.1111720503862169e-5，末步仍有实际master更新。aux实际group LR1e-5、WD0、betas(.9,.999)保持；effective optimizer、κ、override与reference-only标记在正式报告、末500保存manifest及部署provenance逐值一致，未把torch默认字段中的LR/WD误当实际group配置。

冻结持久字段保持exact；250/500常规保存均记录1728状态叶、FP32 masters exact、保存前后optimizer对象／scheduler／RNG不变。原`save_total_limit=1`使250权重随500正常轮换，其审计仍在报告；500 manifest及各保存文件实际大小与报告一致。复用已有save→回训及独立fresh证据，未新增模型加载、GPU验证或exact-resume审计。Trainer实际20,705.5023秒，report含训练／导出20,756.5485秒，外层formal20,774.9371秒（5.7708小时）；独立fresh另26.6726秒，训练loss0.3871627432仅作记录，不作质量取舍。

最终UUID `876b2fd6-d81b-444a-8b36-323b5ee946b3`，[独立fresh](../../artifacts/cat_down_recovery_muon_cmcc_20261103/formal/strict_reload.json)exit0，3675持久字段及同输入BF16 logits exact，group8 warm216/216，无失败，部署digest `48509138cb0868692565401bd002b15afdaee5814bdf554b10caff35fbfe0981`。实际252投影＝216 native＋36 compact，无pending dense／实际LoRA；logical2.4987414816151494／packed2.5121161419412363bpw，含码、decoder、保护值／indices／scales、rotation／route frame，packed含92,897,280 padding bits。全模型tensor4,671,007,680B（含投影外embedding/head及norm），序列化权重4,672,277,077B，完整目录4,674,035,741B，后者已核实际文件大小，不能混称投影位宽。

完整九项17:59:37自动接续GPU3，timeout `2316121/start43736514`→评测worker `2316122/start43736514`；同UUID及metadata SHA与正式导出／fresh一致，保留原8小时绝对deadline。MMLU timeout `2316704/start43740143`、实际worker `2316705/start43740143`已进入loglikelihood主循环，18:02:46实见8961/56168请求；截至本次有限核查尚无完整九项质量结论，原始进度见[evaluation.json](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/evaluation/evaluation.json)和同目录mmlu.log。真实命令与预算仍由formal_plan/formal_run管理；本次只核结束的500＋fresh及自动评测启动，随后停止主动监控，不等待整份九项、不启动新任务或清理正在具体使用的权重。

## 完整结果与核证（11月4日）

11月03日23:54:22固定序列自然结束，权威run为`completed_formal_sequence`／exit0。11月4日CPU实核controller 2079726、正式训练／fresh和full9相关timeout／worker均已退出；正式报告、外层各stage和LB outer均exit0，未只凭PID消失判成功。`nine_task.exit=1`与`not_met`严格对应质量未达，非技术失败；完整九项已完成，不再等待。

最终同UUID `876b2fd6-d81b-444a-8b36-323b5ee946b3` 的[完整八项](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/evaluation/full8/full8_summary.json)、[LongBench21](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/evaluation/longbench/merged/scores.json)、[九项验收](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/evaluation/longbench/merged/nine_task_acceptance.json)与原C0 `cbb15c9f-1c1f-4394-817c-65c226d76ce7`完成配对：双方full8各64叶／24742条原逐题文件SHA、行数全部匹配各自记录，doc_id/doc_hash/prompt_hash/target_hash逐条同身份，M57科14042及其它7叶10700完整；task版本、metric keys、B64/B32、0-shot／limit=None保持。LB双方21项／4750条预测SHA与行数逐项匹配各自记录，逐题_id／row_index／answers／input_ids_sha256、生成参数和截断后输入身份均一致，官方代码／数据revision／32768窗口／seed42／无采样协议及archive SHA相同。只复算已有八项、LB21 task macro和九项均值、累加M正确数，未重新评分或运行模型。

| 完整任务（分） | 原C0 AdamW500 | 类别Muon500 | 本臂−C0（pp） |
| --- | ---: | ---: | ---: |
| mmlu | 62.3985187295 | 58.0828941746 | -4.3156245549 |
| arc_challenge | 53.9249146758 | 50.4266211604 | -3.4982935154 |
| arc_easy | 78.5774410774 | 76.9360269360 | -1.6414141414 |
| boolq | 82.3241590214 | 82.2935779817 | -0.0305810398 |
| openbookqa | 37.4000000000 | 38.4000000000 | +1.0000000000 |
| piqa | 75.1904243743 | 75.4080522307 | +0.2176278564 |
| rte | 74.0072202166 | 79.7833935018 | +5.7761732852 |
| winogrande | 66.6929755328 | 64.9565903710 | -1.7363851618 |
| longbench_v1 | 41.7418399240 | 38.9586834402 | -2.7831564838 |
| 八项均分 | 66.3144567035 | 65.7858945445 | -0.5285621590 |
| 九项均分 | 63.5841659502 | 62.8050933107 | -0.7790726395 |

MMLU为8762→8156/14042（-606题）。九项精确记录62.80509331070435，对C0 -0.779072639498487pp；相对BF16 68.04714491790848仍掉5.242051607204pp，未达原≤2.6bpw／九项掉分严格<0.9pp目标。LB为21任务未舍入宏平均，九项是复合分数，不作统一accuracy解释。

正式成本20756.5485秒，完整九项控制器21243.5570秒（5.9010小时）；外层固定序列42087.5323秒（11.6910小时），原命令／PID/start／日志见[运行原件](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/formal_run.json)。此成本不重复计入已复用上游或短验。最终metadata SHA `bf8452ab6cd3c61da66159dee90d45c22da010529cb126958a4e945f5e368d80`；实际acceptance SHA `4bea71a3cb1af52c3eb30459f63723b93eb39ffb4b528b27d84c218396ae5db5`与sequence质量摘要匹配，配置／metadata／正式report／strict小文件绑定SHA及权重size／mtime_ns／inode均与评测原件一致。本次未重复计算4.67GB权重SHA，原完整SHA与正式fresh证据沿用，未声称本次重加载验证。

实际存储沿前文最终账本：logical2.4987414816151494／packed2.5121161419412363bpw，码／decoder／保护值、indices、scales／rotation／route frame均计入，packed另含92897280 padding bits；full-model tensor4671007680B（含embedding/head/norm），序列化4672277077B，清理前目录实核4674035741B。预算通过、质量未达，两者分别记录。完整组件bits及文件bytes保留于九项验收`bit_budget`，不将目录费用当投影位宽。

本结果属于一阶段原generic目标、末down类别500恢复的固定optimizer配对：九项与M／LB均低于原AdamW控制。按本轮Astra审定结束该固定点、不追加网格；不据此否定VAE optimizer或所有七类别，也不把质量负差当实现异常补GPU审计。经验由主线程统一归并至[压缩选择](../../docs/experience/lessons/compression_choices.md#超参作用与故障诊断2026-10-29)。质量核证后已完成下述有引用核实的固定点收尾；原source/pins及GPU1活动链保持。

收尾：完成short后核正式读者只复用short报告/strict SHA，正式实际载入原raw，且无活动FD/mapping引用，已清理短验生成权重、同输入fixture、optimizer/master/scheduler/RNG、tokenizer重复副本及结束的一次性诊断/重复CPU记录，共39文件，4735291843B逻辑／4735401984B allocated（约4.41GiB）。[精确receipt](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/short_verification.json)内`cleanup`列路径、大小、原因、用途检查和前后SHA；short现在仅保留必要metadata/报告/指标/核心日志，不能再作为可加载模型或resume源。原raw/C0控制、正式source/pins及其它D16/E/SciQ/S4均保护不变；CPU/一次性核证日志归并到`prepare_final.log`，两个已空闲验证tmux关闭，正式tmux保留。

## 已结束分支的产物收尾（11月4日）

按Astra／root的既定结束决定，删除前盘点了本包生成物、外部后继、活动GPU1 D16-Muon的1024个源码pins和新gate的已确认父／控制依赖；候选路径无活跃FD／mapping，且新gate只复用成熟Muon源码、不读取三包载荷。精确路径／原因／逻辑bytes／st_blocks归既有[收尾原件](../../result/qwen3_8b_mmlu65_20261004/cat_down_recovery_muon_cmcc_20261103/startup_verification.json)的`ended_branch_cleanup_20261104`，本包本次删除17个文件，逻辑4734748638B、按删除前st_blocks实际释放4734787584B（4.409615GiB），不重复计入此前short或fixture清理。

`artifacts/cat_down_recovery_muon_cmcc_20261103/formal/final_model`现为metadata/config证据集合，**不能再加载模型**；正式checkpoint-500的master／optimizer／scheduler／RNG及重复tokenizer已清，也不能续训；实际training_args.bin、trainer_state.json、mixed_training_meta.json及save audits保留。所有原始指标／预测／train readout、corelog、metadata/config、report／strict、源码和exit均保留；少量重复wrapper／导入／warmup日志已归并后移除。删除后目标逐项不存在，其余受保护证据SHA精确保持；原C0 raw／控制、D16 raw／最终、VAE-Muon raw／最终、新组合活动模型stat及小文件SHA、GPU1源码pins均不变。此为已结束分支的收尾，不新增训练／评测或算法决定。
