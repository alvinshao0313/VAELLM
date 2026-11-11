# E256 与同新父 C0 的完整后缀对照

共同 after_o 为 `7eeb82b4-3868-4cc5-9fa3-51870ee55894`。本臂直接复用已完成原生 S2×10K 拟合、2559 字段独立 strict、2091 非 gate 对父 exact、完整 MMLU 与 group8 部署读出的 E256 raw2 `66122de5-a5da-41a6-a64a-ca03b2c3868d`，不重训前驱。gate 为 D64/K64/S2、encoder256/decoder128、input64 保护、block32 旋转，其余沿共同配方。

同父控制是[原 C0 fixed2](../fresh_fixed2_cmcc_20261030/README.md)，成品 `cbb15c9f-1c1f-4394-817c-65c226d76ce7`，完整九项 63.58416595020284。E256 原生 raw2 的阶段 MMLU 为 9302/14042（66.24412476855149%），共同 raw2 控制为 9279/14042（66.08033043725965%）；这是阶段指标；本次完整后缀／九项已闭合，排序反转，不能据阶段多23题宣称稳定优胜。

新包仅适配来源与 provenance：直接调用原 C0 的真实 native-to-fixed2 消费者；执行 gate500 → 自身 live up S2×10K＋500 → 自身 live down S2×10K＋500 → 独立 fresh → 同 UUID 完整九项。36 gate 全为普通两阶、length2，无第三阶或 Rice；恢复包含原协议内 future-dense rank12，最终合并 rank0，不新增外置 LoRA。仅 GPU1，单候选，不自动选点／重试／占卡。后缀训练预算估计16–20小时，另加完整九项，实际以日志为准。

来源保留 E256 实际 runtime；相对共同 B 仅允许 gate `base_ch=256` 与其真实 `experiment_runtime` 来源政策差异，其他六类配置与三次恢复规格同 C0。真实模型构造后 seed31；新增零更新读出包在 `fork_rng` 内，显式断言 CPU/CUDA RNG 保持。每个 own-live up/down 使用原 `flow.fit_next`，拟合前 seed31，shuffle500031/600031；两 stage 实际 optimizer 初始化完整摘要、形状、dtype、LR 与 C0 核对。拟合自己的恢复后 live 权重，不借控制后缀。

复用原 E256 及成熟转换／500恢复证据，不新增恢复 short2 或 exact-resume。CPU prepare 只核实际来源、配方、runtime 与已有证据，不加载完整模型、不初始化 CUDA。新增接线由首次恢复前真实 E256→fixed2 保存、独立完整 strict、同256token BF16 logits exact 和 RNG 保持验证闭合；失败即停止后继。后缀各边界保持原独立 fresh 合同，终态账本测 logical／packed、decoder／保护／旋转／route frame、完整 tensor 与目录字节；完整九项沿原 full8／LongBench21共4750／32K及当前 BF16 基线。质量 `not_met` 是合法结果，技术失败停止本链。

调用者先激活 `bitvae`，以下均通过 tmux/nohup 后台运行：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_e256_full_cmcc_20261102.sequence --prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_e256_full_cmcc_20261102.sequence --run
```

固定配置与源码 SHA：`artifacts/cat_e256_full_cmcc_20261102/plan.json`。实际 argv、PID/start、资源、阶段状态／退出：同目录 `run.json`，训练核心日志 `formal.log`，训练报告 `formal/report.json`，评测日志 `evaluation.log`。最终输出为 `formal/final_model`，完整质量为 `result/qwen3_8b_mmlu65_20261004/cat_e256_full_cmcc_20261102/`。每次固定链只允许启动一次；失败需按实际原因另立隔离接续，不复用该入口重试。

本链不读 geometry `prepared.json`、四份 fork 或任何 D128 权重；仅绑定 encoder256 原始 run/fit/strict/MMLU/readout 证据及其正式 raw2、共同 after_o／control。共同导航、经验与唯一执行计划由主线程维护。本次完整结果说明阶段 MMLU 排序不能替代完整配方验收；同新父 D64/K64 条件下不采用 E256，旧 D32 父下 encoder256 比原A高0.5800368048pp的结果仍保留其条件，不泛化为 encoder 宽度无效。

## 实际状态与结果

2026-11-03 14:53 完整闭合：`run.json.status=completed_E256_and_complete_nine`，外层、formal、独立 fresh、MMLU、seven、LongBench outer、bind、merge 均 exit0；质量验收 `not_met`／验收码1只表示目标未达，评测链自身成功。最终 UUID `26615c63-b9ce-4844-b104-4f4a1e7fbbc6` 在训练、独立 fresh、full8、LongBench 和九项报告中一致；controller1908883/start34957076、formal1909225/start34957722、评测1989548/start40447672、预测1990193均已自然退出。

| 指标 | E256 | 同新父 C0 | 差值（pp／分） |
| --- | ---: | ---: | ---: |
| boolq | 82.6605504587 | 82.3241590214 | +0.3363914373 |
| rte | 72.9241877256 | 74.0072202166 | -1.0830324910 |
| winogrande | 68.2715074980 | 66.6929755328 | +1.5785319653 |
| arc_easy | 76.6414141414 | 78.5774410774 | -1.9360269360 |
| arc_challenge | 50.2559726962 | 53.9249146758 | -3.6689419795 |
| openbookqa | 37.2000000000 | 37.4000000000 | -0.2000000000 |
| piqa | 76.1697497280 | 75.1904243743 | +0.9793253536 |
| mmlu | 61.8644067797 | 62.3985187295 | -0.5341119499 |
| longbench_v1 | 41.0198955890 | 41.7418399240 | -0.7219443350 |
| 九项综合 | 63.0008538463 | 63.5841659502 | -0.5833121039 |

八项均分65.74847362845753；MMLU为8687/14042，较C0少75题。完整下游64叶／24742样本及LongBench21／4750预测齐全，收尾CPU重新核文件行数及记录SHA全部一致，未重评分或加载模型。C0与本臂验收协议、full8推理设置和LongBench协议逐对象一致：0-shot／无限额，LongBench官方commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`、数据revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`、32K总窗口，九项为八项百分比分数与LB21任务宏平均的等权综合，并非统一accuracy。BF16综合68.04714491790848，要求严格高于67.14714491790848；本臂差5.046291071612561分，预算满足而质量未达。

三类各500恢复及own-live up/down各S2×10K全部完成；`formal/report.json.suffix_initialization_comparison`中的up/down两阶optimizer初始化摘要与C0均true。后缀实际消费者、恢复协议和初始化匹配，但各自live权重及完整配方不同；本结论是指定完整配方比较，不能解释成纯encoder因果。

最终252投影中216 native＋36 compact，无pending dense，合并rank0。实测logical／packed为2.4987414816151494／2.5121161419412363bpw：含码、decoder、input保护值／indices／scales、旋转和route frame，packed另含92,897,280 padding bits；全模型tensor为4,671,007,680B，包含投影之外embedding/head和norm。序列化权重4,672,277,077B，完整final_model目录apparent4,674,044,051B／allocated4,674,056,192B；目录存储与逻辑投影位宽分开报告，不能直接互换。

formal外层计时54,882.4601秒（15.2451小时），完整评测21,711.2590秒（6.0309小时），单GPU1，合计21.2760小时；不含已复用的gate前驱成本。Astra终态审查及主线程确认：结束此D64/K64 E256完整配置，不采用，不追加Muon、延长或改宽网格。保留各项正负差与旧D32正例边界；本次负综合分不支持“encoder宽度整体无效”的结论。

## 复现与核证证据

固定入口与实际argv见上文及`plan.json`／`run.json`；plan SHA `db63bae7c44c7df717d73b999181fcdb77c2dfd495b01a7bee8ecd38d8ed7922`，共1007源码pins。11月2日17:35 CPU prepare exit0，真实来源／runtime／C0后缀规格通过；启动tmux `CAT_E256_full_GPU1_20261102`。原始外层状态、退出及日志不改写。

首恢复前fixed2初态 `6bfc369e-2161-4932-8b31-bcd31422690d` 已通过2811字段独立strict、同256token BF16 logits exact及CPU/CUDA RNG保持，144/144原生group8 warm成功。原第1步warmup零LR／master位移0，第2步改变11,025,421个master元素，前两次FP32梯度有限和Adam step匹配；峰值allocated40,881,532,928B／reserved41,439,723,520B，首两步约35.57秒/步，仅为早期资源证据。最终独立fresh核3675持久字段及同输入BF16 logits exact，部署digest `5d976fee7298375996178590a49bab13372dc6301f496e02010dfdbeafb8a86e`。这些完整性证据直接复用，收尾未作GPU复评／重导或模型数值审计。

原始结果：[evaluation.json](../../result/qwen3_8b_mmlu65_20261004/cat_e256_full_cmcc_20261102/evaluation.json)、[完整九项](../../result/qwen3_8b_mmlu65_20261004/cat_e256_full_cmcc_20261102/longbench/merged/nine_task_acceptance.json)、[CPU终态核证与精确清理记录](../../result/qwen3_8b_mmlu65_20261004/cat_e256_full_cmcc_20261102/closure_20261103.json)。代码、配置、source pins、必要metadata、核心日志、所有原指标／预测及复现入口保留；本轮已按该JSON精确删除40个生成权重／optimizer／scheduler／RNG／用完数值夹具及tokenizer副本，实释46,481,510,400B（43.2892799GiB，allocated）；序列化配置`training_args.bin`、JSON metadata及所有上述证据保留。C0/common、原BF16、cohort E256 raw2/input及其它活动链全部保护。
