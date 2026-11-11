# 同一C0 after-up上的down几何研究与新候选基线

本包从已完成C0的真实 `05_up_proj/after_recovery` 重新拟合全部36个live dense down。固定前驱已完成原BF16→q/k/v/o→B gate raw2→compact gate500→自身up两阶各10K/500；本轮复用此稳定checkpoint，仅运行新down分支。原C0最终成品作为完整后缀对照，不将其它成品的down移植当作自身live流程，也不合并Rice/E256的独立收益。

本组raw阶段均已闭合：D16八项 `67.0125927269`，控制 `65.9771825912`，相差 `+1.0354101357pp`；方向点 `66.0896203605`，较控制 `+0.1124377693pp`。方向点已按预定阶段结束并完成权重收尾，不接第二条长链；唯一完整候选D16的自身down500及独立fresh已完成exit0，GPU0同UUID完整九项已启动，最终九项尚无结果。完整full8和LongBench21/4750全部完成后，才能比较最终九项。质量不达标是有效结果，不自动改选方向候选或追加搜索。

| 分支 | GPU | down唯一变化 | 固定执行范围 |
| --- | --- | --- | --- |
| 原raw控制 | 2 | 原D32/K48、B8192、transpose=True | 复用已拟合 `06_down_proj/round_base`，补raw full8与相同读出，不重训 |
| `d16` | 0 | D32/K48→D16/K24、B8192→16384，transpose=True | 新shape必要短验→正式fit/fresh→raw full8/读出→自身down500/fresh→完整九项 |
| `transpose_false` | 2 | D32/K48、B8192不变，仅transpose=False | 原raw控制评测之后，正式fit/fresh→raw full8/读出，结束 |

两新臂同3bit主码、`batch×D=262144`每层每次标量曝光；不是等FLOPs或跨shape相同参数初始化。固定原M36 pooled BSQ、S2每阶10K、symmetric E/D128/R0/layer/swish、LR.003/linear/无warmup、原AdamW及BSQ损失、seed31/shuffle600031。保留input256 INT8保护、双侧block32旋转，实际保护indices/values/scales和rotation specs/signs与原raw控制exact；原512×256 Alpaca train校准按原刷新规则消费。D16参数shape改变，记录两阶实际初始化摘要；方向臂同结构两阶摘要必须等原C0记录。

原down500继续更新全部252投影decoder，FP32 master、decoder_lr1e-5、无实际dense LoRA或外置LoRA；stage_seed37/data_seed31、B4/1024、动态padding、GC、原五源train数据、KL及500 cosine/warmup.01和250/500保存均复用原合同。raw评测、读出与恢复各为独立进程，恢复继续使用原canonical Trainer的stage/data seeds，不新增RNG接续算法。

父为实际mixed v2：`adaptive_model_meta.json`＋`pytorch_model.bin`，180native＋36compact＋36dense，UUID `5a94d57f-d3b9-4c97-adef-a099d659eb6a`；原raw控制UUID `ed507bcb-9b83-4e4c-9859-daaef44d80d2`，原最终C0 UUID `cbb15c9f-1c1f-4394-817c-65c226d76ce7`。live dense源digest必须等原C0实际拟合源 `c2ffa5bbaffa477fd0f1b7e9be3a8efb4e32557ee000d8cb45fa4633eba26c05`。非down及非投影完整补集保持exact，未覆盖活动代码。

入口为 `prepare.py` CPU来源/配置/原控制补集核证，`run.py` 真实新拟合和部署fresh，`recover.py` 固定唯一D16后缀，`evaluate.py` 复用原mixed消费者、完整八项/官方九项，`sequence.py` 固定依赖链。D16短验只做真实M36/B16384/S2各2次finite/nonzero/update、编码/账本/补集和独立fresh/group8 BF16部署；成熟500恢复和保存路径复用现有证据，不重复exact-resume或恢复短验。正式拟合始终重新加载父，不接短验权重。

full8保持64叶24742题、MMLU B64及其它B32、0-shot/无限制；最终LongBench为官方21任务4750条/32K同协议。部署局部读出固定原train前8×256 token，用同after-up捕获全36down输入，W取同live dense，以各raw真实BF16部署解码比较；它是固定train局部诊断，不代替下游质量。逻辑≤2.6、packed、padding、decoder/scale/保护/旋转、全tensor/目录费用均由真实mixed账本计量；D16预计约2.49789logical，仅为启动前估计，以保存实测为准。

只使用GPU0/2，不触及GPU1的E256完整评测及GPU6/7的S4。正式fit＋部署/fresh合计timeout90分钟，raw full8共90分钟、读出30分钟、down500＋最终fresh合计十二小时、最终九项八小时；D16必要短验合计30分钟。失败立即停止受影响依赖，不重试、不占卡。参考原同机记录，D16主链预计12–14小时、方向链约2小时，属于估计，实际耗时写入run/report。固定链等待本身由父进程wait实现，无额外监控或重试系统。

调用者shell激活 `bitvae`，以下实际入口均在tmux/nohup后台运行。证据根为 `result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/`，模型根为 `/data1/cmcc/VAELLM/artifacts/down_geometry_baseline_cmcc_20261103/`。新包硬绑定本轮父与独立输出，已有运行禁止重复。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_geometry_baseline_cmcc_20261103.prepare
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=31 python -B -u -m experiments.down_geometry_baseline_cmcc_20261103.run --arm d16 --phase short
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_geometry_baseline_cmcc_20261103.sequence --arm d16
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_geometry_baseline_cmcc_20261103.sequence --arm transpose_false
```

从BF16复现的真实分段来源见 [common入口](../fresh_common_cmcc_20261030/README.md)及其 `run.py --output_dir NEW --gpu N --validation-report ...`，和 [C0入口](../fresh_fixed2_cmcc_20261030/README.md)及其 `run_pipeline --common NEW_COMMON --output NEW_C0`：这些入口的七类配方、源码pins、实际命令分别保存于原common的request/prefix/gate_raw2和C0 report。本轮新分支从该完整链自身after-up接续，D16覆盖参数和父合同见本包bindings/prepare结果。未提供统一 `--from-bf16` CLI，也未在本轮重跑已稳定前驱；未来最终入选配方须以原始BF16为根完成整链复现，不能把本次接续对照写成已重新执行整条链。

## Raw阶段结果与当前恢复状态

11月3日11:17前后有限核实：原raw控制UUID `ed507bcb-9b83-4e4c-9859-daaef44d80d2`，D16正式raw UUID `02292fb6-5041-4f0d-bbb3-9827ec5f3a1f`，均为恢复前的全252投影压缩状态。D16正式S2各10K拟合及独立fresh已exit0，用时988.6184秒；两份raw full8及readout均exit0。各full8均为64叶/24742题（MMLU57叶14042、其它7叶10700），保持B64/B32、0-shot、limit=None；本次独立复算各64份逐题文件SHA及非空行数，并核24742条doc_id/doc_hash/prompt_hash/target_hash逐条配对一致，八项均分与原八任务指标重算相同。原件为[控制full8](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/control/raw_full8/full8_summary.json)、[D16 full8](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_full8/full8_summary.json)及其同目录run/mmlu/seven JSON；没有重新训练或重跑评测。

| raw完整任务（分） | 原控制D32/K48 | D16/K24 | D16−控制（pp） | 仅transpose=False | 方向−控制（pp） |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU | 62.5765560461 | 62.9112662014 | +0.3347101552 | 62.4270047002 | −0.1495513460 |
| ARC-C | 53.4129692833 | 51.6211604096 | −1.7918088737 | 51.8771331058 | −1.5358361775 |
| ARC-E | 77.7356902357 | 78.0723905724 | +0.3367003367 | 76.9360269360 | −0.7996632997 |
| BoolQ | 81.2538226300 | 83.9143730887 | +2.6605504587 | 82.7522935780 | +1.4984709480 |
| OpenBookQA | 37.6000000000 | 38.4000000000 | +0.8000000000 | 38.0000000000 | +0.4000000000 |
| PIQA | 75.1360174102 | 75.1904243743 | +0.0544069641 | 74.5919477693 | −0.5440696409 |
| RTE | 73.6462093863 | 79.0613718412 | +5.4151624549 | 74.7292418773 | +1.0830324910 |
| Winogrande | 66.4561957380 | 66.9297553275 | +0.4735595896 | 67.4033149171 | +0.9471191792 |
| 八项均分 | 65.9771825912 | 67.0125927269 | +1.0354101357 | 66.0896203605 | +0.1124377693 |

MMLU为8787→8834/14042，多47题。七项上升、ARC-C下降；RTE和BoolQ两项贡献净八项增量的97.4941%，其它六项均值净差仅 `+0.0345946953pp`。这是本父/单seed/同标量曝光的raw配方比较，不能称各任务广泛大幅提高，也不能把当前八项正差直接换算为最终九项收益。

方向点MMLU为8766/14042，较控制少21题；四项升、四项降，八项净差仅 `+0.11243776926706062pp`。D16 raw比方向点高 `0.9229723664193159pp`，但方向点ARC-C和Winogrande更高，不能称D16全面支配。新增方向[完整八项原件](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/transpose_false/raw_full8/full8_summary.json)及同目录run/mmlu/seven JSON已单独核证：64份逐题SHA、24742行和与原控制逐条doc/prompt/target身份一致，B64/B32及完整协议保持；复用前次control/D16证据，没有重跑评测或重新扫描旧pins。

同一after-up父、同一固定Alpaca train前8×256 token的部署读出，input SHA均为 `c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98`，完整36down均覆盖，参考权重及输入分母相同：

| 真实BF16部署误差 | 原控制 | D16 | D16相对控制 | 仅transpose=False | 方向相对控制 |
| --- | ---: | ---: | ---: | ---: | ---: |
| weight MSE | 5.0396484503e-5 | 4.0451732325e-5 | −19.73302756% | 5.0381623376e-5 | −0.02948842% |
| weight NMSE | 0.0729190437671 | 0.0585299087622 | −19.73302756% | 0.0728975410939 | −0.02948842% |
| output MSE | 0.127492826172 | 0.111147658970 | −12.82046033% | 0.152184378671 | +19.36701322% |
| output NMSE | 0.0153767240544 | 0.0134053572471 | −12.82046033% | 0.0183547362349 | +19.36701322% |

原件为[控制readout](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/control/raw_readout.json)和[D16 readout](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/raw_readout.json)。weight NMSE在36/36层改善，output NMSE在35/36层改善；L34 output NMSE反而由 `0.0147696213804` 至 `0.0163928392538`（+10.99024702%）。固定train局部权重误差与功能误差不单调；这不是已见实现异常，不构成逐层挑选/拼接或新增对照的依据，也不是盲测质量结论。

方向[readout原件](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/transpose_false/raw_readout.json)已核同父、同input SHA、同前8×256 train token、完整36down。其output恶化集中于L34：output NMSE `0.0147696213804→0.0320447424508`，+116.96387216%，贡献总output MSE净增约96.1774%。只保留固定train局部现象，不据此追加逐层拼接、精度修补或短验；未发现实现/协议异常依据。

| raw实际成本 | 原控制 | D16 | 仅transpose=False |
| --- | ---: | ---: | ---: |
| 本轮fit＋fresh（秒） | 复用，不重训 | 988.6184 | 733.5955 |
| 本轮完整full8（秒） | 473.6263 | 477.6376 | 480.0779 |
| logical bpw | 2.4987414816151494 | 2.497889643130095 | 2.4987414816151494 |
| packed bpw | 2.5121161419412363 | 2.511264303456182 | 2.5121161419412363 |
| 全模型tensor字节 | 4,671,007,680 | 4,670,268,096 | 4,671,007,680 |

全部位宽含既有decoder、scale、保护、旋转及compact frame等组件；D16 tensor少739584B。方向保存时完整目录 `4,674,019,120 B`，与同tensor费用是不同口径。各full8含完整MMLU和其它七项，原预估的20–40分钟未作为实测沿用；方向固定链含补原控制full8/readout共1858.4024秒。

GPU0[正式恢复报告](../../artifacts/down_geometry_baseline_cmcc_20261103/d16/recovery/report.json)已于11月3日17:17自然完成：`completed_own_down500`／exit0，own raw UUID `02292fb6-5041-4f0d-bbb3-9827ec5f3a1f`匹配，原stage合同exact，实际500/500更新。原recover worker `2038741/start41304302`及timeout `2038740/start41304302`均已退出，既定driver `2027643/start41200416`继续后继。实际1728个decoder叶、4,966,272个FP32参数，实际LoRA参数0；第1步master位移0、第2步改变4,963,990元素，末步改变4,966,201元素。冻结持久状态exact，250/500保存均保留FP32 master及原Adam／scheduler／RNG状态，保存前后对象与状态未变；未新增resume审计。stage_seed37/data_seed31、原五源train配比、KL top100/T1/prompt.3、decoder_lr1e-5、B4/累积1、cosine/warmup.01、GC保持。Trainer实际21,689.6579秒，formal report含保存／fresh共21,761.8492秒（6.0450小时），外层down500阶段共21,770.8535秒（6.0475小时）；train_loss=0.2769924616仅记录训练结果，不代替质量。

最终UUID `9b43d213-7147-4939-b04a-4a8c6b2b2300`已保存并通过[独立fresh](../../artifacts/down_geometry_baseline_cmcc_20261103/d16/recovery/strict_reload.json)：exit0、3675持久字段及同输入BF16 logits exact，group8 warm216/216、无失败，digest `172096cffe0bf0c932605d5ac8d8d93418c7efe635154ac639cfa94266aa63fa`。真实账本为216 native＋36 compact、252投影全压缩、无future dense，logical2.497889643130095／packed2.511264303456182bpw，含原码、decoder、保护、旋转与route frame及92,897,280 padding bits；全模型tensor4,670,268,096B，序列化权重4,671,537,493B，完整final_model目录4,673,307,961B。最终metadata SHA与评测绑定一致，终态、fresh与当前评测UUID一致；本次仅CPU核证，未重新加载模型或额外GPU验证。

[权威run](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/run.json)仍为running、`current_stage=final_full9`：17:17:06同GPU0自动进入既定完整九项，timeout `2279916/start43481415`、评测worker `2279917/start43481416`，原成熟allocator `expandable_segments:True`保持。17:24有限检查时，最终[MMLU](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/final_full9/full8/mmlu.json)已exit0，57科／14042题、8886题正确，63.28158381996867%，该阶段274.1660秒；其它七项timeout `2284629/start43511594`、worker `2284630/start43511594`已真实推进loglikelihood至7442/29492请求。该单项不是最终九项结论，LB21及综合验收仍待。当前[评测状态](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/final_full9/evaluation.json)及同目录mmlu/seven日志为事实来源；本次核实后停止主动监控，不等待整份评测、不启动新任务、不改活动源码或清理当前有用途产物。

17:29原链的其余七项已自然exit0，同最终UUID的[完整八项](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/final_full9/full8/full8_summary.json)为66.88120428101816，64叶／24742题、M B64／其它B32保持。LongBench独立predictor `2288488`已实际运行，九项仍待；该八项不是总目标验收。

**中途恢复选点的本次取舍：** 同raw→500八项−.1313884459pp而M8834→8886（+52题），支持提出预算／能力代价问题，尚不足以证明九项下降或250更优。Astra实核[down500.log](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/d16/down500.log)第195行及report：原`save_total_limit=1`已在500保存时自动删除checkpoint-250，当前仅有500与mixed保存审计，不能把保存过的日志当作可导出载荷。MixedCatTrainer的训练包是绑定raw父的FP32增量，仍标`exact_resume_implemented=False`，无现成mixed中点部署入口；500／BF16成品不能还原250。因此不启动所谓“廉价250复评”，本次新增预算0，先完成500完整九项。只有D16仍值得继续且恢复预算成为具体决策时，才另审同raw父、原数据／teacher／范围／种子及**原500日程前250更新**的唯一点；重走原250约3:01:09，是新执行轨迹，并需实际部署／fresh及同模型完整九项，不改成250总日程或扩LR／节点网格。经验归[中途快照保留与实际载荷](../../docs/experience/lessons/checkpoint_lifecycle.md#顺序搜索保留中途最佳部署排名须独立重载2026-10-01收尾)。

**本次活动链的限时缺口（Astra只读核实）：** [外层runtime入口](../down_geometry_runtime_repair_cmcc_20261103.py)的final_full9 timeout28800秒只覆盖评测PGID2279916；[实际LB调度](../adaptive_class_evaluation_20261030/evaluate.py)调用旧start_group的`start_new_session=True`，shell2288486独立SID、timeout2288487／predict2288488属PGID2288487，三者实际start_ticks均43533999；内层实际86580秒、runner86400秒，无8小时共同绝对deadline或父退出清理。17:33:39仍正常、尚未超预算；不能把外层8小时设定称作已约束全部GPU子链。本轮不修改活动源码／timer、不停评测或另建监控。外层截止为北京时间2026-11-04 01:17:06，另20秒kill grace；LB自身hard timeout约当日17:28:52，另20秒grace。恢复时先核真实退出与PID/start：完整自然结束则正常收束；外层超时而LB仍live，明确记录超预算孤立子链，按已有停进程确认边界精确处置，不能假称GPU已释放。若LB已完整则复用同UUID的21／4750、partition_complete／exit0及原外层退出，必要时仅补CPU bind／merge，不重复full8或预测；只有partial不评分、不自动续跑或重试。后续新入口应使用同一绝对deadline传到detached shell／runner，相关经验见[嵌套timeout与会话边界](../../docs/experience/lessons/reproducibility.md#嵌套timeout进程组与有限接续)。

GPU2方向臂[固定链run](../../result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/transpose_false/run.json)于11月3日11:28:01完成，completed_raw_direction_comparison／exit0、全部五个阶段exit0；driver `2027642/start41200416` 及子进程均自然退出。raw UUID `d0f90f14-4d80-4fa7-838b-42c1cd660047` 贯穿真实fit、fresh、full8、readout；实际S2各10K、两阶初始化摘要均匹配原C0，冻结前缀／保护／旋转保持；3675持久字段及同输入BF16 logits strict核证exit0，全部252投影压缩、无future dense。Astra确认该固定阶段点结束：小幅均分正差、M/ARC/PIQA下降及同train功能误差恶化，没有足够新增价值开启第二完整链；没有测它的九项，不能称其最终劣。D16继续既定唯一完整链，无新启动/重启/网格/占卡；本次核实后停止主动监控。

## 实测验证与技术接续

CPU `prepared.json` 已exit0，真实父/控制UUID、live dense digest、非down完整补集、唯一配置差异及原down500合同（除独立输出目录）一致；SHA256为 `a3a535f3e9c5d2916c10408f035817e8027424e55e4dee030dd615b11948867b`。D16短验235.85秒、两stage各2次finite/nonzero梯度和实际参数更新，非down/保护/旋转exact；3675字段与同输入BF16 logits独立fresh exact、原group8预热216/216。实际logical `2.497889643130095`、packed `2.511264303456182`，tensor `4,670,268,096 B`、当时完整目录 `4,673,279,410 B`，峰值allocated `57,440,756,736 B`。这些是新增shape/消费者合同证据，短验不证明质量。

首次控制raw full8在MMLU首个loglikelihood分配18.29GiB时OOM；独占卡当时allocated37.69GiB、reserved空闲28.40GiB。新sequence遗漏了成熟C0 plan和 `cmcc_runtime_20261030.sequence.environment` 中的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。错误与真实exit1保留在原 `transpose_false/run.json`、`control/raw_full8/mmlu.log`；方向拟合未执行。

修复仅由包外 [down_geometry_runtime_repair_cmcc_20261103.py](../down_geometry_runtime_repair_cmcc_20261103.py) 在import torch前恢复原allocator，原包Python源码、原prepared、活动拟合代码和协议不改。新结果根为原result下 `allocator_repair/`，其prepared显式绑定原prepared SHA、原pins和修复入口自身SHA。GPU0原controller由root明确授权做单PID handoff：新driver先持有controller/timeout/worker确切pidfd，核独立timeout PGID，单独SIGTERM本轮CPU controller；原timeout/fit不发信号、不重训。pidfd等待原fit自然完成后，必须核其自报completed/exit0、实际UUID/独立fresh/账本再接续。非亲缘进程的OS退出码无法waitpid取得，记录为None，不能用PID消失推断训练成功。完整身份及状态以 `allocator_repair/handoff.json` 为准。

经Astra核心窄审通过，修复CPU已exit0。GPU0当前tmux为 `down_geometry_D16_handoff_GPU0_20261103`；方向会话 `down_geometry_direction_allocator_GPU2_20261103` 已自然退出。修复后固定接续入口如下；已有运行禁止再次执行：

11月3日10:57实际handoff已完成：新driver PID2027643/start41200416；原controller2019643退出、shell记录143，timeout2019948与fit2019949均保持start41148258/PGID2019948，timeout现PPid1。原fit继续推进到第二stage，未重训；新driver仅以pidfd等待其完成/fresh。GPU2新driver2027642/start41200416，实际B64评测worker2028330的环境已核 `CUDA_VISIBLE_DEVICES=2`、`bitvae`、`expandable_segments:True`；其loglikelihood已推进1025/56168，越过首次OOM的forward，未降batch。该10:57节点停止主动监控时尚未完成raw full8、down500或最终九项；其后已完成项以本篇上方最新核证为准。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_geometry_runtime_repair_cmcc_20261103 --target prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_geometry_runtime_repair_cmcc_20261103 --target handoff
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.down_geometry_runtime_repair_cmcc_20261103 --target sequence --arm transpose_false
```

短验正式任务均不再读取短权重。已删除短验 `raw_model/pytorch_model.bin`、`same_input_deployment.pt` 及被最终CPU版本替代的 `prepared_v1.json/prepare_v1.log`，按这四个文件删除前st_blocks实计释放 `4,672,180,224 B`。保留short metadata/config、report/strict及核心日志；该次短验清理未触及原父、控制与正式产物。短验目录现为证据集合，不能再作为可加载模型。

## 方向阶段结束与产物收尾

固定阶段已完成且无具体后继/复评用途，driver自然退出、`lsof`核两生成载荷无reader，源/活动D16后续不依赖方向权重。删除前均核为批准范围内真实绝对路径、无符号链接、硬链接数1；退出文件均为0且已归并run.json。仅清以下13个精确路径，共逻辑 `4,672,787,455 B`、按删除前st_blocks实际释放 `4,672,831,488 B`，不与前次短验释放量重复计数：

- `/data1/cmcc/VAELLM/artifacts/down_geometry_baseline_cmcc_20261103/transpose_false/formal/` 下 `raw_model/pytorch_model.bin`（4,672,277,077B）与 `same_input_deployment.pt`（307,692B）：已结束、无后继的生成权重及已完成strict验证的临时载荷。
- `/home/cmcc/VAELLM/result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/allocator_repair/transpose_false/` 下 `sequence.lock`、`fit.exit`、`raw_full8.exit`、`raw_readout.exit`、`run.exit`、`raw_full8.log`、`raw_readout.log`、`raw_full8/mmlu.log`、`raw_full8/seven.log`：退役锁、已归并退出码及重复导入/预热/进度条日志。
- `/home/cmcc/VAELLM/result/qwen3_8b_mmlu65_20261004/down_geometry_baseline_cmcc_20261103/` 下 `allocator_direction.log`、`allocator_direction.exit`：已归并的终端导入日志与退出码。

删除后已核目标全部不存在，保留方向metadata/config、原formal/strict报告、核心fit.log、有效full8配置/run/mmlu/seven/summary、64份原逐题文件、readout、源SHA和各阶段退出证据。该方向raw_model目录现仅保存身份/几何证据，不能再次加载；全部活动D16权重/恢复报告、共同父与原control及其指标、源码/pins均保留，不涉及共享数据或其它进程。

经验依据是 [参数作用与历史匹配几何对照](../../docs/experience/lessons/compression_choices.md)和 [同父/相同曝光/完整下游边界](../../docs/experience/lessons/experiment_design.md)：历史down D16、方向均有条件依赖，扩大保护已有匹配负例，因此本轮不扩保护或网格；raw MMLU与其它任务可能异向，故用完整full8而非单M排序。本次阶段结果支持在该同父/曝光条件下继续预定D16完整链，并再次显示局部误差、单项和均分不必同向；方向点按既定范围闭合，D16最终九项仍待，不外推为通用维度或方向规则。
