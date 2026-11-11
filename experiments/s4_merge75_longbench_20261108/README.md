# S4 75% model merge：完整LongBench与同checkpoint九项

2026-11-08用户明确同意给75%合并候选补LongBench。完整21项／4750条及同checkpoint九项已于13:34:17完成，全部进程exit0且GPU4–7释放。当前75%候选同时达到MMLU与九项均值的数值目标；BF16整链及泛化仍未验证。详见[完整结果与本轮结论](#完整结果与本轮结论)。唯一排程见[当前执行计划](../../docs/experience/records/w25_transfer/current_plan.md)。

## 范围、来源与验收

只读使用原生checkpoint `838d2db4-48e7-48cd-b299-fc69790e7a35`，即D64占.25／D128占.75的model merge。已有同UUID完整八项M10146/14042（72.25466458%）、八项70.80321508%，本批只读复用，不重跑八项，不训练、不改变权重／比例。来源与泛化警示见[三比例完整记录](../s4_model_merge_20261108/README.md#完整结果与本轮结论)。该候选与原两donor模型／状态继续保护。

补齐官方LongBench v1全部21任务／4750条，保持原共享archive、32K窗口／原截断及token规则、greedy、各task原生成上限、Samsum停止规则和`use_model_defaults=False`。沿成熟S4 strict loader、BF16原生group8预热、FP32 RoPE及FA2，不用默认2.5bit加载器替代本4bit研究账本。logical=packed4.094031292459239bpw、rank0，含原decoder／保护／旋转，embedding/head不计投影账本。

将LongBench的21-task未舍入宏均值作为第九项，与已完成八项等权平均；不把21任务展开为21个独立综合权重。固定门槛M≥10111且九项均值>67.64714491790848（相对BF16九项68.04714491790848严格损失<.4pp）。九项质量数值是否过线与完整算法验收分开：尚无第二seed／跨模型位宽泛化复核，入选流程也未完成新的BF16整链重放。程序exit0只代表评测完整，质量由quality_acceptance字段报告。

## 资源与执行

仅GPU4–7，各一份同UUID模型，使用既有canonical LB_GROUPS按四卡round-robin分任务，每个任务／样本恰好一次，配置SHA完全相同；0–3留给其他研究。每个worker完整生成对应任务后退出释放卡，不保留占卡进程。参考旧S4全21单卡约6.15GPUh，本批约6–8GPUh／四卡约2–3小时，准备另列，非硬中断。各worker仍沿旧24h between-sample预算与外层86580s，单条已开始生成不缩短。

CPU先核选中merge及同UUID八项完整终态、原S4 full21生成证据、4750 archive／32K几何、模型政策与来源、四分组完整且互斥。新加载入口只绑定现有merge strict loader，生成、cache、协议、评分路径不变，因此复用现有同模型拓扑／精度／峰值长输入证据，不重做无关训练或三案例GPU短测。正式开始有限核对真实输出和加载审计；失败停止受影响依赖，不降低长度／生成上限／batch，不自动重试或部分评分。

每组走成熟一次性outer记录、原生runner；全部退出后逐组调用原bind_terminal.py，完成4750唯一身份／配置／生成协议核对，再用原merge_full21.py合并与未舍入评分。CPU汇总现有MMLU／seven及新LongBench，所有阶段绑定同checkpoint，不拼其他模型指标。结果齐后报告各任务／类别／九项、BF16差值、M门槛、时间／GPUh／显存、原始预测与退出状态；不自动启动第二seed、重训或其他merge比例。

## 产物与入口

独立包不修改旧S4或LongBench正式源码／CLI。产物根`/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/merge75_longbench_20261108/trial_20261108_complete21/`，包含request/config/CPU_validation/merge_plan、launch/run/pipeline.log、groups/part_01–04原生预测、终态绑定、merged/scores.json及nine_task_acceptance.json。已有八项、模型和donor均只读保护，不复制权重。

调用者激活bitvae后各执行一次，不能覆盖重跑：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_merge75_longbench_20261108.prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_merge75_longbench_20261108.launch
```

正式生成正常推进后交付并停止主动轮询，不建监控器；下一次唤醒先核真实运行、输出与退出，再处理终态。泛化警示继续维护于[经验条目](../../docs/experience/lessons/kd_and_data.md#教师保护scale和整数qat的条件证据)。

## 实际启动与验证（历史快照）

CPU准备3.31996秒、CUDA未初始化；同UUID已有八项与既有全S4生成证据通过。正式tmux `S4_merge75_LB21_g4_7_20261108`，controller964061/start84747094；四卡native PID依次964259／964262／964264／964266。实际命令及start ticks在launch.json、run.json与各outer.launch.json，stdout/stderr在pipeline.log和groups/part_01–04/outer.log。

startup_verification.json已核四组严格加载无missing/unexpected、同UUID／配置SHA／实际账本、group8 252/252预热、实际生成选项与有效选项一致且use_model_defaults=False；11:57四组已实际落盘100／142／32／82条，共356条。该数字只是启动时进度，完整21项／4750条及九项仍待后台完成。当前停止主动轮询，无监控自动化。

代码及全部旧依赖SHA绑定到request/config；未修改旧正式源码、已有模型／八项或其他任务。文档INDEX已刷新至360文档＋18别名；本次新增链接未报缺，全库仍有1654项既有缺链，未称全库通过。一次性重复缺链输出已归并后移除，未清理任何模型／训练状态。

## 完整结果与本轮结论

同UUID `838d2db4-48e7-48cd-b299-fc69790e7a35` 的64叶／24742题八项只读复用，本批新增完整LongBench 21项／4750条。LongBench task macro为49.525121743475395，较BF16低.1272405111578223分；它作为第九项等权计入，没有展开成21个独立综合权重。

| 指标 | BF16 | 75% merge | 差值 | 预定门槛 |
| --- | --- | --- | --- | --- |
| MMLU正确／14042 | 10251 | 10146 | -105题 | ≥10111，通过 |
| MMLU% | 73.00242131 | 72.25466458 | -.74775673pp | 损失≤1pp，通过 |
| 八项均值% | 70.34649275 | 70.80321508 | +.45672233pp | 仅筛选参考 |
| LongBench task macro | 49.65236225 | 49.52512174 | -.12724051分 | 计入九项 |
| 九项均值% | 68.04714492 | 68.43898248 | +.39183757pp | >67.64714492，通过 |

**当前候选达到本轮S4约4bit的数值质量目标。** 九项比BF16高.39183757pp，MMLU比BF16低.74775673pp、比M门槛多35题；九项高于严格门槛.79183757pp。保留M、OBQA、RTE与LongBench的单项代价，均值增加不代表所有能力均恢复。logical／packed均4.094031292459239bpw，rank0；这不是≤2.6bit主线的验收，也不代表新的BF16整链已重放。

### 同一模型九项明细

| 任务 | BF16 | 75% merge | 差值pp／LB分 |
| --- | --- | --- | --- |
| mmlu | 73.00242131 | 72.25466458 | -0.74775673 |
| arc_challenge | 56.56996587 | 57.67918089 | +1.10921502 |
| arc_easy | 80.80808081 | 82.36531987 | +1.55723906 |
| boolq | 86.57492355 | 87.40061162 | +0.82568807 |
| openbookqa | 41.80000000 | 41.60000000 | -0.20000000 |
| piqa | 77.52992383 | 78.01958651 | +0.48966268 |
| rte | 77.97833935 | 77.25631769 | -0.72202166 |
| winogrande | 68.50828729 | 69.85003946 | +1.34175217 |
| longbench | 49.65236225 | 49.52512174 | -0.12724051 |

### LongBench全部21任务

原生成协议、原task生成上限、32K截断／token约束、greedy及原生EOS／Samsum规则保持；原始未舍入分数在merged/scores.json，下表仅展示舍入值。

| 任务 | 样本 | 分数 |
| --- | --- | --- |
| narrativeqa | 200 | 24.22638545 |
| qasper | 200 | 46.83193732 |
| multifieldqa_en | 150 | 51.91144857 |
| multifieldqa_zh | 200 | 65.12031536 |
| hotpotqa | 200 | 57.64395150 |
| 2wikimqa | 200 | 45.82798663 |
| musique | 200 | 32.42974980 |
| dureader | 200 | 27.96380282 |
| gov_report | 200 | 33.40329787 |
| qmsum | 200 | 23.55245235 |
| multi_news | 200 | 25.69723441 |
| vcsum | 200 | 16.15739705 |
| trec | 200 | 73.00000000 |
| triviaqa | 200 | 91.19523810 |
| samsum | 200 | 43.47585938 |
| lsht | 200 | 46.00000000 |
| passage_count | 200 | 5.56250000 |
| passage_retrieval_en | 200 | 99.50000000 |
| passage_retrieval_zh | 200 | 99.00000000 |
| lcc | 500 | 67.43000000 |
| repobench-p | 500 | 64.09800000 |

六类宏分数分别为：single_document_qa 47.02252167；multi_document_qa 40.96637269；summarization 24.70259542；few_shot_learning 63.41777437；synthetic 68.02083333；code_completion 65.76400000。九项使用21-task macro49.52512174，不使用六类macro51.64901625。

### 完整性、成本与保留

completion_verification.json核证：四个原生partition均partition_complete／exit0、already_present=0，1100／1000／1000／1650条完整且互斥；四outer、四bind与merge/nine及controller全部exit0。4750条生成设置／配置一致，原成熟binder/merger已对archive逐条身份、输入hash及token规则核对；本次CPU复核42份原生＋合并预测文件SHA及64份八项raw SHA，共24742条八项证据。原checkpoint、八项、源码和数据绑定不变；同UUID strict／252预热／4.094031292459239bpw已核，所有原运行PID退役。CPU核证未初始化CUDA，未重跑评测。merge_plan保留评分前prepared_not_scored快照以维持其SHA，实际完成状态见merge_record、run和本核证，不覆盖历史计划。

| GPU | 预测条数 | 原生秒 | 外壳占用GPUh | 每条生成峰allocated GiB的最大值 | 原生结束北京时间 |
| --- | --- | --- | --- | --- | --- |
| 4 | 1100 | 5477.46 | 1.524688 | 29.507 | 13:26:17 |
| 5 | 1000 | 5307.73 | 1.477349 | 26.325 | 13:23:27 |
| 6 | 1000 | 5889.55 | 1.639179 | 29.508 | 13:33:09 |
| 7 | 1650 | 4314.03 | 1.201492 | 29.523 | 13:06:53 |

原生合计5.83021576GPUh，包含加载启动的四外壳合计5.84270793GPUh，低于6–8GPUh参考；不是四卡等待最慢分组的统一占时。整批墙钟5966.67秒（约99.44分钟），CPU准备3.32秒与代码准备另列。评测阶段最大per-generation allocated 29.523GiB，不将它冒称训练或整个进程初始化的峰值。最晚13:34:17全部汇总结束；GPU4–7已释放。

原始指标在merged/nine_task_acceptance.json、merged/scores.json及同一源模型的post/evaluation/；完整预测、退出与SHA证据原位保留。本批未新增或删除模型权重；75%模型作为S4达标研究候选交付保留，原donor及FP32状态继续保护，其余比例／原控制仍按原批评审边界处置。

**泛化性警示继续有效。** 这是当前同seed、同轨迹、同基座／位宽的模型合并候选达标，不证明.75比例可普遍迁移，也不证明目标改动或保护整数本身足以恢复所有知识。两donor未补完整LB，不能给出merge相对它们的九项净增益。固定recipe从原始BF16重放、第二seed及必要跨模型／位宽复核仍未完成；下一批优先讨论整链验证范围与预算，不自动开新训练、SPACE、比例细扫或主码学习。经验已补入[同轨迹merge与泛化警示](../../docs/experience/lessons/kd_and_data.md#教师保护scale和整数qat的条件证据)。
