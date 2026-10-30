# 夜间固定接续接口

**2026-10-30终态已核：四路均完整完成／exit0，当前严格质量均未达。最终事实见本页末节；此前启动／等待段落只保留历史过程。**

本目录入口只消费真实完成的末点，不等待前驱、不重试、不回退或启动新训练。夜间driver负责按已授权任务的真实PID退出／外退出状态和GPU资源接续；本页不改变中央执行计划。

## CAT A/B 完整九项

调用者激活`bitvae`，项目根cwd，CPU协调器隐藏CUDA；`GPU列表`由root／夜间driver固定分配：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u experiments/overnight_goal_20261029/evaluate_cat.py --arm A --gpus GPU列表
```

B只改`--arm B`及其已分配GPU列表。入口读取各自`/data/star/VAELLM/cat_structure_full_20261029/{A,B}/run.json`，必须正式`completed`、`run.exit=0`且最后保存事件为本支`native/.../final_model`。动态绑定实际UUID、metadata／runtime SHA、权重稳定stat及原prepared源码pins，不预填UUID、不借其它模型。核完整七类别252投影、504bank、3423持久字段，无pending dense／外置LoRA，A gateD32K32或B D64K64、S2 asymmetric H128/R0，其余六类保持原结构；实际位宽由N `checkpoint_bpw.py`读取序列化state计量，logical≤2.6、packed和整模型存储另报。

只有真实末点全部源检查通过才生成`result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/sources/{A,B}.json`。该`verified`范围是实际保存来源、序列化库存和账本；独立原生模型strict读入／GPU预热由随后原full8／full21路径验证，不伪称此前已经fresh strict。评测完整复用原full8的MMLU64／seven32、0-shot／64叶24742题，以及完整LongBench21／4750分区、预测、原样本核验和历史汇总；不按八项筛选。独立输出`.../overnight_goal_20261029/evaluation/{A,B}/`，最后另生成当前strict<0.9pp／logical≤2.6的`transfer_acceptance.json`和`current_goal_result.json`。已有输出拒绝覆盖。

10月29日18:35前后完成bitvae后台CPU导入／GPU参数解析及真实已有metadata检查：A／B当时均为running，最新真实after_up UUID分别`bc17c79a-d3a8-4494-957d-de28ea6697b8`／`86ef4fd2-1128-49b6-af14-32b9c1b2915f`；各216投影规格和432个metadata bank通过N接口检查，两臂未final均立即拒绝。原始CPU记录为同结果根`evaluate_cat_cpu.log`，`final_source_verified=false`、CUDA未初始化，未创建sources或evaluation/A、B。此项不证明真实final账本／模型读入或完整质量；本轮未启动GPU评测，A/B／N及原评测活动源码未改。

## G04固定整家族组装与完整九项接口

本次只做一个既定C1组合：实际scale32全部MLP，实际q64全部Q/K/V/O（完整144个投影，D／bank／q／scale／indices／rotation一起替换），不按逐题选择或追加反向组合。两父同G02 `38a7461e-a2f1-4586-80e2-b99883ef2628`；scale32为`6a446110-a963-45a5-9847-8458f5d88c40`，q64为`993ee918-c14a-4d3e-96f6-7766ba71b37c`。

GPU5实核0MiB／无compute后，在调用者bitvae、原`rotation_factor_probe/source_snapshot_20261005_01` cwd、`VAELLM_ALIGNED_PROFILE=G`／`CUDA_VISIBLE_DEVICES=5`下原样启动`experiments/cat_aligned_protection_20261028/assemble.py`。tmux `G04_C1_assembly_GPU5_20261029`，shell PID604543／start_ticks11036925；真实命令、环境、来源audit SHA／stat与旧源码pins保存于同结果根`G04_assembly.launch.json`。启动记录保持原样，完成依据为`G04_assembly.exit=0`及原`/data/star/VAELLM/cat_aligned_recovery_20261028/G/04_C1/assembly_report.json`。

组装已verified，真实新UUID `4ab63833-418c-4418-8d4a-3ae7317d1a94`，3423字段保存／fresh strict exact、504bank、147非投影exact，实际logical2.5295091711956523／packed2.6391813858695654 bpw，完整tensor bytes4,781,328,384。原始核心日志为`G04_assembly.log`；assembly report SHA `41eac6c7257dc2dcbd66b8fdfcb142d48b2d66aae376fc067dd4a31752946eb6`。

独立薄入口如下，缺真实终态立即拒绝；`--check-only`在18:41已对该真实末点exit0，不占GPU运行模型：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u experiments/overnight_goal_20261029/evaluate_g04.py --gpus 5
```

入口复用原`source('G','04',明确final_model／assembly_report)`及原full8/full21／原样本核验，绑定固定两父来源和当前实际账本，输出`.../overnight_goal_20261029/evaluation/G04/`并生成当前strict<0.9pp／logical≤2.6的transfer报告。CPU检查记录为`G04_entry_cpu.log`／`.exit`，新脚本SHA `eb83bb0c85ab18f2b587ba1564133ae0ffd71deea8d900762011dc9d8e66c2a3`。本次没有启动全九项，由root夜间driver确认资源后接续GPU5；尚无G04质量结果，组装／存储通过不等于精度通过。其它正在运行的评测／A/B／Muon和所有旧源码未改。

## 固定夜间接续已启动

2026-10-29 18:47启动四个独立tmux job。`driver.py --prepare`保存唯一`night_manifest.json`，`--job A|B|Muon|G04`只执行对应固定完整九项。A使用3,6；B使用4,0；Muon使用7,1,2；G04使用5。前驱由真实outer PID/start和退出记录绑定；节点链及q64仅是释放GPU的依赖，各自训练／导出才是必需成功的数据依赖。质量not_met保留为有效结果。

复用已验证Linux pidfd等待，新增有界等待的真实CPU子进程验证已通过：PID错配拒绝、退出事件、超时不杀前驱、已退出可读取。组装shell已成为Z状态且真实exit0落盘，准备时按已完成绑定，不等其空cmdline。等待无轮询、重试或占卡。前驱最晚等至2026-10-30 10:00 Asia/Shanghai；新评测按原24h推理预算外加full8余量，整体执行上限90000秒，不把10点当作正在运行任务的统一终止时刻。GNU timeout仅包新评测命令及其子进程，不停止既有前驱。

配置与启动记录位于项目`result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/`：`night_manifest.json`、`night_launch.json`；各路唯一状态`jobs/<任务>/report.json`及`exit`，驱动日志`<任务>_driver.log`，推理日志`jobs/<任务>/evaluation.log`，最终完整结果`evaluation/<任务>/current_goal_result.json`与`transfer_acceptance.json`。当前18:49实核G04已真实推进MMLU，A/B/Muon三个driver均持实际`anon_inode:[pidfd]`等待自己的训练／导出；不能把等待写成评测已完成。完整配置和当前资源安排统一见[执行计划](../../docs/experience/records/w25_transfer/current_plan.md)。

## 10月30日真实终态与完整结果

09:30重新实核无tmux server，目标训练／评测／夜间driver均已自然退出；四份`jobs/<任务>/report.json`为`complete`，其`exit`和评测stage均0，核心`jobs/<任务>/evaluation.log`末行与最终UUID及分数相符。质量验收返回1表示有效`not_met`，不是技术失败；本节替代此前“等待／尚无完整九项”的状态。

A/B正式`run.json`均`completed/exit_code=0`、外`run.exit=0`；共同原父仍为after_o UUID `b63bfe79-3889-40c6-bf66-1d002893f874`，本支fresh fork和最终来源均未换。gate/up/down各36层、每stage10000更新、S2；`short_recovery={}`，无短验callback残留。训练实际耗时A为27146.497135400772秒、B为24979.84431552887秒。原配置与运行细节见[完整结构对照](../cat_structure_full_20261029/README.md)。

| 成品 | UUID | 八项均分 | LongBench21宏均 | 九项（原报告未舍入值） | BF16掉点 | 完成时间（上海） |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | `26a03a51-efb4-41ec-abc1-a329a01c9fea` | 65.97294670308197 | 43.91590364990137 | **63.52216414161746** | 4.524980776291024 | 2026-10-30 00:28:14 |
| B | `87571461-7310-46e5-bc2e-904baf1818ed` | 66.47916293809848 | 42.2226139557569 | **63.78399082894941** | 4.263154088959069 | 2026-10-29 23:59:09 |
| Muon | `1257125f-7c64-4650-80b0-04ec2deac99f` | 68.81637930163077 | 44.088881010301485 | **66.06887949148307** | 1.9782654264254065 | 2026-10-29 21:25:13 |
| G04 | `4ab63833-418c-4418-8d4a-3ae7317d1a94` | 68.91850176657171 | 44.348562399788804 | **66.18850850359583** | 1.8586364143126464 | 2026-10-29 23:50:23 |

四路都是完整0-shot八项64叶／24742题、LongBench21／4750，所有sample artifact、分区终态、合并预测和评分来源同各自实际UUID。全8路（再含G02/direct/scale32/q64）共享只读复核摘要为[终态与精确配对审计](../../result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/terminal_audit_20261030.json)：共2199个唯一文件SHA通过，metadata／审计SHA及权重size/mtime_ns/inode与保存记录一致；它复用原实际张量位宽账本，不声称本次重新加载权重或GPU重评。严格掉点以原十进制输入的有理数计算；所有未舍入九项分量及12组逐项差值均在该JSON的`score_vector_exact`、`paired_deltas`中，避免表格显示舍入影响判断。

| 成品 | 实测logical bpw | 实测packed bpw | 全模型tensor bytes | rank | 当前验收 |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 2.497367527173913 | 2.5107421875 | 4669814784 | 0 | budget通过／quality未达 |
| B | 2.498731530230978 | 2.5121061905570654 | 4670999040 | 0 | budget通过／quality未达 |
| Muon | 2.5295091711956523 | 2.6391813858695654 | 4781328384 | 0 | budget通过／quality未达 |
| G04 | 2.5295091711956523 | 2.6391813858695654 | 4781328384 | 0 | budget通过／quality未达 |

logical账本含code、decoder、保护indices/scales/weights及rotation，packed再含码padding；embedding/head/norm在全模型tensor bytes中另计，不以logical值冒称整文件压缩率。当前严格目标为九项>67.14714491790848、logical≤2.6，四路都未达质量；完整质量／实际存储验收也不单独证明可迁移流程。

同父同流程A/B的最终差值如下（0–100标度，显示6位；精确值见审计）：

| 指标 | B−A |
| --- | ---: |
| BoolQ | +2.629969 |
| RTE | -2.888087 |
| Winogrande | +0.631413 |
| ARC-E | +0.505051 |
| ARC-C | +1.194539 |
| OBQA | -2.200000 |
| PIQA | +2.339499 |
| MMLU | +1.837345 |
| LongBench21 | -1.693290 |
| 九项 | +0.261827 |

B的本轨迹九项高于A，但LongBench、RTE和OBQA回退，不能据单次配对称所有能力均改善。两支起点是含live dense MLP的类别边界，缺少同口径的共同父完整九项；不补造“相对父最终低比特增量”，也不把节点八项结果当最终九项。Muon相对同G09 Adam和冻结G09、G04相对两父／G02的全部对照统一归[恢复结果终态](../g09_recovery_20261029/README.md#10月30日完整结果与同父配对闭合)。

10月30日主线程与研究线依据本次完整结果，选择B作为下一阶段结构；Muon保留同父正收益证据，G04保留候选，q64固定单支不延长。该取舍限于当前单seed和完整配方，不称B或Muon全面改善全部能力。经验边界归[同父恢复与能力代价](../../docs/experience/lessons/kd_and_data.md#同父恢复须同时看完整净收益与能力代价2026-10-30)。本次结果核证没有重评模型；随后A最终成品的无后续用途权重／runtime／tokenizer副本已清理，实际释放4,709,863,424 B，原始结果与身份完整保留。终态审计的cleanup字段列精确路径、SHA及保留验证；A目录现不可加载，B及活动来源未动。
