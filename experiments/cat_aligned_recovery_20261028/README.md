# C09/G09 与旧 best 的恢复链对齐

本目录独立绑定真实 G09 UUID `e8e5efd4-655e-462c-b6e0-fedf4e2e0346`，不修改旧实验源码、配置或 checkpoint。G09 已有 Q/K K38/full-RHT，保留该几何和全部其他投影。旧 best 的完整来源由主实验记录维护；本目录只管理新对齐阶段。

同配方 C09 UUID `726b7b36-bb95-4418-b626-cc029b4f0aec` 为第二个固定 profile：C 的gate32与旧best同形，G的gate34保留额外容量；两者Q/K K38的432叶/1350144元素完全同形。仅比较两真实源的同后恢复，不增加超参组合。G使用GPU0、C使用GPU1，输出分别在独立`G/`与`C/`目录。

固定阶段依次为：

| 阶段 | 新父来源 | 旧配方证据 | 就绪条件 |
| --- | --- | --- | --- |
| 01 QK-only FineWeb2000 | G09 原生末点 | `general_lm_qwen_k38_qkonly_s2000/launch.prepared.json` | 本阶段 CPU prepared 与 G09 native audit |
| 02 all252 D 在线 OPD32 | 01 实际末点 | [新源接线与旧配方绑定](../cat_aligned_opd_20261028/README.md) | 01 完成2000并核末态冻结、重载和位宽 |
| 03a D+scale Med32；03b D+q QAT64 | 同一个02实际末点，各自 fresh | [保护参数恢复入口与旧配方](../cat_aligned_protection_20261028/README.md) | 02 原生核证；独立核新学生输入、目标及原数据顺序 |
| 04 C1 式完整投影拼接 | 03a 全MLP＋03b 全attention | [同父整投影组装](../cat_aligned_protection_20261028/README.md)；旧 C1 UUID `e87d…` | 两分支完成并按完整投影保留原生几何/保护；新UUID与账本 |
| 05 L18 down J32 | 04 实际 C1 | [局部更新基准](../cat_aligned_selected_20261028/README.md)；旧`eta001_R05` | η=.001/R=.5/top482/trust.01 的32步连续U配方，来源独立绑定 |
| 06 L9 全MLP B32 | 05 实际 J32 | [局部更新基准](../cat_aligned_selected_20261028/README.md)；旧`selected_mlp_d32/B` | 新gate34使18叶共56256元素；新源冻结补集和基线 |
| 07 abs005/Rice7 | 06 实际 B32 | [同源稀疏构建与两卡完整验收](../cat_aligned_sparse_20261028/README.md) | 从新末态重建HF−native BF16残差；实测logical≤2.6/rank0后完整验收 |

01现已完成并独立核证，02已完成短验并启动fresh正式32步，实际状态见末节；03及以后仍须各自闭合真实父模型与其核证，再绑定下一步。这里的后续表是固定来源/就绪清单，不冒充已完成训练结果。

## 阶段收益与简化流程的读出

2026-10-28按可迁移算法目标固定最少节点：每个profile完整评测01、03a、03b、04、06、07，共六个成品；原09已有完整九项，直接复用。02和05保留真实训练、冻结状态、来源和原生导出审计，暂不另做完整九项。若以后出现只能由中间点回答的问题，再补对应读出，不预先铺开全部节点。

| 对照 | 本轮能回答的问题 |
| --- | --- |
| 01相对同源09 | QK专项恢复在新基座上的净效果，以及C/G排序是否变化 |
| 03a／03b／04三个成品 | 固定C1组装是否超过更好的完整母本，不能默认旧模型的微小拼接收益仍存在 |
| 06相对04 | 将05＋06作为一个局部恢复阶段，检验整体效果 |
| 07相对06 | 稀疏补偿在新恢复基座上的净收益与最终预算 |

这些是阶段成品对照，不是删除步骤后的完整因果消融。例如04低于某个母本，只说明当下组装没有优势；不能据此断言从该母本重新训练同一后链必然更好。本轮不按中间分数自动更换05父模型、回退或重训，完整对齐链先作为参考。以后能否合并恢复轮次、删去分叉，要用明确的同源尾段对照判断。

普通3423字段节点通过[独立完整九项入口](../cat_aligned_evaluation_20261028/README.md)复用现有完整八项与原生2.6bit LongBench消费者，生成、评分和分区逻辑保持原协议。01已在真实末态核证及空卡检查后，用GPU2/3评C、GPU4/5评G，与GPU0/1的02训练并行；完整八项初报见该入口末节。两卡分区覆盖官方21项各一次，全部4750条正常完成后归并；不按八项分数跳过LongBench。07使用实际Rice bundle消费者，不能当普通3423模型加载。

统一0-shot，MMLU batch64、其余七项batch32、64叶24742题；LongBench v1完整21项、32K总窗口、官方输出预算、非思考greedy、seed42。九项按`(8×八项均分＋LongBench21宏均)/9`，目标67.2471449179；logical≤2.6、rank0，packed另列。阶段读出不会新增MMLU或单任务保持门槛，也不以局部loss代替完整指标。

## 固定训练配方与运行

02保持原seed137、Maux1024/Qasper32/Hotpot96，M目标只混一次`.5F+.5gold`；在线当前学生原生greedy/full-F OPD，M/QA各`.5`、R0，所有1512个decoder叶、D LR1e-6的原linear500前32、AdamW(.9,.95)/eps1e-8/WD0/clip1.5。03a使用Med512＋aux512、原QA/R、全D＋252保护scale、32更新，D1e-6/scale1e-3；03b使用M1024/Q32/N32/H192/R896、原32组重复两遍、全D＋252保护INT8 q、64更新，D1e-6/q.03。两臂都保留原500步linear前缀；Med32来自旧64训练的32快照，不能改成32步调度。

03的R教师必须绑定新对应02初态；05/06共同绑定新04 C1式初态。不能套用旧e3/C1学生R cache冒称新源对齐。full-F缓存是否复用也须先核实际非投影状态及输入身份。04复用`qwen_scale_qat_scope_merge_20261015/README.md`所述C1方向及成熟`experiments/w25_eval/assemble_categories.py`：03a提供108完整MLP、03b提供144完整attention，同投影的decoder、bank、保护q/scale、indices和rotation一体迁移，非投影骨干须两臂exact。

第一阶段直接复用旧 `launch.prepared.json` 的完整 `training_argv`，仅替换 source/output：FineWeb-Edu `sample-10BT` revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`、train、ODC-By；seed/data_seed0，原生单文档LM无packing/chat，B4/GAS1/max1024；纯 `kl_top_mass` alpha0/top100/T1/prompt0；Q/K-only432个FP32 decoder叶、1350144元素，BF16计算；LR3e-6/cosine2000/warm20/clip1.5，码、保护、norm/head冻结，无LoRA。原生save400/limit1不变；外部每400步完整八项暂不执行，训练数学和调度保持旧配方。

调用者激活 `bitvae`，在 `rotation_factor_probe/source_snapshot_20261005_01` cwd执行：

```bash
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/cat_aligned_recovery_20261028/prepare.py
bash /home/star/VAELLM/experiments/cat_aligned_recovery_20261028/run.sh
CUDA_VISIBLE_DEVICES='' VAELLM_ALIGNED_PROFILE=C python -B /home/star/VAELLM/experiments/cat_aligned_recovery_20261028/prepare.py
bash /home/star/VAELLM/experiments/cat_aligned_recovery_20261028/run_C.sh
```

输出在 `/data/star/VAELLM/cat_aligned_recovery_20261028/G/01_qk_fineweb2000`。`prepared.json`绑定实际源SHA/UUID/504码库、真实几何、成熟runtime及数据；`actual_scope.json`由复用的Q/K启动回调核实际432叶/2991冻结/180缓存；`native_saves.json`记录五次原生保存。`execution.json`、`run.log`、`run.exit`与终态`training.json`区分运行/完成/待独立核证。没有自动下一阶段、评测筛选或重试。

2026-10-28首次CPU准备已修正新入口的Python符号链接比较、G09不存在`assembly_report.json`的误假设及输出目录创建；均发生在GPU训练前。首次G运行也在模型加载前因旧callback构造器强制要求外部export GPU而exit1；新callback只初始化本轮record/source/root，原QK `on_train_begin`完整复用，并在CPU实际构造验证。失败核心日志/执行记录保留在`G/01_qk_fineweb2000_failed_callback_init`，未发生模型FW、BW或更新，正式重新fresh绑定。

修复版CPU实际通过：G耗时8.083633s、prepared SHA `af4137d67b1c090b9e0b87bc3ab143c8f94a0fb57c827b8e4f380e127fbc3975`；C耗时7.855572s、SHA `0e7961425756fd01a0a627491217156dfe0be79191c867409ef70508da759e1b`，两者CUDA未初始化。19:17:35/36分别后台启动tmux `cat_aligned_G_QK2000_GPU0_fix_20261028`与`cat_aligned_C_QK2000_GPU1_20261028`；实际worker为272069/start2626119、272077/start2626145。起跑前GPU0/1均0MiB且无compute，GPU5未触碰。当时两者原生主循环均已完成首个真实optimizer step，状态为**已启动、运行中**，尚无2000步完成或质量提升证据；该启动快照由以下真实末态更新。

## 10月28日：01完成核证，02正式恢复已启动

两profile的01均已真实完成2000更新、`run.exit=0`，五次原生保存与最终训练状态保持。两份`aligned_QK2000_terminal_native_v1`审计均verified：432个FP32 Q/K decoder末点转BF16精确读回，2991冻结字段及504主码bank保持，3423字段保存／fresh strict一致；未重压Q/K或更换几何。原`checkpoint-2000`、末态权重及审计绑定的输入继续保留，供后续真实来源核证。

| profile | 01实际末态UUID | 原生审计 | logical／packed bpw | 后续状态 |
| --- | --- | --- | ---: | --- |
| C | `9a392147-20b2-418d-9cf6-25d2ffff3a5b` | [C native_audit](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/C/01_qk_fineweb2000/native_audit.json) | 2.497367527173913／2.5107421875 | 02短验通过，fresh正式32已启动 |
| G | `44ed6d15-abf0-46b0-a627-998a6406087a` | [G native_audit](../../../../../data/star/VAELLM/cat_aligned_recovery_20261028/G/01_qk_fineweb2000/native_audit.json) | 2.5295091711956523／2.6391813858695654 | 同上 |

01完整八项已落盘，UUID／64叶24742题／原始结果SHA及八项等权算术核对通过；[唯一八项初报](../cat_aligned_evaluation_20261028/README.md#10月28日01完整八项初报九项尚待完整longbench)记录精确分数及同源09对照。C／G八项均略低于各自09，不能据此提前判定完整九项收益或更改后继；固定LongBench21／4750仍按原链在后台执行。

02的short2、正式启动与一次性短patch收尾归[OPD唯一正文](../cat_aligned_opd_20261028/README.md#10月28日short2完成与fresh正式32启动)。root于21:48一次核实两正式臂实际3／32更新后停止监控；本次仅复用该已记录事实，没有轮询新训练进度或GPU。正式32及独立导出尚无本次核实的完成结论，03仍待各自真实02末态核证，不自动推进或按八项改链。无新增算法经验。

## 10月29日：01完整九项与02原生终态已核实

本节替代以上在途状态。C01／G01均完整八项64叶24742题加LongBench21／4750，外层exit0，九项分别64.7082363602538／65.71384898861425，较各自09变化−0.005156345388485306／−0.07036111151528246pp；[唯一完整评测正文](../cat_aligned_evaluation_20261028/README.md#10月29日c01g01完整九项终态)保留身份、位宽、原始指标和核证范围。两者均未达[10月29日当前目标](../../docs/experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md#10月29日当前目标与完成条件)，原报告仍保留≤0.8pp的历史阈值及原判定。

C／G02已完成各自fresh32组及独立原生导出，两层exit0，3423字段fresh strict、1911冻结字段及504主码bank审计verified；[唯一OPD终态](../cat_aligned_opd_20261028/README.md#10月29日正式32与独立原生导出完成)记录新UUID和实际成品。02自身尚无完整九项，03及后续未执行；后继必须消费本profile真实02，并重新生成对应R缓存，不能沿用旧教师概率或把准备代码当作执行结果。当前已核任务没有运行失败或缺失预测需要续跑，本次未启动新任务、修改算法或清理模型。无新增算法经验。

## 10月29日：五个成品读出与可迁移流程的下一步

**同日用户新指示已替代本节的立即执行安排：** 优先围绕向量自适应残差阶数、`A Fix for Softmax and Cross-Entropy`的数值问题、监督微调的几何校准三方向整理递进路线，每步按效果决定下一步。本节五成品批次仅保留为可复用对照材料，除已授权的G初态R缓存外，不据本节自动启动03正式训练或整批评测；先完成对应机制与原生路径的具体设计。新算法仍先隔离研究，正式整合遵守项目确认边界，不能把论文建议当作已验证收益。

用户已激活[当前总目标](../../docs/experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md#10月29日当前目标与完成条件)：保留VAELLM／BSQ主体，logical≤2.6、九项相对同口径BF16绝对下降严格<0.9pp，先Qwen3-8B，再验证至少一个确认的其它系列8B模型；从HF可复现整链与可迁移选择规则都是完成条件。本节取代上方“每个profile固定六个完整读出并继续05／06”的后续安排；已完成原件、原阈值报告和训练状态保持。现有C/G链是研究参考，尚不是满足这些条件的最终算法。

本批仅新增以下五份完整九项，不重新评估09／01，也不重新训练02：

| 成品 | 新执行范围 | 决策价值 |
| --- | --- | --- |
| C02 | 对已完成原生末态做完整九项 | 与同源01比较OPD净效果，检查C/G排序是否随恢复改变 |
| G02 | 同上 | 提供G保护阶段的真实整模控制；原计划跳过02，无法区分OPD与后续保护恢复的贡献 |
| G03a | 从真实G02运行既定D＋scale／Med32，独立导出并完整评估 | 检验一条完整恢复路径能否提高G02，避免只按其MLP片段判断价值 |
| G03b | 同父fresh运行既定D＋q／QAT64，独立导出并完整评估 | 检验另一条完整路径及其任务取舍 |
| G04 | 按既定03a全MLP＋03b全attention规则组装，完整评估 | 判断固定组装是否超过两个完整母本中更好的一个，量化保留分叉的实际价值 |

暂不运行C03／C04。G在已完成09与01两处均比C高约1pp，且logical预算合法，因此先把G作为比较恢复结构的代表；这不证明gate34具有独立因果优势，也不证明C的后恢复上限更低。C02若发生排序反转，记录为下一次决策依据，不据此中途取消已安排的G端点或自动复制一整套C。两份02评测可与G03训练使用独立GPU并行；G两臂共用绑定真实G02的一份R缓存，03a／03b完成核证后才组装04。新接线仅补仍缺证据的实际状态往返与原生出口验证，复用有效既有证据，不追加常规超参冒烟。

所有五份均完成八项64叶24742题及LongBench v1全21／4750，沿原生成、评分、指标和未舍入聚合协议；不能据先出的八项取消LongBench。新<0.9pp判定与历史≤0.8pp报告分开记录，packed及完整文件大小另列，不能把logical上限改成物理上限。03a与03b的数据来源和32／64预算不同，二者比较的是完整配方；本批没有D-only匹配控制，不能把分差单独归为scale或q的因果。04相对更好母本的差值仅证明该组装在此阶段的效果，不证明从母本接后续恢复会得到同样差值。

后续规则化按以下边界推进，当前不自动启动新尾段：

1. **先确定是否需要分叉，再研究固定层号。** 比较G02、两个完整母本及04。没有测得组装净优势时，优先研究保留完整母本的单路径；若组装有优势，保留其真实两父与完整投影规则，不扩大为“attention适合q、MLP适合scale”的模型通则。单次小分差是开发证据，不能称稳健跨模型规律；下一条尾段必须从选定的同一个父节点实际执行，不能把历史增量相加。
2. **暂不接硬编码L18／L9的05／06。** 敏感范围应来自新父自己的train／aux诊断，诊断只提名有限候选，再验证真实D／码恢复，不能把HF条件替换收益当成可训练收益或用小面板硬淘汰。层号、预算和有限候选的选择过程须在该轮运行前记清；本批尚未定义或验证新的自动范围选择器，不正式整合新算法。
3. **把构造规则与数值调参分开。** 可复用的是同父生成候选、原生范围／冻结合同、对应阶段R教师、完整投影迁移、预算计账和有限整模比较；新模型可重新计算几何、tokenizer／模板、统计量，并按固定搜索范围调参。不能依赖旧Qwen donor UUID、手选层号、历史缓存或事后最佳步点。已有九项被用于开发择优，必须如实记录接触和候选数，不能称盲测，也不据测试逐题表现挑训练样本。
4. **最终从HF闭合，而非把边界接续当作从头复现。** 当前C/G共用历史after_o，仍只是有完整来源的阶段接续。配方确定后，统一入口须从指定BF16基座重建顺序CAT及必要恢复／分支，所有donor由本次父状态生成，再实际保存、独立重载并完成九项和位宽审计。迁移时冻结这套阶段与选择规则，重新绑定目标模型数据／统计和结构；两模型各自达标，不能只展示loader兼容、末段正例或跨模型均分达标。

设计复用了[实验设计与代理排序边界](../../docs/experience/lessons/experiment_design.md)、[保护／稀疏及存储经验](../../docs/experience/lessons/compression_choices.md)、[KD与跨模型数据边界](../../docs/experience/lessons/kd_and_data.md)和[编码恢复的选择与确认边界](../../docs/experience/lessons/liftquant_recovery.md)。本批新增读出针对已出现的阶段收益疑问；不重开已完成负例或扩大其它算法研究。上述是下一组设计，不是五份新质量结果或最终迁移性证明。
