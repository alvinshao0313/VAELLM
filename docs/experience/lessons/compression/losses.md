# 重构、输出损失与辅助项尺度

[经验分类入口](../../README.md) · 一级分类：压缩经验（`compression`） · 二级专题：`losses`

本篇按问题收录既有经验。原始日期、配置、数值、假设及证据边界保持；历史“当前”仅指原核查时点。跨专题按链接查阅，不能由分类推导通用算法规定。

## 条目导航

- [输出损失：先校准数值尺度，保留真实目标](#output-losses--输出损失先校准数值尺度保留真实目标)
- [按条件选择动作](#output-losses--按条件选择动作)
- [下次怎么做](#output-losses--下次怎么做)
- [问题与证据](#output-losses--问题与证据)
- [原因与解释边界](#output-losses--原因与解释边界)
- [wa_mse坐标、统计刷新与完整质量](#output-losses--wa_mse坐标统计刷新与完整质量)
- [边界](#output-losses--边界)
- [recon、辅助项和BF16日志平台](#output-losses--recon辅助项和bf16日志平台)
- [BSQ辅助权重、幅度和匹配负例](#output-losses--bsq辅助权重幅度和匹配负例)
- [覆盖率后续节点与延长VAE预算](#output-losses--覆盖率后续节点与延长vae预算)
- [AMSE尺度与stage归一化语义](#output-losses--amse尺度与stage归一化语义)
- [完整输出拟合预算与码建议覆盖的交互](#output-losses--完整输出拟合预算与码建议覆盖的交互)

<a id="output-losses--输出损失先校准数值尺度保留真实目标"></a>

## 输出损失：先校准数值尺度，保留真实目标

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#输出损失先校准数值尺度保留真实目标)。

<a id="output-losses--按条件选择动作"></a>

## 按条件选择动作

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#按条件选择动作)。

| 条件／要解决的问题 | 建议动作 | 证据状态与适用边界 |
| --- | --- | --- |
| 替换MSE、AMSE、Gram或输出目标 | 先校准重构/辅助梯度尺度，保持真实交叉项 | [下次怎么做](#output-losses--下次怎么做) |
| 局部NMSE更低而任务变差 | 查完整局部、整模与跨域对照，保留未证原因 | [局部误差改善与整模型跨任务退化](../evaluation/diagnostics.md#output-losses--局部误差改善与整模型跨任务退化) |
| 用SPACE等几何回退解释能力保持 | 分开密集目标的功能价值、原生拟合和任务结果 | [几何校准的目标价值与低比特可达性](../post_recovery/combination.md#output-losses--几何校准的目标价值与低比特可达性) |
| 改变码覆盖率或更新强度 | 看共同预算曲线，事件数不能替代质量 | [模型级KD、更新强度与覆盖率曲线](../post_recovery/objectives.md#output-losses--模型级kd更新强度与覆盖率曲线) |
| 日志平台或辅助项变小 | 还原系数、归一化及BF16舍入位置 | [recon、辅助项和BF16日志平台](#output-losses--recon辅助项和bf16日志平台) |
| wa_mse/AMSE或统计refresh | 核真实坐标与消费来源，分开实现通过和质量 | [wa_mse坐标、统计刷新与完整质量](#output-losses--wa_mse坐标统计刷新与完整质量) |
| BlockLDLQ或扩大搜索覆盖 | 检查真实全模型结果、缓存及拟合交互 | [BlockLDLQ的完整输出、缓存与全模型结果](../post_recovery/code_search.md#output-losses--blockldlq的完整输出缓存与全模型结果) |

以下按问题查动作；各节保留原核查日期、实测条件、未验证声明及原始来源，历史记录不代表当前运行状态。

<a id="output-losses--下次怎么做"></a>

## 下次怎么做

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#下次怎么做)。

- 替换损失前，测量新旧重构项、梯度及辅助项的相对尺度；将尺度对齐作为独立变量，保持初始化、预算与评测题目一致。
- 分组目标验证 H=I 退化为 MSE、对角 H 退化为 AMSE，并比较数值和梯度；测试通过只说明实现语义。
- 完整 Linear 输出目标应先累加各输入块贡献再取平方，保留组间交叉项。分组二次型没有这些交叉项，不能视为同一目标。[完整输出目标记录](../../records/experiments/linear_output/EXPERIMENT_SUMMARY.md)
- 全局 BSQ entropy 等辅助项不能因显存分块就改成块均值的平均。teacher-input 与 student-input/teacher-target 也应作为不同实验。
- 使用相同题目配对评估；题目 bootstrap 不覆盖训练随机性，多个训练种子也不是新增独立题目。
- 比较校准量时统一构造最大训练前缀和其后的公共留出，再按文档组/token去重；扩大训练集后，已进入训练前缀的旧留出不能继续当独立确认。保留训练、留出、权重误差和优化轨迹，最终仍以完整八任务验证。

<a id="output-losses--问题与证据"></a>

## 问题与证据

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#问题与证据)。

真实 Qwen3-8B 的3个 up_proj、两种子、每阶段300步、三任务各前256题对照中，MSE均值71.4193%，分组输出56.5755%，AMSE44.1406%。在这些固定条件下，直接替换默认目标没有收益。[原始中文报告](../../records/.result/activation_loss_ab/20260922_input32_downstream_v2/REPORT_CN.md)

2026-10-01，真实Qwen3-8B双stage/full rotation的L18/o，以完整原坐标输出MSE、固定decoder、FP32敏感度代理优化两级packed码128步。32个Alpaca训练文档/32独立留出文档上，native训练/留出MSE分别降低74.50%/30.74%，权重MSE却增加62.82%。非码字节、存储位宽及独立重载一致性均通过。这说明有限激活分布上的输出收益可以伴随其他方向误差增大，尚无该配方的整模型准确率结论。[配置、逐步轨迹与原始验证](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)

<a id="output-losses--原因与解释边界"></a>

## 原因与解释边界

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#原因与解释边界)。

真实 Gram 矩阵对角均值在被测层间约为0.0046、0.0625、1.3295；加入 H 不只改变误差方向，也改变重构项相对于固定 BSQ 辅助项的尺度。这个尺度变化有测量证据，但尚未通过独立消融证明它解释了全部掉点。不能据此直接断言“相关性有害”或“校准过拟合”。

<a id="output-losses--wa_mse坐标统计刷新与完整质量"></a>

## wa_mse坐标、统计刷新与完整质量

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#wa_mse坐标统计刷新与完整质量)。

2026-10-03核对并实跑现有wa_mse无旋转+普通input INT8路径：激活权重来自原输入abs-max，随被保护列一同删去对应项；transpose的down须按原矩阵输入坐标映射到块。stage2沿用原通道统计，重构目标则是各stage自行归一化的残差。该实现计算`mean(error² * abs(target) * activation_max)`，没有再将激活向量归一化，因此相对BSQ项的尺度也变。真实原HF layer0三MLP/S2×2步的12次生产gather和独立二维坐标参考精确一致，公式、有限非零梯度、参数更新及INT8保存/部署检查通过；短样本上的wa_mse/MSE约0.385–0.502仅是该层观测，不是收益证据。随后完整108-MLP替换、保留a277 attention的实际全量均分为59.85526529%/2.453262993bpw，低于a277纯CAT最好3.95231038个百分点（七项下降、OBQA持平）；3207持久张量保存/strict与完整64/32协议核验通过。该无旋转+wa_mse组合退役，因旋转/损失/类别组随机流同时变化，不能单独判定wa_mse造成全部下降。样本级实现正确不保证整模型质量改善。同日源码核查：`channel_refresh_after_category=true`在当前实现只写`channel_refresh_stats`供保护排名；训练wa_mse/amse仍只读初始`stats_by_linear`，不是加权损失统计刷新开关。随后原HF/all36 q→k、S2各2步、B8192、512×256的真实纯CAT短验通过：q后刷新使35/36层k保护集合区别于初始统计，36/36最终保存索引与刷新统计重算exact一致，第0层自然不变；证明实际消费链路，不证明准确率改善。不要把wa_mse、AMSE和完整Gram损失混称，也不要直接把原坐标统计搬到旋转坐标；当前旋转CAT仅支持MSE，新增旋转加权目标须另行推导并获授权。[实际训练、对齐和保存证据](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的validation/mlp_wa_mse_no_rotation。

<a id="output-losses--边界"></a>

## 边界

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#边界)。

局部3层、小题集的失败不能外推为全部输出感知优化无效；新增完整MLP候选也尚无超过纯CAT最佳的尺度校准收益证据。逐 Linear 实验的部分模型不能当作全模型2-bit成绩。

<a id="output-losses--recon辅助项和bf16日志平台"></a>

## recon、辅助项和BF16日志平台

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#recon辅助项和bf16日志平台)。

CAT日志需按实现系数解释：当前`recon`为阶段归一化批次MSE×l1_weight×num_models，num_models是同时训练的Linear数量，不是codebook_dim。本轮36个Linear/l1_weight1时，D16的13.5对应约0.375，而非13.5/16；每阶段又会重新归一化残差，跨阶段质量应看全量反归一化后的残差RMS。`commit`是包含熵的辅助项×lfq_weight×num_models，改变K会同时改变1/√K量化/熵尺度和熵求和长度，不能直接把两个K配置的commit数值相减当作纯commitment差异。D16 v类的两阶段真实残差下降及与D32的有限对照见[本轮记录](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)；局部观察仍不能替代完整下游结果。

2026-10-24补充日志精度边界：原CAT在encoder/decoder的autocast之外对BF16输入与重构计算MSE，所得标量乘`l1_weight×num_models`后才在日志中`.float()`，不能恢复此前舍入。M36/K24D64的recon约26.125，该处BF16间隔为.125；平台不能单独证明未学习、已收敛或容量上限，也不能反向断定全由显示精度造成。本次10K端点36/36个非保护旋转矩阵均改善，实际native BF16部署残差经FP32误差归约的聚合MSE较原B32降低27.1445%；结合参数／硬码变化只支持局部拟合有效，不证明平台后仍持续改善或九项收益。随后完整八项＋LB21验证：MMLU较B32多60题，但LB仅+.013770分、九项−.214055个百分点；这次局部MSE改善没有转为综合收益，不据此外推所有新增容量无效。先核日志尺度与舍入位置，复用真实部署误差及更新证据，最终按完整任务取舍，不据平台自动改精度或追加步数。[实际终态与诊断边界](../../../../result/qwen3_8b_mmlu65_20261004/qwen_b32_gate36_k24d64_20261024/README.md#loss平台的只读诊断)。

<a id="output-losses--bsq辅助权重幅度和匹配负例"></a>

## BSQ辅助权重、幅度和匹配负例

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#bsq辅助权重幅度和匹配负例)。

2026-10-03匹配独立down的D16K24/S2/H128R0/BF16/block64、input256保护、MSE/每stage10K，只将lfq_weight从2.5降至1.0，并保留a277其他216投影，完整均分由63.80757566%变为63.21269318%，实际位宽同为2.455285114bpw，固定64/32及保存/strict核查通过。五项下降、RTE/OBQA持平、BoolQ略升。该结果只否定此具体系数在当前条件下的收益；lfq同时缩放commitment和entropy，不能称为仅减弱commitment，也不是已保存0/1码位的蒸馏对照。[原始指标与退役证据](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的reconstruction_d16_down_lfq1。

2026-10-03核对当前BSQ：CLI的`new_quant`默认False，但实际CAT主脚本显式开启，已核50份正式normalized均为True。两分支均先做球面L2归一化并使用相同硬符号STE；True量化到±1/√K、False到±1，K为codebook_bits。切False会同时改变decoder输入幅度和commitment目标，不关闭归一化，也不是新的翻码算法或可学习0/1数值。辅助项为`lfq_weight × num_models × [commitment_loss_weight × C + zeta/inv_temperature × entropy_loss_weight × (gamma0 × Hsample − gamma × Hmean)]`；inv_temperature还影响软概率的sigmoid陡峭程度，不能只按外部系数解释。现有`gamma0=0`仅移除促使单样本符号确定的熵项，保留聚合分布的balance熵、commitment、球面归一化和STE。匹配独立gate/input64的gamma0 1→0对照保持gamma/zeta=1、inv_temperature=100、lfq=2.5、commitment=0.25、entropy=0.01、new_quant=True及其他训练条件不变，完整均分64.27600316%→63.25486099%，固定64/32、3423 strict与实测2.455285114bpw通过，7项下降、MMLU提高25/14042题。两stage日志辅助项下降包含删去损失项的直接效果，recon打印值不变，反归一化残差RMS最终仅增约0.023277%；不能据辅助项降低称优化更好。该具体配方没有支持减弱符号置信度压力的质量收益，也不是已保存硬码的模型级蒸馏或翻码数量消融，不外推所有BSQ系数无效。[源码语义、匹配控制及候选边界](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的reconstruction_d32_gate_gamma0。

<a id="output-losses--覆盖率后续节点与延长vae预算"></a>

## 覆盖率后续节点与延长VAE预算

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#覆盖率后续节点与延长vae预算)。

同父gate筛选要分开fresh日程和追加训练，并按实际部署缓存读出，而非只看stage日志。2026-11-02新after_o／native raw2、36gate D64/K64/S2／M36 pooled BSQ／E/D128/R0、seed31、恢复0下，fresh每stage20K相对10K控制的BF16 group8实际全权重输出NMSE仅改善0.636525%（固定首8条Alpaca train、共2048输入tokens），完整MMLU反少6／14042题；fit1221.48秒，常规10K新臂约666–670秒。20K重建完整线性LR时域，不是控制续训；三新臂stage初始化digest一致，旧控制没有digest，不称逐位匹配。该结果只结束本20K配置，不证明其它预算或几何无效。相同cohort的commitment .25→.5局部误差近乎持平却MMLU少80题，LR减半少35题；不把降低某个loss、微小NMSE改善或单项筛选当完整后缀／九项提名的充分依据，也不由三个负点推出全部超参不敏感。[真实部署读出、完整MMLU、配置与边界](../../../../experiments/cat_hyperparam_cohort_cmcc_20261102/README.md#完整结果与取舍)。

2026-10-02同一100%码位轨迹1600步完整结果补充：均分65.92361780174367%，比1200回升0.726052个百分点；同1600步decoder-only/1%/5%为65.89178271/65.67923702/66.09078131%，因此尚无全覆盖持续优于部分覆盖的证据。所有点使用完整strict实际64/32、2.456988791bpw；新点冻结/硬码/导出契约通过。ARC-E/C提升而BoolQ/RTE/Winogrande下降，说明只看总体回升也会漏掉任务差异。XOR73,036,343位仍是相对原码的终点距离，不是翻转事件累计。2000步仍在运行，既有800退化原因未明，不据此改写为“非100%导致不稳定”。来源：[Qwen3专题](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的 `e2e_bits_block32_a100_mp4_r128_lr4e5/step_1600/`。

2026-10-03相同D16-prefix的最后500步decoder恢复再次提供同轨迹完整证据：65.55210825→66.08632496%，净增0.53421672个百分点，7项提高、RTE持平。1911非decoder张量/504硬码bank保持一致，全部252 decoder发生真实BF16数值更新；因此即使down之后没有dense类别用于融合LoRA，当前decoder仍有实际恢复能力。不过该增益未补回down即时1.61340612个百分点损失，最终仍低于49e，不能由此推断更长预算一定补足。[冻结状态审计及五阶段完整结果](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的down_category_transition_d16_prefix。

同日固定S2延长VAE预算前核查：8K–10K日志的BF16重建值平台不等于全量反归一化残差已不再改善；两阶段最终RMS也不提供阶段内收敛曲线。30K fresh的线性LR时域与10K不同，不能解释成原10K成品继续追加20K。当前deterministic/GPU-resident路径按类别/阶段用独立Generator采样，BSQ前向无随机操作时，步数增加本身不自动改变后续stage初始化随机流；拆分类别会改变catidx/初始化序列。需分别核实这些条件，不把所有训练预算比较泛称为随机流不同。[原始日志、配置检索与实际候选](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的reconstruction_d16_down_s2_s30000；随后该完整对照实测62.98061572%，低于匹配10K的63.80757566%；最终未保护原坐标RMS0.006414791→0.006402920仅下降约0.185%。这组额外预算及其线性LR时域没有形成收益，不能因微小权重RMS改善继续无条件加长。此配方已退役，不外推为所有预算/几何无效。

<a id="output-losses--amse尺度与stage归一化语义"></a>

## AMSE尺度与stage归一化语义

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#amse尺度与stage归一化语义)。

AMSE与wa_mse的系数不能只按名称对齐：现有CAT前者读取原输入sq_mean、后者读取abs-max并另乘目标绝对值，输入向量均不按均值/trace归一化，stage2沿用原统计。2026-10-03正式Alpaca512×256的108MLP统计按实际保护列剔除后，h加权均值约0.1477，down层均值0.00051735–1.483162；原HF九矩阵阶段1完整乘子均值也相差数倍至千倍。固定辅助损失时应明确这个尺度差，并记录所用现有全局系数；乘子均值不等于真实loss/gradient，不能用其保证质量或直接推出应逐层归一化。该l1=6.77完整候选随后实测61.76400485%/2.453262993bpw，低于a277纯CAT最好2.04357081个百分点，具体配置退役；相对WA组合提高1.90873956个百分点，但同时改了目标和系数，MMLU/PIQA/OBQA反而下降。尺度乘子的计算正确并不证明该系数有整体优势，更不能把这一负结果扩展为所有AMSE无效；没有新增归一化算法。[实际统计、完整结果及候选边界](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的reconstruction_mlp_amse_no_rotation。

2026-10-03补充当前pure CAT的尺度边界：53份本目标正式normalized均normalize_weight=True/norm_type=layer；在当前intra_parallel=[1,1]下，每个stage对每个Linear全部未保护、已旋转的权重/残差以FP32求mean/std，使用std.clamp_min(1e-6)，之后训练输入转BF16。下一stage重新统计；不是逐D向量/通道标准化或沿用stage1尺度。日志反归一化残差RMS在恢复尺度并扣除重构之后、BF16 decoder末层融合之前计算，未逆旋转和INT8回填，不能冒充实际部署权重误差。尺度最终融合Wout*=scale、bout=bout*scale+mean，v6没有另一份stage scale；INT8保护scales另计。现有normalize_weight=false同时改变输入幅度及MSE/BSQ相对尺度，不等于关闭内部LayerNorm/BSQ归一化；尚无本目标匹配质量证据，未因此启动新候选。[源码核查和实际配置](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的10:00后续尺度核查。

<a id="output-losses--完整输出拟合预算与码建议覆盖的交互"></a>

## 完整输出拟合预算与码建议覆盖的交互

原专题：[输出损失：先校准数值尺度，保留真实目标](../output_losses.md#完整输出拟合预算与码建议覆盖的交互)。

2026-10-07固定Qwen f927/L18 gate，fresh encoder+decoder完整矩阵WMSE与含保护交叉项的完整Linear输出MSE两臂，保持K32/D32/S2/H128R0/block32/input64 INT8、相同初始化与B8/32文档。将每stage256的线性调度改成fresh1024后，output相对WMSE的部署train/held SSE从约+2.47%/+2.95%变为−.73%/−.09%，但replay近乎持平、整模型KL三域仍正/负/正，NLL三域均较WMSE差。实测证明早期局部排名可随预算改变，也说明局部目标优势不保证最终模型优势；不能由256负例宣称已收敛，或由1024局部反超就集成。未融合FP32→native BF16/group8只使SSE增约.0193%–.0252%，无法解释主要差距。范围仅这一层/两预算/旧dev，未测分项aux梯度，不据此认定辅助项或所有输出目标无效。[真实配对、固定观察与边界](../../history/w25_transfer/2026-10-04_29_research_log.md#10月7日0239八卡接续级数兼容与完整longbench推进)。

2026-10-07同f927/L18 gate mixed128最终KL码建议，覆盖4→16组、保持初态/数据/原条件HF_MLP(xs) decoder500，前4组choices完全一致。改码后pre-D真实mixed128 KL .323281→.309376，六观察域均更低；但同阶段局部MLP train NMSE .194893→.224189，D500后train/held/replay约高14%，完整八项69.519535→69.415959。15:29补齐四格：4组无D为69.307340、16组无D为69.388213，拟合收益分别+.212195／+.027746pp。两者拟合均提高八项，更大覆盖只改码也提高八项；所以不能把结果概括为码位学习有害，或16组D拟合抹去全部收益。局部MLP误差差异在拟合前已经存在，后续恢复收益较小的原因仍待区分。

同批次原生部署评分又确认，D500后mixed128 KL为.323680／.309830，较拟合前仅增加.000400／.000454；16组主要KL优势仍在，Alpaca各域同样更低，但八项排序相反。FineWeb两域KL略高、NLL方向混合，不称跨域统一恶化。须分别检查码位选择、拟合前后完整模型指标及实际部署状态；更低校准KL不保证分类八项或九项更高。末100步loss小幅下降且LR同时趋零，不证明已收敛；未找到同源同局部目标超过500步的匹配完整八项，旧其他初态/权重MSE2000不构成反证。容量、优化、数据适配仍是假设，没有据此宣称加宽或加步数必然有效。完整LB尚未补齐两端点，保留16组无D用于实际全21，不按八项差值直接淘汰。[四格、拟合后KL与部署证据](../../history/w25_transfer/2026-10-04_29_research_log.md#10月7日1529四格码位诊断llama完整九项与宽度配对)。

