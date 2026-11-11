# 表示、结构与容量

[经验分类入口](../../README.md) · 一级分类：压缩经验（`compression`） · 二级专题：`representation`

本篇按问题收录既有经验。原始日期、配置、数值、假设及证据边界保持；历史“当前”仅指原核查时点。跨专题按链接查阅，不能由分类推导通用算法规定。

## 条目导航

- [压缩选择：相同码流位宽不代表相同模型或预算](#compression-choices--压缩选择相同码流位宽不代表相同模型或预算)
- [按条件选择动作](#compression-choices--按条件选择动作)
- [下次怎么做](#compression-choices--下次怎么做)
- [原生stage扩容的累计与冻结契约](#compression-choices--原生stage扩容的累计与冻结契约)
- [向量维度与encoder增宽的完整后缀](#compression-choices--向量维度与encoder增宽的完整后缀)
- [小向量的误差、功能与完整任务收益](#compression-choices--小向量的误差功能与完整任务收益)
- [BSQ球面维数、码长与权重向量长度](#compression-choices--bsq球面维数码长与权重向量长度)
- [同码率结构对照的完整结果与迁移边界](#compression-choices--同码率结构对照的完整结果与迁移边界)
- [问题与证据](#compression-choices--问题与证据)
- [stage扩容的局部收益与完整能力代价](#compression-choices--stage扩容的局部收益与完整能力代价)
- [旧码本与旋转证据的模型和预算边界](#compression-choices--旧码本与旋转证据的模型和预算边界)
- [双侧增宽与固定训练预算](#compression-choices--双侧增宽与固定训练预算)
- [边界](#compression-choices--边界)
- [单stage、增码与残块的配方边界](#compression-choices--单stage增码与残块的配方边界)
- [VAE内部归一化、K38与原生兼容范围](#compression-choices--vae内部归一化k38与原生兼容范围)
- [原生RMS、BSQ幅度与已部署码位相关性的范围](#compression-choices--原生rmsbsq幅度与已部署码位相关性的范围)

<a id="compression-choices--压缩选择相同码流位宽不代表相同模型或预算"></a>

## 压缩选择：相同码流位宽不代表相同模型或预算

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#压缩选择相同码流位宽不代表相同模型或预算)。

<a id="compression-choices--按条件选择动作"></a>

## 按条件选择动作

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#按条件选择动作)。

| 条件／要解决的问题 | 建议动作 | 证据状态与适用边界 |
| --- | --- | --- |
| 不知道先改哪类超参 | 按症状区分拟合、恢复与部署，再选一个变量 | [超参作用与故障诊断](fitting.md#compression-choices--超参作用与故障诊断2026-10-29) |
| 由nominal码率确定候选费用 | 实际拟合并计全部部署组件，分开本类和partial整模口径 | [固定D的码率profile与真实费用](../evaluation/storage.md#compression-choices--固定d的码率profile与真实费用) |
| 动态长度或阶数分配 | 核布局、退出误差下界和完整同cap存储 | [动态阶数的布局、退出下界与同cap对照](bit_allocation.md#compression-choices--动态阶数的布局退出下界与同cap对照) |
| 旋转、D、S或宽度有局部优势 | 先核合法形状及匹配整模证据 | [问题与证据](#compression-choices--问题与证据) |
| 重压、拼接或顺序变化 | 核真实live父源和后继适配，分开替换损失与恢复 | [压缩即时损失、末轮恢复与类别拼接](../evaluation/diagnostics.md#compression-choices--压缩即时损失末轮恢复与类别拼接) |
| 保护、INT4或可学习scale/q | 分开选择轴、scale轴、舍入及真实存储表示 | [保护学习与编码的实际参数语义](protection.md#compression-choices--保护学习与编码的实际参数语义) |
| 稀疏补偿继续加密度或精度 | 固定支持检查选址、幅度和值精度，完整九项取舍 | [稀疏补偿的选址、覆盖与值精度取舍](protection.md#compression-choices--稀疏补偿的选址覆盖与值精度取舍) |
| 向量级有效位宽的新方案 | 查待闭环问题，未完成对照不升级结论 | [向量级位宽分配的待闭环问题](bit_allocation.md#compression-choices--向量级位宽分配的待闭环问题2026-10-29) |

以下按问题查动作；各节保留原核查日期、实测条件、未验证声明及原始来源，历史记录不代表当前运行状态。

<a id="compression-choices--下次怎么做"></a>

## 下次怎么做

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#下次怎么做)。

- 先固定类别、stage、向量维度、保护通道和训练预算；报告payload位宽及包含decoder/dense/norm/head等开销的实际存储。
- 单类压缩加完整模型评测，仍不等于所有类别同时压缩。按类挑出的组合要验证最终全压状态。
- PPL与下游指标冲突时分别报告，事先确定主指标，不事后合成有利分数。
- 旋转后核对坐标、通道保护顺序、矩阵整除及Hadamard支持条件；原坐标AMSE权重不能不经推导直接用于新坐标。
- 码位打包支持某个K，不代表模型支持对应的向量维度D。2026-10-05实际CPU核查Llama gate的D24/K24、input保护96：14336×4000除24余8；按D整数倍调整input/output保护仍不能整除，现有原生路径也不做padding。该候选在保护合法性检查即拒绝，没有GPU试跑或主框架扩展。先核真实剩余矩阵及原生reshape契约，再计算候选位宽；不能只看三字节码或通用kernel可用。[具体尺寸和源码证据](../../history/w25_transfer/2026-10-04_29_research_log.md)。
- 重构NMSE有收益只能作为进一步下游实验的依据，不能省略后者。
- 报告resolved结构，特别核对symmetric对decoder显式深度的覆盖；用最终全压成品、实际存储和独立重载评测确认效果。

<a id="compression-choices--原生stage扩容的累计与冻结契约"></a>

## 原生stage扩容的累计与冻结契约

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#原生stage扩容的累计与冻结契约)。

2026-10-23原生残差stage扩容短验补充：S==1与S>1的packed BF16 decoder在LayerNorm affine、输出matmul/bias和stage累计处存在不同舍入边界。给已恢复S2追加同K/D/H的S3时，残差须在实际S>1累计位置用HF旋转矩阵减逐stage BF16输出的FP32和；不能把末次已舍入的S2总输出或独立S1输出当完全等价对象。新增stage装入真实S3后再验证实际贡献。Qwen四层三MLP短验完成3435保存/fresh strict及原生生成，旧504码库、3351补集和144旧stage叶保持exact；这只验证现有多stage机制的隔离扩容，不证明完整九项收益。新增72个packed decoder键扩展stage维度，冻结合同应比较旧stage切片，不能要求扩形后的整张量哈希不变。实际逻辑2.597262028、含原QK码padding的存储2.610636688，按用户逻辑≤2.6验收、两者分列。[方法、短验及正式运行边界](../../history/w25_transfer/2026-10-04_29_research_log.md#10月23日夜间完整结果与qa蒸馏方向对照)。

<a id="compression-choices--向量维度与encoder增宽的完整后缀"></a>

## 向量维度与encoder增宽的完整后缀

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#向量维度与encoder增宽的完整后缀)。
**向量变长或encoder加宽时，先做真实同父阶段比较，再决定一次完整闭合。** 同新after_o和raw2控制，D64/K64的encoder128→256使完整阶段M多23/14042题、固定train输入output NMSE改善约.3815%；D/K64→128且batch4096→2048使M少456题、output NMSE恶化约34.79%，两臂真实部署及保存重载均正确。前者结合旧D32背景的完整容量正例，支持一次新父完整闭合，不支持当前已是稳定优胜；后者结束本固定配置长链投入，不否定所有长向量。`batch×D`相同只匹配标量曝光，D/K同增还改变BSQ幅度／辅助归约、向量池、初始化及decoder费用，不能作为D的独立因果证据。固定迁移方法为同父／同消费者／有限单因素族及完整读出；按模型重新核矩阵flatten、原生误差、BSQ尺度、实际存储和live输入，D64／128及encoder128／256只是本配置已测范围，跨类别／模型或不同系数尚未验证。[完整结果、预算与边界](../../../../experiments/cat_geometry_cohort_cmcc_20261102/README.md#完整阶段结果与取舍)。

11月3日该D64/K64-E256的唯一完整闭合已结束：沿同新父、相同fixed2消费者与三类各500恢复，own-live up/down各S2×10K、后缀初始化对照通过，完整九项63.00085384629592较C0低.5833121039069198pp，M少75题、LongBench低.7219443350分；logical／packed及tensor费用保持。阶段M+23／局部误差微降没有迁移为完整配方增益，因此本点不采用、不接Muon或延长／宽度网格；旧D32背景encoder256九项+.580037pp仍保留其条件。该单seed正反例支持先用局部阶段有界筛选、再以自身完整后缀取舍，不支持“encoder越宽越好／全部无效”或将不同父的增量相加。固定方法是完整同父比较；按模型／结构重算live目标、布局、初始化与费用，encoder128／256仍只是已测范围。[完整终态与来源](../../../../experiments/cat_e256_full_cmcc_20261102/README.md)及[核证与收尾](../../../../result/qwen3_8b_mmlu65_20261004/cat_e256_full_cmcc_20261102/closure_20261103.json)保留完整指标与边界。

<a id="compression-choices--小向量的误差功能与完整任务收益"></a>

## 小向量的误差、功能与完整任务收益

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#小向量的误差功能与完整任务收益)。
**小向量的权重误差、功能误差和任务收益可能分离。** 同live after-up、原BF16 AdamW／S2各10K／保护旋转、匹配B×D的D32→D16→D8三点中，完整raw八项为65.9771825912／67.0125927269／67.0198762689。D8相对D16只高.0072835420pp，但M多141／14042题、ARC-C高1.02389078pp，另有四项下降；不能称稳定赢家或全面被支配。相同train8×256输入下weight NMSE下降7.574392%，output总NMSE却升9.836607%；34／36层output改善而一个坏层贡献超过总净增，其它层部分抵消。该分解定位了误差集中，未证明这些层导致具体任务损失，不直接提出固定层修补或逐层donor选择。正式含fresh1584.32秒比D16约多60%，完整阶段2166.44秒，均非纯kernel计时。按原卡结束本次阶段、不采用／不接500或九项、不再缩D；仅保留有明确取舍的候选到三份既定九项齐后的一次审定，无具体后缀则清权重。迁移固定同父有界比较及完整质量／费用取舍，按模型重算D/K/B、布局／BSQ尺度／初始化／padding和live统计；本三点单seed／raw证据不外推跨类别、末态九项或越小越好。[完整配对与误差／费用](../../../../experiments/down_d8_neighbor_cmcc_20261103/README.md)保留原件。

11月4日三份原九项齐后的预定审定已结束：D16原Adam后缀九项64.27858659314296，对C0+.6944206429401234pp、M多124题、LB高1.7158051661分，保留为完整几何候选及控制；D32 VAE-Muon九项仅高它.0266818675pp而M／LB有代价，不据微差选通用最优。D8本轮新增整链的决策价值不足，结束唯一raw保留用途并清理生成权重，完整阶段正负事实与费用保留；这是本批预算取舍，不是所有D8／恢复无效的结论。D16原完整预测在父超时后自然结束，已用原评分器CPU闭合，技术修复不改变质量，见[完整九项及原失败边界](../../../../result/qwen3_8b_mmlu65_20261004/down_d16_terminal_recovery_cmcc_20261104/README.md)。

<a id="compression-choices--bsq球面维数码长与权重向量长度"></a>

## BSQ球面维数、码长与权重向量长度

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#bsq球面维数码长与权重向量长度)。
**区分BSQ球面维数、码长和权重向量长度。** [BSQ论文§4.1及附录C](https://arxiv.org/html/2406.07548v1)的球面维数L对应本项目每阶码长K；它定义`2^K`个隐式二进制码，而D是待重构的权重向量长度。固定K增加D不增加码的位数；D/K一起增加则同时改变表示、尺度和预算。球面量化的误差界只作用于归一化latent，不能直接界定经可学习decoder后的权重误差，更不能界定九项能力。论文的视觉码长收益不证明本项目“长向量更好”。[作者源码固定版本](https://github.com/zhaoyue-zephyrus/npq-vit/blob/8523184c151e842247556abbeab9e9989b7c4354/transcoder/models/quantizer/bsq.py)还区分球面幅度及分组／解析熵；当前[项目BSQ实现](../../../../litebsq/bsq.py)使用逐bit边际熵与自身commitment归约，不能照搬上游系数或将条件分布可分解等同于聚合联合熵精确可加。具体调节仍按[损失与尺度](../output_losses.md)的本地公式与实测边界。

<a id="当前先闭合同码率结构对照"></a>

<a id="compression-choices--同码率结构对照的完整结果与迁移边界"></a>

## 同码率结构对照的完整结果与迁移边界

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#同码率结构对照的完整结果与迁移边界)。

**遇到局部重构优胜但整模证据缺失，先查同来源的完整配方结果；本组已于10月30日闭合，10月31日复核原始验收。** A＝D32/K32、B＝D64/K64只改变完整36gate的同主码率结构：共同after_o、S2、asymmetric E128/D128/R0、原CAT M36／每级10K，gate向量batch8192／4096匹配`batch×D=262144`；均经gate500恢复，再压缩本支live up/down并各恢复500。它同时包含K相关BSQ尺度、向量池和后继轨迹变化，应称同码率结构配方比较，不能独立归因于D。

原始[A验收](../../../../result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/evaluation/A/transfer_acceptance.json)与[B验收](../../../../result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/evaluation/B/transfer_acceptance.json)给出九项63.52216414161746／63.78399082894941；logical 2.497367527173913／2.498731530230978，packed 2.5107421875／2.5121061905570654。每臂均由同一最终UUID完成64叶／24742及LB21／4750，质量均未达。B九项高0.26182668733195pp，但LB21低1.69328969414447pp，RTE、OBQA也回退；据预设九项选择B为本批结构，不能写所有能力改善。logical计主码、decoder、保护值／scale／索引及旋转，packed另含padding，非投影与序列化仍另计。[完整条件、成本与分项](../../../../experiments/overnight_goal_20261029/README.md#10月30日真实终态与完整结果)。

原C的独立完整链于2026-11-02闭合：同旧after_o、gate D32/K32/S2、只将asymmetric encoder128→256且decoder仍128，保留原M36／每stage10K及各类500恢复，后继拟合自身live up/down。完整九项64.10220094639317，较A高0.58003680477571pp；MMLU高0.277738213929636pp而LongBench21低1.41554875135782pp。encoder不部署，logical／packed仍为2.497367527173913／2.5107421875bpw；正式训练27067.58秒，九项较BF16仍低3.9449439715153103pp，质量未达。该结果支持保留C作为本条件下有净收益的结构配方，不再用三层probe负误差将其关闭；不同宽度改变初始化和后继RNG消费，不能把完整收益单独归因于encoder容量机制。C作为该旧父结构批次综合最高的阶段成品保留；新父／恢复适配仍须独立验证。[真实完整九项、成本与保留范围](../../../../experiments/cat_structure_C_cmcc_20261101/README.md#完整九项结果与保留用途)。

原D的完整链于11月3日核闭：同旧父、gate D32/K32/S2仅decoder128→256、encoder仍128，完整九项63.72663321687882，较A+.2044690753pp，却低于B／C .0573576121／.3755677295pp；logical／packed2.4987899117／2.5121645720，含实际decoder增量。D比C的M高.98988748pp、LB高.36869856分，不能称被C全面支配或decoder增宽无效。按本批完整综合目标／成本结束该D256固定配置追加，不取代C作为默认结构，也不把D的M、C的其它任务拼成模型。固定方法仍是同父、只动一侧、真实位宽及完整后继；按新模型重算decoder开销和live矩阵，128／256只是已测点，单seed不证明容量上限或跨模型最优。[实际协议、九项和取舍](../../../../experiments/cat_structure_D_cmcc_20261102/README.md)。

**其它类别的同码率几何，先做自身live重拟合并保留任务分解。** 11月3日Qwen同C0 after-up、全36down的D32/K48/B8192→D16/K24/B16384，固定S2各10K／E128-D128／input256保护／block32旋转及原前缀，恢复前八项实测提高1.035410pp；权重与固定train输入输出NMSE分别下降19.7330%／12.8205%，含全部部署组件的logical还少.0008518385bpw。正差主要来自RTE和BoolQ，MMLU仅多47题，ARC-C下降1.791809pp；36层权重NMSE均改善，输出却有一层变差。这支持当前down几何配方的阶段正例，仍不能由局部误差或八项推出最终九项，也不构成数值异常、按层修补或拼接的依据。保持事前指定的自身500恢复／fresh／完整九项，闭合后再决定采用，不为阶段正差追加网格。[同父完整八项、部署读出和在途边界](../../../../experiments/down_geometry_baseline_cmcc_20261103/README.md)。

该down经验中，**固定方法**是同真实父、重新拟合live矩阵、实际部署读出、相同恢复及完整九项取舍；**按模型重算**合法布局、向量池／输入统计、BSQ尺度和完整费用；**有限候选**目前仅D/K＝16/24与32/48、transpose另作一个独立对照。batch×D只匹配标量曝光，D/K/B及参数shape共同变化，不作纯D因果或跨模型最优推荐。该原父和八项均已在研发中观察，不称盲测；当前尚无D16末态九项或跨系列质量证据。

同一after-up父下，只将D32/K48的down向量方向transpose=True→False，完整raw八项小增.112438pp，但MMLU少21题、四项升四项降；weight NMSE仅下降.029488%，固定train输入output NMSE却增加19.3670%，部署费用不变。这修订“W重构近似相同即可认为方向等价”的推断：方向比较应保留真实输入下的功能读出和任务分解；局部误差反例不证明最终恢复必然劣势。按事前方向仅到阶段读出的预算结束此点，不追加完整链或转而按层拼接；D16完整链继续。有限方向范围只在D32/K48覆盖True／False，未测D16/False，也未覆盖其它模型。[同组方向结果与结束范围](../../../../experiments/down_geometry_baseline_cmcc_20261103/README.md)。

**可迁移的是比较方法，尚不是最优配置。** 固定同源父、主码率、保护、有效曝光及恢复规则，按新模型实际矩阵重新核合法D/K、向量数、decoder和保护开销，并分报局部误差、完整九项与实际存储。完整有限范围覆盖Qwen gate D/K两点，以及D32/K32下分别encoder128／256和decoder128／256；新D64-E256后缀已完成、九项尚在评测，不能沿旧父增益直接推荐新父。缺少同协议共同after_o父的完整九项，不能补造A/B/C/D相对父的纯压缩损失或恢复增量；单seed排序也不是跨系列结论。三层probe保留其筛选身份，[原件](../../../../experiments/cat_structure_probe_20261029/README.md)中的未完成描述不覆盖本次完整结果。

<a id="compression-choices--问题与证据"></a>

## 问题与证据

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#问题与证据)。

<a id="compression-choices--stage扩容的局部收益与完整能力代价"></a>

## stage扩容的局部收益与完整能力代价

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#stage扩容的局部收益与完整能力代价)。

2026-10-24完整实测补充：Qwen已恢复B32的L0/8/9/18三类MLP追加原生S3，选中非保护权重的局部MSE下降约67.27%–67.87%，MMLU净增143题、LB21宏均增加0.892766418分，完整九项却下降1.389609458分。新增容量改善局部HF拟合、甚至同时改善MMLU和LB，仍不保证综合收益，必须保留原完整任务组合验收；这是该固定配方的完整同口径证据，不识别HF目标、层范围或恢复偏移各自的因果，也不外推为所有stage扩容无效。随后同源只保留gate新增输出的条件消融，八项较B32+.080303分而LB21−.497006分，九项仅+.016158分；消除明显的短题回退仍不足解决综合缺口，不能按类别八项净收益相加预测联合模型。[来源、完整终态与适用边界](../../history/w25_transfer/2026-10-04_29_research_log.md#10月23日夜间完整结果与qa蒸馏方向对照)。

<a id="compression-choices--旧码本与旋转证据的模型和预算边界"></a>

## 旧码本与旋转证据的模型和预算边界

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#旧码本与旋转证据的模型和预算边界)。

旧单类别压缩A/B摘要记录：同约2bit/weight的32/32/2-stage与64/32/1-stage在不同类别表现不同，down_proj偏向后者，多个其他类别偏向前者，o_proj的PPL和任务均值还存在冲突。2026-10-31核查，原件`docs/experience/records/.result/catlora_codebook_ab/cat_codebook_ab_per_category.md`当前缺失；旧三元组轴顺序未核，保留上述历史数字和条件，不按D/K/S自行解释，不用于参数敏感度归因，也不作为已重新核证的推荐。

局部双侧权重旋转最小测试显示某些NMSE改善，但32维分块没有稳定胜过单侧；去掉保护通道后4096变4064，完整Hadamard不再自动可用。[旋转实现与验证](../../records/docs/exp_results/weight_rotation_preprocess.md)

2026-09-30 Qwen3-8B全252投影、零LoRA、无保护、full局部双侧旋转的K32/D32×2纯重建，实际2.013135827bpw，完整八任务均分仅39.19092797%（同机原模型70.24712234%）。21个真实投影的分组/单独BF16解码及原生前向对照逐元素一致，NMSE约0.13；在所查范围未发现缓存/布局差异，重建路径验证通过不代表质量达标。[本轮原始指标与诊断索引](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)

旧纯VAE54.51%存在保护与旋转等条件差异：邻近运行Git源码显示原精度保护默认32/down128通道、无局部旋转，decoder实际同为symmetric128/0。旧原始checkpoint/日志缺失，不能把不同轮次的差值当作保护或旋转的独立因果效果；也不能按脚本中无效的decoder_num_res_blocks=1推断旧结构更深。

<a id="compression-choices--双侧增宽与固定训练预算"></a>

## 双侧增宽与固定训练预算

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#双侧增宽与固定训练预算)。

保持该FP32无旋转/64-256保护/码率/每stage10000步配方，仅symmetric base_ch128→256、仍0残块，实际2.485818614bpw、完整八任务59.7416829%，较H128下降1.437559个百分点。增宽同时影响训练encoder与部署decoder，固定训练预算下没有体现纯重构质量收益；不能据参数更多推断更高精度，也不能外推为其在后续模型级蒸馏中的容量上限。该比较使用修复前相同部署kernel，修核后的比较须重建同核基线。来源为同一Qwen3专题的`reconstruction_no_rotation_o3_down3_h256_fp32/acceptance.json`。

<a id="compression-choices--边界"></a>

## 边界

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#边界)。

码本结论来自旧配置下的14个单类job，旧单stage总10000步、双stage总20000步；不能迁移为所有模型/数据的默认最优配方。旧旋转记录为有限种子及预算的重构验证；新增全压39.19%只证明该无保护配方不足，不构成旋转方法的普遍否定，也不证明零LoRA目标不可达。

<a id="compression-choices--单stage增码与残块的配方边界"></a>

## 单stage、增码与残块的配方边界

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#单stage增码与残块的配方边界)。

2026-10-02对上述down单stage倾向作新条件验证：以7f19的统一block64、o/down D16、INT8保护64/down256纯CAT为基底，仅36个down从K24/D16/S2各10K换成K48/D16/S1共20K，其余216投影逐字节保留。实际位宽2.455285114→2.455082106bpw，完整同64/32八项均分63.69939178→61.95749320%，八项均下降。新down最终原坐标RMS0.009172752也高于原0.006425134，但不据此建立重建误差与下游的普遍单调关系。两者总更新数相同，单stage潜在几何、优化器阶段重置及独立类别随机流不同，不是等FLOPs比较。本配方不再投入；旧单类、D32、不同训练预算的倾向不能直接迁移到当前全压模型，也不能由此否定所有S1参数配置。[完整对照、保存审计与原始指标](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的 `reconstruction_d16_odown_allblock64_down_s1/`。

2026-10-02在7f19纯CAT基础上，只新鲜训练36个q为K40/D32/S2/H128/block64并替换，其余216严格继承，实际2.498253864bpw，完整同64/32均分63.21563404%，比原63.69939178%低0.483758个百分点；RTE净少10题、MMLU仅多2题。增加某类码率仍需整模型下游验收；K变化也改变encoder/BSQ及随机初始化，不是单纯给既有编码追加位。该配方退役，不能外推为所有q增精度无效。[完整结果与保存审计](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的 `reconstruction_d16_odown_allblock64_q25/`。

同轮H256/R0、统一block64、o/down D16而其他D32的fresh七类纯CAT完整均分62.91584305%/2.464222784bpw，仍低于7f19的H128成品63.69939178%；Winogrande提高而MMLU下降。该结果与早先无旋转FP32增宽未获益方向一致，但几何、精度和类别随机流不同，不合并为通用因果结论。深度R1是不同变量：现有wholefused明确不支持R1，须走已支持native packed残块；增加残块的验证和预算不能用R0 fused测试替代。K24/D16/H128/R1/BF16数值、真实VAE核心更新及v6持久化通过后，完整fresh七类R1于22:26验收 **63.33671491% / 2.474752675bpw**，比7f19低0.362677个百分点；MMLU下降2.891326个百分点、RTE相同，其他任务有升有降。本R1纯重建配方没有更新最佳均分，决定退役；symmetric同时加深训练encoder及部署decoder，且基线为分组拼装，不把差值孤立归因于decoder深度，也不外推为其他蒸馏条件下的容量结论。[实际结果与边界](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)。

<a id="compression-choices--vae内部归一化k38与原生兼容范围"></a>

## VAE内部归一化、K38与原生兼容范围

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#vae内部归一化k38与原生兼容范围)。

2026-10-07源码与原生小CPU对象核实，VAE的R0仍有encoder/decoder各自norm_out及activation；norm/activation并非仅残差块参数。当前Llama50份CAT实际配置全layer+swish，模型级norm恢复结果不能替代VAE norm比较。原生RMS和GELU规格可保存重建；group8会退出whole-fused并走已有逐模块packed路径，叶数须从实际spec/state派生（RMS5叶、LayerNorm6叶），旧实验审计的固定6叶不是核心算法限制。小CPU通过尚不证明CUDA或下游收益；group/batch在M2 CPU并行非连续view失败，暂不直接用于正式试验。[源码、短验边界与未完成事项](../../history/w25_transfer/2026-10-04_29_research_log.md#0813r0归一化实际作用与现有rms参数短验设计)。

2026-10-07 Qwen同fresh HF Q/K、full-RHT、input64 INT8、D32/S2/H128R0/每stage10K的匹配控制中，K32→38使完整八项68.247315→68.691897%、M8370→8557；其余180投影保留f927。实际逻辑2.497367527、物理2.510742188，144个K38 bank高2位padding为零。该容量增配有匹配八项正收益，但仍低于原已恢复f927；形状与随机数消费同时改变，不能把不同压缩历史的差值归为纯码率效应。完整LongBench尚待评，不能从八项或局部检索代替九项结论。[原始计数、存储及后续全21计划](../../history/w25_transfer/2026-10-04_29_research_log.md#0824k38完整八项与八卡接续)。

同日RMS原生8B短验已补齐：真实32gate/B8192/每stage2步均更新，2947保存重载一致，224模块group8 BF16预热成功；L0两stage packed对原生dense最大绝对差1.220703125e-4、4token模型前向有限。确认这套RMS参数与当前v6/packed路径兼容，但2步改变线性LR时域，仅证明功能；同条件fresh10K×2正式质量尚待评。[短验与正式运行](../../history/w25_transfer/2026-10-04_29_research_log.md#0834rms真实短验完成并启动正式对照)。

<a id="compression-choices--原生rmsbsq幅度与已部署码位相关性的范围"></a>

## 原生RMS、BSQ幅度与已部署码位相关性的范围

原专题：[压缩选择：相同码流位宽不代表相同模型或预算](../compression_choices.md#原生rmsbsq幅度与已部署码位相关性的范围)。

2026-10-07，同Llama gate K34/转置True/10K×2条件，仅encoder/decoder LayerNorm→原生RMS，完整八项61.44471128%比匹配控制低.37489733pp，MMLU+14题；逻辑2.48255099bpw。功能与保存正确不等于精度提升；固定端点即LB100九项上界65.72863225仍低66.15709190，可按真实九项上界退役该端点，不能将八项0.8pp当额外门槛或否定所有RMS。

同日32层gateFalse/K32的原生BSQ前128更新，仅new_quant True→False，使commitment/recon encoder梯度范数比从1–2%升至30–84%，并出现少量单批恒定位，128当前批recon未改善。±1与±1/√K同时改变decoder输入/梯度及commitment常数，不能按total loss直接排序；证据不足以升级False为正式质量候选。此诊断没有覆盖后stage、完整收敛或packed部署，保留True不代表False在所有系数下无效。

独立CPU抽样98a8/f927部署gate硬码（64/72 banks，每bank4096行，排除padding）未见恒定位或|corr|>.9位对；有限二阶统计不支持严重重复码解释。协方差PR不是有效bits/熵/表达能力，两模型不同几何和训练历史不可作因果对照，也不解释所有E2E STE失败。无证据时不因一般性“去相关”设想扩大主框架。[原始结果、配置和独审](../../history/w25_transfer/2026-10-04_29_research_log.md#10月7日0909llama完整longbenchrms结果与八卡衔接)。

