# 数值精度、梯度与舍入路径

[经验分类入口](../../README.md) · 一级分类：数值与运行性能经验（`runtime`） · 二级专题：`precision`

本篇按问题收录既有经验。原始日期、配置、数值、假设及证据边界保持；历史“当前”仅指原核查时点。跨专题按链接查阅，不能由分类推导通用算法规定。

## 条目导航

- [Softmax诊断须保留缓存目标与真实前向语义（2026-10-29）](#kd-and-data--softmax诊断须保留缓存目标与真实前向语义2026-10-29)
- [教师RoPE数值路径与匹配控制](#liftquant-recovery--教师rope数值路径与匹配控制)
- [固定κ复用与梯度数值身份](#memory-precision--固定κ复用与梯度数值身份)
- [GPU身份表示与effective dtype](#memory-precision--gpu身份表示与effective-dtype)
- [RoPE原始精度与norm部署跨格预算](#memory-precision--rope原始精度与norm部署跨格预算)
- [单模块精度与历史mid-eval证据边界](#memory-precision--单模块精度与历史mid-eval证据边界)
- [实际激活dtype、master存储与部署缓存](#memory-precision--实际激活dtypemaster存储与部署缓存)
- [pure CAT全FP32与独立down完整结果](#memory-precision--pure-cat全fp32与独立down完整结果)
- [kernel数值检查的dtype和形状范围](#memory-precision--kernel数值检查的dtype和形状范围)
- [同BF16计算下的FP32叶参数对照](#memory-precision--同bf16计算下的fp32叶参数对照)

<a id="kd-and-data--softmax诊断须保留缓存目标与真实前向语义2026-10-29"></a>

## Softmax诊断须保留缓存目标与真实前向语义（2026-10-29）

原专题：[KD 与数据边界：同名参数可能已有不同数学语义](../kd_and_data.md#softmax诊断须保留缓存目标与真实前向语义2026-10-29)。

**触发条件是实际数值路径出现高margin或近教师梯度异常，不是恢复分数停滞本身。** 2026-10-31核对[A Fix for Softmax and Cross-Entropy v3](https://arxiv.org/html/2610.31642v3)：Collapse是高置信主类的`q_t−1`归零而尾部梯度仍非零；Cancellation是近师生分布相减的消减误差。前者修log-softmax／硬标签CE，后者的稳定梯度需要师生raw logits差；仅替换log-softmax不能据此声称两者都修复。论文证据来自小型模运算／grokking及同权初始化蒸馏，没有本项目8B恢复或九项收益证明。

本项目M／QA的BF16 head logits先`.float()`再log-softmax，故应核FP32公式的约16.6 margin条件，不能拿BF16约5.5直接判失效；margin是最大logit减其余类log-sum-exp，不是top1−top2。当前`kl_div(log_softmax(z), cached_logp, log_target=True)`的精确logit梯度是`a*q-p`，其中`p=exp(cached_logp)`、`a=sum(p)`。把缓存logp当raw teacher logits会归一化成`p/a`，改变原目标。M已将教师与gold混合一次；另加硬标签CE或重混目标也不属于纯数值修正。[实际调用链与精度](../../../../experiments/softmax_numerics_20261029/README.md#实际调用链与精度)保存了支持集、causal位置和归约依据。

G02 BoolQ train前4文档／8分支的已完成审计中，FP32公式logit梯度误差L2为6.22e-9，同缓存FP64参照为5.54e-4；跨BF16 logits反传边界后的差为3.49e-10。缓存8/8重播一致，raw恒等Eq.9梯度为零，最大margin1.417。这里真实serial／left-pad路径差远大于公式误差；正常路径差不自动构成缓存bug，也未测参数Jacobian或Adam影响。

原经验仅覆盖R的边界已由[后续真实G09 M32/live结果](../../../../experiments/softmax_numerics_20261029/README.md#g09初态group0-m32真实live结果)补齐：同缓存FP64参照下，公式误差／梯度范数为`1.4738901107e-7`，学生margin最大6.31864，128个梯度元素跨同一BF16边界后全部相同，无零项或非零错号。按预定停止条件未执行参数VJP，也没有优化更新；不把它写成完整M＋QA、参数更新或九项等价。R与该M窗口均不支持将公式误差认定为恢复不足主因，QA等未测路径保持未知。本轮核对实验正文及源码；其旧机raw数值目录未迁入当前artifacts，不声称重新复算张量。

诊断动作固定原缓存、温度、mask、支持集及归约，以同目标FP64作参照，分列绝对误差、真实梯度尺度、缓存质量偏差及BF16边界存活差量；数值精度与优化超参不能同时改。`1e-3`曾是该窗口的研究资源阈值，非通用误差容限。[作者现行代码与勘误](https://jonbarron.info/softmaxfix/)还修正了早期`softplus(log(s))`在仅一个有效logit、`s=0`时的NaN反向，采用`log1p`路径；有限值窗口通过不覆盖该mask极值，勘误也不撤销已有R／M结果。迁移须核实际loss精度及缓存语义，不能全局替换softmax、attention或照搬旧v1实现来宣称保持目标不变；本节不新增诊断运行或修复训练。

<a id="liftquant-recovery--教师rope数值路径与匹配控制"></a>

## 教师RoPE数值路径与匹配控制

原专题：[编码恢复：先验证阈值可达性和真实数值路径](../liftquant_recovery.md#教师rope数值路径与匹配控制)。

2026-10-04准备另一来源的全码位匹配对照时，源码核查进一步定位到教师加载差异：当前`runtime_v6._load_teacher`在layer_mp分支无条件`teacher.to(device="cpu", dtype=...)`，会把原生FP32 RoPE buffer一并转为BF16；DP且teacher_model_offload=none的分支只在参数dtype不符时转换，当前Qwen控制因此保留FP32 RoPE。独立13文件包只修这个冗余cast后，真实两模型×DP/MP/修复MP六臂验证了差异：原MP均有63/64频率舍入，固定B4×1024实际有效因果位上teacher full-KL约0.000912/0.001031；修复MP保留原频率，统一单GPU的整个logits与DP逐位一致。源码仍未改入活动生产依赖，实际导入指向隔离副本，数据和库仍用原路径；未用事后buffer重建。旧全码四卡与单卡控制不能仅归因于覆盖率，但这项加载验证不证明旧质量回退由它造成，也不代替真正多卡训练比较或使旧实测失效。[最小修复、真实六臂及后续边界](../../history/w25_transfer/2026-10-04_29_research_log.md#教师rope加载精度的隔离修复)。

<a id="memory-precision--固定κ复用与梯度数值身份"></a>

## 固定κ复用与梯度数值身份

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#固定κ复用与梯度数值身份)。

GPU正常浮点差异不能自动扩成梯度逐位复现要求；参数初态、packed bits、保存态／RNG和exact-resume仍保留各自严格契约。2026-11-09新版SC-CAT的P倍率轮仅改变保护学习率，LoRA Muon却在首次更新前因梯度SHA不同被挡。真实同配置两次复现及半倍率对照均匹配初始参数SHA和实际输入；梯度相对L2约1.46%，但用于κ的AdamW／Muon候选更新范数相对原控制最大差约0.00219%／0.05754%，κ比值最大差0.05969%。不能把不同SHA等同不同初始化，也不能因此声称梯度逐值相等。

针对该相同Qwen3 PL基底，固定原κ不变，新增raw／类别／数据／随机种子／非倍率设置身份，参数SHA及实际分组继续精确比较；重算首次裁剪梯度的候选更新范数仅作核验，范数及比值相对误差≤0.2%才复用，超界仍停止。0.2%界来自本次重复实测，约为最大比值漂移3.35倍且小于BF16单位舍入误差；不外推为其它模型、分组、损失或普通梯度误差的通用容限。CPU拒绝真实参数、上下文与候选范数变化；实际P倍率半／LoRA LR半各两次GPU更新通过，原κ精确保留。新首类搜索尚未完成，不能把技术修复当质量结论。[首因、三次梯度与实际更新证据](../../records/sc_cat/qwen--qwen3-8b/2026-11-08_172625_s31_results.md#11月9日固定κ复用检查修复)。

<a id="memory-precision--gpu身份表示与effective-dtype"></a>

## GPU身份表示与effective dtype

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#gpu身份表示与effective-dtype)。

2026-10-21新增18D入口在模型加载前因GPU身份检查误拒：本机PyTorch2.6的`_CUuuid.__str__`返回裸UUID，nvidia-smi返回带`GPU-`前缀的同一值。先实读两边表示，修复仅补齐固定前缀后严格比较，保留卡身份约束；不能把表示差异当资源变化，也不应通过删检查或换卡绕过。此处零模型前向／更新，修复后CPU来源绑定不变，GPU训练与导出性质仍须由原定实际短验确认。[失败、最小修复与后续运行](../../history/w25_transfer/2026-10-04_29_research_log.md#10月21日较小码更新终态freshcat末点与最后类别中点)。

2026-10-20四格原生读出在两片教师缓存完成后、学生0次前向时退出：新guard要求全部VAELinear的`_decoder_compute_dtype`显式等于BF16，而原生loader默认传入None，表示沿用实际输入／目标dtype。真实C1 CPU strict加载确认252模块均为None，3423源状态exact、参数仍BF16；不能将可选override字段等同实际计算精度，也不应强设override或整模cast来迁就错误检查。独立修复核effective dtype、原BF16缓存和每次真实输入，同时保留eval、冻结、no_grad、无checkpoint／KV及状态恢复条件；GPU验证范围与终态见[本次记录](../../history/w25_transfer/2026-10-04_29_research_log.md#10月20日夜间配对终态与完整更新的有限干预)。本证据定位的是诊断检查错误，不是训练精度退化或下游收益。

上述10月20日修复随后在GPU3通过四格各一个完整原group、共64学生前向，含最长32561-token prompt；252模块实际BF16输入和cache命中、完整源状态与缓存恢复、释放均通过，0新增teacher前向。原失败已完成且身份完整绑定的teacher缓存可以复用；不因诊断guard修复重算，也不据这个有限验证声称完整四格诊断或下游提升。

<a id="memory-precision--rope原始精度与norm部署跨格预算"></a>

## RoPE原始精度与norm部署跨格预算

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#rope原始精度与norm部署跨格预算)。

2026-10-15 CAT长文本迁移复核：`get_model(torch_dtype='auto')`原本保留HF的FP32 RoPE `inv_freq`，`base_reference`随后无条件`model.to(dtype=BF16)`却会舍入该buffer；student精度配置只转Parameter／BatchNorm，并不能证明teacher也保持相同路径。已有真实Qwen B4×1024对照中63/64频率改变，最大差.00115472，1691有效位置top1变化13个、teacher massTop100 KL约.00091798；既有修法跳过已同参数dtype的冗余整模cast后，与native DP logits逐位一致。forward再`.float()`不能恢复已舍入频率，旧CAT也做过该cast不是保留它的依据。新8K双CPU卸载CAT已复用该修法；真实8B CPU加载→GPU→CPU验证12.694秒／exit0，全参数BF16且冻结eval，`inv_freq`始终与原生配置FP32值精确一致。同步处理空模型构造的冗余cast，400个广播条目的dtype/layout与真实loader一致；空模型仅检查CPU/meta构造，未另启GPU驻留训练或重复logits对照。完整七类资源短验仍是修前教师，应保留版本边界，不能倒填为修后运行。此证据支持教师数值路径修正，不证明下游收益。[原始对照](../../../../result/qwen3_8b_mmlu65_20261004/teacher_rope_precision/comparison.json)、[本次运行范围及迁移证据](../../history/w25_transfer/2026-10-04_29_research_log.md#10月15日下午先准备广覆盖校准与蒸馏数据)。

2026-10-14对e3的145个norm做CPU实际值核查：fresh Adam(.9,.95)、LR1e-5及500步线性日程的前32步，精确算术位移上界约.00032010；308224个BF16源值中仅12.21%的最近半格点在该预算内，final norm只有2/4096。上界不是CUDA严格证书，可达也不等于真实更新；这说明拟定预算可能无法充分检验部署norm效果，不是norm无效的证据。另核自定义Qwen3RMSNorm在现有精度包装下使用FP32 gain乘BF16激活再转输出，与先舍入gain的BF16部署不同；不能只查输出dtype就声称两路径相同。后续若测试norm，需同时明确实际参数跨格预算、原生端点及计算语义，不能把训练loss或参数requires_grad当作有效部署更新。[实际计数、预算假设与源码边界](../../history/w25_transfer/2026-10-04_29_research_log.md#10月14日13时norm实际预算与任务梯度算法对照)。

<a id="memory-precision--单模块精度与历史mid-eval证据边界"></a>

## 单模块精度与历史mid-eval证据边界

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#单模块精度与历史mid-eval证据边界)。

单模块FP32参数存储对照中，两组仍为BF16计算；代表形状的allocated仅增加0.33MiB，reserved增加24MiB。该值不能推算整模型峰值，也不能据实现通过宣称精度提升。[精度记录](../../records/docs/distill_fp32_validation.md)

历史mid-eval OOM计划指出两条风险：no-cache推理仍物化dense解码scratch；直接unwrap模型时连AMP/autocast一起剥离。[排查方案](../../history/plans/2026-09-07-e2e-mid-eval-oom-memory-lifecycle-fix-cn.md)。这份来源是计划，不能由其标题认定所有修复当前已验收。

<a id="memory-precision--实际激活dtypemaster存储与部署缓存"></a>

## 实际激活dtype、master存储与部署缓存

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#实际激活dtypemaster存储与部署缓存)。

- FP32 decoder 存储与 BF16 激活并存时，推理缓存须显式使用实际激活 dtype；按参数 dtype 预热既可能浪费资源，也可能生成前向不能复用的缓存。2026-10-01 Qwen3 K48/H128 的 FP32 分组预热请求229376字节共享内存，超过A800的166912字节；指定BF16后252层通过，FP32参数字节不变。比较缓存正确性时保持原cache策略：关闭cache会优先走packed_u8，正常冷cache走fullfused，这两种计算路径不能混称同一native对照。本次同cache=True的分组/单层解码及前向完全一致，证据见[Qwen3专题](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的 `validation/fp32_eval_cache_v3.json`。

- 2026-10-12原生八步D/P配对进一步证实，decoder输出BF16不代表FP32 master与全BF16存储的解码W逐位相同。两端局部实物中，仅将packed首层bias先舍入BF16再升FP32（保持接口同dtype）就精确复现全部六叶转BF16后的W；相对L2约.00267，保护列完全不变，原BF16 source升FP32则无差异。这是本路径首层FP32 bias累加造成的真实舍入影响，不是保存重载丢状态。保持同存储精度的strict state检查；跨存储精度记录差异并用实际部署成品读出，不新增W hash相等门槛、不为迎合旧hash改kernel。局部W差异本身不能预测任务增减。[实物诊断](../../../../result/qwen3_8b_mmlu65_20261004/qwen_native_protected_trajectory_full9_20261012/deployment_diagnosis/report.json)、[后续完整评测记录](../../history/w25_transfer/2026-10-04_29_research_log.md#10月12日1115单域完整九项与保护整数scale协同)。
- 2026-10-05同一真实QCE7200原生状态的独立完整八任务对照，仅保留1512个decoder为FP32、其余参数BF16/原生FP32 RoPE，实际位宽2.4569887908→2.4676619820（增加9,266,688字节）。正常BF16 group8预热、3423持久状态重载和4token输出均通过；八项68.05382394→68.07310769（+.01928375分），MMLU8071→8080题。部署核仍使用原混合计算路径，多数decoder参数参与运算时投BF16，首层bias保留FP32累加，不能称为全FP32推理。微小收益不支持把该点主要掉分归因于导出舍入，也不否定其他端点的精度影响；不得用已舍入BF16再升FP32冒充真实原生信息保留。[同一步导出、位宽及完整评测](../../history/w25_transfer/2026-10-04_29_research_log.md)。
- 同一真实7200 FP32状态进一步只切换既有packed BF16缓存物化路径，八项68.07310769→68.13458412（+.06147643分），MMLU8080→8058题。训练与缓存的同4token logits完全一致，两次完整评测均复用252个缓存，全部持久state及2.467661982bpw保持；不同于正常wholefused路径约.01054的训练/部署logits相对L2差。消除该数值路径差异没有带来本点MMLU恢复，不应由路径更接近训练自动推断质量更好，也不外推f927或Llama。实现诊断时注意VAELinear的generic `hit`同时统计packed及wholefused成功；本并行stage拓扑的实际packed252对应hit252，不能要求hit0。来源同上。

<a id="memory-precision--pure-cat全fp32与独立down完整结果"></a>

## pure CAT全FP32与独立down完整结果

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#pure-cat全fp32与独立down完整结果)。

- 2026-10-03核查当前pure CAT：`--bf16 true`同时选择BF16 VAE参数和autocast，普通AdamW的两个动量以`zeros_like(p)`初始化，没有FP32主参数副本；`--bf16 false --fp16 false`同时改变参数、梯度、动量、输入和主要计算精度，不能称为仅改变参数存储。BSQ内部部分计算原本就为FP32。现有BF16日志没有实际更新幅度或舍入比例，因此“BF16更新舍入限制精度”仍是假设。K24/D32的FP32存储/BF16计算packed-fused反向失败仍有效，但pure CAT训练走普通VAE前后向，不能把该失败直接外推为D16 FP32训练失败或通过。随后真实all36 down/D16K24/S2/H128R0/block64/B8192、每stage2步的FP32短验通过：四次更新参数/grad/Adam动量均FP32、非零梯度及真实更新；831张量保存/fresh严格字节一致，8个真实down的cache=True冷fused与BF16 group8预热前向完全相同且FP32参数未变，训练峰allocated约15.435GiB。该短验证据支持这一精度路径可运行，不能单独证明完整任务提升。正式匹配独立down每stage10K随后完成：保留a277其余216投影，只将现有BF16开关改为FP32，完整固定64/32八任务均分63.57159726659092%，低于matched BF16 a277的63.80757566115567%，差−0.2359784个百分点；实际2.4562006411345108bpw，3423张量组装保存与fresh strict核验通过。FP32最终训练日志中的反归一化残差RMS为0.006449492，BF16为0.006414791，未见该指标或下游收益；本次没有独立逆旋转并合并保护INT8后计算部署矩阵误差。该结果只限于本次独立down配方，不证明所有类别或所有FP32方案无效，也不能单独归因于参数舍入。[源码、真实验证及完整结果](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的validation/d16_down_fp32与reconstruction_d16_down_fp32。

<a id="memory-precision--kernel数值检查的dtype和形状范围"></a>

## kernel数值检查的dtype和形状范围

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#kernel数值检查的dtype和形状范围)。

2026-10-03扩展K24/D32和K56/D32/H128R0/S2的现有packed/fused数值检查：两者BF16参数路径的前向、梯度及group8均在未改阈值下通过；额外K24 FP32首层weight梯度相对差2.0167427e-5超过测试原1e-5，原失败保留。相同样本诊断的前向完全相同，将参考梯度直接舍入BF16的相对误差0.00166303约为实际差异82.46倍；这只排除了该样本出现BF16舍入量级退化的解释，未证明精确根因，也不将FP32失败改成通过。BF16纯CAT依赖只绑定其实际通过范围，不据此声称新宽度的FP32-KD或整模型恢复已验证；无需为额外dtype失败改变既定BF16训练条件。group8 helper重复同一decoder，不作8个独立checkpoint或冻结状态审计。完整证据、原阈值与未执行范围见[Qwen3专题](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的`validation/mixed_k24_k56_d32/`和BF16后续目录。

<a id="memory-precision--同bf16计算下的fp32叶参数对照"></a>

## 同BF16计算下的FP32叶参数对照

原专题：[显存与精度：分清驻留对象、数值存储和验证范围](../memory_precision.md#同bf16计算下的fp32叶参数对照)。

2026-10-08，Llama全部32个q_proj、K32/D32/S2/fullRHT、固定INT8保护的隔离两级各1000步对照保持BF16前向/部署及共同舍入初态，仅将叶参数、叶梯度积累与Adam动量由BF16改为FP32。原坐标部署加权NMSE下降.26630%，32/32矩阵均小幅改善、存储不变；末步LR3e-6的零更新比例由约99.8%降至.4%–.5%。这证实此数值组合能改善本次拟合，但效果小，不能将全部归因为参数舍入、解释正式10K平台或推出下游收益。隔离时必须同时保持`AutoEncoder.params_dtype`所控制的BSQ→decoder activation cast，不能只看autocast开关；decoder应在原生q-scale/stage-normalization融合前统一落到部署BF16。stage2不同残差是stage1干预的传递，共同probe只供诊断，最终比较真实逆旋转和保护回填后的原坐标误差。旧Qwen down全FP32负例的输入/计算/存储与此不同，继续保留其适用边界。[完整对照、独审及失败观察器修复](../../history/w25_transfer/2026-10-04_29_research_log.md#10月8日0920新完成的完整顺序链恢复轨迹与后续评测)。

