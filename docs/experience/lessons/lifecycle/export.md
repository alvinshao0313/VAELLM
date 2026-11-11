# 结构转换、打包、导出与独立重载

[经验分类入口](../../README.md) · 一级分类：保存与复现经验（`lifecycle`） · 二级专题：`export`

本篇按问题收录既有经验。原始日期、配置、数值、假设及证据边界保持；历史“当前”仅指原核查时点。跨专题按链接查阅，不能由分类推导通用算法规定。

## 条目导航

- [训练包装、compact格式与子集状态的导出](#checkpoint-lifecycle--训练包装compact格式与子集状态的导出)
- [compact依赖闭合与native前缀转换](#checkpoint-lifecycle--compact依赖闭合与native前缀转换)
- [参数与优化器状态精度不同的重载](#checkpoint-lifecycle--参数与优化器状态精度不同的重载)
- [子集解冻、元数据及容器契约](#checkpoint-lifecycle--子集解冻元数据及容器契约)
- [顺序搜索保留中途最佳，部署排名须独立重载（2026-10-01收尾）](#checkpoint-lifecycle--顺序搜索保留中途最佳部署排名须独立重载2026-10-01收尾)
- [结构转换与解码核切换分开验证（2026-09-24）](#checkpoint-lifecycle--结构转换与解码核切换分开验证2026-09-24)
- [全码位多卡断点的独立导出（2026-10-02）](#checkpoint-lifecycle--全码位多卡断点的独立导出2026-10-02)
- [CPU构造、原权重读取与真实导入上下文](#checkpoint-lifecycle--cpu构造原权重读取与真实导入上下文)
- [真实翻码后的独导与回调资源预约](#checkpoint-lifecycle--真实翻码后的独导与回调资源预约)
- [head、norm及配置运行字段的导出边界](#checkpoint-lifecycle--headnorm及配置运行字段的导出边界)
- [残块迁移、扩宽与恢复成品验证](#checkpoint-lifecycle--残块迁移扩宽与恢复成品验证)
- [独立导出须复现配置规范化顺序（2026-10-17）](#checkpoint-lifecycle--独立导出须复现配置规范化顺序2026-10-17)
- [生成参数须检查最终有效配置（2026-10-04）](#checkpoint-lifecycle--生成参数须检查最终有效配置2026-10-04)

<a id="checkpoint-lifecycle--训练包装compact格式与子集状态的导出"></a>

## 训练包装、compact格式与子集状态的导出

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#训练包装compact格式与子集状态的导出)。

2026-10-30自适应dynamic／Rice最后down类别再次命中训练包装边界：无pending dense时不经PEFT融合，直接沿`trainer.model`导出会遗留Accelerate的`convert_outputs_to_fp32`，而`prepare_model_export`不会移除它。CPU真实Accelerate机制证明默认unwrap仍输出FP32；显式`trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)`后原字段不变、BF16同输入输出exact且全部有限。应在decoder／PEFT结构收束前用该既有标准解除包装，不能cast最终logits或放宽BF16要求。另将dtype与finite分开报错：`dtype == BF16 and isfinite(...)`前项失败时没有检测finite，日志不足以证明NaN。

同日新机真实Qwen3-8B两臂`short_dynamic_trl01`／`short_rice_trl01`已补足GPU证据：gate／up／down各2/500更新，step1保存后回训至step2，FP32 master／Adam／scheduler／RNG保存审计及冻结持久字段exact；三个类别分别融合导出并独进程strict，BF16同输入logits exact且有限。最终3927字段、216native／36compact／0dense、无dense最后类别仅训练decoder，真实账本≤2.6bpw；两臂各自fresh后的up/down两阶实际初始化digest／形状／dtype／LR／seed也逐项相同。[实际pair与绑定SHA](../../../../result/qwen3_8b_mmlu65_20261004/cmcc_sequence_20261030/short_pair_verification.json)已通过，替代此前末端GPU／初始化配对pending；[整类接续及原件](../../../../experiments/adaptive_class_20261030/README.md#10月30日-cmcc-新机接续)保留旧失败、修复、依赖与清理边界。此经验限已验证Qwen3混合三阶／Rice路径的包装、保存和初始化合同，不证明方法精度收益、正式500／10K效果或exact resume。

<a id="checkpoint-lifecycle--compact依赖闭合与native前缀转换"></a>

## compact依赖闭合与native前缀转换

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#compact依赖闭合与native前缀转换)。

2026-10-29完整compact格式原型补充：G09父进程会由bindings选择已验证full-RHT源码snapshot；仅把权重和manifest交给独立进程、沿公共rotation重建时，真实4032维因缺252阶固定Hadamard族而失败。保存字段正确不能替代运行代码依赖闭合。隔离新包显式选择同一既有snapshot并在manifest绑定相关源码SHA后，真实251个VAELinear＋1个compact及全部非投影的3437字段在独立Python进程strict exact；有效权重只读取新成品自身单一文件。不得以换block、换等价Hadamard方向或加载旧父权重掩盖缺失依赖，也不把新schema称为公共v6支持。该证据限定Qwen3/D32K48实际端点；完整GPU更新及部署证据按[本原型结果](../../../../experiments/adaptive_native_model_20261029/README.md)维护，不代表跨规格或同父质量收益。

同日native两阶importer实测补充：原生持久decoder已融合BSQ符号尺度和原stage统计，加载后非持久`_q_scale_fused=False`不证明未融合；只复制当前BF16值，原mean/std未保存就明确不可恢复。真实G09四种K32/34/38/48的码/参数/保护/sign/fresh exact后，K34 gate真实8193行的packed→独立stage之和FP32 relative-L2为6.58e-8，BF16约0.566%，约65.10%元素改变；新compact对相同独立getter操作仍exact。字段无损不能替代跨算子函数等价；该数值仅限逆旋转前所测行，不支持全Linear/GPU或下游结论。后续两臂共享新compact前缀并重算各自residual和误差，不加偏移补偿或把转换差归为动态收益。[接口、实际范围与原始证据](../../../../experiments/adaptive_native_model_20261029/README.md#同父-native-两阶前缀-importer纯cpu接口闭合)。

遇到native／compact混合恢复时，先按真实模块所有权和metadata区分矩阵、norm与bias，并分别处理缓存和临时BF16生成。真实B的fixed2 compact decoder是二维`nn.Linear`叶，native仍为按stage打包的Conv1d，不能机械套`.conv`字段或旧1512叶数量。该B接口父的完整8B两次Muon更新已实测：1728D叶、1008独立矩阵，生成↔回训保持master／momentum／Adam／κ／LR，3675字段独立fresh及同输入BF16输出exact。完整gate跨packed／独立算子relative-L2约0.0070633，单独记录而不补偿或当作优化器收益。适用范围是本fixed2/R0真实拓扑；dynamic／Rice仅CPU载荷检查，不能外推为其完整GPU生命周期或新父32步精度收益。固定的方法是实际归属分组及状态合同；各模型重算布局、κ和学生轨迹，优化超参沿已声明范围，不凭接口成功扩网格。[实际源码、数据身份与两步证据](../../../../experiments/muon_compact_cmcc_20261030/README.md)。

<a id="checkpoint-lifecycle--参数与优化器状态精度不同的重载"></a>

## 参数与优化器状态精度不同的重载

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#参数与优化器状态精度不同的重载)。

当Parameter为BF16而优化器内部动量设计为FP32时，不可仅以采用标准`torch.optim.Optimizer.state_dict`认定重载精度不变。本机Torch2.6的`load_state_dict`会把非step浮点state按Parameter dtype转换；先转换丢精、再`.float()`不能恢复原值。新adapter须在标准load前保留其自身schema中原FP32动量，再按真实参数映射恢复原值／设备；辅助Adam的BF16矩状态沿原精度，不全局提升或降低state。

11月3日VAE Muon隔离适配的CPU验证已覆盖真实owner的144独立矩阵、辅助原BF16 AdamW两步exact，以及32个state字段的dtype／逐值、κ／step往返exact，原参数初始化摘要也匹配控制。[实际adapter](../../../../experiments/cat_down_vae_muon_cmcc_20261103/optimizer.py)与[CPU绑定证据](../../../../result/qwen3_8b_mmlu65_20261004/cat_down_vae_muon_cmcc_20261103/prepared.json)保留来源。随后真实M36/B8192的两stage共4更新、冻结状态及模型独立fresh均通过；优化器state重载精度证明仍限CPU，不把模型fresh升级为GPU优化器重载或训练exact-resume证明，也不代表长程质量。迁移固定按参数实际所有权／state精度处理，不固定模型层数；Parameter本来为FP32的类别恢复不因本例新增全链精度变更或重复审计。

<a id="checkpoint-lifecycle--子集解冻元数据及容器契约"></a>

## 子集解冻、元数据及容器契约

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#子集解冻元数据及容器契约)。

2026-10-24子集decoder原型补充：当前`enable_trainable_decode`会重新将整个decoder设为可导；仅在初始化冻结hidden，不能保证native OPD返回后仍然冻结。三α隔离实现因此在每次恢复后重新应用24头可导／48隐藏叶冻结，并把3个独立master与packed carrier分开核查。真实码CPU两次更新、冻结补集及保存/fresh strict已验证；完整GPU OPD往返仍待，不能把此机制证据写成下游收益或完整GPU通过。适用于训练／原生推理切换时只开放decoder子集的路径，不需要修改原先全部decoder可导的实验。[原型证据与待验证范围](../../history/w25_transfer/2026-10-04_29_research_log.md)。

2026-10-21最后类别中点独导补充：`save_v6_full_checkpoint`的返回值是保存摘要，不能当作完整v6 metadata传给`audit_state_dict`。新脚本在真实恢复／保存后因此退出，尚未进入fresh strict或GPU；修复为读取落盘`checkpoint_meta.json`，在独立输出重新导出、账本审计及3423张量独进程strict通过后才接评测。`round_base`本身不是该审计器接受的成品kind，也不能为通过检查伪改kind；须在真实category_boundary保存后用真实元数据审计。入口静态检查不能替代新增调用路径的实际运行，已成功的前序评测无需因后续导出失败重跑。[失败原件、最小修复及实际接续](../../history/w25_transfer/2026-10-04_29_research_log.md#10月21日较小码更新终态freshcat末点与最后类别中点)。

2026-10-24异构stage兼容验证补充：packed decoder的`extract_single`可返回含裸`nn.Linear`的临时decoder，而v6按spec重建的是单模型`ParallelLinear.linear`。参数值正确并不保证保存后的键/容器契约匹配；本次真实fresh strict报16个线性层键不匹配。隔离修复按v6原生单模型容器逐值复制四个线性叶和两个norm叶，保留已融合q_scale，完整3449张量保存/fresh strict随后通过。临时旧S2视图只读compressed坐标时，其自身尺寸也须使用compressed尺寸，保护与rotation由外层只执行一次。异构v6的CPU重载不证明原预热入口兼容或CUDA数值等价；需明确独立适配器及原生零新增对照。[实际失败、修复与CPU证据](../../../../result/qwen3_8b_mmlu65_20261004/qwen_b32_s3_lowrate_compat_20261024/README.md)。

<a id="checkpoint-lifecycle--顺序搜索保留中途最佳部署排名须独立重载2026-10-01收尾"></a>

## 顺序搜索保留中途最佳，部署排名须独立重载（2026-10-01收尾）

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#顺序搜索保留中途最佳部署排名须独立重载2026-10-01收尾)。

SC-CAT前两类的正式完整节点比较进一步支持按保存节点选优：Qwen3-8B／seed31／五源train／原500日程，每125步完整八项，k与其后继o分别选250和125，均不是末点；同配方较各自500末点分别高0.16455055／0.38042596pp。两类按规则追加半decoder LR后仍未超过原LR赢家，因此“早期更好”只触发有限比较，不能直接判断学习率过大。o的raw候选排序在恢复后反转，必须在同一live父上比较各自恢复结果。固定方法为保存原日程有限节点、独立部署与同协议八项选点、完整mutable接续；按模型／父重算步点及配方，LR只在已批准[5e-6,1e-5]范围比较，本次不支持固定250／125、跨模型最优或九项收益结论。[完整32节点对照与边界](../../records/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31_results.md#cat-recovery)。

**本配置实测支持**：0920 checkpoint、rank8、四卡DP搜索于北京时间2026-10-01 02:20:35正常完成；15组配置中12组到5000步（3组1000→5000、9组2500→5000），另3组到2500步。阶段退出码、run_meta和原始指标支持真实续训完成，不等于连续训练/resume逐位审计。[配置、指标与来源](../../records/e2e_0920/2026-09-24_rank8_search.md)

**导出证据边界**：早期baseline `lr3e5/step1000`训练内66.564804%→fresh process四卡strict重载66.554029%，差−0.010775个百分点。收尾前三个step2500快照也各自严格重载成功：pre-MLP权重0为68.009998%→68.024489%，0.03为67.974884%→67.859254%，0.01为67.883982%→67.742981%。排序未变但导出差值不同，不能用早期baseline差值替代各候选复评，更不能推断位级一致。

**动作**：保留中途最佳训练checkpoint直至实际导出重载比较完成，5000步终点不能替代2500步权重；搜索最高分与最终交付排名分别报告。本轮`best_result.json`已指向唯一保留的`deployments/c05_pre_mlp_hidden_loss_weight_1_step2500/model`，重载均分68.024489%，未达到69%。

拟补评中点时，先核实际载荷及生产保存策略，保存成功审计不代表文件仍在。2026-11-03核D16类别恢复：step250曾保存绑定raw父的完整FP32可变参数与Adam／scheduler／RNG，但原`save_total_limit=1`在step500保存时已自动删除它；当前仅存500，日志和mixed_save_audits不是250模型。MixedCatTrainer尚标`exact_resume_implemented=False`，没有现成mixed中点部署入口，不能从500参数或BF16成品还原250，也不因保存字段齐全称已验证GPU exact-resume。本轮八项raw→500微降而M多52题，九项尚未完成，故不为缺失中点追加约3小时重训。未来确需选点时，先确定同父有限节点／完整质量比较与保留用途，确保待评载荷不被轮转，再按原总日程前缀执行、实际导出／fresh；重跑是新轨迹，不冒称旧快照。该规则不要求永久保存全部checkpoint，未测250也不视为优于500。[真实250删除记录及研究边界](../../../../experiments/down_geometry_baseline_cmcc_20261103/README.md)保留来源；本轮不新增训练或评测预算。

**清理与恢复边界**：控制器已清理其他生成权重及全部optimizer训练checkpoint，原数据、初始checkpoint、配置、核心日志和原始指标保留；交付模型仍可推理，但无法恢复原训练的optimizer/scheduler/RNG状态做exact-resume。147条cleanup累计124.8008 GiB为控制器历史删除文件的逻辑大小，不是本次文档收尾手动释放的空间。

<a id="checkpoint-lifecycle--结构转换与解码核切换分开验证2026-09-24"></a>

## 结构转换与解码核切换分开验证（2026-09-24）

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#结构转换与解码核切换分开验证2026-09-24)。

**本配置实测支持，已通过真实导出及fresh strict重载，并在用户授权的隔离worktree落地。** 0910 Qwen3-8B的decoder+LoRA真实8步checkpoint：冻结decoder保持packed路径时logits逐元素相同，转换LoRA相对同路径也逐元素相同；仅由packed切到修复候选fused时，logits相对L2约0.00794。原core校验将结构转换和计算核切换合在一起，不能将这一报错直接解释为LoRA参数丢失。[同模型三阶段证据与边界](../../records/e2e_0910/2026-09-24_rank8_search.md)

先排除真实实现失配：本次旧fused的LayerNorm/SiLU与当前原生decoder不同，FP32 bias与末Linear舍入也有陈旧行为；真实模块对照与CUDA回归通过后才讨论不同核的正常浮点差异。结构校验应在同一部署计算路径比较转换前后，保持原门限与部署缓存；跨核误差单独记录，保存后仍须fresh strict加载复核。不能只关缓存或放宽总门限掩盖错误，也不把不同计算核全链路逐位一致当作通用要求。

**2026-10-01新服务器核查及落地**：上述修复最初只在旧隔离worktree，新block32/INT8底座的8步decoder-only E2E再次在结构finalization失败。真实K32/K48诊断确认同类LN/SiLU失配，旧patch修复H128后，H256/FP32首层W在BF16运算时仍请求229,376字节共享内存，超过A800限额。将内核原有W转换前移至两个launch的临时张量，保留autograd ctx中的原FP32 W和bias，可将H256 native/group8降至132,096字节。旧消费者全部退出后，14:26主源应用修复；16项相关回归通过，H128实际8步decoder与decoder_sparse_bit均完成finalization、独立step4导出及严格重载/冻结状态/实际位宽审计。H256实际8步也完成训练及导出。更换推理解码核后重测了三个未更新底座及joint512成品，旧结果单独保留；不能把重测差值当作训练收益，也不能用旧worktree的“修复完成”替代当前文件哈希。[新源诊断、SHA与真实结果](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)

独立导出训练中间checkpoint时，还须按生产侧同样的类型归一化重建不可变契约。本轮真实HF `TrainingArguments` 将eval/save strategy转成枚举，而normalized JSON保存枚举值；直接对JSON字符串执行lower会把`no`与生产侧`intervalstrategy.no`误判为不同。只在新导出入口恢复相应枚举、再调用原契约校验，未改checkpoint、原始总步数或生产语义；7项真实参数回归及两份实际step4导出通过。不能通过改原max_steps或关契约检查来导出选定步。

<a id="checkpoint-lifecycle--全码位多卡断点的独立导出2026-10-02"></a>

## 全码位多卡断点的独立导出（2026-10-02）

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#全码位多卡断点的独立导出2026-10-02)。

Qwen3-8B全252投影使用100% FP32码位及Adam时，精确训练侧车约194.3 GB。独立推理导出无需在一张卡重建全部score和optimizer：生产侧车的`packed_banks`已经物化当前score符号，CPU mmap按原chunk布局核对bank、sampler、固定尺度、轮状态及全部符号后，可直接恢复硬码，再用一张卡执行原生BF16部署转换。不能把这种导出说成恢复了optimizer或通过了exact-resume。

本轮真实四卡4步/save2完成两次独立导出、冻结状态审计、limited GPU推理和回训；最终独立与原生导出的3423个持久张量及字节一致。同步保存回调等待导出与评测结束，最新训练断点的轮转与独立成品目录分开；正式每400步固定全八任务，诊断limit只在一次性验证驱动中。CPU状态审计虽不做GPU前向，当前FA2装载器仍需CUDA可见；隐藏CUDA引起的是装载前提错误，不是模型状态损坏。[实测、参数和复现证据](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)

<a id="checkpoint-lifecycle--cpu构造原权重读取与真实导入上下文"></a>

## CPU构造、原权重读取与真实导入上下文

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#cpu构造原权重读取与真实导入上下文)。

2026-10-18持续代理8步的独立导出再次遗漏这个已知前提：旧两步在GPU可见进程内成功，不能复用为`CUDA_VISIBLE_DEVICES=''`子进程的构造证据。源码确认Qwen工厂显式FA2，训练权重完整；保留原prepared／训练report／源码pins，只新增无前向CPU构造入口，临时eager构造后立即恢复FA2配置，再走canonical v6装配、保存与fresh strict。真实修复出口0、CUDA未初始化、3423张量全等／3415冻结／位宽不增，部署推理协议未改。该隔离构造适用于已核Qwen3路径，不能当通用模型兼容实现；活动旧链不原地改pins，完成后用明确修复入口接续。[本次复现、限定修复与审计](../../history/w25_transfer/2026-10-04_29_research_log.md#10月18日完整九项终态与下一轮验证)。

2026-10-30新机H192独导补充：完整v6父成品自身已含全部持久参数，canonical便捷工厂先`from_pretrained`读BF16原权重属于重复读取。隔离入口复用已验证的`no_init_weights`／`from_config`模式，从同一原HF metadata以BF16／FA2构造结构，再直接调用完整canonical v6重建／dtype恢复／strict加载，不读HF shard。GPU可见环境下实际完成3423保存／fresh strict全部exact、1911冻结、504码库、原账本及拓扑审计；FP32 RoPE由原config构造并单独核dtype／有限性，没有整模BF16转换。此入口仅覆盖已验证的完整Qwen3-8B252／rank0拓扑，不由该结果推断部分checkpoint、在线teacher或来源校验也不需要原BF16。新机身份与运行源码重新绑定，原训练和旧失败原件不回写。[接口、实测成品和边界](../../../../experiments/g09_hotpot_coverage_20261030/README.md#cmcc-新机接续2026-10-30)。

同次故障还说明只设置快照`PYTHONPATH`或临时插入`sys.path`不足以证明实际导入一致：旧独导的snapshot `weight_preprocess`混入主仓库旧`get_hadK`，4032维在旧96分支断言失败；原完整快照已有正确252阶因子。新入口在会改`sys.path`的helper之前导入所需native模块，并核各模块真实`__file__`、SHA及实际4032→252，再执行真实全量重建。修复不改变旋转spec、保护维度、位宽或检查；不能把该导入失配解释为4032维原算法不支持。来源同上。

若项目包是namespace包，提前导入包本身仍不保证后续子模块来自同一快照：helper再次前插仓库路径后，动态`__path__`会重新排序。应在隔离入口选择完整快照后、导入这些helper前，明确限定所需namespace的子模块搜索，并用真实正式入口的导入链验证来源；只检查一个sources模块会遗漏延迟导入。2026-10-30 C0第一次接续在训练前因`train_utils.cat_train_residual_protection`落到S而被原来源断言拒绝。独立C0包固定四个namespace到N后，实际`run_pipeline`全部导入、正式raw2绑定、七类recipe和512×256校准runtime恢复通过CPU核验，未加载训练权重或执行更新；GPU父源检查仍由正式入口在首次恢复前完成。来源与完整边界见[C0故障修复](../../../../experiments/fresh_fixed2_cmcc_20261030/README.md#验证范围和当前状态)。此规则固定的是代码来源，迁移时重算各模块路径／SHA；没有新增算法超参，不把CPU导入通过外推为模型质量或跨系列验证。

临时调整调用点时，模块同路径不代表同一个对象。11月4日up真实N小fit复现：顶层旧`transformers` LazyModule与实际localimport当前对象不同，hook旧对象后命中0、两步LR仍.003／.0015；直接构造scheduler的CPU测试没有覆盖这条调用链。context进入时用importlib取得实际当前对象、退出恢复，再用真实N两stage小fit核命中2、LR .003／.0029997，才补正式shapeGPU短验；两臂保存／fresh均过后接正式。固定方法是核真实调用对象／命中及有效日程，不能伪造模块属性或用孤立helper通过替代实际入口。此次未追踪具体触发模块替换的调用栈，不将安装源码中的可能机制写成已证根因；适用证据和失败原件见[up接线修复](../../../../experiments/up_vae_muon_cmcc_20261104/README.md)。这不增加常规超参的逐点短验要求。

随后完整八任务实际成功，但CPU汇总才触发`lm_eval`的延迟导入：直接执行位于含`evaluate.py`的目录中的脚本，使本地文件遮蔽HF同名包。只做`--check-only`未覆盖该延迟路径；已跑完的两份full8不能因此重跑。隔离接续使用模块入口与完整包导入，真实检查HF `__file__`，复用原`full8_summary`对64叶／24742及raw哈希做完整验证，只把新摘要写入新根，再接剩余LB。来源同上。该故障属于Python模块搜索路径，不能以改名原CLI或吞掉导入异常处理。

增加独立来源入口时，也不能隐式依赖原外层launcher先完成本机路径初始化。2026-10-31正式BR源码审查发现，裸`bind_C0`直接调用旧reader会让正向`model_inventory`落到默认`/home/star`；只验证short／缺失来源被拒没有触及此路径。修复采用本包一次`configure_readers`：先核真实cmcc support并绑定BASE/REFERENCE/TEMPLATE，再安装BR或BR→E dispatch；后续source调用只校同一绑定，不重复setup而重置已安装的source/pins/command。实际CPU检查读取本机HF的252投影元数据，并证明两种dispatch安装后7条真实拒绝链保持路由，未载权重或伪造正式C0。该证据说明读取上下文已接通，不等于正式训练已经成功；迁移时重绑路径／协议支持，算法超参不变。[正式入口、核验与未验证范围](../../../../experiments/block_reconstruction_formal_cmcc_20261031/README.md#当前验证与未验证范围)。

<a id="checkpoint-lifecycle--真实翻码后的独导与回调资源预约"></a>

## 真实翻码后的独导与回调资源预约

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#真实翻码后的独导与回调资源预约)。

正式400步进一步覆盖了大量真实翻码：CPU核对全部16,024,338,432个score符号后导出成功，最终硬码相对初始源XOR15,749,205位；3423持久tensor的fresh strict重载、1407冻结tensor及实际2.456988791bpw账本均通过。该证据补足4步验证中尚无翻码的边界，不代表恢复训练状态或精度已达标；完整八任务结果仍按该保存点单独验收。

同步回调的GPU预约边界早于实际推理：当前StepExportCallback在进入导出阶段前先检查指定GPU没有compute进程，再依次CPU导出、审计和完整评测。因此给其他任务安排空档应截止于保存后的回调入口，不能把CPU导出耗时当作可继续占卡的时间。2026-10-02的D16 q/k排程漏算此前置检查，按实测速率预计晚于入口结束，主代理及时撤下自己本轮刚启动的q/k组并改到100%800完整回调之后；原100%训练和其他既有任务未被暂停。该事件是已识别的调度错误，不是D16质量失败或未知外部SIGTERM；精确PID、主动停止原因、原始日志与替代安排见[同一专题](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)。

<a id="checkpoint-lifecycle--headnorm及配置运行字段的导出边界"></a>

## head、norm及配置运行字段的导出边界

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#headnorm及配置运行字段的导出边界)。

2026-10-03，Qwen3完整252投影的既有linear head/final norm方法补齐隔离周期导出，均通过真实552282/B4/1024/4步save2的两回调、回训及原生final。head训练映射按原生路径融合进原head，保持BF16形状与字节数；final norm对照只允许原model.norm.weight及decoder变化，冻结head/144其他norm/504码。短验final norm的4095个FP32元素更新，只有2个在BF16保存后保留，因此不能强制“短验BF16必须大量变化”来调高LR，也不能用FP32变化宣布部署改善。导出逐字节核所选FP32 norm转BF16，并分别记录存活量；全部3423保存/fresh、1910冻结及实际2.456988791bpw通过，有限题量仅验证流程。[原生顺序、精度差与完整验证](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的20:00与20:35记录。

同日已有类别组装对552282与bits05导出源的本地config比较因use_cache/pad_token_id差异受阻；真实v6 loader用共同base_model_path的原HF配置构建模型，两个字段属于保存的运行状态，不能因此误判投影架构变化。新增隔离入口仅接受bool cache与None/原HF tokenizer pad id，所有其余config及252部署spec严格相同，并实际核147个非投影张量相等；复用原source-union/保存fresh/账本，未修改仍被训练使用的旧入口。12项CPU配置拒绝/接受测试与两份实际3423张量组装及完整64/32评测通过。该白名单只适用这里已证明的加载语义，不授权宽泛忽略其他config或未经证明的架构差异；正确组装也没有使两候选超过来源最好。[运行字段来源、完整证据与质量边界](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的20:35记录。

<a id="checkpoint-lifecycle--残块迁移扩宽与恢复成品验证"></a>

## 残块迁移、扩宽与恢复成品验证

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#残块迁移扩宽与恢复成品验证)。

2026-10-03的R0→R1零输出残支迁移再次显示结构与计算路径须分开：原3423张量不变、新增1008张量后，同native packed的真实七类完整权重/前向相同；原R0 wholefused对新R1 native仍有最大相对L2约0.0001313的差异。未训练完整均分差+0.00876个百分点不能当作容量优化收益。真实4步/save2、两次导出回训及4431张量native/fresh一致验证通过，正式runner库存必须取新实物而非写死旧3423；没有因此要求正常跨核浮点逐位相同。[迁移、初始完整基线与短验](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的22:58和23:24记录。

2026-10-04隔离H128R0→H256R0采用每stage均匀复制输入Linear/LN、输出[B,0]。均匀复制才能保持LN统计；输出不对称使隐藏副本首步梯度分化。CPU数学、梯度和现有v6 pack/unpack/strict重载通过；真实552282 L0 gate/down完整native BF16矩阵与原native相同，但原wholefused相对新native仍有约0.0001254相对L2差异。此helper替换Parameter对象，应在optimizer/hooks建立前调用。[隔离实现、证据与边界](../../history/qwen3_w25/2026-09-30_10-04_research_log.md)的10月4日00:16记录。

后续01:47–02:22已补齐同552来源、L0 gate/down两个模块的256步真实反传、独立BF16导出与完整八任务：H128/H256均分66.96068%/66.96508%，都低于源67.13033%；H256的MMLU57.94046%也低于H128的58.20396%。因此不能继续称“只有推理验证”，也没有证实扩宽带来质量收益。当时尚未验证全模型扩宽训练、best1200起点或其他更广范围/损失条件，不把该局部负结果推广为容量的普遍上限，预计位宽仍须与实际导出账本区分。来源见上述同一专题的后续扩宽对照。

同日16:19补齐best1200起点的全部108个MLP配对：同task50纯KL/prompt.3、decoder3e-6、原2000/warm20调度前400步，H128/H256完整均分68.51705410%/67.83285084%、MMLU60.66087452%/60.20509899%，均低于各自未训练起点。H256实际2.4615956182bpw，108模块真实更新、其他2775张量及504码库冻结，3423保存/fresh严格一致。两臂40个窗口的token计数相同，H256训练耗时1.4524倍；相同LR不等于相同函数空间步长，不能只归因容量，也不外推所有扩宽方案无效。本配方无收益依据，不延长，退役生成权重并保留原始证据。[完整配对、初始差与资源边界](../../history/w25_transfer/2026-10-04_29_research_log.md#1619-起容量配对完整负结果及-head-1200)。

2026-10-12当前a0e5为symmetric H128、`num_res_blocks=0`，外置LoRA rank0。真实库存预测允许仅gate或仅down的36模块H256，两类一起超logical2.5bit；预测不等于新成品实测。旧widen helper强制asymmetric，不能直接用于symmetric源的单因素加宽；动态save/load可表达H256也不代表当前固定原shape/库存的teacher×scope导出器已支持。旧负结果只约束既有配方，不能证明本轮容量根因；当前四格完整评测继续，首张实际可用卡衔接的既定BIT诊断已完成，四格完整九项与BIT证据齐后，再选择数据或容量方向。[当前预算、实现边界与决定](../../history/w25_transfer/2026-10-04_29_research_log.md#10月11日-train-only-八组完整结果与下一轮覆盖对照)。

<a id="checkpoint-lifecycle--独立导出须复现配置规范化顺序2026-10-17"></a>

## 独立导出须复现配置规范化顺序（2026-10-17）

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#独立导出须复现配置规范化顺序2026-10-17)。

**实测及源码支持。** C1的57科学科重配1200步独导被immutable contract拒绝，差异仅为60个数据混合权重，最大绝对差1.8513e-14。normalized快照在数据构建前保存，训练随后调用`DistillDataConfig.validate()`，重新解析`.12g`格式化的mix并归一化；旧导出器直接拿此前快照比较，遗漏了这次确定性转换。自然union臂恰无此差异，不能由它的短验通过外推所有混合配置。

隔离修正版仅对快照data副本调用原生validate一次，仍用原全字段exact检查；A/B六份真实400／800／1200 checkpoint均通过，改真实概率、source顺序、prompt损失或已存contract概率均拒绝。B1200随后实际独导及原生审计通过：3423保存／fresh张量、1512 decoder、1911冻结、504硬码库不变，rank0／原位宽。此修复不训练、不修改元数据或活动依赖，也不证明恢复精度；不可用放宽容差、循环归一化到相等或复制saved值掩盖不一致。原正式链仍引用旧入口，完成训练后需用修正版接续导出。[根因、隔离验证及运行证据](../../history/w25_transfer/2026-10-04_29_research_log.md#10月17日长预算结果阶段节点比较与文献对照)。

<a id="checkpoint-lifecycle--生成参数须检查最终有效配置2026-10-04"></a>

## 生成参数须检查最终有效配置（2026-10-04）

原专题：[Checkpoint：先持久化可恢复状态，再导出并独立重载](../checkpoint_lifecycle.md#生成参数须检查最终有效配置2026-10-04)。

本机 Transformers 4.51 的真实 Qwen3 LongBench 短测发现：仅传新建的 GenerationConfig，默认值 `do_sample=False` 仍可能被模型自身采样默认值替换。首次 dense 输出虽正常完成，协议不成立，已保留并排除；不能以配置对象的初值声明实际使用 greedy。

独立 LongBench 入口显式传 `use_model_defaults=False`，在真实 `_prepare_generation_config` 路径核对所有已指定选项，并在预测中记录最终有效配置。CPU 回归及 dense/压缩两模型各三个真实长输入、多步 KV、512-token 摘要和 SAMSum 停止检查均通过。固定双方 EOS、pad 与特殊任务 min_length，避免模型保存状态改变比较协议。该结论针对当前版本和入口，不等于其他 API 或所有版本都有相同行为；完整评测仍须全部样本完成后评分。[原始无效短测、修复及复测](../../history/w25_transfer/2026-10-04_29_research_log.md)的 LongBench 部分。

