# Checkpoint：先持久化可恢复状态，再导出并独立重载

## 问题与证据

Linear输出实验在导出第4个模块时触发误差门限，旧runner此前没有保存整分片训练状态，导致后续60个Linear最终状态无法从日志恢复。校验还混合BF16训练、FP32融合与TF32部署路径。[导出故障记录](../records/experiments/linear_output/RECOVERY_NOTES.md)

真实硬件审查还发现：finalization重新unpack已训练packed decoder、提前合并LoRA支路、保留FP32输出wrapper，以及已到max_steps的恢复额外多跑一步等问题。[硬件审查](../history/audits/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md)

## 下次怎么做

- 在不可逆导出前原子保存恢复必需状态；每个模块导出成功立即更新manifest，失败写明FAILED及目标，不能靠过期TRAINING标记判断进程状态。
- 区分同精度同路径的保存等价性与部署精度变化；各自记录误差，不通过放宽一个总门限掩盖语义混用。
- 采用fresh process strict重载；核对压缩范围、码、拓扑、冻结状态、推理输出，不能只验证文件存在。
- step恢复验证optimizer/scheduler/RNG、可变参数和Sparse Bit状态，并比较下一步更新；到达max_steps后应进入finalization。
- 运行缓存要在状态加载后刷新，训练评测、finalization与部署的表示必须明确。低精度矩阵运算重结合不自动保持同值。
- 成功验收后，依据后续用途清理checkpoint；恢复安全不意味着永久保留所有权重。

## 边界

历史硬件审查只通过单卡/单rank链路，真实多卡为hardware-deferred。本次没有复测。结果报告里旧output_alignment_v6_partial192路径与后续distill_init交付名称不同；读取当前模型必须以实际路径和最新交付记录核对，不能照搬历史“当前状态”。[后续交付说明](../records/experiments/linear_output/EXPERIMENT_SUMMARY.md)

## 层边界恢复须重建运行缓存（2026-09-23）

**本配置实测支持**：Qwen3-8B单stage VAE逐层恢复中，block9/10各1步的连续训练与层边界恢复，在native码/decoder、输出、loss/LR、RNG和冻结状态上严格一致；注入7处过期grouped packed副本后，复用native refresh能恢复正确前向。[工程验证及原始证据](../records/experiments/liftquant_recovery/2026-09-23_engineering_equivalence.md)

**动作**：边界原子保存仅完成块的native状态和必要身份/样例/RNG；恢复从原初始模型开始，验证完成列表是目标前缀，匹配实际初始权重、FP teacher权重、代码、配置与token内容。复制persistent状态后同步grouped packed并清除decoded缓存，再严格比较已完成块输出。teacher前缀重放后才在保存边界恢复RNG。

**边界**：新block原本就建立fresh optimizer，因此层边界可不保存前一块Adam；未完成block的连续proxy/Adam/scheduler不能仅从硬码恢复。本轮各1步只验证恢复路径，未证明长训练或下游收益。

## E2E中途评估改变RNG时的阶段恢复（2026-09-24）

**证据状态：CPU精确对照通过、真实双卡流程已覆盖。** 当前lm-eval会重设随机状态，而Trainer的checkpoint原在评估前保存RNG；阶段暂停再恢复可能重放评估前状态，偏离连续训练。修复是在成功评估并恢复训练状态后，将各rank实际RNG写回本次checkpoint，再允许暂停；不改变连续训练原有语义。[验证及边界](../records/e2e_0910/2026-09-24_rank8_search.md)

**当前实现约束**：启用eval_after_save时，save_steps和save_strategy属于不可变恢复契约；评估本身影响RNG，不能把降低评测频率当作无语义变化的续训加速。阶段预算只改stop_after_step，并保持其为原save_steps的整倍数；待评估完成、RNG写回并成功暂停后再恢复。当前恢复入口从checkpoint反推原run目录，`run_root_dir`不会将续训迁移到新目录；状态以resolved output为准，同一实验按step追加结果。依据为`v6_runtime_state.py`的evaluation_execution契约及`runtime_v6_pipeline.py`的阶段验证，具体运行见上述记录。

阶段预算应独立于scheduler总steps。真实CPU Trainer含dropout/累积的连续与恢复参数、Adam、scheduler逐位一致；真实GPU非确定运行在暂停前也有差异，不能据此宣称GPU逐位等价。最终导出核切换问题已独立修复并通过同路径验收，见下节；阶段恢复通过本身不能替代导出验证。

## 顺序搜索保留中途最佳，部署排名须独立重载（2026-10-01收尾）

**本配置实测支持**：0920 checkpoint、rank8、四卡DP搜索于北京时间2026-10-01 02:20:35正常完成；15组配置中12组到5000步（3组1000→5000、9组2500→5000），另3组到2500步。阶段退出码、run_meta和原始指标支持真实续训完成，不等于连续训练/resume逐位审计。[配置、指标与来源](../records/e2e_0920/2026-09-24_rank8_search.md)

**导出证据边界**：早期baseline `lr3e5/step1000`训练内66.564804%→fresh process四卡strict重载66.554029%，差−0.010775个百分点。收尾前三个step2500快照也各自严格重载成功：pre-MLP权重0为68.009998%→68.024489%，0.03为67.974884%→67.859254%，0.01为67.883982%→67.742981%。排序未变但导出差值不同，不能用早期baseline差值替代各候选复评，更不能推断位级一致。

**动作**：保留中途最佳训练checkpoint直至实际导出重载比较完成，5000步终点不能替代2500步权重；搜索最高分与最终交付排名分别报告。本轮`best_result.json`已指向唯一保留的`deployments/c05_pre_mlp_hidden_loss_weight_1_step2500/model`，重载均分68.024489%，未达到69%。

**清理与恢复边界**：控制器已清理其他生成权重及全部optimizer训练checkpoint，原数据、初始checkpoint、配置、核心日志和原始指标保留；交付模型仍可推理，但无法恢复原训练的optimizer/scheduler/RNG状态做exact-resume。147条cleanup累计124.8008 GiB为控制器历史删除文件的逻辑大小，不是本次文档收尾手动释放的空间。

## 结构转换与解码核切换分开验证（2026-09-24）

**本配置实测支持，已通过真实导出及fresh strict重载，并在用户授权的隔离worktree落地。** 0910 Qwen3-8B的decoder+LoRA真实8步checkpoint：冻结decoder保持packed路径时logits逐元素相同，转换LoRA相对同路径也逐元素相同；仅由packed切到修复候选fused时，logits相对L2约0.00794。原core校验将结构转换和计算核切换合在一起，不能将这一报错直接解释为LoRA参数丢失。[同模型三阶段证据与边界](../records/e2e_0910/2026-09-24_rank8_search.md)

先排除真实实现失配：本次旧fused的LayerNorm/SiLU与当前原生decoder不同，FP32 bias与末Linear舍入也有陈旧行为；真实模块对照与CUDA回归通过后才讨论不同核的正常浮点差异。结构校验应在同一部署计算路径比较转换前后，保持原门限与部署缓存；跨核误差单独记录，保存后仍须fresh strict加载复核。不能只关缓存或放宽总门限掩盖错误，也不把不同计算核全链路逐位一致当作通用要求。

**2026-10-01新服务器核查及落地**：上述修复最初只在旧隔离worktree，新block32/INT8底座的8步decoder-only E2E再次在结构finalization失败。真实K32/K48诊断确认同类LN/SiLU失配，旧patch修复H128后，H256/FP32首层W在BF16运算时仍请求229,376字节共享内存，超过A800限额。将内核原有W转换前移至两个launch的临时张量，保留autograd ctx中的原FP32 W和bias，可将H256 native/group8降至132,096字节。旧消费者全部退出后，14:26主源应用修复；16项相关回归通过，H128实际8步decoder与decoder_sparse_bit均完成finalization、独立step4导出及严格重载/冻结状态/实际位宽审计。H256实际8步也完成训练及导出。更换推理解码核后重测了三个未更新底座及joint512成品，旧结果单独保留；不能把重测差值当作训练收益，也不能用旧worktree的“修复完成”替代当前文件哈希。[新源诊断、SHA与真实结果](../records/qwen3_w25/2026-09-30_goal.md)

独立导出训练中间checkpoint时，还须按生产侧同样的类型归一化重建不可变契约。本轮真实HF `TrainingArguments` 将eval/save strategy转成枚举，而normalized JSON保存枚举值；直接对JSON字符串执行lower会把`no`与生产侧`intervalstrategy.no`误判为不同。只在新导出入口恢复相应枚举、再调用原契约校验，未改checkpoint、原始总步数或生产语义；7项真实参数回归及两份实际step4导出通过。不能通过改原max_steps或关契约检查来导出选定步。

## 源码身份拒绝恢复时保留原运行版本（2026-09-24）

**本配置实测支持**：LiftQuant全层恢复被SIGINT中断后，保存边界与当前源代码仅有通用E2E CLI六行差异，严格恢复因此拒绝。隔离源码副本还原这六行后，101个训练依赖文件SHA与边界逐个相同，初始权重、teacher、参数、token、runtime及保存张量检查通过。[中断与重启记录](../records/experiments/liftquant_recovery/2026-09-24_full_layers_lr.md)

**动作**：先定位身份差异，不修改boundary或关闭检查，也不回滚其他活动实验使用的共享文件。已有原内容可精确重建时，在独立源码目录保留原版本，cwd/PYTHONPATH都指向它，模型/数据/产物路径显式指向原位置。运行必须持有明确源码副本或持续保护源码，不能仅依赖启动时已导入模块来保证后续子进程一致。

**边界**：本次CLI差异是重启阻断，不是SIGINT来源。只有KeyboardInterrupt/子进程signal退出时，不能认定OOM或算法失败，也不能推断信号发送者。无控制终端并断开stdin可隔离终端操作，但不能保证外部显式信号永不再次终止任务；仍保留正常的SIGINT/SIGTERM停止能力。本配置在独立session、stdin=/dev/null下仍第二次收到SIGINT，用户明确表示未主动停止；这证明先前终端隔离措施未解决未知发送源，不能将中断归因于SSH断线。最新边界的101个源码SHA及CPU结构校验仍通过，信号中断与源码恢复问题须分别判断。层边界恢复只保留已完成层，未完成层需从该层首步重做。


重复SIGINT中断而缺少发送者审计时，可在既有运行外层加入系统现有的信号记录工具。本次strace 5.16以seccomp过滤全部syscall，仅记录指定中断信号；真实CPU探针验证能给出原始发送者PID/UID，且保持原有SIGINT退出语义。Python最终可能再向自身发SIGINT以还原退出状态，溯源应看第一条外部SI_USER记录，不能把随后的self-signal误判为训练主动中断。此探针未查明此前两次发送者；工具只改善取证，不表示外部中断原因已经修复，也不改变训练协议。

## 全码位多卡断点的独立导出（2026-10-02）

Qwen3-8B全252投影使用100% FP32码位及Adam时，精确训练侧车约194.3 GB。独立推理导出无需在一张卡重建全部score和optimizer：生产侧车的`packed_banks`已经物化当前score符号，CPU mmap按原chunk布局核对bank、sampler、固定尺度、轮状态及全部符号后，可直接恢复硬码，再用一张卡执行原生BF16部署转换。不能把这种导出说成恢复了optimizer或通过了exact-resume。

本轮真实四卡4步/save2完成两次独立导出、冻结状态审计、limited GPU推理和回训；最终独立与原生导出的3423个持久张量及字节一致。同步保存回调等待导出与评测结束，最新训练断点的轮转与独立成品目录分开；正式每400步固定全八任务，诊断limit只在一次性验证驱动中。CPU状态审计虽不做GPU前向，当前FA2装载器仍需CUDA可见；隐藏CUDA引起的是装载前提错误，不是模型状态损坏。[实测、参数和复现证据](../records/qwen3_w25/2026-09-30_goal.md)

正式400步进一步覆盖了大量真实翻码：CPU核对全部16,024,338,432个score符号后导出成功，最终硬码相对初始源XOR15,749,205位；3423持久tensor的fresh strict重载、1407冻结tensor及实际2.456988791bpw账本均通过。该证据补足4步验证中尚无翻码的边界，不代表恢复训练状态或精度已达标；完整八任务结果仍按该保存点单独验收。

同步回调的GPU预约边界早于实际推理：当前StepExportCallback在进入导出阶段前先检查指定GPU没有compute进程，再依次CPU导出、审计和完整评测。因此给其他任务安排空档应截止于保存后的回调入口，不能把CPU导出耗时当作可继续占卡的时间。2026-10-02的D16 q/k排程漏算此前置检查，按实测速率预计晚于入口结束，主代理及时撤下自己本轮刚启动的q/k组并改到100%800完整回调之后；原100%训练和其他既有任务未被暂停。该事件是已识别的调度错误，不是D16质量失败或未知外部SIGTERM；精确PID、主动停止原因、原始日志与替代安排见[同一专题](../records/qwen3_w25/2026-09-30_goal.md)。


2026-10-03，Qwen3完整252投影的既有linear head/final norm方法补齐隔离周期导出，均通过真实552282/B4/1024/4步save2的两回调、回训及原生final。head训练映射按原生路径融合进原head，保持BF16形状与字节数；final norm对照只允许原model.norm.weight及decoder变化，冻结head/144其他norm/504码。短验final norm的4095个FP32元素更新，只有2个在BF16保存后保留，因此不能强制“短验BF16必须大量变化”来调高LR，也不能用FP32变化宣布部署改善。导出逐字节核所选FP32 norm转BF16，并分别记录存活量；全部3423保存/fresh、1910冻结及实际2.456988791bpw通过，有限题量仅验证流程。[原生顺序、精度差与完整验证](../records/qwen3_w25/2026-09-30_goal.md)的20:00与20:35记录。

同日已有类别组装对552282与bits05导出源的本地config比较因use_cache/pad_token_id差异受阻；真实v6 loader用共同base_model_path的原HF配置构建模型，两个字段属于保存的运行状态，不能因此误判投影架构变化。新增隔离入口仅接受bool cache与None/原HF tokenizer pad id，所有其余config及252部署spec严格相同，并实际核147个非投影张量相等；复用原source-union/保存fresh/账本，未修改仍被训练使用的旧入口。12项CPU配置拒绝/接受测试与两份实际3423张量组装及完整64/32评测通过。该白名单只适用这里已证明的加载语义，不授权宽泛忽略其他config或未经证明的架构差异；正确组装也没有使两候选超过来源最好。[运行字段来源、完整证据与质量边界](../records/qwen3_w25/2026-09-30_goal.md)的20:35记录。


2026-10-03的R0→R1零输出残支迁移再次显示结构与计算路径须分开：原3423张量不变、新增1008张量后，同native packed的真实七类完整权重/前向相同；原R0 wholefused对新R1 native仍有最大相对L2约0.0001313的差异。未训练完整均分差+0.00876个百分点不能当作容量优化收益。真实4步/save2、两次导出回训及4431张量native/fresh一致验证通过，正式runner库存必须取新实物而非写死旧3423；没有因此要求正常跨核浮点逐位相同。[迁移、初始完整基线与短验](../records/qwen3_w25/2026-09-30_goal.md)的22:58和23:24记录。

2026-10-04隔离H128R0→H256R0采用每stage均匀复制输入Linear/LN、输出[B,0]。均匀复制才能保持LN统计；输出不对称使隐藏副本首步梯度分化。CPU数学、梯度和现有v6 pack/unpack/strict重载通过；真实552282 L0 gate/down完整native BF16矩阵与原native相同，但原wholefused相对新native仍有约0.0001254相对L2差异。此helper替换Parameter对象，应在optimizer/hooks建立前调用。[隔离实现、证据与边界](../records/qwen3_w25/2026-09-30_goal.md)的10月4日00:16记录。

后续01:47–02:22已补齐同552来源、L0 gate/down两个模块的256步真实反传、独立BF16导出与完整八任务：H128/H256均分66.96068%/66.96508%，都低于源67.13033%；H256的MMLU57.94046%也低于H128的58.20396%。因此不能继续称“只有推理验证”，也没有证实扩宽带来质量收益。当时尚未验证全模型扩宽训练、best1200起点或其他更广范围/损失条件，不把该局部负结果推广为容量的普遍上限，预计位宽仍须与实际导出账本区分。来源见上述同一专题的后续扩宽对照。

同日16:19补齐best1200起点的全部108个MLP配对：同task50纯KL/prompt.3、decoder3e-6、原2000/warm20调度前400步，H128/H256完整均分68.51705410%/67.83285084%、MMLU60.66087452%/60.20509899%，均低于各自未训练起点。H256实际2.4615956182bpw，108模块真实更新、其他2775张量及504码库冻结，3423保存/fresh严格一致。两臂40个窗口的token计数相同，H256训练耗时1.4524倍；相同LR不等于相同函数空间步长，不能只归因容量，也不外推所有扩宽方案无效。本配方无收益依据，不延长，退役生成权重并保留原始证据。[完整配对、初始差与资源边界](../records/w25_transfer/2026-10-04_mmlu65_llama31.md#1619-起容量配对完整负结果及-head-1200)。

2026-10-12当前a0e5为symmetric H128、`num_res_blocks=0`，外置LoRA rank0。真实库存预测允许仅gate或仅down的36模块H256，两类一起超logical2.5bit；预测不等于新成品实测。旧widen helper强制asymmetric，不能直接用于symmetric源的单因素加宽；动态save/load可表达H256也不代表当前固定原shape/库存的teacher×scope导出器已支持。旧负结果只约束既有配方，不能证明本轮容量根因；当前四格完整评测继续，首张实际可用卡衔接的既定BIT诊断已完成，四格完整九项与BIT证据齐后，再选择数据或容量方向。[当前预算、实现边界与决定](../records/w25_transfer/2026-10-04_mmlu65_llama31.md#10月11日-train-only-八组完整结果与下一轮覆盖对照)。

## 生成参数须检查最终有效配置（2026-10-04）

本机 Transformers 4.51 的真实 Qwen3 LongBench 短测发现：仅传新建的 GenerationConfig，默认值 `do_sample=False` 仍可能被模型自身采样默认值替换。首次 dense 输出虽正常完成，协议不成立，已保留并排除；不能以配置对象的初值声明实际使用 greedy。

独立 LongBench 入口显式传 `use_model_defaults=False`，在真实 `_prepare_generation_config` 路径核对所有已指定选项，并在预测中记录最终有效配置。CPU 回归及 dense/压缩两模型各三个真实长输入、多步 KV、512-token 摘要和 SAMSum 停止检查均通过。固定双方 EOS、pad 与特殊任务 min_length，避免模型保存状态改变比较协议。该结论针对当前版本和入口，不等于其他 API 或所有版本都有相同行为；完整评测仍须全部样本完成后评分。[原始无效短测、修复及复测](../records/w25_transfer/2026-10-04_mmlu65_llama31.md)的 LongBench 部分。
