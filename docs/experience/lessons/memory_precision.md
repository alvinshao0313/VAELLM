# 显存与精度：分清驻留对象、数值存储和验证范围

## 问题与证据

单模块FP32参数存储对照中，两组仍为BF16计算；代表形状的allocated仅增加0.33MiB，reserved增加24MiB。该值不能推算整模型峰值，也不能据实现通过宣称精度提升。[精度记录](../records/docs/distill_fp32_validation.md)

历史mid-eval OOM计划指出两条风险：no-cache推理仍物化dense解码scratch；直接unwrap模型时连AMP/autocast一起剥离。[排查方案](../history/plans/2026-09-07-e2e-mid-eval-oom-memory-lifecycle-fix-cn.md)。这份来源是计划，不能由其标题认定所有修复当前已验收。

## 下次怎么做

- FP32 decoder 存储与 BF16 激活并存时，推理缓存须显式使用实际激活 dtype；按参数 dtype 预热既可能浪费资源，也可能生成前向不能复用的缓存。2026-10-01 Qwen3 K48/H128 的 FP32 分组预热请求229376字节共享内存，超过A800的166912字节；指定BF16后252层通过，FP32参数字节不变。比较缓存正确性时保持原cache策略：关闭cache会优先走packed_u8，正常冷cache走fullfused，这两种计算路径不能混称同一native对照。本次同cache=True的分组/单层解码及前向完全一致，证据见[Qwen3专题](../records/qwen3_w25/2026-09-30_goal.md)的 `validation/fp32_eval_cache_v3.json`。

- 分别记录参数、梯度、optimizer moments和计算dtype；加载时先恢复存储精度，避免先低精度舍入再升回FP32。
- 2026-10-03核查当前pure CAT：`--bf16 true`同时选择BF16 VAE参数和autocast，普通AdamW的两个动量以`zeros_like(p)`初始化，没有FP32主参数副本；`--bf16 false --fp16 false`同时改变参数、梯度、动量、输入和主要计算精度，不能称为仅改变参数存储。BSQ内部部分计算原本就为FP32。现有BF16日志没有实际更新幅度或舍入比例，因此“BF16更新舍入限制精度”仍是假设。K24/D32的FP32存储/BF16计算packed-fused反向失败仍有效，但pure CAT训练走普通VAE前后向，不能把该失败直接外推为D16 FP32训练失败或通过。随后真实all36 down/D16K24/S2/H128R0/block64/B8192、每stage2步的FP32短验通过：四次更新参数/grad/Adam动量均FP32、非零梯度及真实更新；831张量保存/fresh严格字节一致，8个真实down的cache=True冷fused与BF16 group8预热前向完全相同且FP32参数未变，训练峰allocated约15.435GiB。该短验证据支持这一精度路径可运行，不能单独证明完整任务提升。正式匹配独立down每stage10K随后完成：保留a277其余216投影，只将现有BF16开关改为FP32，完整固定64/32八任务均分63.57159726659092%，低于matched BF16 a277的63.80757566115567%，差−0.2359784个百分点；实际2.4562006411345108bpw，3423张量组装保存与fresh strict核验通过。FP32最终训练日志中的反归一化残差RMS为0.006449492，BF16为0.006414791，未见该指标或下游收益；本次没有独立逆旋转并合并保护INT8后计算部署矩阵误差。该结果只限于本次独立down配方，不证明所有类别或所有FP32方案无效，也不能单独归因于参数舍入。[源码、真实验证及完整结果](../records/qwen3_w25/2026-09-30_goal.md)的validation/d16_down_fp32与reconstruction_d16_down_fp32。
- 同时报告allocated、reserved和外部GPU观测，注明context、工作区、激活与同卡作业影响。
- 训练能跑不代表中间评测/导出能跑；审计teacher targets、解码cache、scratch及wrapper的完整生命周期。
- offload或sequence chunk优化需要数值与学生梯度对照，保持原损失归一化；“转到CPU”不自动代表GPU峰值降了。
- 多进程队列需要超时和子进程存活检查；最后汇总前所有rank应参与collective。
- 测试矩阵明确CPU、单GPU、单rank、真实多GPU；不得把单rank torchrun或同卡layer_mp当成多卡证明。
- Triton启动应显式使用输入张量的CUDA设备；`fork_rng`只隔离随机数状态，不切换当前设备。2026-10-02四卡100%码位短测在校准第二张卡时触发`Pointer argument ... cannot be accessed from Triton`，张量本身均已在同一卡，不能误判为CPU张量或OOM。packed前向/反向、码位初始化/提交/梯度/Adam/RMS、单层及分组部署的12处launch增加device上下文后，真实双卡5项数值、状态与调用者设备恢复检查通过；随后全Qwen3四卡100%码位4步/save2完成两次独立导出、limited重载推理及回训；第二次反向传播及后续步最高allocated约72.06 GiB、reserved74.44 GiB。此资源证据限定于B4/max1024/FP32 sensitivity+Adam，不等于收敛或下游提升。证据与后续状态见[Qwen3专题](../records/qwen3_w25/2026-09-30_goal.md)。

## 边界

2026-10-04 五个真实 BF16 部署模型的只读区域诊断，teacher 提前留 GPU 时，首个 group8 decoder 预热在 allocated78.56GiB、额外申请1.47GiB处 OOM；该异常时占用不是单独测得的峰值。改成 teacher 初载 CPU，并在每个 student 加载/预热前回 CPU、预热后才回 GPU，同样 group8/BF16/数据/损失下五模型全部完成，最大实测 allocated70,587,747,328B，teacher 返回后常驻约36.93GB，没有逐模型累积。不能将 native 逐模块缓存的约50GiB训练证据外推到 batched group8 的临时解码峰值，也不能将此高 live allocation 当成仅 allocator 碎片。修复只涉及隔离诊断的设备驻留，不是分块损失或精度变更；旧失败证据保留。[完整驻留修复和五模型结果](../records/qwen3_w25/2026-09-30_goal.md)的03:03记录。

现有单模块数值、历史实现审查和硬件冒烟各自只支持限定范围。设计新资源预算时检查当前可用GPU及共享任务，不继承旧日志中的卡号授权。[历史硬件范围](../history/audits/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md)


2026-10-03扩展K24/D32和K56/D32/H128R0/S2的现有packed/fused数值检查：两者BF16参数路径的前向、梯度及group8均在未改阈值下通过；额外K24 FP32首层weight梯度相对差2.0167427e-5超过测试原1e-5，原失败保留。相同样本诊断的前向完全相同，将参考梯度直接舍入BF16的相对误差0.00166303约为实际差异82.46倍；这只排除了该样本出现BF16舍入量级退化的解释，未证明精确根因，也不将FP32失败改成通过。BF16纯CAT依赖只绑定其实际通过范围，不据此声称新宽度的FP32-KD或整模型恢复已验证；无需为额外dtype失败改变既定BF16训练条件。group8 helper重复同一decoder，不作8个独立checkpoint或冻结状态审计。完整证据、原阈值与未执行范围见[Qwen3专题](../records/qwen3_w25/2026-09-30_goal.md)的`validation/mixed_k24_k56_d32/`和BF16后续目录。


2026-10-03两份Qwen3类别快照的完整MMLU64诊断在log_softmax分配失败：allocated约37–46GiB，另有29–34GiB reserved但未用，单次请求约10–18GiB。两者BF16/strict加载、216或252模块预热均已成功，不能直接判断模型尺寸或batch本身放不下。实际诊断环境漏掉原成功流程的PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True；补齐完整成功环境（含该项及既有CUBLAS/PYTHONHASHSEED等设置）后，不改模型、batch64/32、样本或算法，两份完整八任务均成功，约449秒。表现符合碎片化，但此次同时补齐数项环境，不能宣称已单独消融证明allocator是唯一原因。复用显存证据应同时绑定实际启动环境，环境修复后直接完成原协议，不以降低batch代替。[失败日志、启动差异和完整运行证据](../records/qwen3_w25/2026-09-30_goal.md)的down_category_transition_d16_prefix。
