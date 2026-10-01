# 残差 LoRA：初始化、训练开关和恢复语义分别验证

## 已遇问题与证据

additive 为 x+D(x)，B=0 时保持初始输出；replace 为 D(x)，正交初始化得到 rank-r 投影，并非恒等映射。已有残差 checkpoint 的 none 模式冻结新增残差参数，但不删除其推理计算。[实现验证](../records/docs/exp_results/residual_lora_validation.md)

无 hidden 对齐的 rank8 残差对照在1000/2000步分别较普通LoRA高0.2894/0.1152个百分点，未形成持续扩大收益；用户已决定结束该分支并清理无用途权重。[结果与去留](../records/docs/exp_results/residual_lora_ab_20260923.md)

## 下次怎么做

- 比较 additive/replace 前先测初始化后的基线；不能把投影改变信息通路当成学习收益。
- 先构造 PEFT，再确认 decoder/残差参数的 requires_grad、optimizer 分组及真实更新；模块挂上不代表在训练。
- 保存模式、拓扑、权重及 exact-resume 配置；向无残差 checkpoint 首次加模块应新开训练，不沿用旧 step 状态。
- CLI 非法值可能由 argparse 抛 SystemExit(2)，测试应校验对外协议；skip_layers 空值按实际类型契约传入，不能放宽正式解析来迁就错误 fixture。
- 优先查已有验证和结果，再设计新的 hidden-loss 或初始化消融；相同损失项/数据/预算下比较，不能只读总 loss。

## 边界与当前接口

测试验证公式、更新和保存链路，不证明多卡速度或下游提升。历史记录中的 residual_lora_enabled 与“默认 additive”已被后续统一模式替代；当前接口以[模块说明](../../guides/residual_lora.md)和源码为准。hidden0.1/0.1 为独立配置，已于 2026-09-24 核查完成；其结论见下节，不能与无 hidden 损失的结果混为同一实验。


## 隐状态对齐的小幅均分收益需要拆到任务（2026-09-24）

**证据状态：本配置单次实测支持有限均分收益；未证明统计显著性或方法上限。** Qwen3-8B、同一压缩初始 ckpt、rank8 additive 残差、seed0、全量八任务 0-shot、同一 5000 步 cosine 前段，在两种 hidden 对齐权重各 0.1、adaptive_top_3、2000 步时，比普通 LoRA 高约 0.45 个百分点，比残差单独组高约 0.34 个百分点。RTE 的贡献超过整体增益，其余七任务均分略降；实际 Trainer 耗时约为残差单独组的 2.43 倍，包含中途评测等开销。[完整结果、条件与清理决定](../records/docs/exp_results/residual_lora_ab_20260923.md)

已确认的是收益集中在个别任务，并没有形成普遍恢复；不能把均分小幅提高解释为模型整体能力已明显改善。两种对齐损失同时开启，且没有普通 LoRA + 相同对齐损失组，所以具体是哪项损失有效、残差模块是否必要、两者是否存在协同，均仍未验证。不要将这些解释写成根因。

下次动作：均分必须同时检查任务贡献与成本；新增辅助损失后分开读取蒸馏与对齐分量，不能跨目标比较总 loss。当前配方无继续重投入和保留权重的具体用途，已按项目授权收尾；若要重启，先提出新的受控变量或证据。1000→2000 步仍有增益，且末端 LR 未归零，故本实验既不能证明已到上限，也不支持外推到 69+。以上判断仅适用于此次模型、损失权重、数据与训练预算，不否定所有 hidden loss 或残差 LoRA 设置。

## 首次安装 replace 的发散排查（2026-09-24）

**证据状态：残差模式对照已补充；共同发散的独立根因仍未确认。** 0920 checkpoint 本来没有残差 LoRA，新装 `replace` 将 36 层共 72 处恒等 skip 改为 rank8 投影；这不是保持初始函数的普通 LoRA 微调。前两次有效配置只改了 `kd_top_partial → kl_top_partial`，仍发生同类发散。随后additive/none两组前110步恢复到loss约0.4–0.5，却都在120–130步失稳：replace能解释初始劣化，不能作为共同发散的必要原因；none组也不能归因于残差参数的BF16存储。[配置、日志和代码证据](../records/docs/exp_results/residual_lora_replace_instability_20260924.md)

本配置同时解冻decoder；其stage归一化已融合进输出层参数，实读output weight RMS约0.002–0.003，绝对LR不能直接与普通LoRA比较。同日18:26核查，仅将decoder_lr从3e-5降至1e-5的none残差对照已稳定跨过原120–150步失稳段，至230步loss0.4407、grad_norm15.70，支持原decoder更新强度参与早期失稳；单seed短段证据不证明完整训练稳定，也不能据此断言唯一根因。

`train_mode=lora`只冻结decoder等主训练组件，不禁用aux配置里的残差LoRA。随后`181434`同时冻结decoder并将none改回replace，第10步loss32.5，至80步下降到17.19；这是高起点下降的观察，不是“冻结decoder仍在相同条件下突然发散”的证据。检验decoder作用时保持残差none，不能一次同时改两者，也不能用该loss数值直接认定公式有误。

读取当前 E2E 日志时，`grad_norm` 是裁剪前范数；总 `loss` 是日志窗口及多卡平均，分项则来自 rank0 最后一次 loss 计算，不能逐行加权相加验证总数。hidden/pre-MLP 为相对教师能量归一化的误差，爆涨证明相对误差严重异常，但没有逐层张量和梯度记录时，不能据此认定最先失稳的层或参数组。

**后续证据（2026-10-01）。** 同一0920初始模型的普通rank8 LoRA搜索已完成，采用decoder冻结、residual none和head linear，12组完成5000步，最佳2500步模型独立严格重载后均分68.024489%。这支持该具体训练配方可执行；相对前述诊断还改变了head等条件，不能把稳定性改善全部归因于冻结decoder，也不据此否定所有decoder联合优化。[当前配置、正式指标与限制](../records/e2e_0920/2026-09-24_rank8_search.md)
