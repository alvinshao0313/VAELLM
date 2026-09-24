# KD 与数据边界：同名参数可能已有不同数学语义

## 问题与证据

早期prompt KD方案采用所有token的加权均值，后续改成响应区均值加权组合：L_response + w·L_prompt。两份历史审查都通过过各自版本，不能同时当成当前公式。[后续区域归一化审查](../history/reviews/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/final-review.md)

causal KD曾存在一位错位问题：位置t的logits对应t+1目标。硬件审查又发现Qwen3多轮ChatML无法依赖独立prefix tokenization定位真实响应边界。[mask修复计划](../history/plans/2026-08-04-fix-causal-kd-token-mask-alignment.md)、[硬件证据](../history/audits/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md)

## 下次怎么做

- 明确目标位置到logits的单次shift，最后位置、padding及prompt/response边界用具体token例子校验；避免双重shift。
- 改prompt权重时先写数学定义，确认是区域均值组合还是token加权均值。历史同名CLI不保证同样公式。
- CE+KD组合中CE只加入一次；hidden/pre-MLP的mask不能因KD改动而无意改变。
- dense/offload路径比较标量和学生梯度；空区域应有明确语义，不能用任意默认值掩盖问题。
- token统计按学生micro-batch计一次，不能因teacher、重计算或CPU staging重复计数；恢复后的首个日志窗口按实际观察步数计算。
- 历史错误mask训练得到的checkpoint不能当作“干净对照起点”；以新设计明确是否从同一原始模型重训。

## 边界

来源是历史计划、审查和有限硬件验证。本次未核验当前所有loss实现；CLI与支持的loss以当前源码及根README为准。审查通过不证明某个prompt权重能提高下游成绩。


## 2026-09-24 当前实现核查

当前 `train_utils/distill_loss_core.py` 的 prompt reduction 是所有有效token的加权均值 `(sum_response + w*sum_prompt)/(N_response + w*N_prompt)`，不是上文历史审查中的区域均值组合。`kl_top_partial` 是纯KD，`alpha`仅在 `kd*` 的CE/KD组合中生效；不能因命令含 `alpha=0.5` 就认为训练加入了真实答案CE。`kl_top_mass` 在teacher top-K之外加入tail桶，`kl_top`则是top-K条件分布。以上为当前源码核查，不是不同loss精度优劣的实测。[本轮设计与后续证据](../records/e2e_0910/2026-09-24_rank8_search.md)

当前task混合生成器为七任务取train、MMLU取auxiliary_train，评估按lm-eval使用validation/test，split设计隔离；这不等同于完成第三方语料的样本级去重。调整数据权重前区分样本占比和实际有效token贡献。本轮先固定loss/data，避免与学习率/可训练组件混改。
