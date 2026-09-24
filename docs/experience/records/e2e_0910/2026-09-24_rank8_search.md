# 0910 checkpoint 的 rank8 端到端微调搜索（2026-09-24）

## 目标与状态

目标：初始 checkpoint 固定为 `result/catlora/Qwen_Qwen3-8B_20260910_094022/final_model`（id `01457bf3-ef22-49e8-847f-dc721287c2d6`），LoRA rank≤8，在既有八任务全量、0-shot 口径下达到均分≥69%。使用用户本轮授权的物理 GPU 4–7；不使用或停止 0–3 卡的其他任务。

运行状态：配置已准备，阶段续训功能与最小正式形状验证进行中，尚未启动正式筛选。没有声称已达到目标。成功首先以真实生成模型的全量评估为准；若达到69，再复核保存重载与同口径结果，并视余量做第二种子确认，区分单次达标与稳定收益。

## 参考证据与本次差异

- [对照设计](../../lessons/experiment_design.md)：400步只用于筛选，不证明上限；维持同一学习率轨迹。
- [残差与双隐状态结果](../docs/exp_results/residual_lora_ab_20260923.md)及[经验](../../lessons/residual_lora.md)：普通 LoRA 2000步66.4493，残差66.5645，双hidden66.9000；后者收益集中RTE、耗时2.43倍。本轮不继续叠加残差/hidden。
- [KD与数据边界](../../lessons/kd_and_data.md)：以当前源码核对公式，不照抄历史同名参数的含义。
- 0910 初始八任务均分65.42%，证据为原 CAT `linear_by_category.log` 3697–3716行；与目标差约3.58个百分点。普通LoRA仅恢复约1.03个百分点，不能用其他初始模型或仅压down_proj的69.32作达标依据。
- 旧 `decoder_lora_rank8_20260922` 的 step200 均分66.35来自 `eval_limit=256`，400步训练完成但导出结构parity失败，无有效全量终态指标。其普通LoRA LR也是3e-6（本基线的1/33），同时改变dropout、loss、K、prompt、norm/head，且400步cosine末端接近零；不能据此否定decoder联合更新。没有top1000优于top100的受控实测。
- 本次重新尝试decoder的依据：恢复正常强度的LoRA、保留norm/head，decoder独立小LR，decoder参数以FP32存储、BF16计算。不是重复旧失败配方。

## 首轮：更新强度 × decoder 联合训练

所有组：普通LoRA rank8/alpha16/dropout0.1，全部252压缩projection、0–35层；norm all与lm_head LoRA均LR1e-4；残差none、两种hidden权重0。固定 `kl_top_partial`、K100、T1、prompt0.3、原数据混合、seq1024、seed/data_seed0。

| 配置 | 普通LoRA LR | decoder | GPU | 首段预算 |
|---|---:|---:|---|---:|
| A `lora_lr1e4` | 1e-4 | 冻结 | 4,5 | 400步 |
| B `lora_lr3e4` | 3e-4 | 冻结 | 6,7 | 400步 |
| C `decoder_lora_lr1e4` | 1e-4 | LR3e-6，FP32参数 | 4,5 | 400步 |
| D `decoder_lora_lr3e4` | 3e-4 | LR3e-6，FP32参数 | 6,7 | 400步 |

每组2卡DP、每卡batch8、梯度累积2，有效batch32。实际数据为iterable，group_by_length在运行时关闭。同四个microbatch时两卡acc2与原四卡acc1的loss归约均为四个microbatch标量均值；样本分片、dropout流与历史不同，因此以本轮A/B/C/D为主对照，旧结果只作参考。四组统一worker/seed/样本预算，不改数据以图省时。

固定 `steps=5000`、warmup_steps150、cosine num_cycles0.5；`stop_after_step=400`仅限制本阶段执行步数，不改变scheduler终点。每400步保存并执行原八任务全量评估，随后正常停止，保留训练checkpoint供明确的晋级用途；不重复导出大模型、重复评估同一步。不得将`steps=400`训练再换总步数冒充原轨迹续训。

4–5卡依次执行A、C；6–7卡依次执行B、D，前一配置成功完成才启动后一配置，错误立即停止该串行命令，不自动重试。两组快实验先跑、两组慢实验随后跑，避免人为等待另一组的额外波次屏障。正式脚本在 `experiments/e2e_0910_search/run_*.sh`，调用前激活bitvae，后台运行。

## 晋级规则与后续顺序

1. **400步：** 看八任务全量均分、各任务差值、实际GPU时间及是否稳定。训练loss用于排除数值异常，不代替任务精度。保留最高分候选与有质量/成本优势的候选，最多2组进入1200步；微小分差不能宣称显著优劣，400步仍在上升的候选不标为已到上限。
2. **1200步：** 在相同卡数、batch、seed、总调度及评估节奏下从原checkpoint续训。结合400/800/1200轨迹选主线；明显被质量与成本同时支配的配方停止并收尾。
3. **2000步及以后：** 优先把主线推进2000步，仍持续增益则按预算推进3200/5000，不能仅因还没69就立刻淘汰。达到69也先检查可导出/重载与全量指标，不拿训练内模型分数代替最终交付状态。
4. **若更新强度/decoder搜索仍平台：** 固定胜出优化器配置，受控比较 `kl_top_partial` 与 `kl_top_mass`（K先固定100），再考虑`kd_top_mass`的轻量真实答案监督（如alpha0.9）。必要时再比较K100/1000。每轮只回答一个问题，不同时更换loss、数据、LR和训练模块。目标函数/采样的具体变更须按项目边界明确后执行，首轮不包含这些变更。
5. 数据比例最后再动。`vaellm_eval_task=0.5`是样本比例，不代表token损失贡献50%；当前七任务训练split和MMLU auxiliary_train与评估split设计隔离，但未宣称完成第三方语料的样本级去重。

当前`kl_top_partial`是纯KD，`alpha=0.5`不产生CE。prompt按 `(sum_response + w*sum_prompt)/(N_response+w*N_prompt)` 归约。partial未包含尾部项，mass包含tail桶；是否更有效必须实验，不能由公式直接断言精度提升。

## 验证与成本

阶段停止必须覆盖保存→全量评估调用链→恢复训练，以及最终导出。检查发现原mid-eval会改变随机状态，而训练checkpoint保存于评估前；当前修复将评估后的各rank RNG写回同一checkpoint，保留连续训练已有语义。阶段预算不进入数学resume契约；总steps、loss/data、world/batch、调度仍严格锁定。

短测采用真实0910全模型、2卡DP、batch8×acc2、seq1024、decoder+LoRA、FP32参数/BF16计算；仅压缩步数、保存触发间隔和短测eval题数。连续训练与暂停/恢复均经过评估，检查后续参数/optimizer/scheduler及数据进度；最后验证导出路径。短测题数限制不带入正式筛选，短测结果不参与精度排名。

历史普通LoRA四卡约7.7s/步、decoder四卡约18.4s/步；两卡acc2按工作量粗估约15–16/37s每步。每400步约1.7/4.1小时纯训练，两条并行串行链首轮估计约6–7小时（含评估/加载）。这些是启动前估算，新FP32 decoder与并行累积的实际耗时以短测和正式日志为准，不承诺几百步到69。

## 产物与收尾

新结果根目录：`result/compressed_e2e_fintuning/e2e_0910_search_20260924/`。旧结果不覆盖，初始模型不修改。

首段训练checkpoint的具体用途是400→1200→2000的候选晋级，排名前暂时保留；淘汰后只留配置、核心日志、原始指标和脚本，自动删除权重/optimizer/RNG/重复运行文件。赢家仅保留明确续训所需的最新状态及需交付的最终模型。短测结束后先归并验证证据，再清理全部短测模型与一次性脚本。每轮沿用本文更新状态与结果，不按每步另建报告。

本轮源码版本、实际验证、后台PID与结果将在实际完成后补充；当前没有新精度结论。
