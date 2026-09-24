> 归档于 2026-09-23；类型：实验结果/验证/恢复记录；不是独立的通用经验。
> 原路径：`docs/exp_results/residual_lora_ab_20260923.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。归档后于 2026-09-24 核实并补充双隐状态损失实验结果；旧清理记录仍按原日期解读。
> 对应经验：[残差 LoRA](../../../lessons/residual_lora.md)。

# 残差 LoRA 端到端对照结果（2026-09-23—24）

## 实验条件和有效结果

远程工作区 `/home/shaoyuantian/program/VAELLM`，bitvae，4 张 A800（物理卡 4–7）。同一初始 checkpoint `result/catlora/Qwen_Qwen3-8B_20260910_094022/final_model`，checkpoint_id `01457bf3-ef22-49e8-847f-dc721287c2d6`，Qwen3-8B；普通 LoRA rank8/alpha16/dropout0.1，norm all、lm_head LoRA；global batch32，seed/data_seed0，原数据混合和 kl_top_partial 损失不变。实际参数保留在实验目录的 `normalized_e2e_runtime_args.json`，运行脚本保留；原 comparison_manifest 的源码哈希已归并到本文。

新增 additive 残差 LoRA：rank8、alpha16、dropout0、lr1e-4、FP32 参数存储，新增 4,718,592 参数；本组两项隐状态损失权重均为0。

八任务全量、0-shot、eval_limit=None；平均分为八个任务分数的算术平均，分数用百分数、差值用百分点。基线使用用户指定的 `only_lora/Qwen_Qwen3-8B_20260922_024034`，没有重跑。

| 步数 | 普通 LoRA (%) | 加残差 LoRA (%) | 差值（百分点） |
|---|---:|---:|---:|
| 1000 | 66.2803 | 66.5697 | +0.2894 |
| 2000 | 66.4493 | 66.5645 | +0.1152 |

残差组在1000至2000步的均分变化为 -0.0052 个百分点：单次实验有小幅优势，未随训练持续扩大，不能宣称稳定显著提升，更不能外推到69+。原基线在1000步恢复过，full_determinism=false；这是历史同配方对照，不是逐位复现或多种子结论。

最终2001步导出模型均分为 66.4564%。final 与 step2000 的训练步数及评估流程不同；未单独量化多一步更新和导出的影响，不能直接把分差归因为导出误差。用户明确要求精简后，已核实后续实验从原始ckpt重训、不使用本组权重；本组训练checkpoint及导出模型在指标归档后删除。保留原始指标，不再支持直接从本组续训或直接重载复评；如需重新验证，应按保留脚本重训。

去留决定（用户，2026-09-23）：本条不含隐状态对齐损失的残差LoRA分支没有取得值得保留权重、继续投入的结果，距均分69+目标仍有明显差距；用户决定只总结经验并保留必要证据，删除生成权重。这是实验投入和保留策略，不是对其理论上限的证明。双隐状态损失配置作为独立实验判断，其已完成结果见下节。

## 双隐状态对齐 0.1/0.1（2026-09-24 核查）

**状态：已完成，退出码 0，global_step=2001。** 启动于 2026-09-23 08:31:19 UTC，结束于 20:17:40 UTC（北京时间 2026-09-24 04:17:40）。核查时无本实验活动进程。最终导出 checkpoint_id 为 `e0dcc977-dab4-4a2f-936b-8dc7181965f3`。运行标识为 `residual_lora_hidden01_20260923/Qwen_Qwen3-8B_20260923_083129`。

设计假设是：在残差 LoRA 基础上，加入层输出与 attention 残差连接后、MLP 归一化前的隐状态对齐，可补足仅有输出蒸馏的约束。参考此前“小幅 loss 改善不代表下游提升”和“实际 LR 轨迹要对齐”的经验，固定初始 ckpt、rank、数据、评测和 LR，仅将 `hidden_loss_weight`、`pre_mlp_hidden_loss_weight` 从 0 改为各 0.1，保留 `adaptive_top_3`。没有单独消融两个损失，也没有普通 LoRA 加相同 hidden loss 的组，不能归因于某一项损失或残差模块与对齐损失的交互。

实际配置与残差单独组逐字段比较：除这两个损失权重外，仅输出/日志路径改变。seed/data_seed=0，seq_len=1024，bf16 计算，LoRA/norm/lm_head/residual_lora 为 FP32。数据为 `edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_eval_task=0.5`；沿用原数据文件，没有新增数据版本哈希。压缩位宽与原始 ckpt 一致，本轮未重新核算含码本等开销的整体 bpw 或压缩率；残差模块新增的 4,718,592 参数是额外开销。

| 评估点 | 普通 LoRA (%) | 残差 LoRA (%) | 残差 + hidden 0.1/0.1 (%) | 相对普通（百分点） | 相对残差（百分点） |
|---|---:|---:|---:|---:|---:|
| 1000 | 66.2803 | 66.5697 | 66.5816 | +0.3013 | +0.0119 |
| 2000 | 66.4493 | 66.5645 | 66.9000 | +0.4508 | +0.3355 |
| final（2001，导出） | — | 66.4564 | 66.8794 | — | +0.4230 |

2000 步逐任务结果，均沿用原始 JSON 的任务指标：BoolQ/RTE/WinoGrande/MMLU 用 acc，其余四项用 acc_norm。

| 任务 | 普通 LoRA (%) | hidden 0.1/0.1 (%) | 差值（百分点） |
|---|---:|---:|---:|
| BoolQ | 84.7401 | 84.9847 | +0.2446 |
| RTE | 72.2022 | 76.5343 | +4.3321 |
| WinoGrande | 67.1665 | 67.1665 | 0.0000 |
| ARC-Easy | 77.3569 | 77.0623 | -0.2946 |
| ARC-Challenge | 50.8532 | 50.7679 | -0.0853 |
| OpenBookQA | 39.4000 | 39.2000 | -0.2000 |
| PIQA | 77.1491 | 76.8226 | -0.3264 |
| MMLU | 62.7261 | 62.6620 | -0.0641 |

**结论：单次实测有小幅均分收益，未形成广泛提升，未达到 69。** RTE 对八任务均分贡献 +0.5415 个百分点，已超过总收益 +0.4508；其余七任务均分反而下降 0.1037 个百分点；全部八项中 2 项提升、1 项持平、5 项下降。1000→2000 步本组仍提高 0.3185 个百分点，因此不能说已经完全收敛，也不能据此证明方法上限低于 69；当前观察到的收益集中且目标差距仍为 2.1000 分。

实际 1000/2000 步 LR 为 `9.262777704622938e-5` / `6.822540996484631e-5`，与原轨迹一致；不是因将 max_steps 设置成 2001 而在末尾退火至零。Trainer 报告时间由残差单独组的 17,260.4199 秒增加到 41,867.3715 秒，约 4.79→11.63 小时、2.43 倍；此口径包含中途评估等开销，是历史运行的实际耗时比，不是隔离测得的纯训练算子开销。

**去留：本次配置结束，不自动追加训练、调 LR 或保留用于假想续训的权重。** 小幅且集中在 RTE 的收益不足以支撑继续用当前配方重投入冲击 69；没有安排具体续训/复评，也未被指定为交付模型。按项目持续授权保留最小证据并清理生成权重。这是基于当前收益和成本的实验取舍，不是已证明所有残差 LoRA 或 hidden 对齐设置均无效。若重启该方向，须有改变条件或新依据，不能照抄本配方继续试。

经验增量已并入 [残差 LoRA 经验](../../../lessons/residual_lora.md)：检查均分收益由哪些任务贡献，分开解释新增损失的收益与运行代价，保留单种子、组合损失和未证明上限的边界。该节替代此前“hidden 0.1/0.1 尚无质量结论”的状态。

原始证据：

- [实际配置](../../../../../result/compressed_e2e_fintuning/residual_lora_hidden01_20260923/Qwen_Qwen3-8B_20260923_083129/normalized_e2e_runtime_args.json) 与 [核心日志](../../../../../result/compressed_e2e_fintuning/residual_lora_hidden01_20260923/Qwen_Qwen3-8B_20260923_083129/compressed_e2e_fintuning.log)。
- [1000 步指标](../../../../../result/compressed_e2e_fintuning/residual_lora_hidden01_20260923/Qwen_Qwen3-8B_20260923_083129/lm_eval/lm_eval_results_step_1000.json)、[2000 步指标](../../../../../result/compressed_e2e_fintuning/residual_lora_hidden01_20260923/Qwen_Qwen3-8B_20260923_083129/lm_eval/lm_eval_results_step_2000.json)、[最终导出指标](../../../../../result/compressed_e2e_fintuning/residual_lora_hidden01_20260923/Qwen_Qwen3-8B_20260923_083129/lm_eval/lm_eval_results_final.json)。
- 在远程项目目录、已激活 bitvae 的 shell 中执行 `bash experiments/residual_lora_ab/only_lora_hidden01_20260923/run_additive_hidden01.sh` 可重训此配置；原脚本保留。该命令历史使用物理卡 4–7，未来复跑仍须核对资源并使用新的独立输出根目录。

## 对照过程记录

1. **先对齐用户指定基线。** 40步、limit16的smoke_B曾被误选为对照，其损失、dropout和预热也不同。该次新冒烟在加载阶段被停止，不提供有效收益证据。复用已有普通LoRA结果，不重复占卡重训基线。
2. **实际LR比max_steps名字可靠。** 本组停止于2001步，但设置warmup_steps=150、num_cycles=0.19082474226804125，即 `0.5*(2001-150)/(5000-150)`，保留原5000步cosine前段。实测200个日志点与基线的最大LR差为1.36e-20；1000步9.2627777e-5、2000步6.8225410e-5，未提前退火到零。2000步只是对照点，不是收敛上限。
3. **loss下降不能替代下游评估。** 前560步平均loss仅相对下降0.036%，最近100步下降0.077%；后来精度优势也只有零点几分。保留蒸馏分量和同一评估口径，避免根据单步波动下结论。此前100–560步实测每步约7.72s→8.20s，增加约6%；不代表其他损失配置的开销。
4. **正式评估要在准确步数进行。** 当前回调在global_step>=max_steps时跳过中途评估，因此使用2001步上限获取step2000的训练模型评估。不能将最终导出或不同步数指标混标成同一步结果。
5. **新增损失后重新解释总loss。** 后续独立实验 `residual_lora_hidden01_20260923` 仅把hidden_loss_weight和pre_mlp_hidden_loss_weight各改为0.1，保留adaptive_top_3及原学习率，从同一初始ckpt重训。总loss多了两项，不能和本组总loss直接比较；该实验现已完成，质量结论和收尾决定见上节；早期总 loss 变化不能代替完整评估。

## 复现入口与保留记录

- 普通基线：`result/compressed_e2e_fintuning/only_lora/Qwen_Qwen3-8B_20260922_024034/`。
- 残差对照：[脚本](../../../../../experiments/residual_lora_ab/only_lora_20260923/run_additive.sh)，结果 `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/`；仅保留核心训练日志、实际参数和1000/2000/final三个原始指标JSON；无后续用途的权重、启动文件、预检记录、重复进度日志及重复指标表已删除。
- 双隐状态损失：[脚本](../../../../../experiments/residual_lora_ab/only_lora_hidden01_20260923/run_additive_hidden01.sh)，结果 `result/compressed_e2e_fintuning/residual_lora_hidden01_20260923/`。
- 模块语义及历史单元测试：[实现验证记录](residual_lora_validation.md)。本次整理没有重跑训练/单元测试，不改变训练代码或学习率。

## 实现版本

以下SHA256归并自原comparison_manifest，便于核对实验代码版本；不再保留单独的重复参数/清单文件。

| 文件 | SHA256 |
|---|---|
| `compressed_e2e_fintuning/trainer.py` | `5eec2fc69c88a77ebabd542948220b2bab6c50761f481fb215f9b41b6b4c0677` |
| `compressed_e2e_fintuning/runtime_v6_pipeline.py` | `8a63862e7455536d4e7a3700d617bb604839a7ac141f075f6abe7cbbdab5c8ed` |
| `e2e_common/residual_lora.py` | `4c9ee7121c1017f3a127ec26fd376581bd75880b429ba4e1057640b4d629245a` |
| `e2e_common/full_lora.py` | `75aa2a1d8370720f477c2a79ada8b0db92cc9a91467e909357288842abbac1a8` |
| `train_utils/model_level_trainables.py` | `a99e3613fb99a4033bdb68ebb9c6028914da5fb4ec45faff2810b7d38fcd01b6` |
| `train_utils/model_level_optimizer.py` | `851587034200245483433c4d04224f0f2e2c7412503205b62f4cf8ace327c626` |
| `train_utils/config/cli.py` | `c09f90f24a4119635b414fc46f40644dfbfb74a7df8458ee04e9dbfa5285f08e` |
| `train_utils/config/configs.py` | `026bb192a99dda4c683a39ae04419e750d69fb60b4643b57ac384ae4d1c54bb5` |
| `compressed_e2e_fintuning/mid_eval.py` | `4f2a13400ddbefc86456d730484d0d3ee8833265e427d043b829179a41b81a4a` |
| `compressed_e2e_fintuning/args.py` | `870622599240e6598c30324cd7c9d87eedaebe8b91af002d4ff9598dc3d82fbf` |

双隐状态实验启动时的 manifest 记录上述 10 个文件哈希一致；另记录 `train_utils/lora_training.py` SHA256 为 `855ac9182a098954544129ce0607d454b0594f00d354c6b538bad8b1a2c46e45`。本轮未改训练源码。

## 清理记录（2026-09-23）

用户要求已结束实验只留必要结果。上一轮只删除4项小文件并压缩日志，清理力度不足；本轮核实没有任何当前任务使用本组checkpoint，删除3个训练checkpoint及导出模型。新隐状态损失实验仍从初始catlora模型重训，不受影响。

状态：清理完成；5个保留结果文件哈希未变，初始ckpt与当前实验配置未变。

| 删除目标（项目内相对路径） | 原字节数 | 文件数 |
|---|---:|---:|
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/Qwen_Qwen3-8B_20260923_032315/trainer_state` | 1015425981 | 27 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/Qwen_Qwen3-8B_20260923_032315/final_model` | 4824405224 | 9 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/preflight.log` | 608 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/run.log.gz` | 92584 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/README.md` | 576 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/argv.json` | 3291 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/preflight.exit` | 2 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/finished_at.txt` | 26 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/launcher.pid` | 8 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/baseline_original_snapshot.json` | 10867 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/run.exit` | 2 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/comparison_manifest.json` | 2776 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/started_at.txt` | 26 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/Qwen_Qwen3-8B_20260923_032315/run_meta.json` | 2749 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/Qwen_Qwen3-8B_20260923_032315/lm_eval/lm_eval_summary_step_1000.md` | 318 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/Qwen_Qwen3-8B_20260923_032315/lm_eval/lm_eval_summary_step_2000.md` | 318 | 1 |
| `result/compressed_e2e_fintuning/residual_lora_only_lora_ab_20260923/Qwen_Qwen3-8B_20260923_032315/lm_eval/lm_eval_summary_final.md` | 318 | 1 |
| `docs/exp_results/residual_lora_cleanup_20260923.json` | 2999 | 1 |

本轮目标合计 5,839,948,673 字节（5.439 GiB），原分配空间 5,840,105,472 字节；不以共享磁盘总量变化估算释放量。上一轮清理记录的要点归并到本文，单独清单JSON已纳入本轮删除。

最终实验目录只保留5个文件：一份核心训练日志、一份实际参数、三个原始评估JSON。经验文档和复现脚本在源码目录保留。初始ckpt、用户指定的普通LoRA基线记录、其他任务和正在运行的隐状态实验未纳入本轮删除。本次没有改训练代码或运行训练/单元测试。

## 双隐状态实验收尾（2026-09-24）

结果、退出码、起止时间、配置差异与源码哈希已归并到本文。复现脚本 SHA256：`fc76a79c594153515f6b3f8ba84cf37631fabd014f7b3f4d895109290d6c17d9`。删除前确认：退出码 0，无本实验活动命令或同用户进程打开本输出目录的文件；脚本/文档引用只用于复现与记录，没有具体续训/复评使用者。所有目标解析后均在本项目的本实验输出根目录内，未发现符号链接。训练状态与导出权重无具体保留用途；其余启动、预检、重复配置、进度日志、指标 Markdown 的必要信息已归并，原始指标与核心日志保留。

以下目标均相对于 `result/compressed_e2e_fintuning/residual_lora_hidden01_20260923/`，删除已完成并核实：

| 删除目标 | 原字节数 | 文件数 |
|---|---:|---:|
| `Qwen_Qwen3-8B_20260923_083129/trainer_state` | 1015472091 | 27 |
| `Qwen_Qwen3-8B_20260923_083129/final_model` | 4824405228 | 9 |
| `README.md` | 666 | 1 |
| `argv.json` | 3287 | 1 |
| `comparison_manifest.json` | 3545 | 1 |
| `finished_at.txt` | 26 | 1 |
| `launcher.pid` | 8 | 1 |
| `preflight.exit` | 2 | 1 |
| `preflight.log` | 908 | 1 |
| `preflight_config.json` | 6339 | 1 |
| `run.exit` | 2 | 1 |
| `run.log` | 765365 | 1 |
| `started_at.txt` | 26 | 1 |
| `Qwen_Qwen3-8B_20260923_083129/run_meta.json` | 2750 | 1 |
| `Qwen_Qwen3-8B_20260923_083129/lm_eval/lm_eval_summary_step_1000.md` | 318 | 1 |
| `Qwen_Qwen3-8B_20260923_083129/lm_eval/lm_eval_summary_step_2000.md` | 318 | 1 |
| `Qwen_Qwen3-8B_20260923_083129/lm_eval/lm_eval_summary_final.md` | 318 | 1 |

本轮合计 5,840,661,197 字节（5.4395 GiB），文件原分配空间 5,840,822,272 字节。仅统计本轮确认存在的目标，不将共享磁盘变化或先前清理计入本轮。


清理验证通过：本实验只剩 5 个结果文件（配置、核心日志、3 份原始指标），内容哈希均未变；原始 ckpt 的元数据/配置、两个对照的最小结果及复现脚本哈希未变。未修改训练代码、没有新训练或 GPU 测试；文档索引与链接检查单独执行。生成权重已删除，不能直接续训或重载复评；仍可从受保护的初始 ckpt 按脚本重训。
