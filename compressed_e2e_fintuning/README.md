# Compressed E2E v6

正式入口是：

```bash
python -m compressed_e2e_fintuning.main \
  --student_checkpoint_dir /path/to/v6/final_model \
  --dataset_mix "openorca=1.0" \
  --train_mode decoder
```

入口只调用 `runtime_v6`。输入必须是完整 v6 `round_base`、`category_boundary` 或 `final_model`；legacy checkpoint 必须先用 `tools/migrate_checkpoint_v6.py` 转换。

## 训练模式

`--train_mode` 由 `decoder`、`lora`、`sparse_bit` 三种组件组合：

- `none`
- `decoder`
- `lora`
- `sparse_bit`
- `decoder_lora`
- `decoder_sparse_bit`
- `lora_sparse_bit`
- `decoder_lora_sparse_bit`

模型级 LoRA 只有 plain full-space 实现，参数为 `--lora_rank`、`--lora_alpha`、`--lora_dropout`。不支持 DoRA、RSLoRA、AdaLoRA 或 compressed-subspace LoRA。

`--target_layers` 接受 `all`、范围或显式层号；`--target_modules` 接受 `all` 或完整 projection 名集合。两者共同限定训练目标，普通未压缩 Linear 不会被当作 VAELinear decoder 目标。

## Sparse Bit 敏感度代理坐标

`--bit_proxy_coordinates` 默认 `unit`，保持原来的 FP16 `±1` score、零阈值，以及 `bit_lr=auto` 对应的 RMS-SGD `0.05`、Adam/AdamW `0.02`。原模式本来就可以翻码；新选项是按 decoder 敏感度调整训练坐标，并非修复原模式不能翻码。

在已配置的数据、损失及运行命令中，可以显式加入：

```bash
--train_mode decoder_sparse_bit \
--bit_proxy_coordinates decoder_sensitivity \
--bit_lr 2e-5
```

同样支持 `sparse_bit`、`lora_sparse_bit` 和 `decoder_lora_sparse_bit`。这里的 `2e-5` 仅为机制验证的起点，不是推荐最优学习率或正式实验配置；新模式要求显式正数 `--bit_lr`，拒绝 `auto`，因为原坐标的默认步长不能直接沿用。`bit_active_ratio`、`bit_round_steps` 和优化器选择保持原含义，每轮更新预算仍会影响是否发生翻码。

每个编码 bank（一个 module/stage/part）在训练开始时均匀选取最多 256 行，逐个翻转全部 latent bit，使用完整 decoder 的 eval 输出测量权重变化 RMS，得到固定尺度 `s`。该测量与 LiftQuant 恢复共用 `litebsq.bit_sensitivity.measure_scale`。代理值采用 FP32 `p=s×(b−0.5)`，初始翻转距离为 `s/2`，前向仍使用硬 `0/1` bit，STE 梯度按 `1/s` 缩放，优化后将 `p` 限制在 `[-s/2,s/2]`。`s` 在整个训练期间固定，换轮不重算。Sparse Bit 保留原有稀疏采样、换轮、优化器和坐标限幅流程，不能把它视为完整复现 LiftQuant 的训练方法。

E2E 入口根据 AMP 精度确定校准计算 dtype；没有 AMP 时使用输入 embedding dtype，并遵守模块已设置的 decoder 计算精度。直接调用 `SparseBitTuningManager`、且 decoder 参数与输入计算 dtype 不同时，应显式传入 `calibration_dtype`，避免按参数 dtype 校准。

相比 `unit`，FP32 score 和梯度各多 2 字节，每个 active bit 合计多 4 字节；Adam/AdamW 动量原本就是 FP32，不因此增加。另有每 bank 固定尺度和少量运行元数据。未激活 bit 仍保持 packed 存储。

坐标模式进入参数快照；启用敏感度坐标的 Sparse Bit 训练也将其写入严格恢复契约。训练步断点保存 FP32 score、固定 `s`、采样及优化器状态，exact-resume 使用保存的 `s`，不按已更新的 decoder 重算。不能在同一个训练步断点中切换坐标模式或学习率；如需调整，应从完整阶段/最终模型开启新实验。`unit` 及非 Sparse Bit 模式的原恢复契约保持不变。

最终导出前将代理提交为原生 packed bits，并移除 score 和尺度；模型仍只保留原有 packed 编码、decoder 及所选训练组件，不增加此功能的推理开销。该机制可帮助编码跨过阈值，但是否提高下游精度需要实际实验验证。

## 数据与 loss

数据通过 `--dataset_mix` 或 `--train_file` 输入，`--dataset_task` 为 `sft` 或 `lm`。`--model_max_length` 是截断上限，`--dynamic_padding true` 按 micro-batch 动态 padding。

当前模型级 loss 为：

```text
sft, kl, kl_top, kl_top_partial, kl_top_mass, kl_top_mse,
kd, kd_top, kd_top_partial, kd_top_mass
```

Top-K 的 K 用 `--top_k`：`kl_top` 在教师 Top-K 集合内重新归一化；`kl_top_partial` 使用全词表归一化，只保留 Top-K 的 KL 项；`kl_top_mass` 额外把集合外的概率质量合为一项；`kl_top_mse` 在 Top-K KL 上加 logits MSE。`kd*` 为对应 KL 与 CE 的组合，`--alpha` 为 KL 权重；`kl*` 不含 CE，`sft` 只使用 CE。hidden 与 pre-MLP 对齐分别用 `--hidden_loss_weight`、`--pre_mlp_hidden_loss_weight`。

## 并行与正式脚本

`compressed_e2e_fintuning/scripts/e2e_decoder.sh` 只保留 shell 级 `dp`/`layer_mp` 分支：

- `dp`：当前实际默认是`train_mode=lora`、`kd_top_partial`、K=100，并训练norm和linear lm_head；脚本中的“decoder + Sparse Bit”注释不代表当前参数。
- `layer_mp`：`train_mode=decoder_lora`，`kl_top`，K=1000。

`scripts/compressed_e2e_simple.sh` 是单入口示例。旧 `e2e_stage1_pretrain.sh`、`e2e_stage2_instruct.sh` 已删除。

[Qwen3 ≤2.5bit 的无外置LoRA蒸馏入口](../scripts/e2e_w25_vae.sh)显式选择`train_mode=decoder`，冻结norm/head，关闭residual LoRA，使用FP32 decoder参数/BF16计算，最终导出BF16；运行配置与验证状态见[本轮记录](../docs/experience/records/qwen3_w25/2026-09-30_goal.md)。调用者先激活bitvae并设置`CUDA_VISIBLE_DEVICES`，路径可用`STUDENT_CKPT`、`RUN_ROOT_DIR`覆盖。无需传`--lora_rank 0`：当前公共配置要求rank为正数，但decoder模式不创建LoRA，成品应独立审计实际分支和位宽。

## 保存与续训

`training_step` 用于精确恢复 optimizer/scheduler/RNG/组件状态，并通过 checkpoint id 绑定 `round_base`。稳定 `final_model` 可独立加载，所有临时 PEFT proxy 已 finalize，Sparse Bit score 已提交为硬 bit，`lora_config=null`。

分阶段筛选可设 `--steps 4000 --stop_after_step 200 --save_steps 200 --eval_after_save true`，并配置评测任务。`stop_after_step` 是绝对 optimizer step，只控制本次执行预算：完成该步 checkpoint 和中途评测、保存评测后的各 rank RNG 后，以 `status=paused` 正常退出，不执行 finalization 或重复的最终评测，也不生成 `final_model`。停止点必须大于 0、小于 `steps` 且落在整数 `save_steps` 边界上；不设置时保持原有完整训练流程。

晋级时从保留的 `trainer_state/checkpoint-200` 设置 `--resume_from_checkpoint`，将停止点提高（如 `--stop_after_step 400`），或去掉停止点继续到终局。总 `steps`、学习率调度、数据/损失、batch/累积、卡数/并行模式和评测保存间隔等 exact-resume 条件必须保持不变；停止点必须大于断点步数。阶段停止参数自身不进入数学恢复契约。用于已安排晋级的断点保留，淘汰候选按项目规范清理。

Qwen3本轮另提供限定`decoder`/`decoder_sparse_bit`的[中间步导出入口](../experiments/qwen3_w25/export_e2e_selected.py)，适用于全252投影、无外置LoRA、norm/head冻结、单进程DP或`layer_mp`、FP32 decoder参数/BF16计算。它读取原run中的已提交checkpoint并保留原总steps；SparseBit通过CPU mmap读取exact sidecar，核对采样、固定敏感度尺度及score符号后恢复其中已物化的packed bits，不构造训练optimizer、不把完整score搬回GPU、不重标定或重放更新，然后导出新的BF16 v6成品：

```bash
python -m experiments.qwen3_w25.export_e2e_selected \
  --checkpoint_dir <run>/trainer_state/checkpoint-400 \
  --output_dir <new_selected_step400> --device cuda:0
```

中间训练回调采用packed解码，成品使用部署解码，分数不能直接等同。导出后须运行`experiments/qwen3_w25/audit_e2e_decoder.py`和完整八任务评测。原`e2e_w25_vae.sh`保留五个计划保存点，由独立命令导出评测，主入口终点仍执行配置的任务；当前实测状态以[本轮记录](../docs/experience/records/qwen3_w25/2026-09-30_goal.md)为准。

全码位对照使用[独立导出训练入口](../experiments/qwen3_w25/train_with_step_exports.py)及`scripts/e2e_w25_full_bits.sh`：调用者激活bitvae，设置训练`CUDA_VISIBLE_DEVICES=0,1,2,4`，另将GPU7留给`--export_cuda_visible_devices 7`。每400步（含2000步）同步等待独立导出、冻结状态审计、fresh strict八任务全量评测和位宽验收完成后再继续训练。`step_XXXX/`保留阶段日志、退出码、成品和验收结果；`save_total_limit=1`只保留最新可续训checkpoint，独立评测结束前不会进入下一次保存轮转。正式入口没有样本上限，`--eval_tasks ''`仅关闭主入口终点重复评测。该入口已通过25项CPU检查、5项真实双卡检查，以及Qwen3-8B的四卡100%码位4步/save2验证：两次独立导出、冻结状态审计、limited GPU评测和评测后回训完成，末步独立与原生导出的3423个持久张量逐元素及字节一致。短测不代表完整任务精度；正式入口仍固定每400步全量八任务，结果与资源实测见本轮记录。


Qwen3-8B 的既有 head/norm 方法另有两个隔离训练入口，均联合训练 decoder、冻结全部硬码、不创建外置 LoRA。它们使用单进程单训练卡 DP，并由 `--export_cuda_visible_devices` 指定独立导出卡；设置 `--save_steps 400 --save_total_limit 1 --eval_after_save false --eval_tasks ''` 后，每400步依次完成原生部署转换、冻结/存储审计和固定 MMLU64/seven32 完整八任务，再恢复训练。调用者激活 bitvae、设置训练 `CUDA_VISIBLE_DEVICES`，其余数据/损失/训练参数沿用正式入口。

| 独立训练入口 | 组件设置 | 部署与审计 |
| --- | --- | --- |
| [train_aux_with_step_exports.py](../experiments/qwen3_w25/train_aux_with_step_exports.py) | `train_mode=decoder`、`norm_train_mode=none`、`lm_head_train_mode=linear`、`distill_fp32_components=decoder,lm_head` | 训练的4096×4096映射按原生路径融合进原 head；[export](../experiments/qwen3_w25/e2e_aux_export.py) 与 [audit](../experiments/qwen3_w25/e2e_aux_audit.py) 核实没有额外分支、head形状/dtype/字节数不变。 |
| [train_norm_with_step_exports.py](../experiments/qwen3_w25/train_norm_with_step_exports.py) | `train_mode=decoder`、`norm_train_mode=final`、`lm_head_train_mode=none`、`distill_fp32_components=decoder,norm` | 只更新原 `model.norm.weight`；[export](../experiments/qwen3_w25/e2e_norm_export.py) 与 [audit](../experiments/qwen3_w25/e2e_norm_audit.py) 对照所选训练断点，分别记录FP32真实更新与BF16导出存活量，严格冻结其余norm/head/码位。 |

两条路径均已通过真实8B、B4/1024、4步/save2的两次导出、诊断评测、回训和原生final验证。诊断题量不代表精度，正式入口没有样本上限；具体对照参数与完整评测状态见[本轮记录](../docs/experience/records/qwen3_w25/2026-09-30_goal.md)。单独调用 norm 审计还需 `--training_checkpoint_dir` 指向原保存步，以验证最终norm确为该FP32状态的原生BF16投射。

## 按组件保留 FP32 训练参数

CAT 在线压缩后的蒸馏、CAT checkpoint 蒸馏和 E2E 共用 `--distill_fp32_components`，默认 `none`。例如在原实验命令中使用：

```bash
--distill_fp32_components decoder,norm,lm_head \
--bf16 true
```

可选组件为 `lora`、`decoder`、`norm`、`lm_head`，用逗号组合；重复值会去重，未知值以及 `none,norm` 这样的混用会报错。参数只调整已启用训练目标的精度，不会解冻新目标；启动日志列出指定组件、实际生效组件及参数量。

| 组件 | 范围 |
| --- | --- |
| `lora` | 主干普通层和压缩层的 LoRA |
| `decoder` | 当前训练的 VAE decoder，含内部 norm 和 bias |
| `norm` | `--norm_train_mode` 选中的主干 norm |
| `lm_head` | `--lm_head_train_mode` 选中的 linear、full 或 head LoRA |

选中参数及其梯度使用 FP32；普通 AdamW 的两个动量也为 FP32。`--bf16` / `--fp16` 继续决定计算精度，decoder 保留 packed uint8 路径，隐藏张量及重构权重保持对应的低精度。独立 VAE 重构训练、Sparse Bit 编码和 score 不受此参数控制。FP16 GradScaler 若仍遇到 FP16 可训练参数，会在创建 Trainer 前报出需要调整的组件。

训练步断点保留实际参数精度和优化器状态，不转换正在训练的参数。CAT 阶段模型、最终模型在 `bf16=true` 时导出 BF16，在 `fp16=true` 时导出 FP16；两者都未设置时保留实际精度。压缩码、整数索引和格式规定的编码数据保持原格式。CAT 内存中的阶段模型同步应用导出精度；E2E 最终评估也在转换后执行，融合误差和导出转换误差分别记录。

组件选择会进入配置快照和精确恢复约束。更改选择后可以从阶段/最终模型开始新实验，但不能作为同一个训练步断点的精确续训。默认 `none` 兼容旧断点的恢复约束。

显存实测及验证范围见 [FP32 蒸馏验证记录](../docs/experience/records/docs/distill_fp32_validation.md)。
