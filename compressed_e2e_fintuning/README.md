# compressed_e2e_fintuning 三种训练形式

`compressed_e2e_fintuning` 输入 `cat_train.py` 产出的压缩 checkpoint，输出新的压缩 `final_model/`。

核心参数是：

```bash
--finetune_mode decoder   # 默认
--finetune_mode compressed_lora
--finetune_mode both
```

三种模式都使用同一套数据、loss、评估和保存流程；差别只在训练哪些参数。

## Dynamic padding

可选参数：

```bash
--dynamic_padding false   # 默认：pad 到 --model_max_length
--dynamic_padding true    # pad 到本 micro-batch 最长样本，再向上对齐到 8 的倍数
```

`--model_max_length` 始终是单样本截断上限。仅对 `dataset_task=sft/lm` 的 shared sequence collator 生效；`mcqa` 不受影响。

## Packed VQ decoder first-linear kernel

主 VAE decode 默认启用：

```bash
--parallel_stage_decode true
--packed_vq_decoder_linear true
```

该路径在 decoder 优化时不再先把 checkpoint 中的 bit-packed `uint8` VQ code 展开成持久的 BOOL/BF16 grouped VQ，而是在 Triton kernel 中按 K 维分 tile 解 bit，并直接完成 decoder 第一层线性投影。serial stage decode 和 parallel-stage decode 共用同一个 packed first-linear helper。

kernel 本身只依赖 `packed uint8 + first linear weight/bias`，不把实验配置写死：`codebook_bits` 是 K 维并按 tile 循环，可为任意正整数；`codebook_dim` 不设固定值限制。`decoder_type=linear` 时 first linear 即 `decoder.linear`；`symmetric/asymmetric` 时 first linear 为 `decoder.linear_in`，后续 `resblock -> norm -> activation -> linear_out` 仍使用原 decoder，因此 `decoder_num_res_blocks=0/1/...` 都兼容。

训练和推理都可使用 packed helper，但路由优先级不同：decoder 参数需要梯度时优先 packed 路径，自定义 backward 直接从 packed uint8 重新解 bit 计算第一层参数梯度，不保存 dense latent；普通 `no_grad` 推理/prewarm 若满足现有 `resblock=0 + symmetric + layer norm + swish` whole-decoder fused Triton 条件，则继续优先原 whole-decoder fuse，避免为了节省一次性 decode 显存而回退推理速度，并在 decoded weight cache 建好后立即释放该路径临时 materialize 的 dense BOOL/BF16 grouped VQ；旧 whole-decoder fuse 不适用时直接尝试 packed first-linear。若关闭 `--packed_vq_decoder_linear`，或运行环境不是 CUDA/Triton，则回退旧的 dense decode 路径。最终 checkpoint 格式不变，VQ code 仍按原 bitpack uint8 保存。

## Hidden-state 对齐

可选参数：

```bash
--hidden_loss_weight 0.0
--hidden_layer_weighting uniform  # uniform | linear_depth | adaptive_top_k
```

`hidden_loss_weight=0` 表示关闭。开启后会在现有 SFT/KD loss 之外，对齐 teacher 和 student 的所有 transformer block 输出 hidden states，即 `hidden_states[1:]`，跳过 embedding hidden state。

`linear_depth` 会让越靠后的层权重越大，并把平均权重归一到 1。当前 `e2e_decoder.sh` 默认关闭普通 hidden alignment，但保留 `adaptive_top_3` 作为启用 hidden/pre-MLP alignment 时的层选择设置：

```bash
--hidden_loss_weight 0.0
--hidden_layer_weighting adaptive_top_3
```

这个损失会额外保存 teacher/student hidden states，长序列训练时显存压力会增加。

### Pre-MLP hidden 对齐

端到端训练同时支持：

```bash
--distill_pre_mlp_hidden_loss_weight 0.0
```

`0.0` 表示关闭。开启后，teacher/student 都在每个 Transformer block 的 `post_attention_layernorm` **输入端**捕获 hidden，也就是 attention residual 之后、MLP 之前的表示。损失计算直接复用 CAT 压缩训练的 `_compute_named_pre_mlp_hidden_alignment_loss`，因此每层损失、mask、`uniform` / `linear_depth` / `adaptive_top_k` 聚合语义与 CAT 一致：

```text
L_pre_mlp(layer) = mean_mask((student - teacher)^2) / (mean_mask(teacher^2) + 1e-6)
L_total += distill_pre_mlp_hidden_loss_weight * L_pre_mlp
```

`--hidden_layer_weighting` 同时控制普通 hidden alignment 和 pre-MLP hidden alignment。`adaptive_top_k` 时，pre-MLP 层选择也使用 CAT 的同一实现。与 CAT 一致，pre-MLP alignment 使用 `attention_mask` 覆盖整条序列的所有非 padding token，不使用 response-only mask，也不做 causal shift；它与 `prompt_kd_weight` 控制的 logit KD region 是两套独立语义。

### Teacher 权重 CPU offload

```bash
--teacher_output_offload cpu
--distill_teacher_model_offload cpu
```

`--distill_teacher_model_offload none` 为默认值。设为 `cpu` 时，teacher 权重在 teacher forward 前搬到训练设备；本 batch 需要的 logits / hidden targets materialize 到 CPU 后，teacher 权重立即搬回 CPU，再执行 student forward。当前实现要求 teacher model offload 与 `--teacher_output_offload cpu` 一起使用。该模式与 CAT 的 teacher model offload 语义一致：它减少 teacher 权重与 student forward 的显存重叠，但 teacher 本身的一次 forward 仍需完整落在其 forward device 上，并不是 CPU inference 或 teacher layer-wise model parallel。

## Prompt KD weighting

可选参数：

```bash
--prompt_kd_weight 0.0
```

`prompt_kd_weight=0.0`（默认）与当前行为完全一致：只对 response region 做 logit KD。

logit KD 按 **region** 分别计算 token 均值，再线性组合：

```text
L_logit = L_response + prompt_kd_weight * L_prompt
```

- `L_response`：response region（`labels != -100` 且非 padding；causal shift 到 logits 位置）上的 per-token KD token 均值。
- `L_prompt`：prompt region（`labels == -100` 且非 padding；同样 shift）上**独立**计算的 token 均值。

**取值含义**：系数作用于 **region 均值**，与 prompt/response token 数量比无关。例如 `0.03` 表示 prompt-region 均值以系数 0.03 加入总 logit KD；`1.0` 表示两个 region 均值等系数相加。

该权重**只**作用于 teacher-student logit KD（含 EAKLD / forward KL 等），**不**改变 CE 或 `--hidden_loss_weight` 对应的 hidden-state alignment。

EAKLD 在 response / prompt region 上**分别**计算 teacher entropy 与 gamma，再各自得到 region 均值后按上式组合。现有 `eakld/*` 训练日志 telemetry 仍只反映 **response region** 的 EAKLD 统计。

padding 与序列最后一个无 next-token 的 logits 不计入任一 region。预测 EOS 的 logits 仍属于 response region。

`--dataset_task mcqa` 不支持 `prompt_kd_weight != 0`。

以下数值仅作实验示例，**不是**推荐最优值，也**未**经对照实验验证下游收益：

```bash
--prompt_kd_weight 0.03
--prompt_kd_weight 0.1
```

## Token telemetry

E2E 训练日志会周期性输出 `E2E token stats:` 行，语义与类别蒸馏的 `LoRA token stats:` 一致（仅日志前缀不同）。

计数规则（**不是** causal KD mask 计数）：

- 直接统计 truncation 后 `labels` 与 `attention_mask` 上的非 padding 位置。
- prompt token：`attention_mask != 0` 且 `labels == -100`。
- response token：`attention_mask != 0` 且 `labels != -100`（含 EOS）。
- 不做 causal shift；例如 `labels=[-100,-100,-100,A,B,EOS]` 的一个样本计 prompt=3、response=3。

每个 regular logging window 内，所有 gradient accumulation micro-batch 与 DDP rank 的计数先求 global sum，再除以 global sample 数得到 per-sample 均值。输出字段：`step`、`window_optimizer_steps`、`avg_prompt_tokens`、`avg_response_tokens`、`global_samples`。

## 1. `decoder`

只训练 VAELinear 的 decoder。

这是默认模式，等价于旧行为：

```bash
bash compressed_e2e_fintuning/scripts/e2e_decoder.sh \
  --finetune_mode decoder
```

训练内容：

- 训练选中 layer / target module 的 VAE decoder
- 可选训练 `--vae_tune_bias`
- 可选训练 `--tune_final_norm`
- 可选训练 `--use_post_norm_head_linear`

适合场景：

- checkpoint 没有低秩离群保护分支
- 主要想继续优化 VAE 重建质量
- 想保持当前 e2e decoder 微调逻辑

输出：

- `run_dir/final_model/`
- 仍是压缩 checkpoint
- 不产生 PEFT adapter

## 2. `compressed_lora`

只训练已有低秩分支，不训练 decoder。

运行方式：

```bash
bash compressed_e2e_fintuning/scripts/e2e_decoder.sh \
  --finetune_mode compressed_lora \
  --decode_device auto \
  --decode_group_size 8 \
  --tune_final_norm false \
  --use_post_norm_head_linear false \
  --vae_tune_bias false
```

输入要求：

- 输入 checkpoint 必须已经带有 `low_rank_a` 和 `low_rank_b`
- 被 `--decoder_layers` 和 `--target_modules` 选中的 VAELinear 必须全部有完整低秩分支
- 选中模块的低秩 rank 必须一致
- 选中模块的 `low_rank_scope` 必须一致（`full` 或 `compressed_subspace`）；E2E **不新增**第二套 scope CLI，自动沿用 checkpoint 中的 scope

训练逻辑：

- `full`（默认/旧 checkpoint）：先把选中的 VAELinear 解码成 dense `nn.Linear`（含 VAE decoder 与固定 sparse residual，不含 low-rank patch），再 `get_peft_model` 包完整权重并训练 LoRA
- `compressed_subspace`：把选中模块包成 `CompressedSubspacePeftProxy` + O(1) PEFT carrier，再 `get_peft_model` 得到 root `PeftModel`，只在压缩子空间训练 LoRA
- 两种 scope 都保持 root `PeftModel`，继续共用现有 Trainer checkpoint/resume
- 用 checkpoint 里的 effective `low_rank_a/b` 初始化 LoRA（scaling 固定为 1）
- 训练结束后，把 LoRA 写回压缩模型的 `low_rank_a/b`，并校验 `expected_scope`

说明：本功能只覆盖最终写回 `VAELinear.low_rank_a/b` 的 compressed LoRA；block-level PEFT LoRA 不在此范围。

限制：

- 不允许同时打开 `--vae_tune_bias true`
- 不允许同时打开 `--tune_final_norm true`
- 不允许同时打开 `--use_post_norm_head_linear true`

适合场景：

- VAE decoder 已经够稳定
- 只想快速优化离群值低秩补偿
- 希望训练时借用 dense LoRA 的速度，但最终仍保存压缩模型

输出：

- `run_dir/final_model/`
- 仍是压缩 checkpoint
- 更新的是 VAELinear 内部 `low_rank_a/b`
- 不保存 LoRA adapter

## 3. `both`

同时训练 VAELinear decoder 和已有低秩分支。

运行方式：

```bash
bash compressed_e2e_fintuning/scripts/e2e_decoder.sh \
  --finetune_mode both
```

输入要求：

- 被选中的 VAELinear 必须全部有完整 `low_rank_a/b`
- 选中模块的 `low_rank_scope` 必须一致；`both` 直接训练 checkpoint 中的 `low_rank_a/b`（含 `compressed_subspace` shape），不创建 proxy / root PEFT

训练内容：

- 训练选中模块的 VAE decoder
- 同时训练选中模块的 `low_rank_a/b`
- 可选训练 `--vae_tune_bias`
- 可选训练 `--tune_final_norm`
- 可选训练 `--use_post_norm_head_linear`

适合场景：

- 低秩补偿和 decoder 都需要一起调整
- 能接受比 `compressed_lora` 更重的训练成本

输出：

- `run_dir/final_model/`
- 仍是压缩 checkpoint
- decoder 和 `low_rank_a/b` 都会更新

## sparse residual

decoder 微调不会重新计算 sparse residual。输入 checkpoint 已有的 sparse residual 会在训练和最终保存中保持固定。

训练和最终推理顺序都是：

```text
VAE reconstruction -> low_rank patch -> sparse_residual patch
```

## 参数选择建议

- 普通 VAE decoder 继续训：用 `decoder`
- 只修低秩离群补偿：用 `lora`
- decoder 和低秩补偿都明显不够：用 `both`

`lora` 是最窄的优化路径，但它要求输入 checkpoint 已经有低秩分支。没有低秩分支时直接用 `decoder`。
