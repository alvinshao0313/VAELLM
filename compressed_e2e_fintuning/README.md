# compressed_e2e_fintuning 三种训练形式

`compressed_e2e_fintuning` 输入 `cat_train.py` 产出的压缩 checkpoint，输出新的压缩 `final_model/`。

核心参数是：

```bash
--finetune_mode decoder   # 默认
--finetune_mode compressed_lora
--finetune_mode both
```

三种模式都使用同一套数据、loss、评估和保存流程；差别只在训练哪些参数。

## Hidden-state 对齐

可选参数：

```bash
--hidden_loss_weight 0.0
--hidden_layer_weighting uniform  # uniform | linear_depth
```

`hidden_loss_weight=0` 表示关闭。开启后会在现有 SFT/KD loss 之外，对齐 teacher 和 student 的所有 transformer block 输出 hidden states，即 `hidden_states[1:]`，跳过 embedding hidden state。

`linear_depth` 会让越靠后的层权重越大，并把平均权重归一到 1。当前 `e2e_decoder.sh` 使用较保守的：

```bash
--hidden_loss_weight 0.003
--hidden_layer_weighting linear_depth
```

这个损失会额外保存 teacher/student hidden states，长序列训练时显存压力会增加。

## Prompt KD weighting

可选参数：

```bash
--prompt_kd_weight 0.0
```

`prompt_kd_weight=0.0`（默认）与当前行为完全一致：只对 response target 做 logit KD。response target 权重固定为 1.0，不可单独配置。

取值含义：

- `0.0`：response-target-only KD（当前默认行为）
- `0.05`：prompt token 的 KD 相对权重为 response 的 5%
- `1.0`：所有有效 next-token 位置等权

该权重**只**作用于 teacher-student logit KD（含 EAKLD / forward KL 等），**不**改变 CE 或 `--hidden_loss_weight` 对应的 hidden-state alignment。

mask 规则：

- padding 与序列最后一个无 next-token 的 logits 始终排除（权重 0）
- 预测 EOS 的 logits 仍按 response target 计入（权重 1.0）
- EAKLD 的 teacher entropy、gamma 与 KL 项共用同一 weighted mask

`--dataset_task mcqa` 不支持 `prompt_kd_weight != 0`。

以下数值仅作实验示例，**不是**推荐最优值或已验证结论：

```bash
--prompt_kd_weight 0.05
--prompt_kd_weight 0.1
```

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
