# VAE E2E Finetuning 三种训练形式

`vae_e2e_fintuning` 输入 `cat_train.py` 产出的压缩 checkpoint，输出新的压缩 `final_model/`。

核心参数是：

```bash
--vae_train_mode decoder   # 默认
--vae_train_mode low_rank
--vae_train_mode both
```

三种模式都使用同一套数据、loss、评估和保存流程；差别只在训练哪些参数。

## 1. `decoder`

只训练 VAELinear 的 decoder。

这是默认模式，等价于旧行为：

```bash
bash vae_e2e_fintuning/scripts/e2e_vae_decoder.sh \
  --vae_train_mode decoder
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

## 2. `low_rank`

只训练已有低秩分支，不训练 decoder。

运行方式：

```bash
bash vae_e2e_fintuning/scripts/e2e_vae_decoder.sh \
  --vae_train_mode low_rank \
  --decode_device auto \
  --decode_group_size 8 \
  --tune_final_norm false \
  --use_post_norm_head_linear false \
  --vae_tune_bias false
```

输入要求：

- 输入 checkpoint 必须已经带有 `low_rank_a` 和 `low_rank_b`
- 这些低秩分支通常来自 `cat_train.py` 的：
  - `--outlier_protect_mode per_vae_low_rank`
  - 或 `--outlier_protect_mode post_vae_low_rank`
- 被 `--decoder_layers` 和 `--target_modules` 选中的 VAELinear 必须全部有完整低秩分支
- 选中模块的低秩 rank 必须一致

训练逻辑：

- 先把选中的 VAELinear 解码成 dense `nn.Linear`
- dense base 权重包含 VAE decoder 和固定 sparse residual
- dense base 权重不包含 low-rank patch
- 再用 checkpoint 里的 `low_rank_b` 初始化 LoRA A
- 用 checkpoint 里的 `low_rank_a` 初始化 LoRA B
- LoRA scaling 固定为 1
- 训练结束后，把 LoRA 写回压缩模型的 `low_rank_a/b`

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
bash vae_e2e_fintuning/scripts/e2e_vae_decoder.sh \
  --vae_train_mode both
```

输入要求：

- 被选中的 VAELinear 必须全部有完整 `low_rank_a/b`

训练内容：

- 训练选中模块的 VAE decoder
- 同时训练选中模块的 `low_rank_a/b`
- 可选训练 `--vae_tune_bias`
- 可选训练 `--tune_final_norm`
- 可选训练 `--use_post_norm_head_linear`

适合场景：

- 低秩补偿和 decoder 都需要一起调整
- 能接受比 `low_rank` 更重的训练成本

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
- 只修低秩离群补偿：用 `low_rank`
- decoder 和低秩补偿都明显不够：用 `both`

`low_rank` 是最窄的优化路径，但它要求输入 checkpoint 已经有低秩分支。没有低秩分支时直接用 `decoder`。
