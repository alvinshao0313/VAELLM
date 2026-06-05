# `block_vae_lora` 参数说明

入口：

```bash
bash scripts/block_vae_lora_simple.sh
```

实际调用：

```bash
python tools/block_vae_lora_train.py ...
```

参数解析代码：`train_utils/block_vae_lora_args.py`

说明：

- `Parser 默认值` 是 `tools/block_vae_lora_train.py` 不传该参数时的默认值。
- `simple.sh 默认值` 是 `scripts/block_vae_lora_simple.sh` 当前显式传入的值。
- `simple.sh 默认值` 为空表示脚本没有显式传这个参数。
- `block_vae_lora` 当前只支持 Qwen3 block 的 7 个线性层：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。

## 1. override 参数语法

部分 VAE 参数使用 category override 字符串。

格式：

```text
selector=value,selector=value
```

允许的 selector：

```text
default
cat:q_proj
cat:k_proj
cat:v_proj
cat:o_proj
cat:gate_proj
cat:up_proj
cat:down_proj
```

例子：

```bash
--codebook_bits "default=32,cat:k_proj=24"
--vae_steps "default=200,cat:q_proj=500"
--intra_parallel "default=1x1,cat:q_proj=4x1"
```

命中规则：

```text
cat:<当前线性层类别> > default
```

如果没有命中，也没有 `default`，直接报错。

## 2. 基础运行参数

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--model_path` | 必填 | `Qwen/Qwen3-8B` | 原始模型路径或 HF repo id | 当前只支持 Qwen3 |
| `--output_dir` | `.result` | `.result` | run 目录根路径 | 无 |
| `--seed` | `42` | `${SEED}`，默认 `42` | 随机种子 | 整数 |
| `--deterministic` | `false` | `${FULL_DETERMINISM}`，默认 `false` | 是否启用确定性模式 | bool |
| `--train_device` | `cuda` | `cuda` | 训练和 block 前向设备 | 设备必须可用 |
| `--convert_device` | `cuda` | `cuda` | VAE 替换/转换设备 | 设备必须可用 |
| `--unload_vae_original_weights_on_final_save` | `false` | 开启 | 最终保存时卸载 `original_weight` | flag，无显式值 |
| `--access_token` | `None` | 空 | HF private model token | 字符串或不传 |
| `--bf16` | `true` | `true` | VAE 训练使用 bf16 | 不能和 `--fp16 true` 同时开 |
| `--fp16` | `false` | `false` | fp16 开关 | 当前 VAE 训练不支持 `true` |

## 3. VAE 结构参数

这些参数支持 category override。

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--codebook_bits` | `default=32` | `default=32` | 每组 codebook bit 数 | 整数，`>=1` |
| `--codebook_dim` | `default=32` | `default=32` | codebook latent 维度 | 整数，`>=1` |
| `--residual_stages` | `default=2` | `default=2` | residual VAE 阶数 | 整数，`>=1` |
| `--base_ch` | `default=128` | `default=128` | VAE encoder base channel | 整数，`>=1` |
| `--num_res_blocks` | `default=1` | `default=1` | VAE encoder residual block 数 | 整数，`>=0` |
| `--decoder_base_ch` | `default=128` | `default=128` | VAE decoder base channel | 整数 `>=1` 或 `none` |
| `--decoder_num_res_blocks` | `default=1` | `default=1` | VAE decoder residual block 数 | 整数 `>=0` 或 `none` |
| `--norm_type` | `default=layer` | `default=layer` | VAE norm 类型 | `group/batch/layer/no` |
| `--decoder_type` | `default=symmetric` | `default=symmetric` | decoder 结构 | `linear/symmetric/asymmetric` |
| `--recon_loss_type` | `default=mse` | `default=mse` | VAE 重建 loss | `mse/l1/huber/relative_l1/top_k_mse/cosine/w_mse/w2_mse` |
| `--intra_parallel` | `default=1x1` | `default=1x1` | 单个线性层内部切分，格式 `rows x cols` | 如 `1x1`、`4x1` |
| `--intra_part_sort_mode` | `default=none` | `default=none` | part 内排序模式 | 由 `parse_intra_part_sort_mode_text` 校验 |

## 4. VAE 训练参数

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--vae_steps` | `default=20000` | `default=200` | 每个 VAE group 的训练步数，支持 category override | 整数，`>=1` |
| `--vae_batch_size` | `8192` | `all` | VAE 权重训练 batch size | 正整数或 `all` |
| `--vae_log_every` | `100` | `100` | VAE 训练日志间隔 | 整数，`>=1` |
| `--vae_eval_every` | `0` | `0` | VAE 训练中评估间隔；`0` 表示不评估 | 整数，`>=0` |
| `--quantizer_type` | `BSQ` | `BSQ` | 量化器类型 | `LFQ/BSQ` |
| `--gamma0` | `1.0` | `1.0` | 量化器参数 | float |
| `--gamma` | `1.0` | `1.0` | 量化器参数 | float |
| `--zeta` | `1.0` | `1.0` | 量化器参数 | float |
| `--inv_temperature` | `200.0` | `100.0` | 量化器温度倒数 | float |
| `--lr` | `1e-2` | `1e-2` | VAE 优化器学习率 | float，`>0` |
| `--beta1` | `0.9` | `0.9` | optimizer beta1 | float |
| `--beta2` | `0.95` | `0.95` | optimizer beta2 | float |
| `--weight_decay` | `0.0` | `0.0` | VAE optimizer weight decay | float，`>=0` |
| `--optimizer` | `adamw` | `adamw` | VAE optimizer | `adam/adamw/sgd/rmsprop` |
| `--lr_scheduler` | `linear` | `linear` | VAE lr scheduler | `none/linear/cosine/constant/constant_with_warmup` |
| `--lr_warmup_steps` | `0` | `0` | VAE warmup steps | 整数，`>=0` |
| `--l1_weight` | `1.0` | `1.0` | VAE loss 权重 | float |
| `--lfq_weight` | `5.0` | `2.5` | LFQ/BSQ loss 权重 | float |
| `--commitment_loss_weight` | `0.1` | `0.25` | commitment loss 权重 | float |
| `--entropy_loss_weight` | `1e-4` | `1e-2` | entropy loss 权重 | float |
| `--diversity_gamma` | `1.0` | `1.0` | diversity loss 参数 | float |
| `--normalize_weight` | `false` | 开启 | VAE 训练前是否归一化权重 | flag |
| `--use_checkpoint` | `false` | 空 | VAE 内部是否使用 checkpoint | flag |
| `--new_quant` | `false` | 开启 | 使用新版量化逻辑 | flag |

## 5. block 蒸馏数据参数

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--block_distill_dataset` | `fineweb_edu=0.35,race=0.30,sciq=0.20,openorca=0.15` | `openorca=0.24,fineweb_edu=0.18,race=0.24,sciq=0.03,alpaca=0.11,longalpaca=0.10,longalign=0.10` | 蒸馏校准数据混合比例 | 必须包含 `=` |
| `--block_distill_nsamples` | `100` | `5000` | 校准样本数 | 整数，`>0` |
| `--block_distill_seqlen` | `4096` | `4096` | 每个样本 token 长度 | 整数，`>0` |

相关环境变量：

| 变量 | simple.sh 默认值 | 说明 |
|---|---:|---|
| `CAT_LORA_DATASET_NUM_PROC` | `16` | 构建 LoRA/block 校准数据时的数据处理进程数 |
| `TOKENIZERS_PARALLELISM` | `false` | tokenizer 并行开关 |

## 6. block 蒸馏训练模式

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--block_distill_train_mode` | `lora` | `lora` | block 蒸馏训练哪些参数 | `lora/decoder/both` |
| `--block_distill_steps` | `100` | `5000` | 每个被选中 block 的蒸馏步数 | 整数，`>0` |
| `--block_lora_lr` | `1e-4` | `1e-4` | block 蒸馏 optimizer 学习率 | float，`>0` |
| `--block_distill_log_every` | `10` | `10` | block 蒸馏日志间隔 | 整数，`>0` |
| `--block_decode_group_size` | `8` | `8` | PEFT proxy materialize 时的 VAE decode group size | 整数，`>0` |

`--block_distill_train_mode` 取值：

| 值 | 训练参数 | 最终 checkpoint |
|---|---|---|
| `lora` | 只训练 PEFT LoRA adapter | 保留 PEFT proxy adapter |
| `decoder` | 只训练 `VAELinear` decoder | 不允许残留 LoRA adapter |
| `both` | 同时训练 decoder 和 PEFT LoRA adapter | decoder 更新，并保留 PEFT proxy adapter |

说明：

- 三种模式都共用 `--block_distill_steps`。
- 三种模式都共用 `--block_lora_lr`。
- `decoder` 模式虽然不训练 LoRA，但学习率仍由 `--block_lora_lr` 控制。
- `decoder/both` 会调用 `enable_trainable_decode(parallel_stage_decode=True)`。
- `decoder/both` 训练结束后会拆回 parallel stage decoder，并关闭 trainable decode。

## 7. block 蒸馏 loss 参数

总 loss：

```text
loss = alpha * attention_kl + beta * linear_mse + (1 - alpha - beta) * hidden_mse
```

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--block_loss_alpha` | `0.1` | `0.3` | attention map KL loss 权重 | float，`>=0` |
| `--block_loss_beta` | `0.2` | `0.2` | linear output relative MSE loss 权重 | float，`>=0` |
| `--block_attn_query_chunk_size` | `128` | `4096` | attention KL 按 query 维度分 chunk 的大小 | 整数，`>0` |

约束：

```text
block_loss_alpha + block_loss_beta <= 1
```

`hidden_mse` 权重自动等于：

```text
1 - block_loss_alpha - block_loss_beta
```

## 8. LoRA 参数

这些参数在 `--block_distill_train_mode lora` 和 `both` 下生效。

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--block_lora_rank` | `32` | `32` | LoRA rank；AdaLoRA 下是 target rank | 整数，`>0` |
| `--block_lora_variant` | `plain` | `plain` | LoRA 类型 | `plain/rslora/dora/adalora` |
| `--block_lora_alpha` | `None`，归一化后等于 `block_lora_rank` | `32` | LoRA alpha | float，`>0` |
| `--block_lora_dropout` | `0.0` | `0.0` | LoRA dropout | `0 <= value <= 1` |
| `--block_lora_bias` | `none` | `none` | LoRA bias 模式 | `none/lora_only` |
| `--block_lora_hif4_act` | `false` | `false` | student 路径是否启用 HiFloat4 activation 量化 | bool |

`block_lora_variant` 说明：

| 值 | 说明 |
|---|---|
| `plain` | 标准 PEFT LoRA |
| `rslora` | PEFT RSLoRA |
| `dora` | PEFT DoRA |
| `adalora` | PEFT AdaLoRA |

## 9. AdaLoRA 参数

只在 `--block_lora_variant adalora` 时生效。

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--block_adalora_init_rank` | `None`，归一化后等于 `block_lora_rank` | `32` | AdaLoRA 初始 rank | 整数，`>0` |
| `--block_adalora_tinit` | `0` | `0` | AdaLoRA tinit | 整数，`>=0` |
| `--block_adalora_tfinal` | `0` | `0` | AdaLoRA tfinal | 整数，`>=0` |
| `--block_adalora_delta_t` | `1` | `1` | AdaLoRA deltaT | 整数，`>0` |
| `--block_adalora_beta1` | `0.85` | `0.85` | AdaLoRA beta1 | float，`>0` |
| `--block_adalora_beta2` | `0.85` | `0.85` | AdaLoRA beta2 | float，`>0` |
| `--block_adalora_orth_reg_weight` | `0.5` | `0.5` | AdaLoRA orthogonal regularization 权重 | float，`>=0` |

额外约束：

```text
block_adalora_init_rank >= block_lora_rank
```

## 10. block 选择参数

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--block_layers` | `all` | `0` | 要压缩并蒸馏的 block 层 | `all` 或层号/range |

支持写法：

```bash
--block_layers "all"
--block_layers "0"
--block_layers "0-3"
--block_layers "0,2,4-7"
```

约束：

- 空字符串报错。
- 重复层号报错。
- 越界层号报错。
- range 必须满足 `start <= end`。

## 11. transpose 参数

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--transpose_modules` | `q_proj,v_proj,o_proj,down_proj` | `q_proj,v_proj,o_proj,down_proj` | VAE 切分前需要转置的投影名 | 逗号分隔；允许为空字符串 |
| `--skip_layers` | 空字符串 | 空字符串 | 指定哪些 Linear 保留原始权重，不做 VAE 压缩和 block 蒸馏 | 格式为 `layer_idx.category`，例如 `0.down_proj,30.q_proj` |

允许的模块名：

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

约束：

- 不允许未知模块名。
- 不允许重复模块名。
- 空字符串表示全部不转置。

`--skip_layers` 约束：

- category 只能是 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。
- 指定的 layer 必须在 `--block_layers` 选中范围内，否则直接报错。
- 被 skip 的模块保持原始 `nn.Linear`，不会替换成 `VAELinear`，也不会注入 LoRA 或训练 decoder。
- 如果某个已选 block 的 7 个 Linear 全部 skip，该 block 不训练、不蒸馏、不做逐层 eval，只推进 hidden states。

## 12. 每层后评估参数

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 | 约束 |
|---|---:|---:|---|---|
| `--block_eval_after_each_layer` | `false` | `true` | 每个选中 block 蒸馏后是否立即评估 | bool |
| `--block_eval_tasks` | 空字符串 | `boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu` | lm-eval 任务列表 | 逗号分隔 |
| `--block_eval_ppl` | `false` | `false` | 是否评估 PPL | bool |
| `--block_eval_ppl_limit` | `-1` | `-1` | PPL 样本限制；`-1` 表示不限制 | `-1` 或 `>=1`，不能为 `0` |
| `--block_eval_device` | `None` | `cuda` | 评估设备；不传时使用 `train_device` | 设备必须可用 |
| `--block_eval_hif4_act` | `false` | `false` | 评估时是否启用 HiFloat4 activation | bool |

约束：

如果：

```bash
--block_eval_after_each_layer true
```

则必须满足至少一个：

```bash
--block_eval_ppl true
```

或：

```bash
--block_eval_tasks "非空任务列表"
```

## 13. 输出相关

| 输出 | 说明 |
|---|---|
| `block_vae_lora.log` | 训练日志 |
| `normalized_block_vae_lora_args.json` | 归一化后的参数快照 |
| `final_model/` | 最终压缩 checkpoint |

`normalized_block_vae_lora_args.json` 包含：

```text
args
hf_args
training_args
resolved_runtime
```

`final_model/checkpoint_meta.json` 的 `extra_meta` 会包含：

```text
stage = block_vae_lora_final
block_distill = 完整 block 参数
block_distill_train_mode = lora/decoder/both
selected_block_layers = 实际训练层
skip_layers = 实际跳过的 layer.category 列表
target_module_count = 实际压缩和蒸馏的 Linear 数量
```

## 14. simple.sh 环境变量

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `PYTHONPATH` | `.` | Python import 路径 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA allocator 设置 |
| `CUDA_VISIBLE_DEVICES` | `6` | 默认可见 GPU |
| `SEED` | `42` | 训练随机种子 |
| `PYTHONHASHSEED` | `${SEED}` | Python hash seed |
| `TOKENIZERS_PARALLELISM` | `false` | tokenizer 并行 |
| `CAT_LORA_DATASET_NUM_PROC` | `16` | 数据处理进程数 |
| `FULL_DETERMINISM` | `false` | 为 `true` 时额外设置 `CUBLAS_WORKSPACE_CONFIG` |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | 仅 `FULL_DETERMINISM=true` 时设置 |

## 15. 常用覆盖示例

只改训练层：

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_layers "0-3"
```

只训 decoder：

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_distill_train_mode "decoder"
```

同时训 decoder 和 LoRA：

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_distill_train_mode "both"
```

快速连通性检查：

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_layers "0" \
  --vae_steps "default=2" \
  --block_distill_steps "2" \
  --block_distill_nsamples "2" \
  --block_eval_after_each_layer "false"
```

关闭 attention KL，只保留 hidden 和 linear loss：

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_loss_alpha "0.0" \
  --block_loss_beta "0.2"
```

只用 hidden loss：

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_loss_alpha "0.0" \
  --block_loss_beta "0.0"
```
