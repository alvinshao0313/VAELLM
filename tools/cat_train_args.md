# `tools/cat_train.py` 参数说明

本文档按代码真实行为整理（`tools/cat_train.py` + `train_utils/train_args.py` + `train_utils/lora_utils.py`）。

## 1. 参数来源与解析顺序

`cat_train.py` 会按下面顺序解析参数：

1. 脚本私有参数（`build_cat_train_parser`）
2. 通用 VAE/量化/优化参数（`add_model_specific_args` + `add_llm_args`）
3. HuggingFace 参数（`HFArguments` + `transformers.TrainingArguments`）

说明：

- 脚本私有参数用于“按类别训练 + 分组 + LoRA + 保存”等流程控制。
- VAE/量化参数用于 `MultiLayerVAE` 的结构、损失、优化器。
- `TrainingArguments` 是 HF 全量参数集合，本脚本只直接使用其中少数关键项（下文列出）。

## 2. 脚本私有参数（`cat_args`）

| 参数 | 默认值 | 功能 | 备注/约束 |
|---|---:|---|---|
| `--category_order` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | 类别训练顺序 | 可设 `auto`（按发现类别排序）；可包含 `others` 兜底 |
| `--transpose_modules` | `v_proj,o_proj,gate_proj,up_proj,down_proj` | 这些类别在切分前先转置权重 | 影响切分维度与可整除性 |
| `--projection_suffixes` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | 仅 decoder 投影模式下允许的后缀 | 与 `--only_decoder_projections` 联动 |
| `--only_decoder_projections` | `True` | 只收集 decoder layers 中投影层 `nn.Linear` | 当前实现是 `store_true + default=True`，CLI 无法显式关掉 |
| `--include_all_linears` | `False` | 覆盖上项，改为收集模型里全部 `nn.Linear` | 开启后会忽略“只看投影层”的过滤 |
| `--steps_per_category` | `2000` | 每个类别训练步数 | 支持标量或 JSON list（按 residual stage） |
| `--steps_per_group` | `None` | 每个分组训练步数 | 优先级高于 `steps_per_category`；支持标量或 JSON list |
| `--skip_layers` | `""` | 指定某些层在推理时始终走原始权重 | 格式：`layer_idx.category`，例如 `0.down_proj,30.q_proj` |
| `--linear_group_size` | `32` | 同类别跨层分组大小 | 必须 `>=1` |
| `--intra_parallel` | `1` | 单个 Linear 的层内切分 | 支持 `n` / `a,b` / JSON dict（按类别覆盖） |
| `--intra_part_sort_mode` | `l2` | 切分前排序方式 | 支持单值 `mode` 或双值 `row_mode,col_mode`（也支持按 stage 的 JSON list）；`none`/`l2`/`act_l2` |
| `--batch_size` | `256` | VAE 训练和评估 DataLoader batch 大小 | 作用于“块数据”而非 token |
| `--log_every` | `50` | 每多少 step 打印一次训练日志 | `<=0` 等价不打印 |
| `--eval_every` | `0` | 每多少 step 做一次 VAE 评估 | `0` 代表不做中间评估 |
| `--eval_blocks` | `256` | 每次中间评估最多评估多少块 | 与 `eval_every` 联动 |
| `--activation_weight_path` | `None` | 激活 abs-max 字典文件（`.pt`）路径 | `wa_mse`、`act_l2` 常用 |
| `--outlier_protect_ratio` | `0.0` | 保护 top `floor(channel_dim * ratio)` 个通道不参与 VAE 压缩 | `>0` 时依赖 activation 向量；范围 `[0,1)` |
| `--outlier_protect_axis` | `input` | 选择保护 `input` 还是 `output` channel | `input` 会裁输入通道；`output` 会裁输出通道 |
| `--wa_mse_act_mode` | `dynamic` | `wa_mse` 的 `act_max` 来源 | `dynamic` 每组重算；`static` 使用上面的路径 |
| `--wa_mse_calib_dataset` | `wikitext2` | `dynamic` 重算时校准集 | 传给激活采样流程 |
| `--wa_mse_calib_nsamples` | `512` | `dynamic` 校准样本数 | 仅 `dynamic` 有效 |
| `--wa_mse_calib_seqlen` | `512` | `dynamic` 校准序列长度 | 仅 `dynamic` 有效 |
| `--wa_mse_calib_seed` | `0` | `dynamic` 校准随机种子 | 仅 `dynamic` 有效 |
| `--wa_mse_calib_device` | `""` | `dynamic` 校准设备 | 为空时回退到 `--train_device` |
| `--wa_mse_calib_log_every` | `0` | `dynamic` 校准日志间隔 | `0` 为关闭 |
| `--ppl_limit` | `-1` | 每个类别训练后 PPL 评估样本上限 | `-1` 表示全量 |
| `--lora_after_category` | `False` | 每训练完一个类别后，对剩余类别做 LoRA 微调并融合 | 开启后 LoRA 参数生效 |
| `--lora_rank` | `8` | LoRA rank (`r`) | 可被 `lora_schedule` 覆盖 |
| `--lora_alpha` | `16.0` | LoRA alpha | 可被 `lora_schedule` 覆盖 |
| `--lora_dropout` | `0.0` | LoRA dropout | 可被 `lora_schedule` 覆盖 |
| `--lora_steps` | `50` | LoRA 最大训练步数 | `<=0` 时 LoRA 阶段跳过 |
| `--lora_batch_size` | `2` | LoRA 每卡 batch size | 可被 `lora_schedule` 覆盖 |
| `--lora_nsamples` | `128` | LoRA 训练集采样数量 | 从 wikitext2 train 采样 |
| `--lora_lr` | `1e-4` | LoRA 学习率 | 可被 `lora_schedule` 覆盖 |
| `--lora_weight_decay` | `0.0` | LoRA 权重衰减 | 可被 `lora_schedule` 覆盖 |
| `--lora_log_every` | `1` | LoRA 日志间隔 | 可被 `lora_schedule` 覆盖 |
| `--lora_tune_norm` | `False` | LoRA 时额外解冻 norm 参数 | 可被 `lora_schedule` 覆盖 |
| `--lora_tune_lm_head` | `False` | LoRA 时把 `lm_head` 加入目标模块 | 可被 `lora_schedule` 覆盖 |
| `--lora_tune_bias` | `False` | LoRA 时额外训练选中 Linear 的 bias | 可被 `lora_schedule` 覆盖 |
| `--lora_tune_protected_outliers` | `False` | LoRA 时额外训练 `VAELinear` 中被保护的 outlier 权重切片 | 可被 `lora_schedule` 覆盖 |
| `--lora_bias_categories` | `[]` | 允许训练 bias 的 Linear 类别列表（逗号分隔或 JSON 列表） | 为空表示全部 LoRA 目标 Linear |
| `--lora_loss_type` | `sft` | LoRA loss 类型 | 支持：`sft/origin/rkl/kl/mse/kd/r_kl_top[_K]/kl_top[_K]` |
| `--lora_use_dora` | `True` | LoRA 是否启用 DoRA | 解析 true/false 字符串 |
| `--lora_schedule` | `None` | 按“已完成类别”覆盖 LoRA 超参的 JSON | 支持 `default`/`*` 兜底键 |
| `--seed` | `0` | 全流程随机种子 | LoRA 每轮会加上轮次偏移 |
| `--train_device` | `cuda` | VAE 训练与评估设备 | 例如 `cuda` / `cuda:0` / `cpu` |
| `--convert` | `False` | 训练后将目标 Linear 替换为 `VAELinear` | 不开则只训练不替换 |
| `--convert_device` | `cuda` | 构建 `VAELinear` 时的设备 | 替换后会移回 CPU |
| `--save_model` | `False` | 最终保存模型及配置/tokenizer | 需要同时开启 `--convert` |
| `--unload_vae_original_weights_on_final_save` | `False` | 最终保存前卸载 `VAELinear` 里的原始权重缓存 | 减小模型体积 |
| `--output_dir` | `./output_linear_by_category` | 输出根目录 | 实际会创建时间戳子目录 |
| `--allow_tail_group` | `True` | 是否允许最后一个不足组大小的尾组训练 | 当前实现是 `store_true + default=True`，CLI 无法显式关掉 |

## 3. 通用 VAE/量化/优化参数（`vae_args`）

这些参数由 `train_utils/train_args.py` 提供，`cat_train.py` 在训练分组 VAE 时使用。

### 3.1 优化器与训练

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--lr` | `1e-4` | VAE 学习率 |
| `--beta1` | `0.9` | Adam/AdamW 的 `beta1`，SGD 的 momentum |
| `--beta2` | `0.95` | Adam/AdamW 的 `beta2` |
| `--weight_decay` | `1e-2` | 优化器权重衰减 |
| `--optimizer` | `adamw` | `adam/adamw/sgd/rmsprop` |
| `--lr_scheduler` | `none` | `none/linear/cosine` |
| `--lr_warmup_steps` | `0` | 学习率 warmup 步数 |

### 3.2 数据/损失/量化相关

| 参数 | 默认值 | 功能 | 备注 |
|---|---:|---|---|
| `--model_path` | `meta-llama/Llama-2-7b-hf` | 基座模型路径或 HF ID | 会用于加载模型、PPL、LoRA tokenizer |
| `--normalize_weight` | `False` | 切块前做 z-score 标准化 | 转换时会把均值方差融合回 decoder |
| `--recon_loss_type` | `mse` | 重建损失类型 | 支持标量或 JSON list；可选 `mse/l1/huber/relative_l1/top_k_mse/cosine/w_mse/w2_mse/wa_mse` |
| `--distil_loss_type` | `mse` | 蒸馏损失类型 | 当前 `cat_train.py` 主流程未使用 |
| `--distil_loss_weight` | `1.0` | 蒸馏损失权重 | 当前 `cat_train.py` 主流程未使用 |
| `--l1_weight` | `1.0` | 重建损失系数 | 在 `llm_vae` 中参与总损失 |
| `--lfq_weight` | `1.0` | 量化辅助损失系数 | 在 `llm_vae` 中参与总损失 |
| `--commitment_loss_weight` | `0.25` | BSQ commitment 权重 | 传入量化器 |
| `--entropy_loss_weight` | `0.1` | BSQ entropy 权重 | 传入量化器 |
| `--diversity_gamma` | `1.0` | BSQ diversity 超参 | 传入量化器 |
| `--use_checkpoint` | `False` | 是否在 VAE 编解码器中用 gradient checkpointing | 降显存、增算力 |
| `--new_quant` | `False` | BSQ 新量化分支开关 | 传入量化器 |
| `--w_input_batches` | `1` | 权重输入分批数 | 当前 `cat_train.py` 主流程未使用 |

### 3.3 结构参数

| 参数 | 默认值 | 功能 | 备注 |
|---|---:|---|---|
| `--codebook_bits` | `16` | latent bit 维度 | 支持标量整数、按类别 JSON dict，或按 stage 的 JSON list（元素可为前两种） |
| `--codebook_dim` | `8` | 权重切块大小（chunk size） | 支持标量整数、按类别 JSON dict，或按 stage 的 JSON list（元素可为前两种） |
| `--residual_stages` | `1` | 残差量化阶数 | `1` 为原单阶流程；`>1` 时采用逐阶残差量化（每阶完整训练步数） |
| `--base_ch` | `128` | 编解码器共享基础通道数 | 支持标量或 JSON list（按 stage）；encoder 恒定使用它 |
| `--num_res_blocks` | `1` | 编解码器共享残差块数量 | 支持标量或 JSON list（按 stage）；encoder 恒定使用它 |
| `--decoder_base_ch` | `None` | decoder hidden dim | 支持标量或 JSON list；仅 `decoder_type=asymmetric` 独立生效，默认回退 `base_ch` |
| `--decoder_num_res_blocks` | `None` | decoder 残差块数 | 支持标量或 JSON list；仅 `decoder_type=asymmetric` 独立生效，默认回退 `num_res_blocks` |
| `--quantizer_type` | `BSQ` | 量化器类型 | 当前主要支持 `BSQ` |
| `--gamma0` | `1.0` | BSQ 超参 | 传入 BSQ |
| `--gamma` | `1.0` | BSQ 超参 | 传入 BSQ |
| `--zeta` | `1.0` | BSQ 超参 | 传入 BSQ |
| `--inv_temperature` | `100.0` | BSQ 温度倒数 | 传入 BSQ |
| `--norm_type` | `group` | 归一化层类型 | 支持标量或 JSON list；`group/batch/layer/no` |
| `--decoder_type` | `linear` | decoder 结构类型 | 支持标量或 JSON list；`linear/symmetric/asymmetric` |
| `--parallel_layers` | `32` | 并行模型数 | `cat_train.py` 会忽略并给出 warning（实际用 `--intra_parallel`） |

## 4. HuggingFace 参数（`hf_args` + `training_args`）

### 4.1 HF 私有参数

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--access_token` | `None` | 访问 gated 模型（如 Llama）时的 HF token |

### 4.2 `TrainingArguments` 中本脚本直接/间接使用的关键项

| 参数 | 作用位置 | 功能 |
|---|---|---|
| `--bf16` | VAE 训练 dtype、LoRA SFT args | 开启 bf16 精度 |
| `--fp16` | VAE 训练 dtype、LoRA SFT args | 开启 fp16 精度 |
| `--output_dir` | LoRA `TrainingArguments` | LoRA 临时输出目录 |
| `--num_train_epochs` | LoRA `TrainingArguments` | LoRA epoch |
| `--gradient_accumulation_steps` | LoRA `TrainingArguments` | LoRA 梯度累积 |
| `--optim` | LoRA `TrainingArguments` | LoRA 优化器名称 |
| `--max_grad_norm` | LoRA `TrainingArguments` | LoRA 梯度裁剪 |
| `--warmup_ratio` | LoRA `TrainingArguments` | LoRA warmup 比例 |
| `--group_by_length` | LoRA `TrainingArguments` | LoRA 按长度分组 |
| `--lr_scheduler_type` | LoRA `TrainingArguments` | LoRA 学习率调度器类型 |
| `--model_max_length` | LoRA trainer `max_seq_length` | LoRA 样本最大长度 |

说明：

- `transformers.TrainingArguments` 还有大量通用参数，脚本可以接收，但 `cat_train.py` 主流程不一定消费。

## 5. 参数联动与常见报错条件

1. `--save_model` 需要同时开启 `--convert`，否则直接报错。
2. 当 `--recon_loss_type wa_mse` 时：
   - `--wa_mse_act_mode dynamic`：每组动态采集 act_max。
   - `--wa_mse_act_mode static`：必须提供 `--activation_weight_path`。
3. 当 `--outlier_protect_ratio > 0` 时，必须存在 activation 向量来源（`--activation_weight_path` 或可复用的 `wa_mse` dynamic act_max）；`--outlier_protect_axis input/output` 都使用输入激活对权重做加权打分；保护后若破坏 `intra_parallel` 或 `codebook_dim` 可整除性，会直接报错。
4. 当 `--intra_part_sort_mode` 在任一启用维度上使用 `act_l2`（例如 `act_l2,none` 且该维切分份数 `>1`）时，需要 activation 向量来源（通常是 `--activation_weight_path` 或 `wa_mse` 动态流程）。
5. `--skip_layers` 格式必须是 `<layer_idx>.<category>`，否则解析失败。
6. `--intra_parallel` 若用 JSON dict，必须命中当前类别或提供 `default` / `*` 兜底键。
7. 切分可整除性必须满足：
   - 若该类别在 `transpose_modules` 中：`in_features % row_parts == 0` 且 `out_features % col_parts == 0`
   - 否则：`out_features % row_parts == 0` 且 `in_features % col_parts == 0`
8. 切分后每个 part 的展平长度必须能被该类别生效的 `codebook_dim` 整除。

## 6. 关键复杂参数详解

### 6.1 `--intra_parallel`

支持三种写法：

1. `--intra_parallel 2`
2. `--intra_parallel 2,4`（等价 `[2,4]`）
3. `--intra_parallel '{"default":[2,1],"q_proj":[4,1],"k_proj":2}'`

语义：

- 第一维是 `row_parts`，第二维是 `col_parts`，总切分份数是 `row_parts * col_parts`。
- dict 模式可按类别覆盖；未命中类别会尝试 `default`，再尝试 `*`。

### 6.2 `--lora_schedule`

是一个 JSON 对象，按“已经完成的类别（after_category）”覆盖 LoRA 超参。

示例：

```json
{
  "default": {"rank": 8, "alpha": 16, "steps": 300, "loss_type": "sft"},
  "q_proj": {"rank": 64, "alpha": 128, "steps": 1000, "loss_type": "r_kl_top_1000", "use_dora": false}
}
```

可用键（规范名）：

- `rank`（`>=1`）
- `alpha`（`>0`）
- `dropout`（`>=0`）
- `steps`（`>=0`）
- `batch_size`（`>=1`）
- `nsamples`（`>=1`）
- `lr`
- `weight_decay`
- `log_every`（`>=1`）
- `tune_norm`（bool）
- `tune_lm_head`（bool）
- `tune_bias`（bool）
- `tune_protected_outliers`（bool）
- `bias_categories`（string/list）
- `loss_type`（与 `--lora_loss_type` 同集合）
- `use_dora`（bool）

同时支持别名键：`r/lora_rank`、`lora_alpha`、`lora_steps`、`lora_batch_size`、`lora_nsamples`、`lora_lr`、`lora_tune_protected_outliers` 等。

### 6.3 `--codebook_bits` / `--codebook_dim`（按类别 + 按 stage）

两者都支持三种写法：

1. 标量整数（全类别共用）
2. JSON dict（按类别覆盖）
3. JSON list（按 residual stage 覆盖；每个元素可为标量整数或 JSON dict）

示例：

```bash
--codebook_bits 32 \
--codebook_dim 16
```

```bash
--codebook_bits '{"default":32,"q_proj":24,"k_proj":24}' \
--codebook_dim '{"default":16,"q_proj":8,"down_proj":32}'
```

dict 模式下：

- 命中优先级：`category` > `default` > `*`
- 若某类别未命中且没有 `default/*`，会直接报错

按 stage 示例（2 阶）：

```bash
--residual_stages 2 \
--codebook_bits '[16,{"default":12,"q_proj":14}]' \
--codebook_dim '[16,8]'
```

### 6.4 输出目录实际结构

`--output_dir` 是根目录，真实运行目录会自动生成为：

`<output_dir>/<safe_model_name>_<YYYYmmdd_HHMMSS>/`

例如：`./output_linear_by_category/meta-llama__Llama-2-7b-hf_20260303_103000/`。

### 6.5 `--residual_stages`（多阶残差量化）

- 默认 `1`，保持当前单阶行为。
- 当设置为 `N>1` 时，训练流程为逐阶残差：第 1 阶重构后计算残差，第 2..N 阶继续量化残差。
- 当前实现中每一阶都会使用完整步数；可通过 `--steps_per_category`/`--steps_per_group` 的 JSON list 做按阶覆盖。
- 结构/损失参数可按阶覆盖：`codebook_bits/codebook_dim/base_ch/num_res_blocks/decoder_base_ch/decoder_num_res_blocks/norm_type/decoder_type/recon_loss_type/intra_part_sort_mode`。
- 当 `--normalize_weight` 开启时，会对“每一阶当前 residual”分别计算 `(mean,std)` 做标准化训练，并在该阶 decoder 内融合回去。
- 当 `intra_part_sort_mode` 按阶给出不同值时，切分排序仅在 stage1（stage0 索引）执行，后续 stage 复用同一切分顺序。

`intra_part_sort_mode` 补充说明（转置后执行）：

- 单值：`l2` 等价于 `l2,l2`（两维同模式）。
- 双值：`l2,none` 表示第一维排序、第二维不排序；`none,l2` 则相反。
- `l2/act_l2` 会先按该维 L2 分数降序，再按 `codebook_dim` 宽度做蛇形交错分配，分散高分通道。
- 即使 `--intra_parallel 1`，只要模式不是 `none,none` 也会执行对应维度排序。
- 按阶覆盖：可传 JSON list；若某阶需要双值，可写成嵌套形式，例如 `[['l2','none'], 'l2']`。

## 7. 快速示例

### 7.1 最小可跑（只训练不替换）

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --steps_per_category 200 \
  --batch_size 256 \
  --train_device cuda
```

### 7.2 典型压缩流程（训练 + 替换 + 保存）

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --save_model \
  --output_dir ./output_linear_by_category \
  --steps_per_category 2000 \
  --linear_group_size 32 \
  --intra_parallel 2,1 \
  --codebook_bits 16 \
  --codebook_dim 8 \
  --residual_stages 1 \
  --recon_loss_type mse \
  --train_device cuda \
  --bf16 True
```

### 7.3 两阶残差量化示例

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --steps_per_category 2000 \
  --codebook_bits 16 \
  --codebook_dim 8 \
  --residual_stages 2 \
  --train_device cuda
```

### 7.4 `wa_mse + dynamic act_max`

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --recon_loss_type wa_mse \
  --wa_mse_act_mode dynamic \
  --wa_mse_calib_dataset wikitext2 \
  --wa_mse_calib_nsamples 512 \
  --wa_mse_calib_seqlen 512 \
  --train_device cuda
```
