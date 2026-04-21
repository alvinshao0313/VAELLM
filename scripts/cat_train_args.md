# `tools/cat_train.py` 参数说明

本文档按当前代码真实行为整理，覆盖以下模块：

- [tools/cat_train.py](/home/shaoyuantian/program/VAELLM/tools/cat_train.py)
- [train_utils/cat_train_args.py](/home/shaoyuantian/program/VAELLM/train_utils/cat_train_args.py)
- [litebsq/vae_args.py](/home/shaoyuantian/program/VAELLM/litebsq/vae_args.py)
- [litebsq/autoencoder.py](/home/shaoyuantian/program/VAELLM/litebsq/autoencoder.py)
- [train_utils/lora_utils.py](/home/shaoyuantian/program/VAELLM/train_utils/lora_utils.py)

## 1. 参数来源与解析顺序

`cat_train.py` 的参数分三层解析：

1. 脚本私有参数：`build_cat_train_parser()`  
   位置：[train_utils/cat_train_args.py](/home/shaoyuantian/program/VAELLM/train_utils/cat_train_args.py)
2. VAE / 量化 / 优化器参数：`_build_cat_train_vae_parser()`  
   位置：[train_utils/cat_train_args.py](/home/shaoyuantian/program/VAELLM/train_utils/cat_train_args.py)
3. HuggingFace 参数：`HFArguments + CatTrainHFTrainingArguments`  
   位置：[train_utils/train_args.py](/home/shaoyuantian/program/VAELLM/train_utils/train_args.py) 和 [train_utils/cat_train_args.py](/home/shaoyuantian/program/VAELLM/train_utils/cat_train_args.py)

说明：

- 第 1 层负责类别训练流程、分组、LoRA、保存、校准等控制逻辑。
- 第 2 层负责 `MultiLayerVAE` 使用的结构、量化、损失、优化器参数。
- 第 2 层解析出的结构参数在真正构建 VAE 前，会由 [litebsq/vae_args.py](/home/shaoyuantian/program/VAELLM/litebsq/vae_args.py) 做默认值归一化。
- 第 3 层只接收当前 `cat_train` 真正会用到的少量 HF / LoRA trainer 参数，不再暴露整包 `TrainingArguments`。

## 2. 统一 override 语法

`cat_train` 现在的复杂参数不再接受 JSON dict / JSON list / 旧的 stage schedule 字符串。统一改为 `selector=value` 风格。

### 2.1 按类别覆盖

用于：

- `steps_per_category`
- `joint_decoder_steps`
- `joint_decoder_lr`
- `joint_decoder_group_size`
- `intra_parallel`
- `intra_part_sort_mode`
- `outlier_protect_count`
- `outlier_residual_top_p`
- `codebook_bits`
- `codebook_dim`
- `residual_stages`
- `base_ch`
- `num_res_blocks`
- `decoder_base_ch`
- `decoder_num_res_blocks`
- `recon_loss_type`
- `norm_type`
- `decoder_type`

写法：

```bash
--codebook_bits default=16,cat:q_proj=24
--steps_per_category default=2000,cat:q_proj=500
--intra_parallel default=1x1,cat:q_proj=4x1
--outlier_residual_top_p default=0.01,cat:down_proj=0.02
```

规则：

- 允许的 selector 只有 `default` 和 `cat:<category>`
- 命中顺序：`cat:<category>` > `default`
- 未命中且没有 `default` 会报错
- 所有 category key 会在发现真实线性层类别后统一校验

### 2.2 按 after-category 覆盖

用于所有 LoRA 参数：

- `lora_rank`
- `lora_alpha`
- `lora_dropout`
- `lora_steps`
- `lora_batch_size`
- `lora_nsamples`
- `lora_lr`
- `lora_weight_decay`
- `lora_log_every`
- `lora_temperature`
- `lora_loss_alpha`
- `lora_loss_type`
- `lora_use_dora`

写法：

```bash
--lora_rank default=8,after:q_proj=16
--lora_steps default=50,after:q_proj=200
```

规则：

- 允许的 selector 只有 `default` 和 `after:<category>`
- 命中顺序：`after:<category>` > `default`
- 这里的 `<category>` 指“已经完成训练的类别”

### 2.3 值的编码规则

| 类型 | 写法示例 |
|---|---|
| 整数 | `16` |
| 浮点 | `1e-4` / `0.1` |
| 布尔 | `true` / `false` |
| 可空整数 | `none` / `256` |
| `intra_parallel` | `1x1` / `4x1` |
| `intra_part_sort_mode` | `none` / `spectral_cosine` / `act_spectral_cosine` |

### 2.4 仍保留旧字符串语法的参数

以下参数没有并入 override 系统，仍然使用原有简单字符串格式：

- `category_order`
- `transpose_modules`
- `projection_suffixes`
- `skip_layers`
- `outlier_residual_codec`
- `outlier_residual_index_bits`
- `outlier_residual_value_bits`

其中：

- `category_order` 支持 `auto` 或逗号分隔列表
- `skip_layers` 格式仍是 `layer_idx.category`，例如 `0.down_proj,30.q_proj`

## 3. 脚本私有参数（`cat_args`）

| 参数 | 默认值 | 功能 | 备注 |
|---|---:|---|---|
| `--category_order` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | 类别训练顺序 | 可设 `auto`；可包含 `others` |
| `--transpose_modules` | `v_proj,o_proj,gate_proj,up_proj,down_proj` | 这些类别在切分前先转置权重 | 影响切分方向与整除性 |
| `--projection_suffixes` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | 默认 projection-only 模式下允许的后缀 | 仅在未开启 `include_all_linears` 时生效 |
| `--include_all_linears` | `False` | 关闭默认的 projection-only 过滤，改为收集模型中全部 `nn.Linear` | 开启后忽略 `projection_suffixes` 过滤 |
| `--steps_per_category` | `default=2000` | 每个 group 的训练步数 | 类别 override；名字保留但语义就是每组步数 |
| `--joint_decoder_steps` | `default=none` | 多阶段训练完成后的 decoder 联合微调步数 | 类别 override；`none` 表示回退到该类别的 `steps_per_category` |
| `--joint_decoder_lr` | `default=none` | 多阶段 decoder 联合微调的学习率 | 类别 override；`none` 表示回退到全局 `--lr` |
| `--joint_decoder_group_size` | `default=none` | 多阶段 decoder 联合微调时的子分组大小 | 类别 override；`none` 表示回退到 `linear_group_size` |
| `--skip_layers` | `""` | 指定某些层在推理时始终走原始权重 | 格式必须是 `layer_idx.category` |
| `--linear_group_size` | `32` | 同类别跨层分组大小 | 必须 `>=1` |
| `--intra_parallel` | `default=1x1` | 单个 Linear 的层内切分 | 类别 override |
| `--intra_part_sort_mode` | `default=none` | 每个 part 内的列排序方式 | 类别 override；支持 `none/spectral_cosine/act_spectral_cosine` |
| `--sort_prep_workers` | `0` | 排序预处理并行 worker 数 | 全局单值；`0=auto, 1=串行, >1=显式 CPU 多进程`；只影响 `spectral_cosine/act_spectral_cosine` |
| `--batch_size` | `256` | VAE 训练与评估 DataLoader batch 大小 | 作用于块数据，不是 token batch |
| `--log_every` | `50` | 每多少 step 打印一次训练日志 | `<=0` 等价关闭 |
| `--eval_every` | `0` | 每多少 step 做一次 VAE 中间评估 | `0` 表示不做中间评估 |
| `--eval_blocks` | `256` | 每次中间评估最多评估多少块 | 与 `eval_every` 联动 |
| `--outlier_protect_count` | `default=0` | `channel` 模式下保护 top-N channel 不参与压缩 | 类别 override；`residual_sparse` 模式要求它对所有类别都为 `0` |
| `--outlier_protect_mode` | `channel` | 离群值保护模式 | `channel` / `residual_sparse`，两者互斥 |
| `--outlier_residual_top_p` | `default=0.0` | `residual_sparse` 模式下保留最终重构残差 top-p 比例元素 | 类别 override；对所有参与训练的类别要求 `0 < p <= 1` |
| `--outlier_residual_score` | `abs` | `residual_sparse` 模式下的选点打分方式 | `abs` / `input_act_weighted_abs` / `original_weight_abs` / `input_act_weighted_original_weight_abs`；原始权重打分只影响选点，最终保存的仍是对应 residual |
| `--outlier_residual_min_abs` | `1e-6` | `residual_sparse` 模式下 residual 的最小绝对值门槛 | 全局单值；若 `|original-reconstructed| < threshold`，该位置会从 top-p 中剔除，并继续往后补 |
| `--outlier_residual_codec` | `coo_fp16` | `residual_sparse` 的存储格式 | `coo_fp16` / `blocked_quantized` |
| `--outlier_residual_index_bits` | `8` | `blocked_quantized` 的块内索引位宽 | `8` / `4`；`4` 位时 block 边长必须 `<=16` |
| `--outlier_residual_value_bits` | `8` | `blocked_quantized` 的残差 value 位宽 | `8` / `4` |
| `--outlier_protect_axis` | `input` | 保护输入还是输出通道 | `input` / `output` |
| `--wa_mse_calib_dataset` | `wikitext2` | 动态采集时的校准集 | 供 `wa_mse / act_spectral_cosine / channel outlier protect / residual_sparse(activation-weighted score)` 共用；支持 `wiki / wikitext2 / fineweb_edu / openorca / redpajama / alpaca`，其中 `wikitext2` 是 `wiki` 别名 |
| `--wa_mse_calib_nsamples` | `512` | 动态采集样本数 | 供 `wa_mse / act_spectral_cosine / channel outlier protect / residual_sparse(activation-weighted score)` 共用 |
| `--wa_mse_calib_seqlen` | `512` | 动态采集序列长度 | 供 `wa_mse / act_spectral_cosine / channel outlier protect / residual_sparse(activation-weighted score)` 共用 |
| `--wa_mse_calib_seed` | `0` | 动态采集随机种子 | 供 `wa_mse / act_spectral_cosine / channel outlier protect / residual_sparse(activation-weighted score)` 共用 |
| `--wa_mse_calib_device` | `""` | 动态采集设备 | 为空时回退 `train_device` |
| `--wa_mse_calib_log_every` | `0` | 动态采集日志间隔 | `0` 表示关闭 |
| `--eval_ppl` | `true` | 是否在类别后评估阶段运行 PPL | 现在和 `convert` 解耦；`false` 时不跑 PPL |
| `--eval_tasks` | `""` | 类别后 lm_eval 任务列表 | 逗号分隔；空串表示不跑下游任务；当前固定 `fewshot=0`、`batch_size=auto`、`limit=None` |
| `--ppl_limit` | `-1` | 每个类别训练后 PPL 评估样本上限 | `-1` 表示全量 |
| `--lora_after_category` | `False` | 每训练完一个类别后，对剩余类别做一次 LoRA 微调并融合 | 开启后才会进入 LoRA 阶段 |
| `--lora_dataset` | `wiki` | LoRA 补偿训练数据集 | 支持 `wiki / fineweb_edu / openorca / redpajama / alpaca` |
| `--lora_rank` | `default=8` | LoRA rank | after-category override |
| `--lora_alpha` | `default=16.0` | LoRA alpha | after-category override |
| `--lora_dropout` | `default=0.0` | LoRA dropout | after-category override |
| `--lora_steps` | `default=50` | LoRA 最大步数 | `0` 表示跳过该轮 LoRA |
| `--lora_batch_size` | `default=2` | LoRA 每卡 batch size | after-category override |
| `--lora_nsamples` | `default=128` | LoRA 训练样本数 | 从 `--lora_dataset` 指定的数据集采样 |
| `--lora_lr` | `default=1e-4` | LoRA 学习率 | after-category override |
| `--lora_weight_decay` | `default=0.0` | LoRA 权重衰减 | after-category override |
| `--lora_log_every` | `default=1` | LoRA 日志间隔 | after-category override |
| `--lora_temperature` | `default=1.0` | LoRA 蒸馏温度 | after-category override |
| `--lora_loss_alpha` | `default=0.5` | LoRA 蒸馏混合权重 | after-category override |
| `--lora_loss_type` | `default=sft` | LoRA loss 类型 | after-category override |
| `--lora_use_dora` | `default=true` | LoRA 是否启用 DoRA | after-category override |
| `--lora_hif4_act` | `false` | 是否只在 LoRA 阶段对 student 线性层输入启用 HiFloat4 激活伪量化 | 全局开关，不参与 after-category override |
| `--eval_hif4_act` | `false` | 是否在 cat_train 内部类别后评估阶段启用 HiFloat4 激活伪量化 | 同时作用于 PPL 和 lm_eval；不影响训练 |
| `--seed` | `0` | 全流程随机种子 | LoRA 每轮会叠加轮次偏移 |
| `--train_device` | `cuda` | VAE 训练与评估设备 | 例如 `cuda` / `cuda:0` / `cpu` |
| `--rot_llm` | `False` | 压缩前先做一次离线旋转融合 | 调用 rotation 流程 |
| `--resume_from_checkpoint` | `None` | 从已有 `cat_train` checkpoint 继续训练 | 可传 run 目录、`final_model` 目录，或 `checkpoint_meta.json` |
| `--convert` | `False` | 训练后把目标 Linear 替换为压缩后的 `VAELinear` | 只控制是否替换模型权重；不再隐式控制 PPL |
| `--convert_device` | `cuda` | 构建 `VAELinear` 时的设备 | 替换完成后会移回 CPU |
| `--save_model` | `False` | 最终保存模型 state_dict / config / tokenizer | 需要同时开启 `convert` |
| `--unload_vae_original_weights_on_final_save` | `False` | 最终保存前卸载 `VAELinear` 中缓存的原始权重 | 用于减小保存体积 |
| `--output_dir` | `./output_linear_by_category` | 输出根目录 | 实际会创建时间戳子目录 |
| `--allow_tail_group` | `True` | 是否允许最后一个不足组大小的尾组训练 | 可显式传 `false` 跳过尾组 |

## 4. VAE / 量化 / 优化器参数（`vae_args`）

这些参数由 [train_utils/cat_train_args.py](/home/shaoyuantian/program/VAELLM/train_utils/cat_train_args.py) 解析，最终由 [litebsq/vae_args.py](/home/shaoyuantian/program/VAELLM/litebsq/vae_args.py) 和 [litebsq/autoencoder.py](/home/shaoyuantian/program/VAELLM/litebsq/autoencoder.py) 消费。

### 4.1 结构参数

下列参数全部使用“按类别 override”语法：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--codebook_bits` | `default=16` | latent bit 维度 |
| `--codebook_dim` | `default=8` | 权重切块大小（chunk size） |
| `--residual_stages` | `default=1` | 每个类别的残差量化阶数 |
| `--base_ch` | `default=128` | encoder hidden dim |
| `--num_res_blocks` | `default=1` | encoder 残差块数 |
| `--decoder_base_ch` | `default=none` | asymmetric decoder hidden dim；`none` 回退到 `base_ch` |
| `--decoder_num_res_blocks` | `default=none` | asymmetric decoder 残差块数；`none` 回退到 `num_res_blocks` |
| `--recon_loss_type` | `default=mse` | 重建损失类型 |
| `--norm_type` | `default=group` | `group/batch/layer/no` |
| `--decoder_type` | `default=linear` | `linear/symmetric/asymmetric` |

重要语义：

- `residual_stages` 允许按类别不同。
- 同一类别内，如果 `residual_stages > 1`：
  - 先做逐阶残差量化；若 `intra_part_sort_mode != none`，每个 stage 都会基于当前 residual 重新排序
  - 所有 stage 仍复用同一组已解析结构参数：
    - `steps`
    - `codebook_bits`
    - `codebook_dim`
    - `recon_loss_type`
    - `intra_part_sort_mode`
    - `base_ch`
    - `num_res_blocks`
    - `decoder_base_ch`
    - `decoder_num_res_blocks`
    - `norm_type`
    - `decoder_type`
  - 全部 stage 训练完后，会再做一次 decoder 联合微调，步数由 `joint_decoder_steps` 控制
- `joint_decoder_steps=none` 时，回退到该类别解析后的 `steps_per_category`
- `joint_decoder_lr=none` 时，回退到全局 `lr`
- `joint_decoder_group_size=none` 时，联合微调直接沿用当前类别的 `linear_group_size`
- `decoder_type=linear` 或 `decoder_type=symmetric` 时，decoder 的 hidden dim / residual blocks 会强制对齐 encoder。
- `decoder_type=asymmetric` 时，才独立使用 `decoder_base_ch` / `decoder_num_res_blocks`。

### 4.2 优化器与训练相关

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--lr` | `1e-4` | VAE 学习率 |
| `--beta1` | `0.9` | Adam/AdamW 的 `beta1`，SGD 的 momentum |
| `--beta2` | `0.95` | Adam/AdamW 的 `beta2` |
| `--weight_decay` | `1e-2` | 权重衰减 |
| `--optimizer` | `adamw` | `adam/adamw/sgd/rmsprop` |
| `--lr_scheduler` | `none` | `none/linear/cosine` |
| `--lr_warmup_steps` | `0` | warmup 步数 |

### 4.3 数据 / 损失 / 量化相关

| 参数 | 默认值 | 说明 | 备注 |
|---|---:|---|---|
| `--model_path` | `meta-llama/Llama-2-7b-hf` | 基座模型路径或 HF ID | 用于加载模型、PPL、LoRA tokenizer |
| `--normalize_weight` | `False` | 每个 residual stage 对当前 residual 做 z-score 标准化训练 | 转换时会把 `(mean,std)` 融合回 decoder |
| `--l1_weight` | `1.0` | recon loss 系数 | 参与 VAE loss |
| `--lfq_weight` | `1.0` | quantizer 辅助损失系数 | 参与 VAE loss |
| `--commitment_loss_weight` | `0.25` | BSQ commitment 权重 | 传入量化器 |
| `--entropy_loss_weight` | `0.1` | BSQ entropy 权重 | 传入量化器 |
| `--diversity_gamma` | `1.0` | BSQ diversity 超参 | 传入量化器 |
| `--quantizer_type` | `BSQ` | 量化器类型 | 当前实现支持 `BSQ` 和 `Identity` |
| `--gamma0` | `1.0` | BSQ 超参 | |
| `--gamma` | `1.0` | BSQ 超参 | |
| `--zeta` | `1.0` | BSQ 超参 | |
| `--inv_temperature` | `100.0` | BSQ 超参 | |
| `--use_checkpoint` | `False` | 是否在 VAE 编解码器中启用 gradient checkpointing | 降显存，增算力 |
| `--new_quant` | `False` | BSQ 新量化分支开关 | 影响 decoder q-scale 融合逻辑 |

## 5. HuggingFace 参数（`hf_args` + `training_args`）

### 5.1 HF 私有参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--access_token` | `None` | 访问 gated 模型时的 HF token |

### 5.2 `cat_train` 实际使用的 trainer 参数

| 参数 | 作用位置 | 说明 |
|---|---|---|
| `--bf16` | VAE dtype / LoRA SFT args | 为 `cat_train` 设置 `vae_weight_dtype=bf16` 和 `vae_autocast_dtype=bf16`，LoRA 也会透传 |
| `--fp16` | VAE 数据 dtype / LoRA SFT args | 影响 VAE 训练数据张量 dtype 和 LoRA SFT args；不会覆盖 `vae_weight_dtype` |
| `--lora_gradient_accumulation_steps` | LoRA `TrainingArguments` | LoRA 梯度累积 |
| `--lora_optim` | LoRA `TrainingArguments` | LoRA 优化器名称 |
| `--lora_max_grad_norm` | LoRA `TrainingArguments` | LoRA 梯度裁剪 |
| `--lora_warmup_ratio` | LoRA `TrainingArguments` | LoRA warmup 比例 |
| `--lora_group_by_length` | LoRA `TrainingArguments` | LoRA 按长度分组 |
| `--lora_lr_scheduler_type` | LoRA `TrainingArguments` | LoRA 学习率调度器类型 |
| `--lora_model_max_length` | LoRA trainer | LoRA 样本最大长度 |
| `--lora_hif4_act` | LoRA trainer | 是否只在 LoRA 阶段对 student 线性层输入启用 HiFloat4 激活伪量化；默认 `false` |
| `--eval_hif4_act` | cat_train eval | 是否在内部类别后评估时启用 HiFloat4 激活伪量化；默认 `false` |

## 6. 关键运行时语义

### 6.1 类别解析与分组

- 线性层先按 `category_order` 排序，再按层号排序。
- 默认只收集 decoder projection 相关的 `nn.Linear`；如果传 `--include_all_linears`，才会放开成全量 `nn.Linear`。
- `skip_layers` 会在发现真实 `(layer_idx, category)` 后校验；如果有未知项，直接报错。
- `linear_group_size` 控制同类别跨层分组大小。
- 如果最后一个 group 不足 `linear_group_size`：
  - `allow_tail_group=true`：照常训练
  - `allow_tail_group=false`：直接跳过尾组

### 6.2 `residual_stages`

- `residual_stages` 是按类别解析的。
- 某类别为 `N>1` 时，会做逐阶残差量化：当前 stage 重建后，从 residual 中扣除该 stage 重建结果，再进入下一 stage。
- 若 `intra_part_sort_mode != none`，每个 stage 都会基于当前 residual 重新排序，而不是复用 stage1 顺序。
- 全部 stage 训练完后，会额外做一次 decoder 联合微调。
- 但同一类别的各个 stage 不再有单独结构参数配置，自由度只剩“stage 数量”、`joint_decoder_steps`、`joint_decoder_lr` 和 `joint_decoder_group_size`。

### 6.3 `steps_per_category`

- 当前实现里它表示“这个类别里每个 group 训练多少步”
- 它不是“整个类别总步数”

### 6.4 `joint_decoder_steps`

- 只在 `residual_stages > 1` 时生效。
- 它控制“全部 stage 训练完之后”的 decoder 联合微调步数。
- `default=none` 表示自动回退到该类别已解析的 `steps_per_category`。

### 6.5 `joint_decoder_group_size`

- 只在 `residual_stages > 1` 时生效。
- 它控制联合微调时一次打包多少条 linear 一起做 full-batch。
- `default=none` 表示自动回退到 `linear_group_size`。
- 当它小于训练 group 时，前面的 VAE stage 训练仍按原 group 跑，只在最后的联合微调阶段拆小。
- 当前 `recon_loss_type=cosine` 和 `relative_l1` 不支持把联合微调 group 拆小。

### 6.6 `joint_decoder_lr`

- 只在 `residual_stages > 1` 且 `joint_decoder_steps > 0` 时生效。
- 它控制“全部 stage 训练完之后”的 decoder 联合微调学习率。
- `default=none` 表示自动回退到全局 `--lr`。

### 6.7 类别后评估

- 类别后评估由 `--eval_ppl` 和 `--eval_tasks` 共同控制。
- `--eval_ppl=true` 时跑 PPL。
- `--eval_tasks` 非空时跑 lm_eval，并在日志里记录：
  - 每个任务的 `metric_key` 和分数
  - 所有有效任务分数的简单平均值（忽略 `N/A`）
- `--eval_ppl=false` 且 `--eval_tasks=""` 时，整个类别后评估阶段直接跳过。
- 如果配置了 `--eval_tasks`，但所有任务结果都是 `N/A`，脚本会直接报错。

### 6.8 `convert` 与评估的关系

- `--convert` 只控制是否把训练出的压缩结果替换回模型里的目标 `Linear`。
- 不开 `--convert` 时：
  - 不会把模块替换成 `VAELinear`
  - 仍然可以按 `--eval_ppl/--eval_tasks` 跑类别后评估
- `--lora_after_category` 和 `--save_model` 仍然要求 `--convert` 开启，因为它们依赖“压缩结果已经写回模型”。

### 6.9 `normalize_weight`

- 训练时会对当前 stage 的 residual 计算 `(mean, std)` 并标准化训练
- 转换时会把该 stage 的 `(mean, std)` 融合进 decoder
- 多阶 residual 会分别计算和融合各自的标准化统计量

### 6.10 `wa_mse` 与 activation 依赖

- `recon_loss_type=wa_mse` 时，会在当前 group 上动态重算 `act_max`
- `outlier_protect_mode=channel` 且 `outlier_protect_count > 0` 时，也复用同一条动态 activation 路径
- `outlier_protect_mode=residual_sparse` 且 `outlier_residual_score` 使用 activation 加权模式时，也复用同一条动态 activation 路径
- `intra_part_sort_mode` 若启用 `act_spectral_cosine`，也复用同一条动态 activation 路径
- 当前 `cat_train` 不再支持通过静态 activation 字典驱动这三类逻辑

### 6.11 保存与输出目录

- `save_model` 需要同时开启 `convert`
- 真实运行目录会自动变成：

`<output_dir>/<safe_model_name>_<YYYYmmdd_HHMMSS>/`

- LoRA trainer 的临时输出目录固定写到：

`<run_output_dir>/lora_trainer_state/`

- 运行目录中会额外写出：

`normalized_cat_train_args.json`

里面包含：

- 规范化后的 `cat_args`
- 规范化后的 `vae_args`
- `training_args`
- 每个类别的 resolved runtime config

### 6.12 从已保存 checkpoint 继续训练

- `--resume_from_checkpoint` 会优先加载已有 checkpoint，而不是重新从 `--model_path` 拉起纯基座模型
- 支持三种输入：
  - 上一次 `cat_train` 的 run 目录
  - run 目录下的 `final_model/`
  - 直接传 `checkpoint_meta.json`
- 恢复后，已经被转换成 `VAELinear` 的模块不会再被 `collect_linears()` 收集，因此这些类别会自动跳过
- 继续训练时，剩余仍是 `nn.Linear` 的类别会照常进入后续类别循环
- 恢复时会优先使用 checkpoint meta 里的 `base_model_path`；如果 meta 没写，才回退到 `--model_path`
- `--resume_from_checkpoint` 不能和 `--rot_llm` 同时使用，因为 checkpoint 已经包含要续训的模型权重

## 7. 已删除或不再支持的参数

以下旧参数已经从当前 `cat_train` CLI 中移除：

- `--lora_schedule`
- `--lora_bias_categories`
- `--lora_tune_bias`
- `--lora_tune_norm`
- `--lora_tune_protected_outliers`
- `--only_decoder_projections`
- `--parallel_layers`
- `--w_input_batches`
- `--num_train_epochs`
- `--per_device_train_batch_size`
- `--activation_weight_path`
- `--wa_mse_act_mode`

替代方式：

- LoRA 超参统一改为 `default=...,after:<category>=...`

例如：

```bash
--lora_rank default=8,after:q_proj=16 \
--lora_steps default=50,after:q_proj=200
```

以下旧复杂语法也不再支持：

- 复杂参数使用 JSON dict
- 复杂参数使用 JSON list
- 旧的 stage schedule 风格

## 8. 常见报错条件

1. `save_model` 没有配合 `convert`
2. `skip_layers` 中包含不存在的 `layer_idx.category`
3. override 参数缺少 `default`，且当前类别没有显式值
4. override 参数包含未知 category / after-category key
5. 动态 activation 重算失败，导致当前 group 缺少 activation 向量
6. `outlier_protect_mode=channel` 且 `outlier_protect_count > 0`，或 `act_spectral_cosine`，或 `residual_sparse + activation 加权 score` 启用，但当前 group 的动态 activation 采集失败
7. 切分后某个 part 的展平长度无法被该类别的 `codebook_dim` 整除
8. `linear_group_size < 1`

## 9. 快速示例

### 9.1 最小可跑

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --steps_per_category default=200 \
  --batch_size 256 \
  --train_device cuda
```

### 9.2 典型压缩流程

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --save_model \
  --output_dir ./output_linear_by_category \
  --steps_per_category default=2000 \
  --linear_group_size 32 \
  --intra_parallel default=2x1 \
  --codebook_bits default=16 \
  --codebook_dim default=8 \
  --residual_stages default=1 \
  --recon_loss_type default=mse \
  --train_device cuda \
  --bf16 true
```

### 9.3 按类别覆盖结构参数

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --residual_stages default=1,cat:q_proj=2 \
  --codebook_bits default=16,cat:q_proj=24 \
  --codebook_dim default=8,cat:q_proj=16 \
  --base_ch default=128,cat:q_proj=192 \
  --decoder_type default=linear,cat:q_proj=asymmetric \
  --decoder_base_ch default=none,cat:q_proj=256 \
  --steps_per_category default=2000,cat:q_proj=1000
```

### 9.4 `wa_mse + dynamic act_max`

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --recon_loss_type default=wa_mse \
  --wa_mse_calib_dataset wikitext2 \
  --wa_mse_calib_nsamples 512 \
  --wa_mse_calib_seqlen 512 \
  --train_device cuda
```

### 9.5 LoRA after-category 覆盖

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --lora_after_category \
  --lora_dataset openorca \
  --lora_rank default=8,after:q_proj=16 \
  --lora_steps default=50,after:q_proj=200 \
  --eval_ppl true \
  --eval_tasks boolq,rte,piqa \
  --lora_hif4_act false \
  --eval_hif4_act false \
  --lora_use_dora default=true,after:q_proj=false
```

### 9.6 从已完成 `q_proj` 的保存结果继续训练其它类别

```bash
python tools/cat_train.py \
  --resume_from_checkpoint ./output_linear_by_category/your_run/final_model \
  --convert \
  --save_model \
  --output_dir ./output_linear_by_category_resume \
  --steps_per_category default=2000 \
  --train_device cuda
```

说明：

- 如果上一次已经把 `q_proj` 转成了 `VAELinear` 并保存，这次恢复后 `q_proj` 会自动跳过
- 脚本会继续训练还保持为 `nn.Linear` 的类别

### 9.7 只跑下游任务，不跑 PPL

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --eval_ppl false \
  --eval_tasks boolq,rte,piqa \
  --train_device cuda
```
