# `tools/cat_train.py` 参数说明

本文档按当前代码真实行为整理，覆盖以下模块：

- [tools/cat_train.py](/home/shaoyuantian/program/VAELLM/tools/cat_train.py)
- [train_utils/cat_train_args.py](/home/shaoyuantian/program/VAELLM/train_utils/cat_train_args.py)
- [litebsq/vae_args.py](/home/shaoyuantian/program/VAELLM/litebsq/vae_args.py)
- [litebsq/autoencoder.py](/home/shaoyuantian/program/VAELLM/litebsq/autoencoder.py)
- [train_utils/cat_train_residual_protection.py](/home/shaoyuantian/program/VAELLM/train_utils/cat_train_residual_protection.py)
- [train_utils/lora_utils.py](/home/shaoyuantian/program/VAELLM/train_utils/lora_utils.py)

当前排序代码和 joint decoder 联合优化代码已关闭；旧实现只以 `#` 注释形式保留在源码中。

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
- `activation_type`
- `decoder_type`

写法：

```bash
--codebook_bits default=16,cat:q_proj=24
--steps_per_category default=2000,cat:q_proj=500
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
- `distill_steps`
- `distill_batch_size`
- `distill_lr`
- `distill_weight_decay`
- `distill_log_every`
- `distill_temperature`
- `distill_loss_alpha`
- `distill_loss_type`
- `distill_hidden_loss_weight`
- `lora_use_dora`

写法：

```bash
--lora_rank default=8,after:q_proj=16
--distill_steps default=50,after:q_proj=200
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
| `intra_part_sort_mode` | 排序代码已关闭；只保留内部固定 `none` |

### 2.4 仍保留旧字符串语法的参数

以下参数没有并入 override 系统，仍然使用原有简单字符串格式：

- `target_categories`
- `transpose_modules`
- `skip_layers`
- `outlier_residual_codec`
- `outlier_residual_index_bits`
- `outlier_residual_value_bits`

其中：

- `target_categories` 必须是显式逗号分隔列表，不支持 `auto` / `others`
- `skip_layers` 格式仍是 `layer_idx.category`，例如 `0.down_proj,30.q_proj`

## 3. 脚本私有参数（`cat_args`）

| 参数 | 默认值 | 功能 | 备注 |
|---|---:|---|---|
| `--target_categories` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | 要压缩的类别及顺序 | 必须显式列出；不支持 `auto` / `others`；指定但模型中未发现会直接报错 |
| `--transpose_modules` | `v_proj,o_proj,gate_proj,up_proj,down_proj` | 这些类别在切分前先转置权重 | 影响切分方向与整除性 |
| `--include_all_linears` | `False` | 关闭默认 decoder projection 路径限制 | 仍只收集 `target_categories` 指定的类别 |
| `--steps_per_category` | `default=2000` | 每个 group 的训练步数 | 类别 override；名字保留但语义就是每组步数 |
| `--joint_decoder_steps` | 已关闭 | joint decoder 联合优化旧参数 | 不再注册；传入会报错 |
| `--joint_decoder_lr` | 已关闭 | joint decoder 联合优化旧参数 | 不再注册；传入会报错 |
| `--joint_decoder_group_size` | 已关闭 | joint decoder 联合优化旧参数 | 不再注册；传入会报错 |
| `--joint_decoder_batch_size` | 已关闭 | joint decoder 联合优化旧参数 | 不再注册；传入会报错 |
| `--skip_layers` | `""` | 指定某些层在推理时始终走原始权重 | 格式必须是 `layer_idx.category` |
| `--linear_group_size` | `32` | 同类别跨层分组大小 | 必须 `>=1` |
| `--intra_part_sort_mode` | 已关闭 | 排序旧参数 | 不再注册；传入会报错 |
| `--sort_prep_workers` | 已关闭 | 排序预处理旧参数 | 不再注册；传入会报错 |
| `--batch_size` | `256` | VAE 训练与评估 DataLoader batch 大小 | 作用于块数据，不是 token batch |
| `--log_every` | `50` | 每多少 step 打印一次训练日志 | `<=0` 等价关闭 |
| `--eval_every` | `0` | 每多少 step 做一次 VAE / joint decoder 中间评估 | `0` 表示不做中间评估；joint decoder 前后完整重建损失仍会记录 |
| `--eval_blocks` | `256` | 每次 VAE / joint decoder 中间评估最多评估多少块 | 与 `eval_every` 联动；joint decoder 前后完整重建损失不受它限制 |
| `--outlier_protect_count` | `default=0` | `channel/channel_residual_vae` 模式下选择 top-N channel | 类别 override；`residual_sparse` 要求它对所有类别都为 `0` |
| `--outlier_protect_mode` | `channel` | 离群值保护模式 | `none` / `channel` / `channel_residual_vae` / `residual_sparse`，互斥 |
| `--outlier_channel_scope` | `layer` | channel 计数范围 | `layer` 为每层独立 top-N；`category` 为同类所有层全局排序，总预算为 `N * 有效 linear 数` |
| `--outlier_residual_top_p` | `default=0.0` | `residual_sparse` 模式下保留最终重构残差 top-p 比例元素 | 类别 override；`residual_sparse` 要求 `0 < p <= 1` |
| `--outlier_rank_metric` | `sparse_residual_abs` | 离群选择排序指标 | `residual_sparse` 只允许 `sparse_*`；`channel_residual_vae` 只允许 `channel_*`；`channel` 实际保护通道时只允许 `channel_weight_*`；`actmax` 使用激活绝对值最大值，`actmean` 使用激活绝对值均值 |
| `--outlier_mlp_rank_metric` | `none` | MLP gate/up/down 专用选道指标 | `none` 时 MLP 仍走 `--outlier_rank_metric`；`mlp_intermediate_aligned_actrms` / `actmean_abs` / `actrms_abs` 时按 SwiGLU intermediate path 共享保护通道 |
| `--outlier_mlp_fuse_weights` | `1,1,1` | MLP aligned 选道的 up/gate/down 融合权重 | 格式 `alpha_up,alpha_gate,alpha_down`；对所有 aligned MLP metric 生效 |
| `--outlier_residual_min_abs` | `1e-6` | `residual_sparse` 模式下 residual 的最小绝对值门槛 | 全局单值；若 `|original-reconstructed| < threshold`，该位置会从 top-p 中剔除，并继续往后补 |
| `--outlier_residual_codec` | `coo_fp16` | `residual_sparse` 的存储格式 | `coo_fp16` / `blocked_quantized` |
| `--outlier_residual_index_bits` | `8` | `blocked_quantized` 的块内索引位宽 | `8` / `4`；`4` 位时 block 边长必须 `<=16` |
| `--outlier_residual_value_bits` | `8` | `blocked_quantized` 的残差 value 位宽 | `8` / `4` |
| `--outlier_protect_axis` | `input` | 保护输入还是输出通道 | `input` / `output` |
| `--outlier_protect_channel_quant` | `none` | `channel` 模式下 protected channel 权重存储格式 | `none` / `fp8_e4m3` / `fp8_e5m2` / `int8`；per-channel 对称量化，scale 为 bf16 |
| `--outlier_residual_vae_stages` | `default=1` | protected channel residual VAE 阶数 | 类别 override |
| `--outlier_residual_vae_batch_multiplier` | `1` | protected residual VAE batch 放大倍数 | 只影响 protected residual VAE，不影响 base VAE stage1/stage2；category shared residual VAE 推荐 `32`，实际 batch 会被 residual block 总数限制 |
| `--outlier_residual_vae_steps` | `0` | protected residual VAE 独立训练步数 | `0` 表示继承 base VAE residual stage steps；category shared residual VAE 推荐 `1000~1500` |
| `--outlier_residual_vae_lr` | `0.0` | protected residual VAE 独立学习率 | `0` 表示继承 base VAE lr；category shared residual VAE 推荐 `0.001~0.003`，当前建议 `0.002` |
| `--activation_calib_dataset` | `""` | 动态采集时的校准混合数据集 | 供 `wa_mse / channel outlier protect / MLP aligned outlier protect / residual_sparse(activation-weighted score)` 共用；启用动态校准时必填；只支持 `alias=weight,...`，例如 `wiki=1.0`、`openorca=1.0` 或 `openorca=0.5,fineweb_edu=0.5` |
| `--activation_calib_nsamples` | `512` | 动态采集样本数 | 供 `wa_mse / channel outlier protect / MLP aligned outlier protect / residual_sparse(activation-weighted score)` 共用 |
| `--activation_calib_seqlen` | `512` | 动态采集序列长度 | 供 `wa_mse / channel outlier protect / MLP aligned outlier protect / residual_sparse(activation-weighted score)` 共用 |
| `--activation_calib_seed` | `0` | 动态采集随机种子 | 供 `wa_mse / channel outlier protect / MLP aligned outlier protect / residual_sparse(activation-weighted score)` 共用 |
| `--activation_calib_device` | `""` | 动态采集设备 | 为空时回退 `train_device` |
| `--activation_calib_log_every` | `0` | 动态采集日志间隔 | `0` 表示关闭 |

`--activation_calib_*` 语义：

- 校准 forward **全 run 只执行一次**，在首个 category 开始 VAE 优化前，对全部 target linears 挂 hook 并跑一遍校准数据。
- 统计来源是**尚未被 VAE 替换的基座 Linear** 输入激活；hook 内以 float32 累积 `max` / `abs_mean` / `sq_mean`。
- 后续 category / group 只做 lookup 复用，不再重复跑校准 forward。
- 供 `wa_mse` / `amse` 损失加权、通道重要性判定、`residual_sparse` activation-weighted 打分等共用同一份统计。

| `--eval_ppl` | `true` | 是否在类别后评估阶段运行 PPL | 现在和 `convert` 解耦；`false` 时不跑 PPL |
| `--eval_tasks` | `""` | 类别后 lm_eval 任务列表 | 逗号分隔；空串表示不跑下游任务；当前固定 `fewshot=0`、`batch_size=auto`、`limit=None` |
| `--ppl_limit` | `-1` | 每个类别训练后 PPL 评估样本上限 | `-1` 表示全量 |
| `--distill_after_category` | `none` | 每训练完一个类别后的蒸馏模式 | `none` / `remaining_lora` / `compressed_lora` / `decoder` / `both`；非 `none` 要求开启 `--convert` |
| `--distill_dataset` | `""` | 每类后蒸馏训练混合数据集 | `--distill_after_category != none` 时必填；只支持 `alias=weight,...`，例如 `wiki=1.0`、`openorca=1.0` 或 `openorca=0.5,fineweb_edu=0.5`；alias 对齐 dense_e2e 的 `dataset_mix` |
| `--lora_rank` | `default=8` | LoRA rank | after-category override |
| `--lora_alpha` | `default=16.0` | LoRA alpha | after-category override |
| `--lora_dropout` | `default=0.0` | LoRA dropout | after-category override |
| `--distill_steps` | `default=50` | LoRA 最大步数 | `0` 表示跳过该轮 LoRA |
| `--distill_batch_size` | `default=2` | LoRA 每卡 batch size | after-category override |
| `--distill_lr` | `default=1e-4` | LoRA 学习率 | after-category override |
| `--distill_weight_decay` | `default=0.0` | LoRA 权重衰减 | after-category override |
| `--distill_log_every` | `default=1` | LoRA 日志间隔 | after-category override |
| `--distill_temperature` | `default=1.0` | LoRA 蒸馏温度 | after-category override |
| `--distill_loss_alpha` | `default=0.5` | LoRA 蒸馏混合权重 | after-category override |
| `--distill_loss_type` | `default=eakld` | LoRA loss 类型（含 `eakld` / `eakld_kd` / `kd_top_*` 等） | after-category override |
| `--distill_eakld_confidence_k` | `16` | EAKLD 熵归一化 K（非 vocab top-k） | 全局 |
| `--distill_hidden_loss_weight` | `default=0.0` | LoRA hidden-state 对齐辅助损失权重 | after-category override；`0` 表示关闭；开启后对齐所有 transformer block 输出 hidden states，跳过 embedding hidden state |
| `--distill_hidden_alignment_layer_weighting` | `uniform` | LoRA hidden-state 对齐的层权重模式 | 全局单值；`uniform` 等权全层；`linear_depth` 后层权重线性增大并归一到平均权重为 1；`adaptive` 默认选 cosine 最低的 3 层；`adaptive_top_<K>` 仅对 teacher 相邻层变化最大的 K 层计算对齐损失 |
| `--lora_use_dora` | `default=true` | LoRA 是否启用 DoRA | after-category override；仅 `remaining_lora` 支持 DoRA，`compressed_lora` / `both` 若解析到 `true` 会直接报错 |
| `--distill_tune_final_norm` | `false` | 每类后 LoRA 蒸馏是否同时微调最终 norm | 仅 `--distill_after_category=remaining_lora` 支持；`compressed_lora` / `decoder` / `both` 会直接报错 |
| `--distill_use_post_norm_head_linear` | `false` | 每类后 LoRA 蒸馏是否训练 post-norm head linear | 仅 `--distill_after_category=remaining_lora` 支持；最终保存前会融合回 `lm_head` |
| `--distill_hif4_act` | `false` | 是否只在 LoRA 阶段对 student 线性层输入启用 HiFloat4 激活伪量化 | 全局开关，不参与 after-category override |
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
| `--norm_type` | `default=group` | `group/batch/layer/rms/no`；`rms` 为 RMSNorm |
| `--activation_type` | `default=swish` | `swish/relu/none/sigmoid/gelu/hard_swish`；控制 encoder/decoder/res block 内激活 |
| `--decoder_type` | `default=linear` | `linear/symmetric/asymmetric` |

重要语义：

- `residual_stages` 允许按类别不同。
- 权重训练数据的基本单位是 block，张量形状是 `[num_blocks, num_models, codebook_dim]`：
  - 单个模型/part 的 1 个 block 包含 `codebook_dim` 个连续权重元素
  - 一个 group 的同一个 block 横跨 `num_models = linears_in_group * intra_parallel_parts` 个模型/part
  - 因此一次评估 `B` 个 blocks，实际覆盖 `B * num_models * codebook_dim` 个权重值
- 同一类别内，如果 `residual_stages > 1`：
  - 先做逐阶残差量化；排序代码已关闭，每个 stage 都保持普通切分顺序
  - 所有 stage 仍复用同一组已解析结构参数：
    - `steps`
    - `codebook_bits`
    - `codebook_dim`
    - `recon_loss_type`
    - `base_ch`
    - `num_res_blocks`
    - `decoder_base_ch`
    - `decoder_num_res_blocks`
    - `norm_type`
    - `activation_type`
    - `decoder_type`
  - joint decoder 联合优化代码已关闭；全部 stage 训练完后直接进入保存/替换流程
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
| `--lr_scheduler` | `constant` | `constant/linear/cosine`；`constant` 表示固定学习率 |
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
| `--distill_gradient_accumulation_steps` | LoRA `TrainingArguments` | LoRA 梯度累积 |
| `--distill_optim` | LoRA `TrainingArguments` | LoRA 优化器名称 |
| `--distill_max_grad_norm` | LoRA `TrainingArguments` | LoRA 梯度裁剪 |
| `--distill_warmup_ratio` | LoRA `TrainingArguments` | LoRA warmup 比例 |
| `--distill_group_by_length` | LoRA `TrainingArguments` | LoRA 按长度分组 |
| `--distill_lr_scheduler_type` | LoRA `TrainingArguments` | LoRA 学习率调度器类型；`constant` 且 `distill_warmup_ratio>0` 会直接报错，需改用 `constant_with_warmup` |
| `--distill_model_max_length` | LoRA trainer | LoRA 样本最大长度 |
| `--distill_teacher_logits_cpu_staging` | LoRA trainer | teacher forward 后把 logits 暂存 CPU（bf16），算 KL 前搬回 GPU；默认 `true` |
| `--distill_hif4_act` | LoRA trainer | 是否只在 LoRA 阶段对 student 线性层输入启用 HiFloat4 激活伪量化；默认 `false` |
| `--eval_hif4_act` | cat_train eval | 是否在内部类别后评估时启用 HiFloat4 激活伪量化；默认 `false` |

## 6. 关键运行时语义

### 6.1 类别解析与分组

- `target_categories` 同时指定要压缩的类别集合和类别训练顺序。
- 默认只在 decoder projection 路径中收集 `target_categories` 指定类别；如果传 `--include_all_linears`，只放开路径限制，仍不会收集目标类别以外的 `nn.Linear`。
- `skip_layers` 会在发现真实 `(layer_idx, category)` 后校验；如果有未知项，直接报错。
- `linear_group_size` 控制同类别跨层分组大小。
- 如果最后一个 group 不足 `linear_group_size`：
  - `allow_tail_group=true`：照常训练
  - `allow_tail_group=false`：直接跳过尾组

### 6.2 `residual_stages`

- `residual_stages` 是按类别解析的。
- 某类别为 `N>1` 时，会做逐阶残差量化：当前 stage 重建后，从 residual 中扣除该 stage 重建结果，再进入下一 stage。
- 排序代码已关闭，每个 stage 都保持普通切分顺序。
- joint decoder 联合优化代码已关闭，全部 stage 训练完后不会额外微调 decoder。
- 同一类别的各个 stage 不再有单独结构参数配置，自由度只剩“stage 数量”和共享 VAE 结构参数。

### 6.3 `steps_per_category`

- 当前实现里它表示“这个类别里每个 group 训练多少步”
- 它不是“整个类别总步数”

### 6.4 joint decoder 联合优化

- joint decoder 联合优化代码已关闭。
- `--joint_decoder_steps`、`--joint_decoder_lr`、`--joint_decoder_group_size`、`--joint_decoder_batch_size` 不再注册；传入会直接报错。

### 6.8 `eval_every` / `eval_blocks`

- `eval_every` 同时作用于 VAE stage 和 joint decoder 的中间评估。
- `eval_every=0` 表示不做中间评估。
- `eval_blocks` 只限制中间评估最多使用多少个 blocks。
- VAE stage 中间评估会输出当前 stage 的 `mse` 和 `top_k_mse(k=100)`。
- joint decoder 中间评估会输出当前 `recon_loss_type` 下的 `recon_loss`，并使用前 `eval_blocks` 个 blocks，不是随机采样。
- joint decoder 在联合优化前后一定会记录完整权重重建损失，不受 `eval_every/eval_blocks` 限制：
  - `full_recon_loss_before`：联合优化前，全量 blocks 的重建损失
  - `full_recon_loss_after`：联合优化后，全量 blocks 的重建损失
  - `delta = after - before`
  - `ratio = after / before`
- 因为前后 full loss 都使用完整 `target_common`，所以比较的是同一批权重元素。

### 6.9 类别后评估

- 类别后评估由 `--eval_ppl` 和 `--eval_tasks` 共同控制。
- `--eval_ppl=true` 时跑 PPL。
- `--eval_tasks` 非空时跑 lm_eval，并在日志里记录：
  - 每个任务的 `metric_key` 和分数
  - 所有有效任务分数的简单平均值（忽略 `N/A`）
- `--eval_ppl=false` 且 `--eval_tasks=""` 时，整个类别后评估阶段直接跳过。
- 如果配置了 `--eval_tasks`，但所有任务结果都是 `N/A`，脚本会直接报错。

### 6.10 `convert` 与评估的关系

- `--convert` 只控制是否把训练出的压缩结果替换回模型里的目标 `Linear`。
- 不开 `--convert` 时：
  - 不会把模块替换成 `VAELinear`
  - 仍然可以按 `--eval_ppl/--eval_tasks` 跑类别后评估
- `--distill_after_category != none` 和 `--save_model` 仍然要求 `--convert` 开启，因为它们依赖“压缩结果已经写回模型”。

### 6.11 每类后蒸馏模式

`--distill_after_category` 控制每个类别压缩完成后的补偿训练：

| 模式 | 训练目标 | 保存语义 | 主要限制 |
|---|---|---|---|
| `none` | 不做每类后蒸馏 | 只保存 VAE 压缩结果 | 无 |
| `remaining_lora` | 尚未压缩的剩余 dense `nn.Linear`，可选最终 norm / post-norm head | LoRA 训练后融合回 dense Linear；post-norm head 在最终保存前融合回 `lm_head` | 保留旧行为，支持 DoRA |
| `compressed_lora` | 当前刚压缩类别的 `VAELinear` proxy LoRA | 先预解码 dense base，再训练 LoRA delta；训练后导出为 `VAELinear.low_rank_a/b` 并恢复普通 `VAELinear` | v1 不支持 DoRA；不训练 final norm / post-norm head |
| `decoder` | 当前刚压缩类别的 `VAELinear` decoder 参数 | 训练后关闭 trainable decode、拆回普通 decoder 并清 cache | 不训练 final norm / post-norm head |
| `both` | 当前刚压缩类别的 decoder + proxy LoRA | decoder 收尾同 `decoder`；LoRA delta 导出为 `low_rank_a/b`；最终不保留 proxy | v1 不支持 DoRA；不训练 final norm / post-norm head |

注意：

- `compressed_lora` / `both` 如果 `--lora_use_dora` 解析为 `true` 会直接报错；v1 不做 DoRA 到低秩补丁的近似 SVD。
- `compressed_lora` / `decoder` / `both` 如果开启 `--distill_tune_final_norm` 或 `--distill_use_post_norm_head_linear` 会直接报错，避免每类后移动最终 logits 路径。
- 最终普通 cat checkpoint 不保留 `PeftVAELinearProxy`；保存前若仍有 proxy 残留会直接报错。

### 6.12 `normalize_weight`

- 训练时会对当前 stage 的 residual 计算 `(mean, std)` 并标准化训练
- 转换时会把该 stage 的 `(mean, std)` 融合进 decoder
- 多阶 residual 会分别计算和融合各自的标准化统计量

### 6.13 `wa_mse` 与 activation 依赖

- `recon_loss_type=wa_mse` 时，会在当前 group 上动态重算 activation stats，其中 `act_max` 仍供 `wa_mse` 使用。
- `outlier_protect_mode=channel/channel_residual_vae` 且 `outlier_rank_metric` 使用 activation 加权或 second-moment 模式时，也复用同一条动态 activation 路径
- `outlier_protect_mode=residual_sparse` 且 `outlier_rank_metric` 使用 activation max/mean 加权模式时，也复用同一条动态 activation 路径
- 当前 `cat_train` 不再支持通过静态 activation 字典驱动这三类逻辑

### 6.13 离群保护模式

`outlier_protect_mode` 当前支持 4 个值：

- `none`：不做离群保护。
- `channel`：VAE 压缩前保护 top-N input/output channel，保护数量来自 `outlier_protect_count`。可选 `--outlier_protect_channel_quant` 对 protected 权重做 per-channel 量化存储（scale 为 bf16）。
- `channel_residual_vae`：主 VAE 仍压缩完整权重；训练后只取选中 channel 的 `original - reconstructed` residual，并为每个 linear 单独训练额外多阶 VAE patch。
- `residual_sparse`：VAE / joint decoder 训练结束后，保存最终重建残差里的 top-p 稀疏补丁。

`channel/channel_residual_vae` 的通道选择：

- `outlier_channel_scope=layer` 时，每个 linear 独立保护 `outlier_protect_count` 个 channel。
- `outlier_channel_scope=category` 时，同一 category 的有效 linear 一起排序，总预算为 `outlier_protect_count * 有效 linear 数`，不同层可分配到不同数量的 channel。
- `channel_weight_abs/channel_weight_actmax_abs/channel_weight_actmean_abs` 按原始权重通道范数排序；`actmax` 乘校准集激活绝对值最大值，`actmean` 乘校准集激活绝对值均值。
- `channel_residual_abs/channel_residual_actmax_abs/channel_residual_actmean_abs/channel_residual_actrms_abs` 在 base VAE stages 结束后，按 final residual 的通道误差排序。
- `outlier_residual_vae_batch_multiplier` 只放大 protected residual VAE 的训练 batch；默认 `1` 保持旧行为，`category` shared residual VAE 建议 `32`。
- `outlier_residual_vae_steps/outlier_residual_vae_lr` 只控制 protected residual VAE；默认 `0` 分别继承 base VAE 的 steps/lr。

#### MLP aligned channel selection（`--outlier_mlp_rank_metric`）

v1 仅支持 `outlier_protect_mode=channel` 且 `outlier_channel_scope=layer`。

- 仅作用于 `gate_proj` / `up_proj` / `down_proj`；attention 类别仍使用 `--outlier_rank_metric` 的 per-linear 逻辑。
- 同一层 MLP 共享一组 intermediate channel indices：`gate_proj`/`up_proj` 保护 output row，`down_proj` 保护 input column。
- `--outlier_protect_count` 在 MLP 三类上必须相同，表示每层 intermediate channel 保护数 `k`。
- `--outlier_protect_axis` 在 MLP aligned 模式下被忽略，由框架按 category 自动映射。
- 一次 MLP block 校准同时采集 `abs_mean` / `sq_mean`（in 与 mid），三种 metric 共用，不重复跑 forward。
- 运行目录会写出 `mlp_channel_selection_summary.json`（含 `rank_metric`）。

三种 `--outlier_mlp_rank_metric` 公式（up/gate 用 MLP 输入 `x` 统计，down 用 SwiGLU 中间态 `z` 统计）：

| 值 | 对齐 per-linear | up/gate（row i） | down（col i） |
|----|-----------------|------------------|---------------|
| `mlp_intermediate_aligned_actrms` | 文档原算法 | `mean_j(\|W_{ij}\| · RMS(x_j))` | `mean_r(\|W_{ri}\| · RMS(z_i))` |
| `mlp_intermediate_aligned_actmean_abs` | `channel_weight_actmean_abs` | `\|W_{i,:} ⊙ mean(\|x\|)\|_2` | `\|W_{:,i} ⊙ mean(\|z\|)\|_2` |
| `mlp_intermediate_aligned_actrms_abs` | `channel_residual_actrms_abs` 权重公式 | `\sum_j W_{ij}^2 · E[x_j^2]` | `\sum_r W_{ri}^2 · E[z_i^2]` |

注意：`actrms` 是 **RMS 标量乘子 + L1 mean**；`actrms_abs` 是 **E[x²] 乘子 + W² 求和**，二者不同。

示例（仅 metric 行不同）：

```bash
--outlier_protect_mode channel \
--outlier_channel_scope layer \
--outlier_rank_metric channel_weight_actmean_abs \
--outlier_mlp_rank_metric mlp_intermediate_aligned_actmean_abs \
--outlier_mlp_fuse_weights 1,1,1 \
--outlier_protect_count "default=32,cat:gate_proj=32,cat:up_proj=32,cat:down_proj=32" \
--activation_calib_dataset "alpaca=1"
```

`residual_sparse` 的约束：

- `outlier_protect_count` 必须为 `0`。
- `outlier_residual_top_p` 必须满足 `0 < p <= 1`。
- 只有 activation 加权 score 需要 activation 校准。

推理时 `VAELinear` 的重建顺序固定为：

```text
VAE reconstruction -> sparse_residual patch
```

若使用 `channel_residual_vae`，重建顺序为：

```text
VAE reconstruction -> protected_channel_residual_vae patch -> low_rank patch -> sparse_residual patch
```

### 6.14 保存与输出目录

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

### 6.15 从已保存 checkpoint 继续训练

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
--distill_steps default=50,after:q_proj=200
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
6. `outlier_protect_mode=channel` 且 `outlier_protect_count > 0`，或 `residual_sparse + activation 加权 score` 启用，但当前 group 的动态 activation 采集失败
7. 切分后某个 part 的展平长度无法被该类别的 `codebook_dim` 整除
8. `linear_group_size < 1`
9. 传入已关闭的排序或 joint decoder CLI 参数

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
  --activation_calib_dataset wiki=1.0 \
  --activation_calib_nsamples 512 \
  --activation_calib_seqlen 512 \
  --train_device cuda
```

### 9.5 residual stages

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --residual_stages default=2 \
  --eval_every 100 \
  --eval_blocks 256 \
  --train_device cuda
```

说明：

- 排序代码已关闭，不再传 `--intra_part_sort_mode`。
- joint decoder 联合优化代码已关闭，不再传 `--joint_decoder_*`。

### 9.7 每类后蒸馏覆盖

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --distill_after_category compressed_lora \
  --distill_dataset wiki=1.0 \
  --lora_rank default=8,after:q_proj=16 \
  --distill_steps default=50,after:q_proj=200 \
  --distill_hidden_loss_weight default=0.01 \
  --distill_hidden_alignment_layer_weighting adaptive_top_3 \
  --eval_ppl true \
  --eval_tasks boolq,rte,piqa \
  --distill_hif4_act false \
  --eval_hif4_act false \
  --lora_use_dora default=false
```

如果需要保留旧的“补偿剩余 dense Linear”行为：

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --distill_after_category remaining_lora \
  --distill_dataset wiki=1.0 \
  --lora_rank default=8,after:q_proj=16 \
  --distill_steps default=50,after:q_proj=200 \
  --distill_hidden_loss_weight default=0.01 \
  --distill_hidden_alignment_layer_weighting adaptive_top_3 \
  --eval_ppl true \
  --eval_tasks boolq,rte,piqa \
  --distill_hif4_act false \
  --eval_hif4_act false \
  --lora_use_dora default=true,after:q_proj=false
```

### 9.8 从已完成 `q_proj` 的保存结果继续训练其它类别

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

### 9.9 只跑下游任务，不跑 PPL

```bash
python tools/cat_train.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --convert \
  --eval_ppl false \
  --eval_tasks boolq,rte,piqa \
  --train_device cuda
```

## 10. 评估脚本（`scripts/eval.sh`）

`scripts/eval.sh` 是对 [tools/cat_eval.py](/home/shaoyuantian/program/VAELLM/tools/cat_eval.py) 的一层薄封装。

- 必填环境变量：`CHECKPOINT_DIR`
- 可选环境变量：`ADAPTER_DIR`

示例：

```bash
CHECKPOINT_DIR=.result/your_cat_run/final_model \
bash scripts/eval.sh
```

评估 dense e2e adapter（先重建 dense 再 merge adapter）：

```bash
CHECKPOINT_DIR=.result/your_cat_run/final_model \
ADAPTER_DIR=.result/dense_e2e_run/final_adapter \
bash scripts/eval.sh
```

约束：

- 传 `ADAPTER_DIR` 时，`cat_eval` 会做 checkpoint 与 adapter 的指纹匹配，不匹配直接报错。
- 传 `ADAPTER_DIR` 时，不支持 `--eval_linear_mse`。
