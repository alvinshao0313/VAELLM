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

当前版本只支持 Qwen3 block 路径。可选 Linear 类别是：

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

## 1. Pipeline 模式

`--block_vae_pipeline_mode` 控制 VAE 和逐层蒸馏的执行方式。

| 模式 | 行为 | 输出 |
|---|---|---|
| `inline` | 旧流程：每层先训练该层 VAE，再立刻蒸馏该层 | `final_model/` |
| `pretrain` | 按 `--block_vae_categories` 顺序跨层同类别训练 VAE，并替换成全 VAE 模型 | `<run_output_dir>/block_vae_cache/` |
| `distill` | 不训练 VAE，从已有全 VAE checkpoint、上一轮 block `final_model` 或 block resume checkpoint 开始逐层蒸馏 | `final_model/` |
| `pretrain_distill` | 先执行完整 `pretrain`，保存全 VAE checkpoint，然后在内存中直接逐层蒸馏 | `<run_output_dir>/block_vae_cache/` 和 `final_model/` |

`scripts/block_vae_lora_simple.sh` 默认使用：

```bash
--block_vae_pipeline_mode "distill"
```

## 2. 初始化 checkpoint

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 |
|---|---:|---:|---|
| `--vae_pretrained_checkpoint` | 空 | `.result/Qwen_Qwen3-8B_20260610_021016/block_vae_cache` | `distill` 第一轮输入：全 VAELinear checkpoint，要求 stage 为 `block_vae_category_pretrained` |
| `--block_init_checkpoint` | 空 | 空，按需手动替换 `--vae_pretrained_checkpoint` | `distill` 新一轮输入：上一轮 block `final_model`，要求 stage 为 `block_vae_lora_final` |
| `--block_vae_pretrain_devices` | 空，实际回退到 `train_device` | `cuda` | VAE pretrain worker 使用的设备列表 |
| `--block_vae_pretrain_workers` | 空，实际等于设备数 | `1` | VAE pretrain worker 数 |
| `--block_vae_linear_group_size` | `32` | `36` | 每个同类别 VAE group 包含多少个 Linear |
| `--block_vae_allow_tail_group` | `true` | `true` | 是否允许不足 group size 的尾组 |
| `--block_vae_categories` | 7 类全量 | 7 类全量 | 要训练和蒸馏的 Linear 类别；输入顺序就是 pretrain 类别顺序 |

`distill` 初始化入口三选一，不能混用：

- `--vae_pretrained_checkpoint`：从全 VAE checkpoint 开始第一轮逐层蒸馏。
- `--block_init_checkpoint`：从上一轮 block `final_model` 开始新一轮逐层蒸馏。不是 resume，会从第 0 层重新走一遍。
- `--block_resume_from_checkpoint`：从 `<run_output_dir>/block_checkpoints/block_XXXX/` 继续中断的同一轮逐层蒸馏。

`--block_init_checkpoint` 只接受 block final checkpoint，不能传逐层 checkpoint，也不能传 `block_vae_cache`。

`pretrain/pretrain_distill` 不再保存 category group payload。VAE payload 只在内存中流转：每个 group 训练完立刻 apply，所有类别替换完成后才写一个全 VAELinear checkpoint：

```text
<run_output_dir>/block_vae_cache/
```

`--block_vae_allow_tail_group false` 在 block 流程中要求每个类别的 active Linear 数能被 `--block_vae_linear_group_size` 整除；否则直接报错，避免生成部分层未 VAE 化的 checkpoint。

`--block_vae_pretrain_workers 1` 时，category pretrain 会复用主进程已经加载的模型连续训练各类别，不会每训练完一类就重新加载 base model。多 worker 模式仍使用独立 worker 进程。

## 3. 类别、层选择和 skip

`--block_vae_categories` 同时控制 VAE pretrain 和后续逐层蒸馏目标。没有列入的类别不会训练 VAE，也不会做 LoRA/decoder/both 蒸馏。

默认：

```bash
--block_vae_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
```

自定义顺序：

```bash
--block_vae_categories "down_proj,up_proj,q_proj"
```

约束：

- 空类别列表报错。
- 重复类别报错。
- 当前只允许 Qwen3 7 类子集。
- `--skip_layers` 的类别必须属于 `--block_vae_categories`。
- `--block_layers`、`--block_vae_categories`、`--skip_layers` 组合后如果没有任何有效目标，直接报错。

`--block_layers` 支持：

```bash
--block_layers "all"
--block_layers "0"
--block_layers "0-3"
--block_layers "0,2,4-7"
```

`--skip_layers` 格式：

```bash
--skip_layers "0.down_proj,30.q_proj"
```

被 skip 的模块保持原始 `nn.Linear`，不会替换成 `VAELinear`，也不会注入 LoRA 或训练 decoder。

## 4. VAE 参数

部分参数支持 category override：

```text
selector=value,selector=value
```

允许 selector：

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

命中规则：

```text
cat:<当前类别> > default
```

常用 VAE 参数：

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 |
|---|---:|---:|---|
| `--vae_steps` | `default=20000` | `default=5000` | 每个 VAE group 训练步数 |
| `--vae_batch_size` | `8192` | `32768` | VAE 训练 batch size；`all` 表示训练时使用全量权重 batch |
| `--vae_gpu_resident_data` | `false` | `true` | VAE 训练数据是否常驻 GPU；只影响数据搬运，不改变 batch size 和 cache 语义 |
| `--vae_log_every` | `100` | `100` | VAE 日志间隔 |
| `--vae_eval_every` | `0` | `0` | VAE 训练中评估间隔；`0` 表示不评估；`>0` 表示每 N step 对当前 residual stage 做全量重构评估 |
| `--recon_loss_type` | `default=mse` | `default=mse` | 重建 loss |
| `--intra_parallel` | `default=1x1` | `default=1x1` | 单 Linear 内部切分 |
| `--intra_part_sort_mode` | `default=none` | `default=none` | part 内排序模式 |
| `--codebook_bits` | `default=32` | `default=32` | codebook bit 数 |
| `--codebook_dim` | `default=32` | `default=32` | codebook latent 维度 |
| `--residual_stages` | `default=2` | `default=2` | residual VAE 阶数 |
| `--lr` | `1e-2` | `3e-3` | VAE 优化器学习率 |
| `--quantizer_type` | `BSQ` | `BSQ` | 量化器类型 |
| `--normalize_weight` | `false` | 开启 | VAE 训练前是否归一化权重 |
| `--new_quant` | `false` | 开启 | 使用新版量化逻辑 |

当前 block 路径没有 activation runtime。动态 activation 依赖功能不在 block VAE pretrain 中启用。

`--vae_gpu_resident_data true` 会把当前 residual stage 的权重数据预先放到 `train_device`，数值 batch 仍按 `--vae_batch_size` 切分。它是纯性能开关，不写入 VAE cache manifest hash；切换该值不会让旧 cache 失效。

`--vae_eval_every` 触发的是训练中监控 eval，不改变训练数据。每次 eval 都会覆盖当前 residual stage 的全部权重块，不再按 `eval_blocks` 抽样截断。为了避免 OOM，eval 前向仍按 batch 执行：`--vae_batch_size N` 时 eval batch 为 `N`；`--vae_batch_size all` 时训练仍走全量 batch，但 eval 使用内部安全 batch `8192`。

## 5. 蒸馏参数

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 |
|---|---:|---:|---|
| `--block_distill_train_mode` | `lora` | `lora` | 蒸馏训练模式：`lora/decoder/both` |
| `--block_distill_steps` | `100` | `5` | 每个 selected block 的蒸馏步数 |
| `--block_distill_dataset` | `fineweb_edu=0.35,race=0.30,sciq=0.20,openorca=0.15` | `openorca=0.24,fineweb_edu=0.18,race=0.24,sciq=0.03,alpaca=0.11,longalpaca=0.10,longalign=0.10` | 蒸馏校准数据 |
| `--block_distill_nsamples` | `100` | `5000` | 校准样本数 |
| `--block_distill_seqlen` | `4096` | `4096` | 校准序列长度 |
| `--block_lora_variant` | `plain` | `dora` | LoRA 类型：`plain/rslora/dora/adalora` |
| `--block_lora_rank` | `32` | `32` | LoRA rank |
| `--block_lora_lr` | `1e-4` | `1e-4` | 蒸馏 optimizer 学习率 |
| `--block_lora_alpha` | rank | `32` | LoRA alpha |
| `--block_lora_dropout` | `0.0` | `0.0` | LoRA dropout |
| `--block_lora_bias` | `none` | `none` | LoRA bias 模式 |
| `--block_loss_alpha` | `0.1` | `0.3` | attention KL 权重 |
| `--block_loss_beta` | `0.2` | `0.2` | linear relative MSE 权重 |
| `--block_attn_query_chunk_size` | `128` | `4096` | attention KL query chunk |
| `--block_decode_group_size` | `8` | `8` | PEFT proxy decode group size |

loss 公式：

```text
loss = alpha * attention_kl + beta * linear_mse + (1 - alpha - beta) * hidden_mse
```

约束：

```text
block_loss_alpha + block_loss_beta <= 1
```

训练模式：

| 模式 | 训练内容 |
|---|---|
| `lora` | 只训练 PEFT LoRA adapter |
| `decoder` | 只训练 `VAELinear` decoder |
| `both` | 同时训练 decoder 和 PEFT LoRA adapter |

## 6. 评估、checkpoint 和输出

| 参数 | Parser 默认值 | simple.sh 默认值 | 说明 |
|---|---:|---:|---|
| `--block_eval_after_each_layer` | `false` | `false` | 每层蒸馏后是否评估 |
| `--block_eval_tasks` | 空 | `boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu` | lm-eval 任务列表 |
| `--block_eval_ppl` | `false` | `false` | 是否评估 PPL |
| `--block_eval_ppl_limit` | `-1` | `-1` | PPL 样本限制 |
| `--block_eval_device` | `None` | `cuda` | 评估设备 |
| `--block_keep_last_checkpoints` | `3` | `1` | 保留最近多少个逐层 checkpoint |
| `--block_resume_from_checkpoint` | `None` | 空 | 从逐层 checkpoint 继续 |

输出：

| 路径 | 说明 |
|---|---|
| `<run_output_dir>/block_vae_lora.log` | 训练日志 |
| `<run_output_dir>/normalized_block_vae_lora_args.json` | 归一化参数快照 |
| `<run_output_dir>/block_vae_cache/` | 全 VAE checkpoint，仅 `pretrain/pretrain_distill` 会保存 |
| `<run_output_dir>/block_checkpoints/block_XXXX/` | 逐层 resume checkpoint |
| `<run_output_dir>/final_model/` | 最终模型，仅 `inline/distill/pretrain_distill` 会保存 |

checkpoint meta 会记录 `block_vae_categories`、`block_vae_pretrain_manifest_hash`、`resume_from_checkpoint` 和 `block_init_checkpoint`。resume 时会校验类别列表、层选择、skip、训练模式和 manifest hash；`block_init_checkpoint` 会校验类别列表和 manifest hash，避免混用不匹配 checkpoint。

## 7. simple.sh 环境变量

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `PYTHONPATH` | `.` | Python import 路径 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA allocator 设置 |
| `CUDA_VISIBLE_DEVICES` | `6` | 默认可见 GPU |
| `SEED` | `42` | 随机种子 |
| `PYTHONHASHSEED` | `${SEED}` | Python hash seed |
| `TOKENIZERS_PARALLELISM` | `false` | tokenizer 并行 |
| `CAT_LORA_DATASET_NUM_PROC` | `16` | 数据处理进程数 |
| `FULL_DETERMINISM` | `false` | 为 `true` 时额外设置 `CUBLAS_WORKSPACE_CONFIG` |

## 8. 不同模式样例

默认脚本：从脚本里写死的 `--vae_pretrained_checkpoint` 做逐层 `lora` distill。

```bash
bash scripts/block_vae_lora_simple.sh
```

旧 inline 流程：每层 VAE 后立刻蒸馏该层。

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_vae_pipeline_mode "inline"
```

只训练并保存全 VAE checkpoint，不做蒸馏。输出路径是本次 run 目录下的 `block_vae_cache/`。

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_vae_pipeline_mode "pretrain"
```

复用已有全 VAE checkpoint 做逐层蒸馏。
把 `scripts/block_vae_lora_simple.sh` 里的 `--vae_pretrained_checkpoint ...` 那一行改成：

```bash
  --vae_pretrained_checkpoint ".result/<pretrain_run>/block_vae_cache" \
```

从上一轮 block `final_model` 开始新一轮 decoder distill。这里不是 resume，会从第 0 层重新逐层蒸馏。
使用时把 `scripts/block_vae_lora_simple.sh` 里的 `--vae_pretrained_checkpoint ...` 那一行替换成下面这一行。

```bash
  --block_init_checkpoint ".result/<lora_run>/final_model" \
```

并把脚本里的训练模式改成：

```bash
  --block_distill_train_mode "decoder" \
```

从第二轮 block `final_model` 开始新一轮 both distill。已有 LoRA adapter 会被复用，不会叠第二套 adapter。
同样把 `--vae_pretrained_checkpoint ...` 那一行替换成：

```bash
  --block_init_checkpoint ".result/<decoder_run>/final_model" \
```

并把脚本里的训练模式改成：

```bash
  --block_distill_train_mode "both" \
```

自定义类别和顺序。

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_vae_categories "down_proj,up_proj,q_proj"
```

多卡并行 VAE pretrain。

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_vae_pretrain_devices "cuda:0,cuda:1,cuda:2,cuda:3" \
  --block_vae_pretrain_workers "4"
```

只训练 decoder。

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_distill_train_mode "decoder"
```

同时训练 decoder 和 LoRA。

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_distill_train_mode "both"
```

从逐层 checkpoint 继续。`pretrain_distill` 不支持和 resume 混用；继续蒸馏时使用 `distill`。

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_vae_pipeline_mode "distill" \
  --block_resume_from_checkpoint ".result/<run>/block_checkpoints/block_0003"
```

快速连通性检查。

```bash
bash scripts/block_vae_lora_simple.sh \
  --block_layers "0" \
  --vae_steps "default=2" \
  --block_distill_steps "2" \
  --block_distill_nsamples "2" \
  --block_eval_after_each_layer "false"
```
