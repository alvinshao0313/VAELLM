# `scripts/catlora_distill_from_checkpoint.sh` 使用说明

这个脚本用于从已经完成 VAE 压缩的 cat checkpoint 继续做逐类别蒸馏。

它不会重新训练 VAE，也不会重新把 `nn.Linear` 替换成 `VAELinear`。它会读取 checkpoint 里已有的 `VAELinear`，按 `--target_categories` 顺序重放 `catlora_simple.sh` 里“每类刚压缩完后”的蒸馏步骤。

## 入口链路

```text
scripts/catlora_distill_from_checkpoint.sh
  -> tools/cat_distill_from_vae_checkpoint.py
  -> train_utils/cat_checkpoint_distill.py
  -> train_utils/cat_after_category_distill.run_after_category_distill(...)
```

脚本仍然复用 `cat_train` 的参数解析，因此 `--target_categories`、`--distill_after_category`、`--distill_steps`、`--lora_rank` 等参数含义和 `scripts/catlora_simple.sh` 保持一致。

## 适用场景

适合这种流程：

1. 已经用 `scripts/catlora_simple.sh` 或同等入口完成目标类别的 VAE 压缩。
2. 保存出来的 checkpoint 里已经有目标类别的 `VAELinear`。
3. 之前没有做蒸馏，或者想从这个全 VAE checkpoint 重新开始逐类别蒸馏。

不适合这种流程：

- 输入 checkpoint 里还没有目标类别的 `VAELinear`。
- 想继续没压缩完的 `cat_train` run。这种情况仍然用 `tools/cat_train.py --resume_from_checkpoint`。
- 想做 compressed e2e 全模型微调。这种情况使用 `compressed_e2e_fintuning/scripts/e2e_decoder.sh`。

## 运行方式

先进入 `bitvae` 环境：

```bash
conda activate bitvae
```

然后传入要继续蒸馏的 VAE checkpoint：

```bash
VAE_CKPT=.result/catlora/<run>/final_model \
bash scripts/catlora_distill_from_checkpoint.sh
```

`VAE_CKPT` 可以是：

- run 目录，例如 `.result/catlora/Qwen_Qwen3-8B_20260618_xxxxxx`
- run 下的 `final_model/`
- 直接指向 `checkpoint_meta.json`

脚本默认输出到：

```text
.result/catlora_distill/<model_name>_<timestamp>/final_model
```

运行日志在新 run 目录下：

```text
linear_by_category.log
```

## 当前脚本默认配置

当前脚本默认只蒸馏这三个类别：

```bash
--target_categories "q_proj,k_proj,v_proj"
```

默认蒸馏模式是：

```bash
--distill_after_category "compressed_lora"
```

这表示每个类别会训练当前类别 `VAELinear` 上的 proxy LoRA，训练后导出到 `VAELinear.low_rank_a/b`，最终 checkpoint 不保存 PEFT adapter。

如果要换 GPU，直接改脚本里的：

```bash
export CUDA_VISIBLE_DEVICES=4
```

如果要换输入类别顺序，直接改：

```bash
--target_categories "q_proj,k_proj,v_proj"
```

如果要改蒸馏步数、数据或 LoRA rank，直接改脚本里的对应 CLI 参数，例如：

```bash
--distill_steps "default=5000"
--distill_dataset "openorca=0.2,fineweb_edu=0.18,race=0.24,sciq=0.14,alpaca=0.04,longalpaca=0.1,longalign=0.1"
--lora_rank "default=128"
```

按类别覆盖仍然使用 `after:<category>`：

```bash
--distill_steps "default=5000,after:k_proj=3000"
--lora_rank "default=128,after:v_proj=64"
```

## 推理路径切换顺序

假设：

```bash
--target_categories "q_proj,k_proj,v_proj"
```

运行时会按下面顺序切换 `VAELinear` 的推理路径：

1. 蒸馏 `q_proj` 时：`q_proj` 走 VAE 压缩路径，`k_proj/v_proj` 走原始权重路径。
2. 蒸馏 `k_proj` 时：`q_proj/k_proj` 走 VAE 压缩路径，`v_proj` 走原始权重路径。
3. 蒸馏 `v_proj` 时：`q_proj/k_proj/v_proj` 都走 VAE 压缩路径。

在 `compressed_lora` 模式下，每个类别蒸馏前会预热当前 active prefix 的 decoded weight cache。`decoder` / `both` 模式会训练 decoder 参数，因此不会做这个预热。

## 输入 checkpoint 的原始权重

如果 checkpoint 里保存了 `VAELinear.original_weight`，运行时会使用 checkpoint 中的原始权重。

如果 checkpoint 因为 `--unload_vae_original_weights_on_final_save` 没有保存 `original_weight`，加载时会从 base model 权重补回。这个逻辑要求 checkpoint metadata 或脚本参数能确定 base model 路径。

如果目标类别在 checkpoint 中没有对应 `VAELinear`，启动时会直接报错，例如：

```text
target_categories contains categories without VAELinear in checkpoint: v_proj
```

## 常见问题

### `VAE_CKPT` 没设置

脚本会直接报错：

```text
set VAE_CKPT to cat VAE checkpoint
```

按下面方式传入：

```bash
VAE_CKPT=.result/catlora/<run>/final_model bash scripts/catlora_distill_from_checkpoint.sh
```

### 想只跑某些类别

直接改脚本里的：

```bash
--target_categories "q_proj,k_proj,v_proj"
```

注意：这里的顺序就是逐类别蒸馏顺序，也决定 active prefix 的打开顺序。

### 想关闭类别后评估

当前脚本已经设置：

```bash
--eval_ppl "false"
```

但仍然保留了：

```bash
--eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"
```

如果完全不跑类别后下游任务评估，把它改成空串：

```bash
--eval_tasks ""
```
