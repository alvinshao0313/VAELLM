# `scripts/catlora_distill_from_checkpoint.sh` 使用说明

这个脚本用于从已经完成 VAE 压缩的 cat checkpoint 继续做逐类别蒸馏。

它不会重新训练 VAE，也不会重新把 `nn.Linear` 替换成 `VAELinear`。它会读取 checkpoint 里已有的 `VAELinear`，按 `--target_categories` 顺序重放“每类刚压缩完后”的蒸馏步骤。

## 入口链路

```text
scripts/catlora_distill_from_checkpoint.sh
  -> tools/cat_distill_from_vae_checkpoint.py
  -> train_utils/cat_checkpoint_distill.py
  -> train_utils/cat_after_category_distill.run_after_category_distill(...)
```

脚本复用 `cat_train` 的参数解析，因此 `--target_categories`、`--distill_after_category`、`--distill_steps`、`--lora_rank` 等参数含义和 `scripts/catlora_simple.sh` 保持一致。

## 适用场景

适合：

1. 已经用 `scripts/catlora_simple.sh` 或同等入口完成目标类别的 VAE 压缩。
2. 保存出来的 checkpoint 里已经有目标类别的 `VAELinear`。
3. 之前没有做蒸馏，或者想从这个全 VAE checkpoint 开始逐类别蒸馏。
4. 从中途 `after_<category>/` checkpoint 继续后续类别。

不适合：

- 输入 checkpoint 里还没有目标类别的 `VAELinear`。
- 想继续没压缩完的 `cat_train` run（用 `tools/cat_train.py --resume_from_checkpoint`）。
- 想做 compressed e2e 全模型微调（用 `compressed_e2e_fintuning/scripts/e2e_decoder.sh`）。

## 运行方式

先进入 `bitvae` 环境：

```bash
conda activate bitvae
```

编辑脚本里的 `--resume_from_checkpoint`，指向 VAE 压缩结果，然后运行：

```bash
bash scripts/catlora_distill_from_checkpoint.sh
```

也可以在命令行覆盖：

```bash
bash scripts/catlora_distill_from_checkpoint.sh \
  --resume_from_checkpoint .result/catlora/<run>/final_model
```

`--resume_from_checkpoint` 可以是：

- run 目录，例如 `.result/catlora/Qwen_Qwen3-8B_20260618_xxxxxx`
- run 下的 `final_model/`
- 中途保存的 `after_<category>/`
- 直接指向含 `checkpoint_meta.json` 的目录

输出目录：

```text
.result/catlora_distill/<model_name>_<timestamp>/
  after_q_proj/
  after_k_proj/
  ...
  final_model/
  linear_by_category.log
```

每个类别蒸馏成功后会立刻落盘到 `after_<category>/`（不卸载 `original_weight`，便于续跑）。全部完成后写 `final_model/`。

## 当前脚本默认配置

```bash
--target_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
--distill_after_category "compressed_lora"
```

`compressed_lora`：训练当前类别 `VAELinear` 上的 proxy LoRA，训练后导出为 `VAELinear.low_rank_a/b`，最终不保留 PEFT adapter。

改 GPU：

```bash
export CUDA_VISIBLE_DEVICES=4
```

改步数 / 数据 / rank：直接改脚本里的对应 CLI，例如：

```bash
--distill_steps "default=5000"
--distill_dataset "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009"
--lora_rank "default=4"
```

学习率调度：若设置了 `--distill_warmup_ratio > 0`，必须用 `constant_with_warmup`（或其它支持 warmup 的 scheduler）。`constant` 会忽略 warmup，启动时会直接报错。

## 自动跳过与续跑

### 1. 已有 `low_rank_a/b` 自动跳过

对 `compressed_lora` / `both`：若当前类别全部目标 `VAELinear` 已有完整 `low_rank_a/b`，会自动跳过，不必再手写 `after:xxx=0`。

### 2. 从 `after_<category>/` 续跑

中途 checkpoint 的 `checkpoint_meta.json` 里会写入：

```json
{
  "extra_meta": {
    "stage": "after_category",
    "category": "q_proj",
    "completed_categories": ["q_proj"],
    "distill_after_category": "compressed_lora"
  }
}
```

续跑时把 `--resume_from_checkpoint` 指到该 `after_<category>/`，并保持完整 `target_categories` 前缀：

```bash
bash scripts/catlora_distill_from_checkpoint.sh \
  --resume_from_checkpoint .result/catlora_distill/<run>/after_q_proj \
  --target_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
```

行为：

1. 读取 `completed_categories`，跳过已完成类别的蒸馏。
2. 已完成且已进入 active 前缀的类别会物化为 `TemporarySwitchLinear`（student=decoded，teacher=共享 original bank）；完整 `VAELinear` 卸到 CPU，仅存盘前 restore。
3. 当前未完成类别保留 GPU 上的 `VAELinear` 供蒸馏。
4. 从未完成类别继续训练，并继续写新的 `after_*` / 最终 `final_model`。
5. `after_*` 保存前仍会全量 restore 为完整 `VAELinear` 图，ckpt 格式与续跑所需的 `original_weight` 不变。

训练期 residency（显存）三分态：

| 状态 | 模型里 | stash |
|---|---|---|
| 已完成 ∩ active | `TemporarySwitchLinear` | 完整 `VAELinear` 在 CPU |
| active 且未完成 | `VAELinear`（可训） | 无 |
| 未进入 active | 冻结 `nn.Linear`（weight 引用 original bank） | 完整 `VAELinear` 在 CPU |

全层 `original_weight` 只保留一份（`original_weight_bank`），teacher/student 仍通过 `set_temporary` 切换，不改变 KD 语义。

### 3. 不要只写后续类别

不要：

```bash
--target_categories "o_proj,gate_proj,up_proj,down_proj"
```

这样会把前面已压缩类别 stash 成 original dense，训练期 progressive 状态错误。

应始终保留完整前缀；已完成类别靠自动跳过或 `completed_categories` 处理。

## 模式说明

| `--distill_after_category` | 含义 |
|---|---|
| `compressed_lora` | 只训 proxy LoRA，导出 `low_rank_a/b` |
| `decoder` | 只训 decoder |
| `both` | 同时训 decoder + LoRA |

checkpoint distill **不支持** `remaining_lora` / `none`。

## 评估

- `--eval_ppl false` 且 `--eval_tasks ""`：跳过类别后评估。
- `--eval_tasks` 非空：即使 `eval_ppl=false` 也会跑下游任务评估。

## 相关文档

- 优化清单：`docs/cat_distill_optimization.md`
- 参数说明：`docs/cat_train_args.md`
- 蒸馏数据：`docs/edgerazor_dataset.md`
