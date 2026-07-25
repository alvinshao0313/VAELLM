# 蒸馏回归测试说明

覆盖 decoder / `compressed_lora` / `both` 每类后蒸馏与 checkpoint 蒸馏相关改动。

## 环境

```bash
conda activate bitvae
cd /path/to/VAELLM
export PYTHONPATH=.
```

## 快速跑（推荐）

只跑蒸馏相关单测：

```bash
python -m unittest \
  tests.test_cat_after_category_distill \
  tests.test_cat_checkpoint_distill \
  tests.test_peft_proxy_low_rank_export \
  tests.test_compressed_e2e_args \
  tests.test_cat_train_args_distill_after_category \
  -v
```

## 本轮优化对应的断言

| 优化项 | 测试位置 | 断言要点 |
|---|---|---|
| O1/O2 中间保存与跳过 | `test_cat_checkpoint_distill` / `test_cat_after_category_distill` | 前缀激活；已有 `low_rank` 自动 skip；`completed_categories` 路径 |
| O2 再蒸一轮 | 手工冒烟 / peft_proxy 相关 | `distill_reset_completed=true` 时从已有 `low_rank` 初始化 LoRA 并允许覆盖写回 |
| O3 warmup | `test_cat_train_args_distill_after_category`（或下方新增） | `constant` + `warmup_ratio>0` 报错 |
| O6 both 效率 | `test_peft_proxy_low_rank_export` | A/B delta 与 `peft-dense` 数值一致；both 前向 = decoder + delta |
| O14 命名 | `test_compressed_e2e_args` | `--finetune_mode compressed_lora`；拒绝旧值 `lora` |
| O15 | `test_cat_checkpoint_distill` | `compressed_lora` 忽略 `vae_decoder_checkpoint` |

## 手工冒烟（可选，需 GPU + 数据）

### A. 类别级续跑（`reset=false`）

确认能启动、能 skip、能写 `after_*`：

```bash
# 1) 从已有 VAE ckpt 只跑 1 个类别、极少 step
bash scripts/catlora_distill_from_checkpoint.sh \
  --resume_from_checkpoint .result/catlora/<vae_run>/final_model \
  --target_categories "q_proj" \
  --distill_steps "default=2" \
  --distill_batch_size "default=1" \
  --distill_log_every "default=1" \
  --eval_tasks "" \
  --eval_ppl false

# 2) 确认产出
ls .result/catlora_distill/*/after_q_proj/checkpoint_meta.json
ls .result/catlora_distill/*/final_model/checkpoint_meta.json

# 3) 从 after_q_proj 续跑，确认 q_proj 被 skip
bash scripts/catlora_distill_from_checkpoint.sh \
  --resume_from_checkpoint .result/catlora_distill/<run>/after_q_proj \
  --target_categories "q_proj,k_proj" \
  --distill_reset_completed false \
  --distill_steps "default=2" \
  --eval_tasks "" \
  --eval_ppl false
```

日志中应出现类似：

- `已在 completed_categories 中，跳过蒸馏` 或 `已有 low_rank_a/b，自动跳过`
- `Saved model to .../after_k_proj`

### B. 再蒸一轮（`reset=true`，含 LoRA 续训）

在已有蒸馏 ckpt 上确认不会跳过、会从 `low_rank` 初始化：

```bash
bash scripts/catlora_distill_from_checkpoint.sh \
  --resume_from_checkpoint .result/catlora_distill/<run>/final_model \
  --target_categories "q_proj" \
  --distill_reset_completed true \
  --distill_steps "default=2" \
  --distill_batch_size "default=1" \
  --distill_log_every "default=1" \
  --eval_tasks "" \
  --eval_ppl false
```

日志中应出现类似：

- `distill_reset_completed=true` / `从已有 low_rank_a/b 初始化 LoRA 续蒸`
- `已用 low_rank_a/b 初始化 LoRA adapters=...`
- `exported compressed LoRA adapters to low_rank_a/b=... allow_overwrite=true`

不应再出现对该类的 `已有 low_rank_a/b，自动跳过`。

## 通过标准

1. 上述 unittest 全部通过。
2. 冒烟 A 能写出 `after_<category>/` 与 `final_model/`；续跑不会对已有 `low_rank_a/b` 的类别再次 export 报错。
3. 冒烟 B 在 `reset=true` 时对已有 `low_rank` 类真正再训，并能覆盖写回。
