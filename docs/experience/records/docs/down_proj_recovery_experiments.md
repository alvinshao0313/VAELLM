> 归档于 2026-09-23；类型：实验结果/验证/恢复记录；不是独立的通用经验。
> 原路径：`docs/down_proj_recovery_experiments.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../README.md) 查阅。

# down_proj 恢复实验记录

更新时间：2026-07-28  
结果根目录：`/root/data/ckpts/result/`  
目标：弄清楚 VAE 压缩后如何更好恢复 `down_proj`。

入口：

- VAE 压缩：`scripts/catlora_simple.sh` → `tools/cat_train.py`
- 蒸馏：`scripts/catlora_distill_4gpu_res0.sh` → `tools/cat_distill_from_vae_checkpoint.py`
- 蒸馏说明：`docs/catlora_distill_from_checkpoint.md`

评测：`boolq, rte, winogrande, arc_easy, arc_challenge, openbookqa, piqa, mmlu`（8 项均值）。  
下文默认用百分数。

---

## 1. 结论（截至 2026-07-28）

1. 全类压缩后，`down_proj` 掉点最重：`up_proj` 后 60.97 → `down_proj` 后 **54.51**（约 -6.5pp）。
2. 当前蒸馏主设定是 `distill_independent_categories=true`：只激活 `down_proj`，其它类恢复为未压缩 Linear。该设定下压缩后基线固定为 **67.93**。不要拿全类 54.51 当蒸馏前对照。
3. 独立 `down_proj` 蒸馏已完成对照：

| run | 关键变量 | 蒸馏后 avg | Δ vs 67.93 |
|---|---|---:|---:|
| `20260724_233524` | dataset A + `kd_top_1000` | 69.07 | +1.14 |
| `20260726_104522` | dataset B + `kd_top_1000` | 69.08 | +1.15 |
| `20260727_091039` | dataset B + `kl_top_1000` | **69.32** | **+1.39** |
| `20260725_225513` | 在 233524 上再蒸（`linear_depth` / 1000 step） | 68.59 | 无效 |

4. 目前最佳：`.../Qwen_Qwen3-8B_20260727_091039`（`kl_top_1000` + dataset B）。dataset A→B 几乎无差；`kd`→`kl` 有小幅增益。
5. 从 VAE-only ckpt 开蒸必须 `distill_reset_completed=true`，否则会跳过 `down_proj`。

---

## 2. 目录

```text
/root/data/ckpts/result/
  catlora/
    Qwen_Qwen3-8B_20260723_181339/   # 中断
    Qwen_Qwen3-8B_20260724_091805/   # 到 after_gate_proj
    Qwen_Qwen3-8B_20260724_185647/   # 短暂 resume
    Qwen_Qwen3-8B_20260724_190531/   # ★ VAE 基座 final_model
  catlora_distill/
    res0-bf16-protect-channel-vae/   # down_proj 蒸馏簇
      Qwen_Qwen3-8B_*/               # 各次 run
```

`res0-bf16-protect-channel-vae` 含义：encoder `num_res_blocks=0` + bf16 + channel outlier protect 的 VAE ckpt。

---

## 3. VAE 基座

路径：`/root/data/ckpts/result/catlora/Qwen_Qwen3-8B_20260724_190531/`

### 3.1 配置

| 项 | 值 |
|---|---|
| model | `Qwen/Qwen3-8B` |
| target / transpose | 全类；`q_proj,v_proj,o_proj,down_proj` 转置 |
| outlier | `channel` / `channel_weight_actmean_abs` / axis=`input` / scope=`layer` |
| protect_count | 默认 32；**down_proj=128** |
| residual_stages | 2 |
| num_res_blocks | 0（encoder） |
| decoder_num_res_blocks | 1 |
| steps_per_category | 10000 |
| distill_after_category | none |
| seed | 31 |
| resume | `.../20260724_091805/after_gate_proj` |

### 3.2 逐类累积评测

`gate_proj` 已保存 `after_gate_proj`，但评测日志中断，分数缺失。

| 压缩完该类后 | boolq | rte | wino | arc_e | arc_c | obqa | piqa | mmlu | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_proj | 85.38 | 74.01 | 64.80 | 79.88 | 54.95 | 41.40 | 77.58 | 70.79 | 68.60 |
| k_proj | 84.71 | 75.09 | 65.35 | 79.59 | 55.29 | 42.80 | 77.69 | 70.67 | 68.90 |
| v_proj | 83.36 | 74.01 | 63.85 | 76.14 | 52.82 | 40.00 | 77.09 | 68.06 | 66.92 |
| o_proj | 81.35 | 73.65 | 62.67 | 75.97 | 50.34 | 40.80 | 76.66 | 63.43 | 65.61 |
| gate_proj | — | — | — | — | — | — | — | — | 缺失 |
| up_proj | 78.81 | 76.17 | 61.09 | 68.94 | 42.83 | 36.20 | 71.06 | 52.68 | 60.97 |
| down_proj / final | 71.38 | 69.31 | 57.06 | 60.40 | 35.15 | 31.40 | 67.03 | 44.34 | **54.51** |

### 3.3 VAE 中断链

| run | 状态 |
|---|---|
| `20260723_181339` | 训到 gate，评测未完成 |
| `20260724_091805` | 写出 `after_gate_proj`，gate 评测未完成 |
| `20260724_185647` | resume 后几乎无产出 |
| `20260724_190531` | 续完 up/down，写出 `final_model` |

---

## 4. down_proj 蒸馏

### 4.1 公共设定

除非表中另写，以下为公共项：

- 4 卡；`target_categories=down_proj`；`transpose_modules=down_proj`
- `distill_after_category=both`
- `distill_independent_categories=true`
- `distill_reset_completed=true`（有效 run）
- LoRA r16 / α16 / dropout 0.03 / DoRA off
- `loss_alpha=0.5`，`temperature=1.0`
- `hidden_loss_weight=0.03`，`pre_mlp_hidden_loss_weight=0.0`
- AdamW lr `2e-5`，wd `0.001`，max_grad_norm `1.3`，warmup_ratio `0.05`
- bf16；grad accum 2；seqlen 1024；gc on
- resume 默认：`catlora/...190531/final_model`

数据配比：

- **dataset A**：`edgerazor_ii_7m=0.409,...,vaellm_eval_task=0.401`
- **dataset B**：`edgerazor_ii_7m=0.676,...,vaellm_eval_task=0.009`

### 4.2 独立 down_proj 基线（蒸馏前）

多次复现：

| 设置 | avg |
|---|---:|
| 全类 VAE | 54.51 |
| 仅 down_proj VAE，其它类 original | **67.93** |

分项：boolq 86.42 / rte 80.51 / wino 65.35 / arc_e 75.93 / arc_c 51.88 / obqa 39.40 / piqa 75.95 / mmlu 68.04。

### 4.3 Run 总表

| run | 状态 | 相对基线的关键改动 | pre | post |
|---|---|---|---:|---:|
| `20260724_213750` | 跳过蒸馏 | `reset_completed=false` | — | 54.53（全类状态） |
| `20260724_215638` | 评测后中断 | reset=true；bs=12；`cosine_with_warmup`；A | 67.93 | — |
| `20260724_222540` | 同上 | 重试 | 67.93 | — |
| `20260724_225526` | 同上 | 重试 | 67.93 | — |
| `20260724_231427` | trainer 启动后中断 | sched=`cosine`；bs=12 | 67.93 | — |
| `20260724_233524` | **完成** | bs=10；2000；`kd_top_1000`；A；`adaptive_top_3` | 67.93 | **69.07** |
| `20260725_222104` | 评测中断 | steps=1000；`linear_depth` | — | — |
| `20260725_225513` | **完成** | 从 233524 再蒸；1000；`linear_depth`；A | 69.00 | **68.59** |
| `20260726_104522` | **完成** | dataset **B**；`kd_top_1000`；2000 | 67.93 | **69.08** |
| `20260727_091039` | **完成 ★当前最佳** | dataset B；**`kl_top_1000`**；2000 | 67.93 | **69.32** |

### 4.4 已完成 run 分项

**`20260724_233524`**（dataset A + kd）

| 阶段 | boolq | rte | wino | arc_e | arc_c | obqa | piqa | mmlu | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 86.42 | 80.51 | 65.35 | 75.93 | 51.88 | 39.40 | 75.95 | 68.04 | 67.93 |
| post | 86.61 | 76.53 | 68.82 | 79.80 | 55.97 | 39.80 | 77.75 | 67.31 | 69.07 |
| Δ | +0.19 | -3.98 | +3.47 | +3.87 | +4.09 | +0.40 | +1.80 | -0.73 | +1.14 |

**`20260726_104522`**（dataset B + kd）

| 阶段 | boolq | rte | wino | arc_e | arc_c | obqa | piqa | mmlu | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 86.42 | 80.51 | 65.35 | 75.93 | 51.88 | 39.40 | 75.95 | 68.04 | 67.93 |
| post | 86.12 | 75.81 | 68.03 | 80.51 | 55.97 | 40.40 | 77.86 | 67.97 | 69.08 |
| Δ | -0.30 | -4.70 | +2.68 | +4.58 | +4.09 | +1.00 | +1.91 | -0.07 | +1.15 |

与 233524 几乎持平：换 dataset B（降低 `vaellm_eval_task`）对均值影响可忽略。

**`20260727_091039`**（dataset B + kl，当前最佳）

| 阶段 | boolq | rte | wino | arc_e | arc_c | obqa | piqa | mmlu | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 86.42 | 80.51 | 65.35 | 75.93 | 51.88 | 39.40 | 75.95 | 68.04 | 67.93 |
| post | 86.36 | 78.70 | 68.51 | 79.76 | 55.03 | 40.60 | 77.09 | 68.49 | **69.32** |
| Δ | -0.06 | -1.81 | +3.16 | +3.83 | +3.15 | +1.20 | +1.14 | +0.45 | **+1.39** |

相对 kd 版（104522）：rte / mmlu 更好，arc_c / piqa 略弱，均值 +0.24pp。

**`20260725_225513`**（233524 上再蒸）

| 阶段 | avg |
|---|---:|
| pre（读入 233524） | 69.00 |
| post | 68.59（-0.41） |

改动：`linear_depth` + 1000 step。结论：再蒸无效。

### 4.5 对照解读

在独立 `down_proj` 设定下：

1. `both` + LoRA r16 + 2000 step 可稳定拿到约 +1.1～1.4pp。
2. dataset A vs B：均值几乎无差。
3. `kl_top_1000` 略优于 `kd_top_1000`（当前最佳 69.32）。
4. 同 ckpt 再蒸 + 换 `linear_depth`：有害。
5. 上述分数都不是全类压缩模型上的恢复效果；全类仍是 54.51 量级，需要另开实验。

### 4.6 已知坑

1. **`distill_reset_completed`**：VAE ckpt 的 `completed_categories` 表示压缩完。从 `190531` 开蒸必须 `true`，否则跳过（见 `213750`）。
2. **独立类 vs 全类**：`independent=true` 的评测只反映「只坏 down_proj」；不能和 54.51 横比。
3. **中断 run**：`215638`/`222540`/`225526`/`231427`/`222104` 无完整 Traceback，更像人工中断或环境中断。

---

## 5. 脚本现状（注意与历史 run 不同）

`scripts/catlora_distill_4gpu_res0.sh` 当前默认已改成全类蒸馏方向，**不等于**上表里已完成的独立 down_proj 实验：

| 项 | 已完成最佳 `091039` | 当前脚本默认 |
|---|---|---|
| resume | `...190531/final_model` | `.result/catlora/res0-bf16-protect-channel-vae/final_model` |
| target_categories | `down_proj` | 全类 |
| reset_completed | true | false |
| loss_type | `kl_top_1000` | `eakld` |
| lora_rank / alpha | 16 / 16 | 8 / 4 |
| batch_size | 10 | 4 |
| hidden_loss_weight | 0.03 | 0.1 |
| dataset | B | B |

若要复现当前最佳独立 down_proj 结果，不要直接按脚本现状跑，需按 `091039` 参数覆盖。

---

## 6. 建议下一步

1. 在独立设定下复现/扩展 `kl_top_1000`：扫 rank、steps、`hidden_loss_weight`。
2. 对比 `compressed_lora` only / `decoder` only / `both`。
3. 关 `independent_categories`，在全类压缩模型上只蒸 `down_proj`，看能否从 54.51 往回拉（部署相关指标）。
4. 补 `gate_proj` 累积评测，确认 MLP 掉点从哪一类开始加重。

---

## 7. 路径速查

| 用途 | 路径 |
|---|---|
| VAE final | `/root/data/ckpts/result/catlora/Qwen_Qwen3-8B_20260724_190531/final_model/` |
| 当前最佳蒸馏 | `/root/data/ckpts/result/catlora_distill/res0-bf16-protect-channel-vae/Qwen_Qwen3-8B_20260727_091039/` |
| dataset B + kd | `.../Qwen_Qwen3-8B_20260726_104522/` |
| dataset A + kd | `.../Qwen_Qwen3-8B_20260724_233524/` |
| 再蒸更差 | `.../Qwen_Qwen3-8B_20260725_225513/` |
| 当前脚本 | `scripts/catlora_distill_4gpu_res0.sh` |
