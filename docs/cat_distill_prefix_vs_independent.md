# 实验记录：checkpoint distill 前缀累加 vs 独立类别

日期：2026-07-28  
模型：Qwen3-8B  
压缩基线：`res0-bf16-protect-channel-vae`（channel outlier protect）  
评估：`boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu` 八任务均值（%）

## 1. 问题

`--distill_independent_categories` 控制蒸馏时的压缩前缀：

| 取值 | 行为 |
|---|---|
| `false`（前缀累加） | 蒸当前类时，已完成类保持压缩态，active 前缀不断变长 |
| `true`（独立类别） | 每轮只激活当前类；已完成类恢复为稠密 Linear；最终评估才一次性全压 |

目标：在同一 VAE 全压 ckpt 上，对比两种策略的最终全压效果，并解释差异。

## 2. 参与对比的主要 run

### 2.1 全压不蒸馏基线（对照）

| 项 | 值 |
|---|---|
| 日志 | `.result/catlora/res0-bf16-protect-channel-vae/linear_by_category.log` |
| 设定 | `distill_after_category=none`，前缀累加压缩，无蒸馏 |
| 最终全压 mean | **54.29** |

前缀轨迹（每多压一类后评估）：

| 已压前缀终点 | mean |
|---|---:|
| q | 69.59 |
| q+k | 69.16 |
| +v | 67.32 |
| +o | 64.72 |
| +gate | 61.49 |
| +up | 59.24 |
| +down（全压） | **54.29** |

### 2.2 前缀累加蒸馏（independent≈false）

由多次续跑拼成完整七类（均为 `decoder` 蒸馏，非 `both`）：

| 阶段 run | resume | 本段训练类 | steps / rank |
|---|---|---|---|
| `.../20260717_032410` | VAE `final_model` | q, k | 500 / 4 |
| `.../20260719_030713` | 上一段 `after_k_proj` | v, o, gate（up 中断） | 500 / 4 |
| `.../20260722_010515` | `.../030713/after_gate_proj` | up, down | 500 / 4 |

数据配比：`edgerazor_ii_7m=0.676,...`（eval_task 权重约 0.009）  
日志中未显式写 `independent=true`，行为是前缀累加（跳过类仍物化为 TemporarySwitchLinear 并留在 active 前缀）。

**最终全压 mean：57.34（相对不蒸馏 +3.05）**

### 2.3 独立类别蒸馏（independent=true）

| 项 | 值 |
|---|---|
| 日志 | `.result/catlora_distill/.../Qwen_Qwen3-8B_20260724_111408/linear_by_category.log` |
| 设定 | `both`，`independent=true`，steps=2000，rank=8 |
| 数据 | eval_task 权重提到 0.401 |
| 中途单类评估 mean | 约 68–71（虚高，见下） |
| **最终全压 mean** | **40.36（相对不蒸馏 −13.93）** |

### 2.4 其它相关但不作主对比的 run

| run | 说明 |
|---|---|
| `catlora/Qwen_Qwen3-8B_20260724_111822` | 另一套 VAE（`residual_sparse` protect，无 channel protect），全压 mean 35.83；与 res0 基线不可比 |
| `.../20260724_105337` | independent=true 的短冒烟（仅 q） |
| `.../20260728_015034` | `both` + `independent=false` 新跑，写本文时仅完成 q 前评估，尚无最终结论 |

## 3. 核心结果

### 3.1 最终全压（唯一公平终点）

| 设定 | mean | boolq | rte | wino | arcE | arcC | obqa | piqa | mmlu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE 全压不蒸馏 | 54.29 | 72.26 | 64.26 | 56.75 | 62.75 | 36.52 | 31.40 | 68.66 | 41.71 |
| 前缀累加蒸馏（decoder 链） | **57.34** | 76.15 | 69.31 | 59.98 | 65.70 | 38.91 | 33.00 | 72.20 | 43.48 |
| 独立类别蒸馏（both） | **40.36** | 56.27 | 46.21 | 50.43 | 36.15 | 25.60 | 29.20 | 54.35 | 24.66 |
| 前缀 vs 不蒸馏 | **+3.05** | +3.89 | +5.05 | +3.23 | +2.95 | +2.39 | +1.60 | +3.54 | +1.77 |
| 独立 vs 不蒸馏 | **−13.93** | −16.0 | −18.1 | −6.3 | −26.6 | −10.9 | −2.2 | −14.3 | −17.1 |

结论一句话：

- **前缀累加蒸馏：最终全压略好于不蒸馏。**
- **独立类别蒸馏：最终全压明显差于不蒸馏，也远差于前缀累加。**

### 3.2 前缀长度对齐的轨迹（累加蒸馏 vs 不蒸馏）

取「该类蒸馏结束后、前缀已含至该类」的 mean，与 VAE 同前缀对比：

| 前缀终点 | VAE 不蒸馏 | 前缀累加蒸馏后 | Δ |
|---|---:|---:|---:|
| q | 69.59 | 70.02 | +0.43 |
| k | 69.16 | 70.04 | +0.88 |
| v | 67.32 | 69.10 | +1.78 |
| o | 64.72 | 67.21 | +2.49 |
| gate | 61.49 | 63.89 | +2.40 |
| up | 59.24 | 61.87 | +2.63 |
| down / 全压 | 54.29 | 57.34 | +3.05 |

前缀越长，蒸馏相对不蒸馏的优势略增大；全程没有出现「蒸完反而低于同前缀 VAE」的情况。

### 3.3 独立模式下的误导性中途分

independent=true 时，中途评估也只激活当前类，因此 mean 长期停在 ~68–71：

| 类 | 蒸前（单类压） | 蒸后（单类压） | Δ |
|---|---:|---:|---:|
| q | 69.60 | 70.15 | +0.55 |
| k | 70.24 | 70.75 | +0.51 |
| v | 68.55 | 70.58 | +2.03 |
| o | 68.80 | 69.73 | +0.93 |
| gate | 68.20 | 68.79 | +0.59 |
| up | 67.75 | 69.03 | +1.28 |
| down | 67.41 | 68.74 | +1.33 |

单类设定下蒸馏确实略升；但最终七类全激活后落到 **40.36**。  
**不能用 independent 中途分数判断最终全压质量。**

## 4. 原因分析

### 4.1 训练 / 推理分布不一致（主因）

independent=true：

1. 蒸 `k` 时，`q` 已恢复稠密 → 学生几乎只面对「单类量化误差」。
2. 最终评估突然七类全压 → 误差沿层叠加。
3. 各类 decoder/LoRA 从未在「其它类也压缩」的状态下联合适应。

前缀累加：

1. 蒸到第 k 类时，前 k 类都已压缩。
2. 训练态与最终全压同分布（逐步逼近）。
3. 后面类是在已有压缩噪声上继续适应。

### 4.2 为何独立中途分还更高

中途只压一类，其余 dense，任务分自然接近「轻度压缩」水平（~70），与「全压 ~54」不是同一状态机。

### 4.3 对比时的混杂因素（需诚实写出）

两组蒸馏**不是**严格单变量 A/B：

| 因素 | 前缀累加链 | 独立 run |
|---|---|---|
| `distill_after_category` | `decoder` | `both` |
| steps / rank | 500 / 4 | 2000 / 8 |
| 数据中 eval_task 权重 | ~0.009 | 0.401 |
| 是否一次跑完七类 | 多次续跑拼接 | 一次跑完 |

因此不能把「−14 vs +3」全部归因于 independent 开关 alone；但方向足够清楚：

- 前缀累加：在更弱的蒸馏预算（decoder、500 step）下仍相对基线为正。
- 独立：在更强预算（both、2000 step）下最终仍大幅为负。

若 independent 无害，更强预算应更容易打平或超过基线——实际相反，说明 **independent 的 train/test mismatch 是主导伤害**。

## 5. 完备结论

1. **最终全压是唯一可靠指标。** independent 中途 mean 会系统性虚高。
2. **前缀累加蒸馏有效：** 相对同 VAE 全压不蒸馏约 **+3 mean**；各前缀长度上均为非负增益。
3. **独立类别蒸馏有害（对该配置）：** 相对同基线约 **−14 mean**；mmlu 掉到近随机（24.66）。
4. **机制是分布错配，不是「蒸馏算子本身坏了」。** 单类 before→after 仍多为小幅正增益。
5. **默认策略应使用 `distill_independent_categories=false`。** 除非另有「全压联合精修」收尾，否则不要用纯 independent 出最终 ckpt。
6. **待补实验（严格对照）：** 同一 `both` / steps / rank / 数据配比下，只改 independent true/false 各跑完整七类；`.../20260728_015034`（both + indep=false）可用于补齐。

## 6. 改善建议（按优先级）

1. 脚本默认：`independent=false`（前缀累加）。
2. 若曾用 independent 跑完：用 `distill_reset_completed=true` 在前缀累加设定下再蒸一轮，或加「全类 active」短精修。
3. 评估：independent 中途也应可选「已完成前缀全激活」评估，避免误判。
4. 不要指望靠加大 independent 下单类 step/rank 挽回最终全压。

## 7. 日志索引

```text
基线 VAE 全压不蒸馏
  .result/catlora/res0-bf16-protect-channel-vae/linear_by_category.log

前缀累加蒸馏链
  .result/catlora_distill/res0-bf16-protect-channel-vae/Qwen_Qwen3-8B_20260717_032410/
  .result/catlora_distill/res0-bf16-protect-channel-vae/Qwen_Qwen3-8B_20260719_030713/
  .result/catlora_distill/res0-bf16-protect-channel-vae/Qwen_Qwen3-8B_20260722_010515/   # final 57.34

独立类别蒸馏
  .result/catlora_distill/res0-bf16-protect-channel-vae/Qwen_Qwen3-8B_20260724_111408/   # final 40.36

进行中（both + indep=false，待补全）
  .result/catlora_distill/res0-bf16-protect-channel-vae/Qwen_Qwen3-8B_20260728_015034/
```
