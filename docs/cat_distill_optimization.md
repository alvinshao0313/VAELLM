# Decoder / LoRA 蒸馏工作流优化清单

本文档整理「训练 decoder」与「仅 LoRA（`compressed_lora`）」链路中已确认的问题与改法。  
范围：checkpoint 蒸馏与每类后蒸馏。

**落地状态（2026-07-16）**

| ID | 状态 |
|---|---|
| O1 | 已做：每类后写 `after_<category>/` |
| O2 | 已做：`completed_categories` + 已有 `low_rank` 自动跳过；文档已去 `VAE_CKPT` |
| O3 | 已做：`constant`+warmup>0 启动报错；脚本改为 `constant_with_warmup` |
| O4 | 已做：decoder/both 只预热前缀，跳过当前类 |
| O5 | 已做：去掉 checkpoint distill 外层预热，只保留 after-category 内层 |
| O6 | 已做：both 前向直接算 LoRA A/B delta（数学等价） |
| O7 | 已做：无可训参数时拆 proxy / finalize decoder |
| O8 | 已做：蒸馏脚本去掉无效 `outlier_protect_mode` |
| O9 | 已做：iterable 时强制关闭 group_by_length 并打日志；脚本改为 false |
| O10 | 已做：run 级缓存 distill dataset |
| O11 | 已做：保留 `find_unused_parameters=True` 并打日志说明原因 |
| O12 | 已做：`init_mode=peft_default` |
| O13 | 已做：重写 `docs/catlora_distill_from_checkpoint.md` |
| O14 | 已做：e2e `--finetune_mode compressed_lora`，无 `lora`/`low_rank` mode 兼容 |
| O15 | 已做：`compressed_lora` 忽略 `vae_decoder_checkpoint` |
| O16 | 已做：completed 物化为 `TemporarySwitchLinear` + 共享 original bank，VAE CPU stash |

回归测试见 `docs/cat_distill_test.md`。

相关入口：

- VAE 压缩：`scripts/catlora_simple.sh` → `tools/cat_train.py`
- 从 ckpt 蒸馏：`scripts/catlora_distill_*.sh` → `tools/cat_distill_from_vae_checkpoint.py`
- 核心逻辑：`train_utils/cat_checkpoint_distill.py`、`train_utils/cat_after_category_distill.py`

---

## 1. 调用链

```text
scripts/catlora_simple.sh
  → tools/cat_train.py
  → train_utils/cat_train_pipeline.run_cat_train()
       默认 --distill_after_category=none（只做 VAE 压缩）

scripts/catlora_distill_from_checkpoint.sh
scripts/catlora_distill_4gpu_res0.sh
  → tools/cat_distill_from_vae_checkpoint.py
  → train_utils/cat_checkpoint_distill.run_cat_checkpoint_distill()
       按 target_categories 逐类激活压缩路径
       → train_utils/cat_after_category_distill.run_after_category_distill()
            compressed_lora | decoder | both
```

模式含义：

| `--distill_after_category` | 训练目标 | 收尾 |
|---|---|---|
| `compressed_lora` | 当前类 VAELinear 上的 proxy LoRA | 导出 `low_rank_a/b`，去掉 proxy |
| `decoder` | 当前类 VAELinear 的 decoder 参数 | `disable_trainable_decode`，清 cache |
| `both` | decoder + proxy LoRA | 两者都做收尾 |

---

## 2. 明确不处理的项

以下已确认**不在本优化清单内**：

1. `eval_ppl=false` 但仍配置了 `eval_tasks` 时会跑下游评测 —— 这是预期行为，不是问题。
2. `catlora_simple.sh` 在 `distill_after_category=none` 时仍保留蒸馏 CLI —— 使用者习惯，不改。
3. 多套蒸馏脚本超参不一致 —— 实验配置自行维护，不强制统一。

---

## 3. 优化项总表

| ID | 主题 | 优先级 |
|---|---|---|
| O1 | 无按类中间保存 | 高 |
| O2 | 续跑靠手工 `steps=0`；文档仍写 `VAE_CKPT` | 高 |
| O3 | `constant` + `warmup_ratio` 无效 | 高 |
| O4 | decoder 内层无效预热 | 中 |
| O5 | LoRA 双重预热 | 中 |
| O6 | `both` 前向实现冗余（效率，数学等价） | 中 |
| O7 | 无可训参数提前 return 残留 proxy | 中 |
| O8 | 蒸馏脚本夹杂无效 VAE/outlier 参数 | 低 |
| O9 | `group_by_length` 对 iterable 无效 | 低 |
| O10 | 每类重复准备 distill dataset | 低 |
| O11 | DDP `find_unused_parameters=True` | 低 |
| O12 | `init_mode="zero"` 名不副实 | 低 |
| O13 | 蒸馏文档过时 | 中 |
| O14 | cat vs e2e mode 命名不统一 | 高 |
| O15 | 纯 LoRA 下 `vae_decoder_checkpoint` 无意义 | 低 |

---

## 4. 优化项详情

### O1. 无按类中间保存

**现象 / 证据**

- `train_utils/lora_utils.py` 里 `_build_sft_args` 固定 `save_strategy="no"`。
- `cat_checkpoint_distill.py` 只在全部类别跑完后、且 `save_model` 时写一次 `final_model`。
- 实际日志：`q_proj` 单类可跑近数天；中途崩溃则整轮结果丢失。

**影响**

- 可恢复性极差；长跑实验风险高。

**建议改法**

1. 每个类别蒸馏成功后立刻落盘，例如 `after_<category>/` 或等价目录。
2. 保存内容与最终 `final_model` 同格式（含已导出的 `low_rank_a/b` / 已 finalize 的 decoder）。
3. 续跑入口读取该中间 ckpt，从未完成类别继续（与 O2 联动）。

**优先级**：高

---

### O2. 续跑靠手工 `steps=0`；文档与脚本不一致

**现象 / 证据**

- 现有续跑约定（见 `docs/catlora_distill_from_checkpoint.md`）：已完成类别必须留在 `target_categories` 前缀，并设 `distill_steps ... after:xxx=0`。
- 若只写后续类别，inactive 类会被 stash 成 `original_weight` 稠密 Linear，训练期 progressive 状态错误。
- 若前缀 steps 不为 0 且已有 `low_rank_a/b`，旧逻辑下 `export_peft_proxy_lora_to_low_rank` 会拒绝覆盖。
- 现状：`--distill_reset_completed false` 时已有 `low_rank` 自动 skip；`true` 时用已有 `low_rank` 初始化 LoRA 再训并允许覆盖写回（见 `docs/catlora_distill_from_checkpoint.md`）。
- 文档仍写 `VAE_CKPT=... bash scripts/...`；脚本实际硬编码 `--resume_from_checkpoint`。

**影响**

- 续跑易配错；文档会直接误导启动方式。

**建议改法**

1. 代码：若某类已有完整 `low_rank_a/b`（`compressed_lora`/`both`），或已判定无需再训 decoder，则自动 skip，不必手工 `steps=0`。
2. 脚本与文档统一使用 `--resume_from_checkpoint`；删除 `VAE_CKPT` 叙述。
3. 与 O1 中间保存联动：支持「从 after_q_proj 继续 k_proj」。
4. `--distill_reset_completed true`：在已有蒸馏参数上再蒸一轮（LoRA 从 `low_rank` 初始化并覆盖写回）。

**优先级**：高

---

### O3. `constant` + `warmup_ratio` 无效

**现象 / 证据**

- 蒸馏脚本常见配置：`--distill_lr_scheduler_type constant` + `--distill_warmup_ratio 0.05/0.1`。
- HuggingFace 中 `constant` 调度器忽略 warmup；`constant_with_warmup` 才会升温。
- 实测：`constant` 从 step 0 起就是满 LR。

**影响**

- 超参与真实训练不一致；以为有 warmup，实际没有。

**建议改法**

- 需要 warmup：改成 `constant_with_warmup`。
- 不需要 warmup：删掉 `distill_warmup_ratio`，避免误导。
- 可在参数校验里：`scheduler=constant` 且 `warmup_ratio>0` 时直接报错。

**优先级**：高

---

### O4. decoder 内层无效预热

**现象 / 证据**

- 外层 `cat_checkpoint_distill.py`：仅 `mode == "compressed_lora"` 时做 `prime_named_vae_linear_cache`；`decoder`/`both` 故意跳过。
- 内层 `cat_after_category_distill._run_compressed_category_distill`：对所有 compressed 模式都调用 `prime_model_vae_linear_cache`。
- 随后 `_enable_only_decoder_params` → `enable_trainable_decode()` 会 `cache_decoded_weight=False` 并 `clear_decoded_weight_cache()`。

**影响**

- 当前类预热白做；浪费时间，并模糊「训练期应无 cache」的语义。

**建议改法**

- `decoder`/`both`：跳过**当前类别**的 prewarm。
- 若需要，仅对已冻结、仍走 cache 的前缀类别预热。

**优先级**：中

---

### O5. LoRA 双重预热

**现象 / 证据**

- 外层已对 active prefix 做 `prime_named_vae_linear_cache(clear_existing=True)`。
- 内层再次 `prime_model_vae_linear_cache`。
- 日志示例（训 `k_proj` 时）：`warmed=36 skipped=36` —— 当前 proxy 被 skip，前缀 `q_proj` 被重复 warm。

**影响**

- 多余 decode / 显存抖动；逻辑重复。

**建议改法**

- `compressed_lora`：以内层 materialize 为准时，去掉外层 prewarm；或外层保留、内层跳过已 warm 模块。二选一，不要两边都做。

**优先级**：中

---

### O6. `both` 前向实现冗余（效率，数学等价）

**现象 / 证据**

`e2e_common/peft_proxy.py` 中 `PeftVAELinearProxy.forward`（`_train_decoder_with_adapter=True`）：

```python
decoder_out = self.base_layer(x)
adapter_delta = peft_linear(x) - dense_base(x)
return decoder_out + adapter_delta
```

含义：

- `base_layer(x)`：可微分解码，训 decoder 必需。
- `peft_linear(x) - dense_base(x)`：为抠出 LoRA delta，多走了完整稠密 Linear，并依赖 materialize 的 dense base。

优化目标本身是：

```text
y = decoder(x) + LoRA_delta(x)
```

**影响**

- 算力、显存偏高；**不是**实验设定错误。

**建议改法（已确认）**

1. 保持优化目标不变：联合训 decoder + LoRA，`y = decoder(x) + LoRA_delta(x)`。
2. 直接用 LoRA A/B 计算 delta，再加到 `decoder_out`。
3. 去掉 `peft_linear(x) - dense_base(x)` 这条为抠 delta 而做的稠密减法路径。
4. 允许极小浮点差；不改实验语义。
5. **不要**改成「只训一边」或取消 `both`。

**优先级**：中

---

### O7. 无可训参数提前 return 残留 proxy

**现象 / 证据**

- `_run_compressed_category_distill` 在 `use_lora` 时先 wrap `PeftVAELinearProxy` 并 inject adapter。
- 之后若 `_enable_compressed_trainable_params` 得到空列表，直接 `return`，不走 export。
- 最终保存会检查 leftover proxy 并报错（`cat_checkpoint_distill._save_final_model`）。

**影响**

- 少见，但一旦触发会导致整轮保存失败或模型状态脏。

**建议改法**

- 提前 return 前：拆掉 proxy，恢复裸 `VAELinear`；或
- 在确认有可训参数之前不要 wrap。

**优先级**：中

---

### O8. 蒸馏脚本夹杂无效 VAE/outlier 参数

**现象 / 证据**

- `catlora_distill_*.sh` 仍出现 `--outlier_protect_mode` 等；checkpoint distill 不重训 VAE，这些参数基本不参与。
- 参数解析仍打印大量 VAE 默认值，增加阅读噪音。

**影响**

- 可维护性差；易误以为蒸馏阶段还在做 outlier/VAE 训练。

**建议改法**

- 蒸馏脚本只保留蒸馏相关 CLI；或
- 加载后对无关项打明确 log：`ignored for checkpoint distill: ...`。

**优先级**：低

---

### O9. `group_by_length` 对 iterable 无效

**现象 / 证据**

- 脚本常写 `--distill_group_by_length true`。
- `_build_sft_args`：若 `train_is_iterable`，强制 `group_by_length=False`（EdgeRazor lazy Dataset 即此情况）。

**影响**

- 配置与真实行为不符。

**建议改法**

- iterable 场景脚本改为 `false`；或启动时 log：`group_by_length ignored because dataset is iterable`。

**优先级**：低

---

### O10. 每类重复准备 distill dataset

**现象 / 证据**

- 每个类别都调用 `prepare_distill_datasets(...)`，mix 相同。
- 日志每类重复打印同一组 alias/weight/raw_rows。

**影响**

- 重复开销与日志噪音；lazy 下通常不致命。

**建议改法**

- run 级缓存 dataset / tokenizer，按类复用。

**优先级**：低

---

### O11. DDP `find_unused_parameters=True`

**现象 / 证据**

- `_build_sft_args` 在分布式时固定 `ddp_find_unused_parameters=True`。
- 原因：只解冻当前类参数，其它参数 unused。

**影响**

- DDP 额外开销；长跑会放大。

**建议改法**

- 确认计算图静态、无条件分支导致的 unused 后尝试关闭；或
- 进一步缩小参与 DDP 的可训模块集合，减少 unused。

**优先级**：低

---

### O12. `init_mode="zero"` 名不副实

**现象 / 证据**

- `_run_compressed_category_distill` 调用 `ensure_peft_vae_proxy_adapter(..., init_mode="zero")`。
- plain LoRA 路径：`init_mode != "gaussian"` 时传 `init_lora_weights=True`（PEFT 默认，通常 A 随机、B 零），并非双侧全零。
- AdaLoRA 才有真正的 zero 初始化分支。

**影响**

- 命名误导；排障时以为从全零 delta 起步。

**建议改法**

- 改名为 `default` / `peft_default`，或实现真正的双侧 zero 并写清语义。二选一，不要名实不符。

**优先级**：低

---

### O13. 蒸馏文档过时

**现象 / 证据**

- `docs/catlora_distill_from_checkpoint.md` 仍写：
  - `VAE_CKPT=...`
  - 默认 `target_categories=q_proj,k_proj,v_proj`
  - 旧数据 mix 示例
- 当前脚本：硬编码 `--resume_from_checkpoint`，全 7 类，EdgeRazor mix。

**影响**

- 按文档操作会失败或训错范围。

**建议改法**

- 重写「启动 / 续跑 / 跳过已完成类别」三节，与现脚本一致；去掉 `VAE_CKPT`。

**优先级**：中

---

### O14. cat vs e2e mode 命名不统一

详见第 5 节。

**优先级**：高

---

### O15. 纯 LoRA 下 `vae_decoder_checkpoint` 无意义

**现象 / 证据**

- 蒸馏脚本常开 `--vae_decoder_checkpoint true`。
- `compressed_lora` 前向走 materialize 的 `per_decoded_linear`，不跑 decoder。
- 该开关主要影响 decoder 内部 `use_checkpoint`。

**影响**

- 配置噪音；让人以为 LoRA 路径也在做 decoder activation checkpoint。

**建议改法**

- `compressed_lora`：忽略该开关并 log。
- 仅 `decoder` / `both` 生效。

**优先级**：低

---

## 5. 命名统一（O14）

### 现状

| 位置 | 名称 |
|---|---|
| cat `--distill_after_category` | `compressed_lora` / `decoder` / `both` |
| e2e CLI `--finetune_mode` | `lora` / `decoder` / `both` |
| e2e 内部 `vae_train_mode` | `low_rank` / `decoder` / `both`（`lora` → `low_rank` 映射） |

映射代码：`compressed_e2e_fintuning/args.py` 中 `_FINETUNE_MODE_TO_VAE_TRAIN_MODE`。

### 统一方案（不留兼容）

以 **cat 侧为准**：

- 对外与对内统一为：`compressed_lora` / `decoder` / `both`
- e2e：
  - `--finetune_mode lora` 改为 `compressed_lora`
  - 删除 `lora` → `low_rank` 映射
  - 内部 `vae_train_mode` / `train_mode` 一律使用 `compressed_lora`，不再出现 `low_rank` 作为 mode 名
- 所有相关脚本、文档、日志、meta 字段同步改名
- **不留别名，不留旧值解析，不留兼容分支**

### 明确不改的名字

- `VAELinear.low_rank_a` / `low_rank_b`：这是权重存储字段，不是 finetune mode，保持不变。
- 函数名里若仅表示「读写 low_rank 载荷」（如 `extract_low_rank_payloads_from_lora`），可随后按需重命名，但不阻塞 mode 统一；mode 统一优先。

---

## 6. 建议落地顺序

1. **O1 + O2**：中间保存与可续跑 —— 直接影响长跑能否完成  
2. **O3**：修好 LR warmup 配置 —— 否则超参与真实训练不一致  
3. **O4 + O5**：去掉无效/重复预热 —— 降耗、语义更清晰  
4. **O14**：e2e 命名统一到 `compressed_lora` —— 消除跨入口混淆  
5. **O6**：`both` 前向效率（数学等价）  
6. **O7 + O13**：proxy 泄漏防护 + 文档同步  
7. **其余低优先级项**（O8–O12、O15）按需处理  

---

## 7. 关键文件索引

| 文件 | 角色 |
|---|---|
| `scripts/catlora_simple.sh` | VAE 压缩实验脚本 |
| `scripts/catlora_distill_from_checkpoint.sh` | 单卡 checkpoint 蒸馏 |
| `scripts/catlora_distill_4gpu_res0.sh` | 多卡 checkpoint 蒸馏 |
| `tools/cat_train.py` | 压缩入口 |
| `tools/cat_distill_from_vae_checkpoint.py` | 蒸馏入口 |
| `train_utils/cat_checkpoint_distill.py` | residency / 前缀激活 / 外层预热 / 保存 |
| `train_utils/cat_after_category_distill.py` | decoder / LoRA / both 分支 |
| `train_utils/lora_utils.py` | Trainer 参数、freeze、数据集侧辅助 |
| `e2e_common/peft_proxy.py` | proxy、materialize、export、`both` 前向 |
| `litebsq/vae_linear.py` | `enable_trainable_decode` / cache / `low_rank_a/b` |
| `compressed_e2e_fintuning/args.py` | e2e mode 枚举与映射 |
| `docs/catlora_distill_from_checkpoint.md` | 蒸馏用法文档（待按 O13 更新） |
