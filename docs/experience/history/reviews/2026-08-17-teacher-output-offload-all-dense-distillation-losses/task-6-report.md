> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-6-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6 Report: Closed-Loop Regression Verification

**Status:** DONE  
**Commits created:** none（禁止提交）  
**Production/test files modified by this task:** none

工作区相对 HEAD 仍只有 Tasks 1–5 的 5 个预期文件：

```text
 M compressed_e2e_fintuning/trainer.py
 M e2e_common/dense_loss.py
 M tests/test_distill_losses.py
 M tests/test_e2e_teacher_first.py
 M train_utils/distill_losses.py
```

`git diff HEAD --stat`：5 files changed, 714 insertions(+), 144 deletions(-)。无额外 production/test 文件。

---

## Step 1: Environment

在当前 shell 激活 `bitvae` 后检查：

```bash
which python
python -V
```

结果：

- `which python` → `/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`
- `python -V` → `Python 3.11.13`
- `CONDA_DEFAULT_ENV=bitvae`

后续所有 pytest 均在该环境、工作目录 `/home/shaoyuantian/program/VAELLM`、`PYTHONPATH=.` 下执行。

---

## Step 2: Low-level dense loss suite

```bash
pytest -q tests/test_distill_losses.py
```

**结果：** 93 passed in 5.31s（exit 0）

覆盖通用 chunk executor、CPU-offload vs legacy loss/gradient parity（26 种 dense loss）、chunk H2D、EAKLD 路径。

---

## Step 3: Teacher target utility suite

```bash
pytest -q tests/test_teacher_target_offload.py
```

**结果：** 18 passed in 5.79s（exit 0）

---

## Step 4: Teacher-first trainer suite

```bash
pytest -q tests/test_e2e_teacher_first.py
```

**结果：** 44 passed in 13.28s（exit 0）

含 `test_cpu_kl_top_1000_teacher_before_student_and_backward`、26 种 dense loss trainer backward、非 EAKLD metadata skip、hidden alignment、training_step cleanup、legacy-vs-CPU trainer parity。

---

## Step 5: Dynamic-padding regression

```bash
pytest -q tests/test_distill_dynamic_padding.py
```

**结果：** 14 passed in 5.02s（exit 0）

batch-longest + multiple-of-8 的 mask/loss invariant 未因本任务破坏。

---

## Step 6: Token telemetry regression

```bash
pytest -q tests/test_e2e_distill_token_stats.py
```

**结果：** 11 passed in 5.45s（exit 0）

---

## Step 7: Smoke suites

```bash
pytest -q tests/smoke/test_loss_pipeline_smoke.py
```

**结果：** 3 passed in 4.65s（exit 0）

```bash
pytest -q tests/smoke/test_one_step_train_smoke.py
```

**结果：** 5 passed in 6.07s（exit 0）

---

## Step 8: Exact bug regression alone

```bash
pytest -q tests/test_e2e_teacher_first.py::test_cpu_kl_top_1000_teacher_before_student_and_backward
```

**结果：** 1 passed in 5.05s（exit 0）

工作树中已删除：

- trainer 侧 `"teacher_output_offload=cpu supports only sft/origin hidden alignment and EAKLD-family losses."`
- dense_loss 侧 `"teacher_output_offload=cpu supports only EAKLD-family losses."`

全仓库 `*.py`/`*.md` 已无上述字符串。`kl_top_1000` 走 teacher-first CPU path，不再报 unsupported。

---

## Step 9: Audit — forbidden full-logit restore

审计对象：`git diff HEAD -- train_utils/distill_losses.py e2e_common/dense_loss.py compressed_e2e_fintuning/trainer.py`。

### 新增 H2D（允许）

`train_utils/distill_losses.py` 新增 `_make_checkpointed_token_mean_chunk_forward()`，唯一 H2D 是现有 helper：

```735:741:train_utils/distill_losses.py
    def chunk_forward(active_student_chunk: torch.Tensor) -> torch.Tensor:
        teacher_chunk = copy_teacher_logit_chunk_to_device(
            teacher_logits_cpu,
            start=fixed_start,
            end=fixed_end,
            target_device=active_student_chunk.device,
        )
```

`copy_teacher_logit_chunk_to_device()` 实现仍是 `teacher_logits_cpu[:, start:end, :].to(device=target)`（`compressed_e2e_fintuning/teacher_targets.py` 116–140 行，本任务未改该文件）。

既有 EAKLD chunk forward `_make_checkpointed_eakld_chunk_forward()` 同样只调用该 helper（912–917 行），未新增 full restore。

`compute_chunked_token_mean_from_cpu_teacher_logits()` 只按 `iter_token_chunk_ranges()` 切 sequence 维，不把完整 `[B,L,V]` 搬到 student device。

### dense_loss / trainer 未做 full restore

- `e2e_common/dense_loss.py` 的 `_compute_offloaded_non_eakld_region_loss()` 把完整 `teacher_logits_cpu` 传给上述 executor；本文件无 `.to(device=...)`、无 `copy_teacher_logit_chunk_to_device` 调用（H2D 全部在 executor 内）。
- CPU 路径 `_compute_teacher_first_cpu_loss()` 把 `targets.logits_cpu` 作为 `teacher_logits_cpu` 传入 `compute_dense_loss_from_offloaded_teacher()`（807–810 行），无 `.to(student device)`。
- `_build_cpu_teacher_targets()` 对完整 logits 只做 D2H：`copy_detached_tensor_to_cpu(teacher_logits, ...)`（529–532 行）。

### 全量搜索

在三个 production 文件中搜索 `teacher_logits_cpu.to(` / `logits_cpu.to(` / `teacher_logits_cpu.cuda` / `.to(student`：**无匹配**。

`distill_losses.py` 中 `copy_teacher_logit_chunk_to_device` 仅两处：新 token-mean executor（736）与既有 EAKLD executor（912）。

### 非 CPU-offload 路径（对照，未改）

- `_compute_legacy_dense_loss` 与 HEAD **字节级相同**（700 行仍有 `teacher_logits = get_output_logits(...).to(device=logits.device)`，这是 `teacher_output_offload=none` 既有行为）。
- `_compute_choice_kd_loss` 与 HEAD **字节级相同**（639 行仍有 teacher logits `.to(device=student_logits.device)`，choice 路径，非 dense CPU offload）。

**结论：** CPU offload path 未新增完整 `teacher_logits_cpu` → student device restore；唯一允许的 H2D 仍是 `copy_teacher_logit_chunk_to_device()`。

---

## Step 10: Audit — EAKLD metadata gating

`_compute_teacher_first_cpu_loss()` 计算：

```734:737:compressed_e2e_fintuning/trainer.py
        eakld_metadata_required = (
            loss_type in {"eakld", "eakld_kd"}
            or is_eakld_top_loss(loss_type)
        )
```

`is_eakld_top_loss("kl_top_1000")` 实测为 `False`（只匹配 `eakld_top*` / `eakld_topk*`）。因此 `kl_top_1000` + 任意 `prompt_kd_weight` 时 `eakld_metadata_required=False`。

`_build_cpu_teacher_targets()` 中 `compute_teacher_entropy_mean_and_gamma()` 仅在该 flag 为 True 时调用：

1. 533 行 `if eakld_metadata_required:` 内，response 区域第一次（535–539 行）。
2. 同分支内 `if self.prompt_kd_weight > 0.0:` 的 prompt 区域第二次（554–558 行）。

`logits_required` 分支在 flag 为 False 时仍会 `copy_detached_tensor_to_cpu` 完整 logits，但 **不** 建 regions、**不** 调 entropy/gamma。

下游 gating 一致：

- 787 行：gamma/entropy 缺失检查包在 `if eakld_metadata_required and (...)`。
- 797 行：prompt 标量检查包在 `if eakld_metadata_required and self.prompt_kd_weight > 0.0 and (...)`。
- 822 行：`telemetry_out=telemetry if eakld_metadata_required else None`。
- 833–834 行：`_record_eakld_telemetry` 仅在 flag 为 True 时执行。

测试证据：`test_cpu_kl_top_1000_skips_eakld_metadata_and_copies_logits_once` 在 `prompt_kd_weight=0.03` 下断言 `compute_teacher_entropy_mean_and_gamma` **0 次**。该测试随 Step 4 套件通过。

**结论：** `_build_cpu_teacher_targets()` 中 entropy/gamma 只在 `eakld_metadata_required=True` 时运行；普通 `kl_top_1000` + prompt KD 为 0 次调用。

---

## Step 11: Audit — choice KD untouched

`git diff HEAD -- compressed_e2e_fintuning/trainer.py` 中出现 `choice` 的唯一 hunk 是删除 `_is_cpu_offload_supported_loss()`，上下文落在 `compute_choice_kd_loss_from_scores()` **之后**（diff 第 5 行 `@@ -286,11 +286,6 @@`）。该函数本体与 HEAD **字节级相同**（1723 bytes）。

函数级对比 vs HEAD：

| 函数 | 与 HEAD 相同 |
|---|---|
| `compute_choice_kd_loss_from_scores` | 是 |
| `_compute_choice_kd_loss` | 是（2445 bytes） |
| `compute_loss` | 是（1200 bytes） |
| `_compute_legacy_dense_loss` | 是 |

`compute_loss` 调度顺序未变（881–886 行，与 HEAD 相同）：

1. `"choice_input_ids" in inputs` → `_compute_choice_kd_loss`（先于 dense offload）
2. `teacher_output_offload == "none"` → `_compute_legacy_dense_loss`
3. `teacher_output_offload == "cpu"` → `_compute_teacher_first_cpu_loss`

**结论：** `_compute_choice_kd_loss()` 无改动；choice dispatch 仍在 teacher_output_offload dense dispatch 之前。

---

## Step 12: Final combined focused suite

```bash
pytest -q \
  tests/test_distill_losses.py \
  tests/test_teacher_target_offload.py \
  tests/test_e2e_teacher_first.py \
  tests/test_distill_dynamic_padding.py \
  tests/test_e2e_distill_token_stats.py \
  tests/smoke/test_loss_pipeline_smoke.py \
  tests/smoke/test_one_step_train_smoke.py
```

**结果：** 188 passed in 14.55s（exit 0）

与分步合计一致：93 + 18 + 44 + 14 + 11 + 3 + 5 = 188。0 failed / 0 skipped。

---

## Final Acceptance Checklist（自检）

| 项 | 证据 |
|---|---|
| `cpu` + `kl_top_1000` 进入 teacher-first，不再报 unsupported | Step 8 通过；unsupported 字符串已删除 |
| `kl_top_1000` teacher forward 在 student 之前 | `test_cpu_kl_top_1000_teacher_before_student_and_backward` |
| student forward 前 GPU teacher outputs 已释放，保留 CPU target | 同上 + `_active_teacher_targets` 生命周期测试 |
| CPU teacher logits 完整 `[B,L,V]`、原 dtype | `_build_cpu_teacher_targets` 单次 `copy_detached_tensor_to_cpu`；测试断言 `logits_cpu` 在 CPU |
| loss 只按 sequence chunk H2D | Step 9；executor 只用 `copy_teacher_logit_chunk_to_device` |
| backward checkpoint 重算时再读对应 CPU chunk | `torch_checkpoint.checkpoint(..., use_reentrant=False)` 包住 chunk_forward |
| 无一次性 full CPU logits → GPU restore | Step 9 |
| 全部 dense distillation loss/gradient parity | `CPU_OFFLOAD_DENSE_DISTILL_LOSS_TYPES` 26 种，见下 |
| `kd/kd_top*/dual_kd/dual_kd_top*` CE 在 chunk 外混合一次 | `dense_loss.py` `_is_ce_blended_dense_loss` + 区域 loss 后再 `ce*(1-alpha)+region*alpha` |
| response/prompt = `L_response + prompt_kd_weight * L_prompt` | `_combine_region_loss` 未改语义 |
| EAKLD 仍用 region-global gamma | EAKLD 分支仍把预计算 `teacher_gamma_cpu` 传入 chunked EAKLD |
| EAKLD response telemetry 保留；prompt 不覆盖 | prompt 调用 `telemetry_out=None` |
| 非 EAKLD 不计算 entropy/gamma | Step 10 |
| 非 EAKLD + positive prompt weight 不要求 EAKLD metadata | 797 行 gated；dense_loss 非 EAKLD 不再校验 prompt gamma |
| 非 EAKLD + hidden alignment teacher-first/backward | `test_cpu_kl_top_1000_adaptive_top_2_hidden_collectors` |
| `sft/origin` + hidden=0 不跑 teacher | 既有 teacher-first 测试仍通过 |
| `sft/origin` + hidden>0 只缓存 hidden | 既有测试仍通过 |
| `teacher_output_offload=none` legacy 未改 | `_compute_legacy_dense_loss` 与 HEAD 相同 |
| `teacher_output_chunk_tokens` 对所有 CPU-offloaded dense loss 生效 | trainer 传入 `sequence_chunk_size=int(self.teacher_output_chunk_tokens)` |
| dynamic padding / causal shift / mask 未改 | Step 5 14 passed |
| `_active_teacher_targets` 在 checkpoint backward 期间有效，step 后释放 | training_step cleanup 测试 |
| `choice_kd` / `choice_kd_ce` 未修改 | Step 11 |
| teacher residency / ckpt / LoRA/VAE / optimizer / 保存逻辑未改 | 仅预期 5 文件有 diff |
| 最终 focused suite 全部通过 | 188 passed |

---

## Required 7-item completion summary

1. **实际修改文件（本任务）：** 无。工作区相对 HEAD 仍是 Tasks 1–5 的 5 个预期文件。未创建临时测试脚本。
2. **通用 CPU sequence-chunk executor 最终接口：**
   `compute_chunked_token_mean_from_cpu_teacher_logits(*, student_logits, teacher_logits_cpu, mask, sequence_chunk_size, chunk_loss_fn) -> Tensor`  
   内部用 `_make_checkpointed_token_mean_chunk_forward` + `copy_teacher_logit_chunk_to_device`。`dense_loss._compute_offloaded_non_eakld_region_loss()` 以 per-chunk `compute_dense_loss_from_logits` 作为 `chunk_loss_fn`。
3. **`kl_top_1000` teacher-first CPU 数据流：** 符合计划。不再被 supported-loss 白名单拒绝；teacher 先、student 后；完整 logits 一次 D2H 到 CPU；loss 按 chunk H2D；无 full restore。
4. **非 EAKLD 不再计算 entropy/gamma：** 是。仅 `eakld_metadata_required=True` 时调用 `compute_teacher_entropy_mean_and_gamma()`；`kl_top_1000` + prompt KD 为 0 次。
5. **loss/gradient parity 覆盖的 loss_type（26）：**  
   `kl, rkl, dual_rkl, mse, kd, kd_top, kd_top_7, dual_kd_top, dual_kd_top_7, dual_kl, dual_kd, eakld, eakld_kd, eakld_top, eakld_top_7, eakld_topk, eakld_topk_7, r_kl_top, r_kl_top_7, dual_r_kl_top, dual_r_kl_top_7, kl_top, kl_top_7, kl_top_1000, dual_kl_top, dual_kl_top_7`
6. **pytest：** 见上 Steps 2–8、12。分步全部 pass；合并 188 passed。
7. **未通过测试 / 偏离计划：** 无。先前 ledger 中 Task 2 的 empty-chunk NaN guard / executor 专项测试、Task 5 的 tuple 重复，均属已记录的计划内/延期项，本任务未扩大范围。

---

## Self-review

- 每个 listed pytest 都实际跑过，记录的 pass 数来自本次输出，不是推断。
- 审计基于 `git diff HEAD` 与函数级字节对比，不是“看起来没问题”。
- 未改 production/test；未 git add/commit；未建 worktree/切分支。
- 未发现必须扩 scope 的失败。
