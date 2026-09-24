> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-6-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6: Closed-Loop Regression Verification

**Files:**
- No additional production changes expected.

**Interfaces:**
- Consumes: completed implementation.
- Produces: final acceptance evidence.

- [ ] **Step 1: Verify environment before tests**

```bash
which python
python -V
```

必须确认是 `bitvae`；若不是，先在当前 shell 激活 `bitvae` 再执行测试。

- [ ] **Step 2: Run low-level dense loss suite**

```bash
pytest -q tests/test_distill_losses.py
```

- [ ] **Step 3: Run teacher target utility suite**

```bash
pytest -q tests/test_teacher_target_offload.py
```

- [ ] **Step 4: Run teacher-first trainer suite**

```bash
pytest -q tests/test_e2e_teacher_first.py
```

- [ ] **Step 5: Run dynamic-padding regression**

```bash
pytest -q tests/test_distill_dynamic_padding.py
```

必须确认本任务没有改变之前 batch-longest + multiple-of-8 下的 mask/loss invariant。

- [ ] **Step 6: Run token telemetry regression**

```bash
pytest -q tests/test_e2e_distill_token_stats.py
```

- [ ] **Step 7: Run smoke suites**

```bash
pytest -q tests/smoke/test_loss_pipeline_smoke.py
pytest -q tests/smoke/test_one_step_train_smoke.py
```

- [ ] **Step 8: Run exact bug regression alone**

```bash
pytest -q tests/test_e2e_teacher_first.py::test_cpu_kl_top_1000_teacher_before_student_and_backward
```

必须通过，且不再出现“CPU offload only supports EAKLD-family”类错误。

- [ ] **Step 9: Audit for forbidden full-logit restore**

检查本次 production diff。CPU offload path 不得出现“把整个 `teacher_logits_cpu` 一次性搬到 student device”的新增代码；唯一允许的 H2D 路径是现有 per-sequence-chunk helper。

- [ ] **Step 10: Audit EAKLD metadata gating**

确认 `_build_cpu_teacher_targets()` 中 `compute_teacher_entropy_mean_and_gamma()` 只在 `eakld_metadata_required=True` 分支执行。普通 `kl_top_1000` + prompt KD 也必须是 0 次调用。

- [ ] **Step 11: Audit choice KD untouched**

检查 `git diff -- compressed_e2e_fintuning/trainer.py`；确认 `_compute_choice_kd_loss()` 无改动，choice dispatch 位置不变。

- [ ] **Step 12: Run final combined focused suite**

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

Expected: all pass。

---

## Final Acceptance Checklist

Cursor 完成后必须逐项自检；任何一项失败都不能宣称完成：

- [ ] `--teacher_output_offload cpu --loss_type kl_top_1000` 正常进入 teacher-first CPU path，不再主动报 unsupported error。
- [ ] `kl_top_1000` teacher forward 在 student forward 之前。
- [ ] student forward 开始前，完整 teacher GPU outputs/references 已释放；保留的是 CPU target。
- [ ] CPU teacher logits 仍是完整 `[B,L,V]` 和 teacher 原 dtype。
- [ ] loss 阶段只按 sequence chunk 将完整-vocab teacher logits搬到 student device。
- [ ] backward checkpoint 重算时重新读取对应 CPU chunk。
- [ ] 没有一次性 full CPU teacher logits → GPU restore。
- [ ] 所有当前 dense distillation loss 通过 CPU-offload vs legacy loss parity。
- [ ] 所有当前 dense distillation loss 通过 student-gradient parity。
- [ ] `kd/kd_top*/dual_kd/dual_kd_top*` 的 CE 只在 chunk 外混合一次，alpha 语义不变。
- [ ] response/prompt 仍严格是 `L_response + prompt_kd_weight * L_prompt`。
- [ ] EAKLD-family 仍使用 region-global gamma，未改成 per-chunk gamma。
- [ ] EAKLD response telemetry 保持；prompt branch 不覆盖 response telemetry。
- [ ] 非 EAKLD 不计算 response/prompt EAKLD metadata。
- [ ] 非 EAKLD + positive prompt weight 不要求 EAKLD metadata。
- [ ] 非 EAKLD + hidden alignment 正常 teacher-first、backward。
- [ ] `sft/origin + hidden_loss_weight=0` 仍不跑 teacher。
- [ ] `sft/origin + hidden_loss_weight>0` 仍只缓存 hidden targets。
- [ ] `teacher_output_offload=none` legacy path 未改变。
- [ ] `teacher_output_chunk_tokens` 对所有 CPU-offloaded dense loss 生效。
- [ ] dynamic padding / causal shift / prompt-response mask 未改变。
- [ ] `_active_teacher_targets` 在 checkpoint backward 所需期间保持有效，并在 training step 后释放。
- [ ] `choice_kd` / `choice_kd_ce` 未修改。
- [ ] teacher weight residency、checkpoint 格式、LoRA/VAE 参数、optimizer、保存逻辑未修改。
- [ ] 最终 focused test suite 全部通过。

## Expected Changed Files

正常完成后只应出现以下代码/测试改动：

```text
train_utils/distill_losses.py
e2e_common/dense_loss.py
compressed_e2e_fintuning/trainer.py
tests/test_distill_losses.py
tests/test_e2e_teacher_first.py
```

以及本计划文件。若 Cursor 认为必须修改其它 production 文件，必须先证明是哪一个现有测试因本任务接口变化失败，以及为何不能在上述 production 文件内正确解决；禁止为了方便扩散范围。

## Required Completion Report from Cursor

完成后汇报以下 7 项即可：

1. 实际修改文件。
2. 通用 CPU sequence-chunk executor 最终接口。
3. `kl_top_1000` 的 teacher-first CPU 数据流是否符合本计划。
4. 非 EAKLD 是否确认不再计算 entropy/gamma metadata。
5. loss parity / gradient parity 实际覆盖的 loss_type。
6. 执行过的 pytest 命令和结果。
7. 是否存在未通过测试或任何偏离本计划的地方。

不要创建 commit；保留工作区 diff 供人工 review。
