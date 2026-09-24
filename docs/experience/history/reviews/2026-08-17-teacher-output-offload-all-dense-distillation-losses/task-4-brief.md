> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-4-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4: Make Trainer Teacher Targets Loss-Agnostic

**Files:**
- Modify: `compressed_e2e_fintuning/trainer.py`
- Test: `tests/test_e2e_teacher_first.py`

**Interfaces:**
- Consumes: generalized offloaded dense dispatcher.
- Produces: teacher-first CPU path that caches exactly the data required by each dense loss.

- [ ] **Step 1: Delete `_is_cpu_offload_supported_loss()` and its early gate**

不要改成“更大的 whitelist”；直接删除第二套 CPU loss whitelist。CPU dense path的能力由统一 dense dispatcher 决定。

- [ ] **Step 2: Compute exact requirements in `_compute_teacher_first_cpu_loss()`**

固定：

```python
loss_type = self.loss_type
hidden_required = float(self.hidden_loss_weight) > 0.0
logits_required = loss_type not in {"sft", "origin"}
eakld_metadata_required = (
    loss_type in {"eakld", "eakld_kd"}
    or is_eakld_top_loss(loss_type)
)
needs_teacher = hidden_required or logits_required
```

choice path 不进入这里，保持 `compute_loss()` 顶部当前提前 dispatch。

- [ ] **Step 3: Extend `_build_cpu_teacher_targets()` call/signature**

显式传 `eakld_metadata_required=eakld_metadata_required`。

- [ ] **Step 4: Separate full-logit cache from EAKLD metadata**

`logits_required=True` 时所有 dense distillation loss都执行：获取 teacher logits → `copy_detached_tensor_to_cpu()` 一次。

只有 `eakld_metadata_required=True` 才：构建 teacher-side response/prompt regions用于 entropy metadata；调用 `compute_teacher_entropy_mean_and_gamma()`；填 response gamma/entropy/count；prompt weight > 0 时填 prompt metadata。

非 EAKLD 的 `TeacherTargetBatch` 必须满足：`logits_cpu` 存在；全部 EAKLD response/prompt metadata fields 为 None。

- [ ] **Step 5: Preserve exactly one full teacher-logits CPU copy**

对 EAKLD 和 non-EAKLD 都不得复制第二份完整 teacher logits。EAKLD metadata 直接从已经生成的 CPU logits cache 计算。Hidden hook 自己复制 hidden tensor不计入 logits copy次数。

- [ ] **Step 6: Generalize post-student validation**

所有 non-SFT dense distillation 统一先只检查 `targets` 和 `targets.logits_cpu` 存在，错误文本可固定为：`Dense distillation with teacher_output_offload=cpu requires teacher logits on CPU.`

仅当 `eakld_metadata_required=True` 时，再检查 response gamma/entropy/count；仅当同时 `prompt_kd_weight>0` 时再检查 prompt EAKLD metadata。

- [ ] **Step 7: Call generalized offloaded loss for all non-SFT dense losses**

保留现有 student-side `regions = self._build_distill_token_regions(inputs, logits)`，将 target 中 optional EAKLD fields原样传给 generalized `compute_dense_loss_from_offloaded_teacher()`；`sequence_chunk_size` 固定继续传 `teacher_output_chunk_tokens`。

- [ ] **Step 8: Record EAKLD telemetry only for EAKLD-family**

只有 `eakld_metadata_required=True` 才调用 `_record_eakld_telemetry(telemetry)`。普通 KL/RKL/MSE/dual/top-k 不生成伪 EAKLD telemetry。

- [ ] **Step 9: Preserve hidden alignment and CPU target lifetime**

不得改变 teacher/student hidden collectors、selected hidden alignment、`_active_teacher_targets` 和 `training_step` finally cleanup。只要 loss requires grad，non-EAKLD CPU teacher logits也必须被 `_active_teacher_targets` 持有到 backward 完成，因为 checkpoint backward 会再次读取 CPU cache。

- [ ] **Step 10: Keep choice branch byte-for-byte behaviorally unchanged**

`compute_loss()` 中检测到 `choice_input_ids` 后，仍必须直接调用 `_compute_choice_kd_loss(model, inputs, return_outputs=bool(return_outputs))`，并且该 dispatch 继续发生在 teacher_output_offload dense dispatch 之前；不要修改 `_compute_choice_kd_loss()`。

---

