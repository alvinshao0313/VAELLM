> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-4-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4 Report: Make Trainer Teacher Targets Loss-Agnostic

**Status:** DONE_WITH_CONCERNS  
**Commits created:** none（禁止提交）

## What I implemented

只改了 `compressed_e2e_fintuning/trainer.py`。未改 `offload.py`、checkpoint、`build_distill_token_mask` / `build_distill_token_regions` 实现、`_compute_choice_kd_loss()`，也未新增 Task 5 测试。

`tests/test_e2e_teacher_first.py` 没有直接调用 `_build_cpu_teacher_targets()` 的站点，因此没有改测试文件。所有现有调用都走 `compute_loss()` → `_compute_teacher_first_cpu_loss()`，由 trainer 显式传入新关键字。

### 1. 删除 `_is_cpu_offload_supported_loss()` 及其 early gate

函数和 `_compute_teacher_first_cpu_loss()` 里的白名单 `ValueError` 都已删除。没有扩大 whitelist。CPU dense path 的能力交给统一 `compute_dense_loss_from_offloaded_teacher()`。

### 2. `_compute_teacher_first_cpu_loss()` 需求标志与 brief 一致

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

`sft/origin` + `hidden_loss_weight=0` 仍跳过 teacher；`hidden_loss_weight>0` 仍只缓存 hidden，不缓存 logits。

### 3–5. `_build_cpu_teacher_targets()` 拆开 logits cache 与 EAKLD metadata

签名新增必填关键字 `eakld_metadata_required: bool`。唯一生产调用点显式传入，不从 `logits_required=True` 推断。

- `logits_required=True`：取 teacher logits，调用一次 `copy_detached_tensor_to_cpu()`。
- 仅当 `eakld_metadata_required=True`：用 teacher logits 建 response/prompt regions，再从**同一份** CPU logits cache 调 `compute_teacher_entropy_mean_and_gamma()`；`prompt_kd_weight>0` 时再填 prompt metadata。
- 非 EAKLD：`logits_cpu` 有值，全部 EAKLD response/prompt 字段保持 `None`。
- 没有第二次完整 teacher logits copy。Hidden hook 自己的 hidden copy 不计入 logits copy 次数。

### 6–8. post-student 校验、dispatcher、telemetry

所有 non-SFT dense distillation 先只检查 `targets` 和 `targets.logits_cpu`，错误文本固定为：

`Dense distillation with teacher_output_offload=cpu requires teacher logits on CPU.`

仅当 `eakld_metadata_required=True` 再检查 response gamma/entropy/count；仅当同时 `prompt_kd_weight>0` 再检查 prompt EAKLD metadata。

student-side 仍 `regions = self._build_distill_token_regions(inputs, logits)`，optional EAKLD fields 原样传给 `compute_dense_loss_from_offloaded_teacher()`，`sequence_chunk_size=int(self.teacher_output_chunk_tokens)`。

只有 `eakld_metadata_required=True` 才传 `telemetry_out` 并调用 `_record_eakld_telemetry()`。普通 KL/RKL/MSE/dual/top-k 不生成伪 EAKLD telemetry。

### 9–10. 保持 hidden lifetime 与 choice dispatch

teacher/student hidden collectors、selected hidden alignment、`_active_teacher_targets`、`training_step` finally cleanup 行为未改。`compute_loss()` 仍在 teacher_output_offload dense dispatch 之前检测 `choice_input_ids` 并调用 `_compute_choice_kd_loss()`。

## Tests / results

环境：`/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`，Python 3.11.13。`PYTHONPATH=.`。

```bash
pytest -q tests/test_e2e_teacher_first.py
# 1 failed, 13 passed in 7.68s
```

失败用例：`test_unsupported_kl_raises_before_student_forward`。原因是 `loss_type="kl"` 现在可以走 CPU teacher-first 路径，不再抛白名单 `ValueError`（`DID NOT RAISE`）。按任务说明保留该测试，交给 Task 5 替换。其余 13 个现有测试通过，包括 EAKLD teacher-first、sft skip/hidden-only、entropy helper 1/2 次、单次 logits CPU copy、telemetry accumulator、training_step cleanup。

未跑 Task 5 新测试（禁止本任务实现）。未提交。

## Self-review

- 需求标志、错误文本、`eakld_metadata_required` 关键字与 brief 一致。
- 非 EAKLD 不再计算 entropy/gamma；EAKLD 仍从同一份 CPU logits 计算 metadata。
- choice 路径、hidden collectors、target lifetime 未改。
- 过时的 unsupported-KL 测试失败是预期的，不是实现缺陷。

## Concerns

- `tests/test_e2e_teacher_first.py::test_unsupported_kl_raises_before_student_forward` 会失败，直到 Task 5 删除/替换它。这是任务说明中的预期结果。
