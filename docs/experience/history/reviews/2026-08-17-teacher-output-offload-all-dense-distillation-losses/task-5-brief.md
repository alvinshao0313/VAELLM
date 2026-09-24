> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-5-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5: Add Trainer-Level Coverage for Every Dense Loss

**Files:**
- Modify: `tests/test_e2e_teacher_first.py`

**Interfaces:**
- Consumes: generalized trainer CPU path.
- Produces: integration proof that every dense loss can traverse teacher-first CPU offload and backward.

- [ ] **Step 1: Replace the obsolete unsupported-KL test**

删除当前 `test_unsupported_kl_raises_before_student_forward()`。新增测试函数 `test_cpu_kl_top_1000_teacher_before_student_and_backward(tmp_path)`，函数体按下面的固定断言完整实现。

固定断言：

- `events == ["teacher_forward", "student_forward"]`。
- `_active_teacher_targets` 存在直到 backward/release。
- `logits_cpu` 存在且 device.type == `cpu`。
- response/prompt EAKLD metadata 全 None。
- loss finite。
- backward 成功。
- student 至少一个 trainable parameter grad 非 None 且 finite。
- finally 调 `_release_active_teacher_targets()` 后为 None。

- [ ] **Step 2: Add all-dense-loss parameterized trainer test**

在本文件复制 Task 1 的 exact tuple，避免 test module 互相 import。新增参数化测试 `test_cpu_offload_all_dense_distillation_losses_backward(tmp_path, loss_type)`，使用 `@pytest.mark.parametrize("loss_type", CPU_OFFLOAD_DENSE_DISTILL_LOSS_TYPES)`；函数体按下面固定 trainer 参数和断言完整实现。

固定 trainer 参数：`hidden_loss_weight=0.0`、`prompt_kd_weight=0.03`、`teacher_output_offload="cpu"`、`teacher_output_chunk_tokens=2`。每个 case 必须 teacher first、student second、loss finite、backward 成功、student grad finite。每个 case finally release active targets。

- [ ] **Step 3: Add non-EAKLD metadata skip regression**

使用 `loss_type="kl_top_1000"`, `prompt_kd_weight=0.03`。同时 monkeypatch `compressed_e2e_fintuning.trainer.compute_teacher_entropy_mean_and_gamma` 和 `copy_detached_tensor_to_cpu` 为 counting wrapper。断言 entropy/gamma helper 调用次数为 0；在 `hidden_loss_weight=0` 条件下完整 teacher logits CPU copy 次数恰好为 1；`logits_cpu` 存在；所有 EAKLD response/prompt metadata 为 None。

该测试是必要验收：不能只“允许普通 KL 通过”，却仍做无用的 full-vocab entropy/gamma 计算，也不能重复复制完整 teacher logits。

- [ ] **Step 4: Preserve existing EAKLD metadata-count semantics**

现有 zero prompt weight → entropy/gamma helper exactly once；positive prompt weight → exactly twice（response + prompt）的测试必须继续通过。若 `_build_cpu_teacher_targets()` 新增参数，只更新调用方式，不改 assertion。

- [ ] **Step 5: Add non-EAKLD + hidden alignment integration test**

用 `kl_top_1000`、`hidden_loss_weight=0.1`、`hidden_layer_weighting="adaptive_top_2"`、`prompt_kd_weight=0.03`。断言 teacher/student 都以 `output_hidden_states=False` forward；selected hidden count == 2；teacher CPU logits 存在；EAKLD metadata None；总 loss backward 成功。

- [ ] **Step 6: Add training-step lifecycle regression for `kl_top_1000`**

直接走 `trainer.training_step(student, inputs)`，断言 step loss finite，并且返回后 `_active_teacher_targets is None`。防止 non-EAKLD checkpoint backward 后 CPU cache 残留到下一 microbatch。

- [ ] **Step 7: Add Trainer-level legacy-vs-CPU parity for the exact bug case**

让测试 helper 支持显式传 `teacher_output_offload="none" | "cpu"`，默认仍可保持 `cpu`，不要改变其它现有 test semantics。用相同 random seed 分别构造两套 teacher/student，构造后逐参数确认对应初始参数相同；输入、`loss_type="kl_top_1000"`、`prompt_kd_weight=0.03`、temperature、alpha 都完全相同，hidden loss 固定为 0。

分别调用两个 trainer 的 `compute_loss()`，比较 scalar loss；分别 backward 后比较 `student.lm_head.weight.grad`。容差使用与低层 parity 相同数量级（loss `atol=2e-6, rtol=2e-5`；grad `atol=3e-6, rtol=3e-5`）。finally 清理 CPU trainer active targets。

这项测试必须验证的是完整 Trainer 路径，而不是再次直接调用两个 dense-loss helper。

- [ ] **Step 8: Re-run the complete trainer test file**

```bash
pytest -q tests/test_e2e_teacher_first.py
```

Expected: all pass。

---

