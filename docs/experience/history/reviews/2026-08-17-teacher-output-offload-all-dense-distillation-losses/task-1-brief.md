> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-1-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1: Write Failing Low-Level Parity Tests First

**Files:**
- Modify: `tests/test_distill_losses.py`

**Interfaces:**
- Consumes: existing `compute_dense_loss_from_logits()` and current EAKLD-only `compute_dense_loss_from_offloaded_teacher()`.
- Produces: complete mathematical contract before production code is changed.

- [ ] **Step 1: Add the exact CPU-offload loss matrix**

固定测试 tuple，不允许只挑 family representative：

```python
CPU_OFFLOAD_DENSE_DISTILL_LOSS_TYPES = (
    "kl",
    "rkl",
    "dual_rkl",
    "mse",
    "kd",
    "kd_top",
    "kd_top_7",
    "dual_kd_top",
    "dual_kd_top_7",
    "dual_kl",
    "dual_kd",
    "eakld",
    "eakld_kd",
    "eakld_top",
    "eakld_top_7",
    "eakld_topk",
    "eakld_topk_7",
    "r_kl_top",
    "r_kl_top_7",
    "dual_r_kl_top",
    "dual_r_kl_top_7",
    "kl_top",
    "kl_top_7",
    "kl_top_1000",
    "dual_kl_top",
    "dual_kl_top_7",
)
```

`sft/origin` 不属于“teacher logits distillation math parity”测试；它们由 trainer integration tests 保证。`choice_kd*` 不加入。

- [ ] **Step 2: Build deterministic parity fixtures**

固定 `B=2, L=6, V=17`、float32 logits、`sequence_chunk_size=2`、`temperature=1.3`、`alpha=0.4`、`prompt_kd_weight=0.03`、`eakld_confidence_k=16`。response/prompt mask 都必须非空，并让至少一个 chunk 对其中一个 region 的 mask 全零，以验证空 region chunk 不产生 NaN。

对 CE-blended loss，legacy/offload 两边必须各自从对应 student tensor 生成相同可微 scalar CE surrogate，例如 student logits 的平方均值；不能共享同一个 CE tensor，否则 gradient parity 不完整。

- [ ] **Step 3: Build EAKLD metadata conditionally**

EAKLD-family 判断固定为：`loss_type in {"eakld", "eakld_kd"} or is_eakld_top_loss(loss_type)`。只有这些 case 才调用 `compute_teacher_entropy_mean_and_gamma()` 生成 response/prompt metadata；其它 case 给 offloaded helper 的所有 EAKLD metadata 传 `None`。

- [ ] **Step 4: Add parameterized loss and student-gradient parity test**

每个 loss 都执行：同一个 base student logits → 两个独立 `requires_grad=True` clone；同一个 detached teacher logits；同一 masks/hparams。legacy 调 `compute_dense_loss_from_logits()`；offload 调 `compute_dense_loss_from_offloaded_teacher()`，并把 `sequence_chunk_size` 固定为 2。

断言：

```python
torch.testing.assert_close(offload_loss, legacy_loss, atol=2e-6, rtol=2e-5)
```

分别 backward 后：

```python
torch.testing.assert_close(offload_student.grad, legacy_student.grad, atol=3e-6, rtol=3e-5)
```

若实际仅因 chunk sum 浮点顺序略超出，可依据观测误差小幅放宽，但禁止放宽到 `1e-3` 数量级。

- [ ] **Step 5: Replace obsolete non-EAKLD rejection test**

删除当前 `test_offloaded_teacher_dense_loss_rejects_non_eakld()`。替换为 `test_offloaded_teacher_dense_loss_supports_kl_top_1000_without_eakld_metadata()`：传 `loss_type="kl_top_1000"`，并且**完全省略**所有 EAKLD-only metadata keyword，断言 loss finite、backward 成功、student grad finite。该测试用于证明 non-EAKLD 调用接口已经真正与 EAKLD 参数解耦。

- [ ] **Step 6: Verify tests fail before implementation for the intended reason**

```bash
which python
python -V
pytest -q tests/test_distill_losses.py -k "offloaded_teacher or cpu_offload_dense"
```

预期：现有 EAKLD cases 继续通过；non-EAKLD 因当前 EAKLD-only dispatcher/gamma requirement 失败。

---

