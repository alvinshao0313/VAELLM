> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-2-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2: Implement the Generic Sequence-Chunk Executor

**Files:**
- Modify: `train_utils/distill_losses.py`
- Test: `tests/test_distill_losses.py`

**Interfaces:**
- Consumes: `iter_token_chunk_ranges()`、`copy_teacher_logit_chunk_to_device()`、`_default_token_mask()`、`torch_checkpoint.checkpoint()`。
- Produces: `compute_chunked_token_mean_from_cpu_teacher_logits()`。

- [ ] **Step 1: Import `Callable` only; add no dependency**

- [ ] **Step 2: Split generic validation**

将当前 EAKLD validator 的 student/teacher ndim、CPU device、shape、chunk-size 检查移动到 `_validate_cpu_teacher_logits_inputs()`。EAKLD implementation 调 generic validator 后再保留 gamma scalar 检查。不要删除现有错误检查。

- [ ] **Step 3: Implement fixed-binding chunk-forward factory**

严格按前述接口固定 `start/end/mask_chunk/valid_count`，避免 checkpoint backward 时 closure 读到 loop 最终边界。factory 内通过 `copy_teacher_logit_chunk_to_device()` 获取 teacher chunk，执行 `chunk_loss_fn`，确认返回 scalar，再乘 valid count。

- [ ] **Step 4: Implement global token-weighted executor**

固定顺序：generic validate → `_default_token_mask` → global denominator → `iter_token_chunk_ranges` → factory → checkpoint/direct call → stack/sum numerator → divide global denominator。

禁止：平均 chunk mean；沿 vocab chunk；保存 teacher GPU full logits；对 student chunk detach。

- [ ] **Step 5: Keep EAKLD specialized executor structurally separate**

现有 `_compute_eakld_from_cpu_teacher_logits_impl()` 只改为复用 generic validator。不要为了“统一”把它改成新 token-mean executor，因为 EAKLD 需要同时累计 forward/reverse numerator 并用 region-global gamma 合成。

- [ ] **Step 6: Run existing EAKLD utility regression**

```bash
pytest -q tests/test_teacher_target_offload.py
pytest -q tests/test_distill_losses.py -k "eakld and offloaded"
```

Expected: existing EAKLD behavior remains green。

---

