> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-3-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3: Generalize `compute_dense_loss_from_offloaded_teacher()`

**Files:**
- Modify: `e2e_common/dense_loss.py`
- Test: `tests/test_distill_losses.py`

**Interfaces:**
- Consumes: new generic chunk executor and existing `compute_dense_loss_from_logits()`.
- Produces: one CPU-offloaded dense dispatcher for all current dense distillation loss types.

- [ ] **Step 1: Import `compute_chunked_token_mean_from_cpu_teacher_logits()`**

禁止增加每个 loss 一个 `*_from_cpu_teacher_logits` function。

- [ ] **Step 2: Add one exact EAKLD-family classifier**

```python
def _is_eakld_family_loss(loss_type: str) -> bool:
    norm = str(loss_type or "").strip().lower()
    return norm in {"eakld", "eakld_kd"} or is_eakld_top_loss(norm)
```

- [ ] **Step 3: Add exact KD-to-region-primitive mapping**

非 EAKLD 的 CE-blended KD 在 chunk 内不能调用 `kd` 自己，因为 chunk 内不应重复 CE。固定映射：

```text
kd              -> kl
kd_top           -> kl_top
kd_top_K         -> kl_top_K
dual_kd          -> dual_kl
dual_kd_top      -> dual_kl_top
dual_kd_top_K    -> dual_kl_top_K
```

其它 non-EAKLD loss 名保持不变，例如 `kl_top_1000 -> kl_top_1000`、`rkl -> rkl`、`mse -> mse`。

建议 helper：

```python
def _offloaded_region_loss_type(loss_type: str) -> str:
    norm = str(loss_type or "").strip().lower()
    if norm == "kd":
        return "kl"
    if norm.startswith("kd_top"):
        return "kl_top" + norm[len("kd_top"):]
    if norm == "dual_kd":
        return "dual_kl"
    if norm.startswith("dual_kd_top"):
        return "dual_kl_top" + norm[len("dual_kd_top"):]
    return norm
```

- [ ] **Step 4: Add exact CE-blended loss classifier**

```python
def _is_ce_blended_dense_loss(loss_type: str) -> bool:
    norm = str(loss_type or "").strip().lower()
    return (
        norm == "kd"
        or norm.startswith("kd_top")
        or norm == "dual_kd"
        or norm.startswith("dual_kd_top")
        or norm == "eakld_kd"
    )
```

禁止把普通 `kl_top*` 判成 KD。

- [ ] **Step 5: Add one generic non-EAKLD region helper**

它必须：先得到 `region_loss_type = _offloaded_region_loss_type(loss_type)`；定义 `chunk_loss_fn`；chunk 内调用现有 `compute_dense_loss_from_logits()`，参数固定为当前 region primitive、`ce_loss=None`、当前 chunk mask、原 temperature、`prompt_mask=None`、`prompt_kd_weight=0`、`telemetry_out=None`；再交给 `compute_chunked_token_mean_from_cpu_teacher_logits()`。

核心约束：**不能在该 helper 里重新写 softmax、KL、MSE、dual scaling 或 top-k 公式。** CPU/legacy 两条路径必须共享现有 dense loss 数学实现。

- [ ] **Step 6: Make EAKLD metadata validation conditional**

`compute_dense_loss_from_offloaded_teacher()` 开始先 normalize loss name、validate prompt weight、判断 `is_eakld`。

只有 EAKLD-family：

- 要求 response gamma 非 None。
- telemetry 开启时要求 response entropy/count。
- prompt weight > 0 时要求 prompt gamma/entropy/count。
- 校验 `eakld_confidence_k >= 2`。

非 EAKLD：即使 prompt weight > 0，也不能要求任何 EAKLD metadata。

- [ ] **Step 7: Preserve existing EAKLD specialized math**

`eakld/eakld_kd/eakld_top*/eakld_topk*` 继续调用现有 CPU EAKLD helpers；response telemetry 仍写入，prompt telemetry 不写；`eakld_kd` 的 alpha 混合不变。

- [ ] **Step 8: Implement generic non-EAKLD response/prompt aggregation**

固定逻辑：response 调 generic region helper；prompt 仍通过 `_combine_region_loss()`，仅 `prompt_kd_weight>0` 时执行第二次 region helper；之后如果 `_is_ce_blended_dense_loss(norm)`，要求 `ce_loss` 存在，并按当前 `(1-alpha)*CE + alpha*distill` 返回；普通 distillation loss直接返回 region loss。

- [ ] **Step 9: Remove EAKLD-only error text from this function**

不再保留 `teacher_output_offload=cpu supports only EAKLD-family losses.`。未知 loss 最终由现有 `compute_dense_loss_from_logits()` 的统一 `Unsupported dense loss type` 报错。

- [ ] **Step 10: Run the complete low-level parity suite**

```bash
pytest -q tests/test_distill_losses.py -k "offloaded_teacher or cpu_offload_dense"
```

Expected: Task 1 全部 loss + gradient parity 通过，包括 `kl_top_1000` with all EAKLD metadata None。

---

