> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-5-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 5: Compute Student Top-k KL by Direct K-Way Gather on Device

**Files:**
- Modify: `mix_bit/kl_metric.py`
- Modify: `mix_bit/cost_search.py`
- Modify: `mix_bit/tests/test_kl_metric.py`
- Modify: `mix_bit/tests/test_cost_table.py`
- Modify: `mix_bit/tests/test_tiny_integration.py`

**Preserved public interface:**

```python
def per_sample_teacher_topk_forward_kl(
    *,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    token_offsets: torch.Tensor,
    shifted_student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one teacher-top-k forward-KL value per sample on the student device."""
```

### Required direct gather algorithm

假设：

```text
student logits: [B,T,V]
teacher indices/probs: [N_valid,K]
valid mask: [B,T]
offsets: [B+1]
```

必须执行：

```python
device = shifted_student_logits.device
mask = valid_mask.to(device=device, dtype=torch.bool)
indices_flat = teacher_topk_indices.to(device=device, dtype=torch.long)
probs_flat = teacher_topk_probs.to(device=device, dtype=torch.float32)

B, T, _ = shifted_student_logits.shape
K = indices_flat.shape[1]
padded_indices = torch.zeros((B, T, K), dtype=torch.long, device=device)
padded_indices[mask] = indices_flat
selected_all = shifted_student_logits.gather(-1, padded_indices)
selected_valid = selected_all[mask].float()
student_log_prob = torch.log_softmax(selected_valid, dim=-1)

row_mass = probs_flat.sum(dim=-1, keepdim=True)
probs_flat = probs_flat / row_mass
teacher_log_prob = probs_flat.log()
token_kl = (probs_flat * (teacher_log_prob - student_log_prob)).sum(-1)
```

随后按 offsets 逐样本平均，输出 `[B]`，输出保留在 student logits device。调用方可只把 `[B]` 搬到 CPU。

保留现有全部输入验证：student logits/mask shape、indices/probs shape、offsets 长度和首尾值、每样本 offsets 与 mask token 数一致、indices 在 `[0,V)`、每行 teacher mass 为正、每个样本至少一个 valid token。验证只能操作小型 mask/offsets/indices/probs，不得为了验证构造 `[N_valid,V]`。

### Forbidden operations

生产 top-k 路径禁止：

```python
shifted_student_logits.cpu()
shifted_student_logits.float()[mask]
shifted_student_logits[mask]  # creates [N_valid,V]
```

`cost_search.evaluate_student_per_sample_kl` 必须把原 device logits 传入：

```python
shifted_student_logits=shifted_student
```

不得 `.detach().cpu()`。函数已在 `torch.inference_mode()`，无需 detach；若保留 detach，只能 `shifted_student.detach()`。

### Memory assertion

新增 private helper：

```python
def _gather_topk_student_logits(
    shifted_student_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
) -> torch.Tensor:
    """Return [N_valid,K], never [N_valid,V]."""
```

测试直接断言 helper 输出 shape，后续 KL 使用该 helper；不得在 KL 函数中复制第二套 gather 实现。

- [ ] **Step 1: Add CPU equivalence tests against the existing dense definition**

覆盖：

- batch 1 和 batch 3；
- 非连续 valid mask；
- K=1、K=3、K=V；
- teacher probs bfloat16；
- 不同样本 token 数；
- negative/positive logits。

- [ ] **Step 2: Add helper shape test**

构造 `V=1000,K=4`，断言 helper shape 仅为 `[N_valid,4]`。

- [ ] **Step 3: Add CUDA device test**

断言 input/output 都留在 CUDA，结果与 CPU dense reference 一致。

- [ ] **Step 4: Add source guard for cost worker**

`inspect.getsource(evaluate_student_per_sample_kl)` 不得包含：

```text
shifted_student.detach().cpu()
shifted_student.cpu()
```

- [ ] **Step 5: Run focused tests and confirm old source guard fails**

- [ ] **Step 6: Implement `_gather_topk_student_logits` and refactor KL**

不要改变 `per_sample_exact_forward_kl`。

- [ ] **Step 7: Remove full-logits CPU transfer in `cost_search.py`**

只允许最终 `kl.detach().cpu()`。

- [ ] **Step 8: Run focused tests**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_kl_metric.py \
  mix_bit/tests/test_cost_table.py \
  mix_bit/tests/test_tiny_integration.py -q
```

- [ ] **Step 9: Commit Task 5 files**

```bash
git add mix_bit/kl_metric.py mix_bit/cost_search.py mix_bit/tests/test_kl_metric.py mix_bit/tests/test_cost_table.py mix_bit/tests/test_tiny_integration.py
git commit -m "perf: gather student top-k logits on device"
```

---

