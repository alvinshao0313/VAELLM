> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-4-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 4: Build Teacher Top-k Cache Without Full-Logits CPU Transfer

**Files:**
- Modify: `mix_bit/teacher_cache.py`
- Modify: `mix_bit/tests/test_kl_metric.py`
- Modify: `mix_bit/tests/test_tiny_integration.py`

**Preserved public interface:**

```python
def build_teacher_topk_chunk(
    *,
    sample_ids: Sequence[int],
    shifted_teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    teacher_topk: int,
    cache_prob_dtype: str,
) -> dict[str, Any]:
    """Return the existing compact cache schema with CPU [N_valid,K] payloads."""
```

### Required implementation

`build_teacher_topk_chunk` must use this order：

```python
logits_device = shifted_teacher_logits.device
batch = int(shifted_teacher_logits.shape[0])
mask_device = valid_mask.to(device=logits_device, dtype=torch.bool)
counts_device = mask_device.sum(dim=-1, dtype=torch.int64)

# top-k over every [B,T] row; output is only [B,T,K]
top_values, top_indices = torch.topk(
    shifted_teacher_logits,
    k,
    dim=-1,
    largest=True,
    sorted=True,
)
valid_top_values = top_values[mask_device]
valid_top_indices = top_indices[mask_device]
probs = torch.softmax(valid_top_values.float(), dim=-1)

# Only compact tensors move to CPU
indices_cpu = valid_top_indices.to(dtype=torch.int32, device="cpu").contiguous()
probs_cpu = probs.to(dtype=prob_dtype, device="cpu").contiguous()
counts_cpu = counts_device.to(device="cpu")
offsets_cpu = torch.zeros(batch + 1, dtype=torch.int64, device="cpu")
offsets_cpu[1:] = counts_cpu.cumsum(dim=0)
```

禁止出现：

```python
shifted_teacher_logits.cpu()
shifted_teacher_logits.float()[mask]
shifted.detach().cpu()  # before build_teacher_topk_chunk
```

`build_teacher_cache` 调用处必须改为：

```python
shifted_teacher_logits=shifted.detach()
```

不得改为 CPU。

### Numerical contract

- 保留现有全部输入验证：logits 必须 `[B,T,V]` floating tensor；mask shape 必须匹配；sample_ids 数量等于 B 且无重复；`1 <= K <= V`；每个样本至少一个 valid token；cache dtype 只能使用现有允许集合。
- teacher top-k indices 与旧 reference 完全相同；同值 tie 依赖 PyTorch `topk`，测试不得构造 tie。
- teacher probs 必须等于对 selected top-k logits 做 float32 softmax。
- 存储为 bfloat16 时允许量化误差，但加载后 KL 继续 float32 归一化。
- cache 文件字段、shape、顺序和无 tail contract 不变。

- [ ] **Step 1: Add a dense reference helper in tests only**

```python
def dense_teacher_topk_reference(logits, mask, k):
    valid = logits.float()[mask]
    values, indices = valid.topk(k, dim=-1, sorted=True)
    return indices.to(torch.int32), values.softmax(dim=-1)
```

该 helper 只能存在于测试，生产代码不得保留 full-row reference。

- [ ] **Step 2: Add CPU numerical equivalence tests**

覆盖 batch>1、不同有效长度、float32 和 bfloat16 cache dtype。

- [ ] **Step 3: Add CUDA transfer-boundary test**

完整实现 `test_teacher_topk_only_returns_compact_cpu_tensors`，并使用：

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
```

断言：

- input logits 在 CUDA；
- output indices/probs/offsets/sample_ids 在 CPU；
- indices/probs shape 是 `[N_valid,K]`；
- 与 dense reference 一致。

- [ ] **Step 4: Add a source guard test**

使用 `inspect.getsource(build_teacher_cache)`，断言不存在：

```text
shifted.detach().cpu()
shifted_teacher_logits=shifted.detach().cpu()
```

该 guard 只防止完整 logits CPU 搬运回归，不检查任意 `.cpu()`，因为 compact outputs 必须转 CPU。

- [ ] **Step 5: Run tests and confirm source guard fails on old code**

- [ ] **Step 6: Implement device-side top-k**

不要改变 cache index version/fields。

- [ ] **Step 7: Run focused tests**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_kl_metric.py \
  mix_bit/tests/test_tiny_integration.py -q
```

- [ ] **Step 8: Commit Task 4 files**

```bash
git add mix_bit/teacher_cache.py mix_bit/tests/test_kl_metric.py mix_bit/tests/test_tiny_integration.py
git commit -m "perf: keep teacher top-k extraction on device"
```

---

