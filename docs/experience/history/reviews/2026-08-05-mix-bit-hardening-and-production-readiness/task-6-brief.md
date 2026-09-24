> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-6-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 6: Replace Full State Cloning with Streaming State Fingerprints

**Files:**
- Create: `mix_bit/state_fingerprint.py`
- Create: `mix_bit/tests/test_state_fingerprint.py`
- Modify: `mix_bit/assembler.py`
- Modify: `mix_bit/tests/test_assembler.py`
- Modify: `mix_bit/tests/test_tiny_integration.py`

**Interfaces:**

```python
STATE_FINGERPRINT_KIND = "mix_bit_state_fingerprint_v1"
STATE_FINGERPRINT_CHUNK_BYTES = 16 * 1024 * 1024


def fingerprint_tensor(
    tensor: torch.Tensor,
    *,
    chunk_bytes: int = STATE_FINGERPRINT_CHUNK_BYTES,
) -> dict[str, object]:
    """Return dtype/shape/numel/SHA metadata using bounded CPU chunks."""


def fingerprint_model_state(
    model: nn.Module,
    *,
    chunk_bytes: int = STATE_FINGERPRINT_CHUNK_BYTES,
) -> dict[str, object]:
    """Fingerprint every state_dict tensor without retaining tensor copies."""


def compare_state_fingerprints(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> None:
    """Raise ValueError on kind/key/dtype/shape/numel/hash mismatch."""


def write_state_fingerprint_manifest(
    path: str | Path,
    payload: Mapping[str, object],
) -> str:
    """Write canonical JSON atomically and return the absolute path."""


def verify_saved_checkpoint_state(
    *,
    resolved: ResolvedRunConfig,
    output_dir: str | Path,
    expected_fingerprint: Mapping[str, object],
    expected_converted_module_names: Sequence[str],
) -> None:
    """Strict-load one saved checkpoint, compare fingerprints and reject original weights."""
```

### Fingerprint format

```json
{
  "kind": "mix_bit_state_fingerprint_v1",
  "chunk_bytes": 16777216,
  "key_count": 123,
  "entries": {
    "model.layers.0.self_attn.q_proj._parallel_stage_decoder.linear_in.conv.weight": {
      "dtype": "bfloat16",
      "shape": [256, 16, 1],
      "numel": 4096,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  }
}
```

### Tensor hashing algorithm

- require strided 且 contiguous tensor；其他 layout 或 non-contiguous tensor 明确失败，避免为了 canonicalize 非连续 view 创建整张 contiguous copy；
- `detached = tensor.detach()`；
- hash header 中的 dtype、shape、numel；
- `flat = detached.view(-1)`；由于前面已要求 contiguous，该操作不得复制整张 tensor；
- 计算每个 chunk 的最大元素数：

```python
bytes_per_element = max(1, tensor.element_size())
chunk_numel = max(1, chunk_bytes // bytes_per_element)
```

- 每次处理 `flat[start:end]`；
- 只把当前 chunk 搬到 CPU；
- 为支持 bfloat16，先在 PyTorch 中 `chunk.view(torch.uint8)`，再 `.cpu().numpy().tobytes(order="C")`；不得对 bfloat16 直接 `.numpy()`；
- 不得调用 `.clone()`；
- 完成一个 chunk 后立刻释放局部 CPU tensor。

### Assembler integration

在 `save_full_checkpoint_from_assignments`：

1. build model；
2. 调用 `save_model_checkpoint`，沿用现有参数并固定 `unload_vae_original_weights=True`，等待其临时 decoder pack/unpack context 完整退出；
3. 调用 `write_reference_logits`；
4. 在保存函数返回后计算 `saved_source_fingerprint = fingerprint_model_state(model)`；这与旧实现保存后再抓取 `reference_state` 的时点一致，确保 original weight 已卸载且 module 已恢复到正常运行态；
5. 把 `saved_source_fingerprint` manifest 原子写到：

```text
<final_model>/state_fingerprint.json
```

6. 保存 converted module name 顺序后删除 model 并回收；
7. 调用 `verify_saved_checkpoint_state`；该 helper 必须通过 profile adapter 加载 base model、调用现有 `load_checkpoint_into_model` 并固定 `strict=True`、计算 reload fingerprint、调用 `compare_state_fingerprints`，并逐个检查 expected converted module 是 `VAELinear` 且 `original_weight is None`；
8. helper 的 finally 必须删除 reload model、`gc.collect()`，CUDA 可用时 `torch.cuda.empty_cache()`；
9. 删除 `reference_state` dict 和逐 key tensor compare；
10. return payload 新增 `state_fingerprint` path。

最终 fingerprint manifest 保存 checkpoint 写入完成并退出临时 pack context 后的 source model 版本；reload 版本无需落盘，因为已比较相等。

### Existing-output skip contract

当前 `assemble_optimal_mixed_checkpoint` 在 provenance 相同且 state/meta 存在时会直接返回。修复后，`skipped_identical=True` 前必须：

1. require `pytorch_model.bin`、`checkpoint_meta.json`、`reference_logits.pt`、`state_fingerprint.json` 全部存在；
2. 读取 fingerprint manifest，require `kind == STATE_FINGERPRINT_KIND`；
3. 从 allocation entries 得到 expected converted module names；
4. 调用 `verify_saved_checkpoint_state`；
5. 任一文件缺失、manifest 非法、strict load 失败或 fingerprint mismatch 时抛 `ValueError`，提示使用显式 `--overwrite` 重新组装；不得自动补写 fingerprint 或直接 skip。

Task 9 还会在同一 skip 路径追加 final tokenizer local-only/fingerprint 验证。只有 state 和 tokenizer 两类验证都通过后才允许 `skipped_identical=True`。

- [ ] **Step 1: Write tensor fingerprint tests**

覆盖 float32、bfloat16、uint8、bool、int64、零长度 tensor，并断言 non-contiguous tensor 被明确拒绝。

- [ ] **Step 2: Write mutation detection tests**

单值变化、dtype 变化、shape 变化、key 缺失、额外 key 都必须被 compare 拒绝。

- [ ] **Step 3: Write no-clone regression test**

使用 monkeypatch 使 `torch.Tensor.clone` 在 `fingerprint_model_state` 调用时抛错；fingerprint 必须仍成功。

- [ ] **Step 4: Run tests and confirm module missing**

- [ ] **Step 5: Implement streaming fingerprint module**

不得把完整 state dict tensor copy 存在 list/dict 中；manifest 只保存字符串和整数。

- [ ] **Step 6: Add assembler test**

现有 toy final checkpoint 测试必须断言：

- `state_fingerprint.json` 存在；
- return payload 包含 path；
- fingerprint key count 与 reload model state key count 相等；
- `inspect.getsource(save_full_checkpoint_from_assignments)` 不包含 `reference_state`、`.cpu().clone()` 或逐 key tensor clone；`.clone()` monkeypatch 只作用于 `fingerprint_model_state` 单测，不包围 checkpoint 保存逻辑；
- provenance 相同但缺失 `state_fingerprint.json` 时不得 `skipped_identical`；
- fingerprint manifest 被篡改时不得 skip；
- state/meta/reference/fingerprint 全部有效且 strict reload hash 相同时才返回 `skipped_identical=True`。

- [ ] **Step 7: Replace full clone in assembler**

搜索并确保以下代码完全删除：

```text
value.detach().cpu().clone()
reference_state = {
```

- [ ] **Step 8: Run focused tests**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_state_fingerprint.py \
  mix_bit/tests/test_assembler.py \
  mix_bit/tests/test_tiny_integration.py -q
```

- [ ] **Step 9: Commit Task 6 files**

```bash
git add mix_bit/state_fingerprint.py mix_bit/assembler.py mix_bit/tests/test_state_fingerprint.py mix_bit/tests/test_assembler.py mix_bit/tests/test_tiny_integration.py
git commit -m "fix: verify final checkpoints with streaming fingerprints"
```

---

