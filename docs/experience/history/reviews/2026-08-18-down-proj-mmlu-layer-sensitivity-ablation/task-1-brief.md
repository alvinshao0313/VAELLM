> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-1-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1: Build the Isolated Core and Lock Down Layer-State Semantics

**Files:**
- Create: `experiments/down_layer_sensitivity/__init__.py`
- Create: `experiments/down_layer_sensitivity/core.py`
- Create: `experiments/down_layer_sensitivity/tests/test_core.py`

**Interfaces:**
- Consumes: `litebsq.vae_linear.VAELinear`, `train_utils.model_checkpoint_io.load_model_checkpoint`, `e2e_common.proxy_trainables.iter_named_vae_module_refs`, `litebsq.vae_linear.NamedVAELinearTarget`, `litebsq.vae_linear.prime_named_vae_linear_cache`.
- Produces:
  - `DownLayerRef`
  - `discover_down_layers(model)`
  - `reset_all_vae_to_compressed(model)`
  - `set_down_restore_set(down_layers, restore_layers)`
  - `assert_down_restore_set(down_layers, restore_layers)`
  - `unload_non_down_original_weights(model, down_names)`
  - `prewarm_compressed_weights(model, device, group_size)`
  - `compute_down_weight_metrics(down_layers)`
  - `load_worker_model(checkpoint_dir, device, prewarm_group_size)`

- [ ] **Step 1: Define `DownLayerRef` and exact name parsing**

In `core.py` define:

```python
from dataclasses import dataclass
import re
from litebsq.vae_linear import VAELinear

_DOWN_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj$")


@dataclass(frozen=True)
class DownLayerRef:
    layer_idx: int
    name: str
    module: VAELinear
```

`discover_down_layers(model)` must:

1. read `expected_layers = int(model.config.num_hidden_layers)`;
2. iterate `model.named_modules()`;
3. keep only names matching `_DOWN_RE`;
4. require matched module `isinstance(module, VAELinear)`;
5. sort by `layer_idx`;
6. require `expected_layers == 36` for this formal experiment;
7. require indexes exactly `list(range(36))`;
8. require every down `original_weight is not None`;
9. require every down `always_use_original == False`;
10. return 36 `DownLayerRef` objects.

Do not make this generic to arbitrary model families in this task; the experiment is intentionally bound to the specified Qwen3-8B checkpoint.

- [ ] **Step 2: Write tests for discovery failure modes**

`test_core.py` 的 discovery 测试固定覆盖以下 6 个 case；不要自行删减、合并或用其它 case 替代：

```text
- valid contiguous down refs are sorted by layer index
- missing layer raises ValueError
- duplicate/non-contiguous layer index raises ValueError
- matched down module not VAELinear raises TypeError
- down original_weight=None raises ValueError
- always_use_original=True raises ValueError
```

For unit tests, build a minimal synthetic module tree with a small real `VAELinear`; do not alter production classes and do not add test-only branches to production experiment code.

- [ ] **Step 3: Implement state reset and restore-set application**

Exact semantics:

```python
def reset_all_vae_to_compressed(model) -> None:
    for module in model.modules():
        if isinstance(module, VAELinear):
            if bool(getattr(module, "always_use_original", False)):
                raise ValueError("Formal sensitivity run requires no always_use_original VAELinear.")
            module.set_temporary(True)


def set_down_restore_set(
    down_layers: list[DownLayerRef],
    restore_layers: set[int],
) -> None:
    valid = {ref.layer_idx for ref in down_layers}
    unknown = sorted(set(restore_layers) - valid)
    if unknown:
        raise ValueError(f"Unknown down layer indices: {unknown}")

    for ref in down_layers:
        ref.module.set_temporary(ref.layer_idx not in restore_layers)
```

Remember existing semantics:

```text
set_temporary(True)  -> compressed path
set_temporary(False) -> original weight path
```

- [ ] **Step 4: Add a strict state assertion**

```python
def assert_down_restore_set(
    down_layers: list[DownLayerRef],
    restore_layers: set[int],
) -> None:
```

For each down:

```text
expected temporary = layer_idx not in restore_layers
actual temporary must equal expected
if original path expected, original_weight must still be non-None
```

Raise immediately on mismatch.

- [ ] **Step 5: Test no cross-job state leakage**

Test sequence exactly:

```text
reset -> restore {3} -> assert only 3 original
reset -> restore {7} -> assert only 7 original
reset -> restore {}  -> assert all compressed
reset -> restore {0, 1, 2} -> assert exactly 0,1,2 original
```

This test is mandatory because a leaked `temporary=False` would invalidate all subsequent MMLU jobs.

- [ ] **Step 6: Implement non-down original-weight unloading**

```python
def unload_non_down_original_weights(
    model,
    down_names: set[str],
) -> dict:
```

Rules:

- iterate all named `VAELinear`;
- if name in `down_names`, never unload its original weight;
- otherwise call existing `module.unload_original_linear()` exactly once and inspect the returned bool;
- if it returns `True`, require `module.original_weight is None`，计入 `non_down_original_unloaded`；
- if it returns `False` because `original_weight is None`，计入 `non_down_already_unloaded`；
- if it returns `False` and `original_weight is not None`，只允许 `module.protect_original_weight is True`，计入 `non_down_protected_original_kept`；若 `protect_original_weight=False` 却仍保留 original，才 raise `RuntimeError`；
- return counts exactly: `total_vae`, `down_original_kept`, `non_down_original_unloaded`, `non_down_already_unloaded`, `non_down_protected_original_kept`.

This is **memory optimization only, not a scientific validity gate**. Existing `VAELinear.unload_original_linear()` explicitly allows `protect_original_weight=True` modules拒绝卸载，因此计划不能把这种合法状态误判成实验失败。无论是否存在 protected original kept，都必须保证所有非-down VAE 仍处于 `temporary=True` compressed forward path。

- [ ] **Step 7: Test unloading does not change compressed forward**

On a small real `VAELinear`:

```text
set_temporary(True)
run compressed forward -> y_before
unload_original_linear()
run compressed forward -> y_after
assert allclose(y_before, y_after)
```

Do not test original forward after unload; it is intentionally unavailable for non-down modules.

- [ ] **Step 8: Implement one-time prewarm**

`prewarm_compressed_weights(model, device, group_size)` must reuse existing grouped prewarm logic:

```python
from e2e_common.proxy_trainables import iter_named_vae_module_refs
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
```

Before prewarm:

```text
reset_all_vae_to_compressed(model)
model.to(device)
model.eval()
```

Build `NamedVAELinearTarget` from `iter_named_vae_module_refs(model)` and call:

```text
prime_named_vae_linear_cache(... group_size=8, compute_device=device)
```

Require `failed == 0` in returned stats.

Do not clear decoded caches between jobs.

- [ ] **Step 9: Implement down weight metrics**

For each down under `torch.no_grad()`，**直接复用 prewarm 后的 decoded-weight cache，禁止为了统计 NMSE 再 decode 一次**：

```python
if ref.module._cached_weight is None:
    raise RuntimeError(f"Missing prewarmed decoded-weight cache for {ref.name}")
w_orig = ref.module.original_weight.detach().float()
w_comp = ref.module._cached_weight.detach().float()
err = w_comp - w_orig
sse = err.pow(2).sum(dtype=torch.float64)
orig_ss = w_orig.pow(2).sum(dtype=torch.float64)
```

这样 weight diagnostic 使用的就是正式 MMLU compressed forward 实际复用的 cached reconstructed weight，同时避免 36 个大 `down_proj` 被重复解码。

Return exact fields:

```text
layer_idx
name
numel
mse
nmse
relative_fro_error
original_rms
error_rms
```

Require `orig_ss > 0` and both tensors have identical shape.

- [ ] **Step 10: Implement `load_worker_model()` in the exact order**

Fixed order:

```text
1. load_model_checkpoint(checkpoint_dir, map_location="cpu", strict=True)
2. model.eval()
3. reset_all_vae_to_compressed(model)
4. discover_down_layers(model)
5. unload_non_down_original_weights(...)
6. model.to(device)
7. prewarm_compressed_weights(... group_size=8)
8. assert_down_restore_set(... restore_layers=set())
9. return model/meta/down_layers/prewarm stats
```

Do not load a second teacher/base model. The original down weights inside the checkpoint are the intervention target.

- [ ] **Step 11: Run core unit tests**

Before testing:

```bash
conda activate bitvae
which python
python -V
```

Run:

```bash
pytest -q experiments/down_layer_sensitivity/tests/test_core.py
```

Expected: all tests pass.

---

