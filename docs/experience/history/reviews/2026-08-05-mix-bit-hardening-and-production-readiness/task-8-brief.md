> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-8-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 8: Make `--pool_manifest` the Authoritative Candidate Pool Location

**Files:**
- Modify: `mix_bit/checkpoint_pool.py`
- Create: `mix_bit/tests/test_cli_pool_manifest.py`
- Modify: `mix_bit/cli/prepare_uniform_baseline.py`
- Modify: `mix_bit/cli/compute_cost_table.py`
- Modify: `mix_bit/cli/solve_allocation.py`
- Modify: `mix_bit/cli/assemble_mixed_model.py`
- Modify: `mix_bit/cli/validate_mixed_model.py`
- Modify: `mix_bit/cost_table.py`
- Modify: `mix_bit/README.md`

**New and changed interfaces:**

```python
def build_candidate_pool_index(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    *,
    output_root: str | None = None,
    write_manifest: bool = True,
) -> CandidatePoolIndex:
    """Validate artifacts; write candidate_manifest.json only when explicitly enabled."""


def build_candidate_pool_index_from_manifest(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    manifest_path: str | Path,
) -> CandidatePoolIndex:
    """Read and validate an existing manifest without rewriting it."""
```

同时从现有 `build_candidate_pool_index` 提取：

```python
def _candidate_manifest_payload(index: CandidatePoolIndex) -> dict[str, Any]:
    """Return the exact canonical manifest payload from a validated index."""
```

### Exact path and immutability rules

1. resolve `manifest_path` to absolute；
2. require file name exactly `candidate_manifest.json`；
3. require file exists；
4. 读取原始 bytes，记录 `supplied_sha256` 并解析 JSON object；
5. require `kind == "mix_bit_candidate_pool_manifest"`；
6. set `pool_root = manifest_path.parent`；
7. call `build_candidate_pool_index(resolved, inventory, output_root=str(pool_root), write_manifest=False)`；
8. require `Path(index.manifest_path).resolve() == manifest_path`；
9. 用 `_candidate_manifest_payload(index)` 构建 expected payload，要求 supplied JSON 与 expected payload 完全相等，包括 artifact 顺序、绝对路径和每个 SHA；
10. 再次计算 on-disk SHA，要求仍等于 `supplied_sha256`；
11. return index。

调用 `build_candidate_pool_index` 且固定 `write_manifest=False` 时，不得创建、修改或 touch manifest。默认 `write_manifest=True` 保持现有 canonical inventory 阶段行为。不得先调用 canonical `build_candidate_pool_index(resolved, inventory)`，不得在 supplied manifest 不匹配时自动重写或修复文件。

### CLI wiring

- `prepare_uniform_baseline.py`：直接调用 new helper。
- `compute_cost_table.py`：直接调用 new helper。
- `assemble_mixed_model.py`：直接调用 new helper。
- `validate_mixed_model.py`：直接调用 new helper。
- `solve_allocation.py`：
  - 若 `--pool_manifest` 提供，调用 new helper；
  - 若未提供，保持 canonical build；
  - help text 改为“optional; provided path is authoritative”。

### Spawn worker wiring

Cost parent 已持有 `pool_index.manifest_path`。必须把它写入 baseline init 和每个 worker args：

```python
"pool_manifest_path": str(Path(pool_index.manifest_path).resolve())
```

`_baseline_init_process_main` 和 `_worker_process_main` 必须使用 `build_candidate_pool_index_from_manifest`，不得 canonical rebuild。

### Tests

- [ ] **Step 1: Build a custom-root tiny candidate pool fixture**

路径必须不是 `resolved.canonical_run_root/candidate_pool`。

- [ ] **Step 2: Test helper loads custom root**

完整实现：

- `test_index_from_manifest_uses_manifest_parent`
- `test_index_from_manifest_rejects_wrong_filename`
- `test_index_from_manifest_rejects_missing_file`
- `test_index_from_manifest_does_not_modify_manifest_bytes`
- `test_index_from_manifest_rejects_stale_payload_without_rewrite`
- `test_build_index_write_manifest_false_creates_no_manifest`

- [ ] **Step 3: Test every CLI uses the authoritative helper**

对五个 CLI 的 `main(argv)` 使用 monkeypatch；断言 helper 收到 supplied manifest，且 canonical builder 未被调用。

测试名固定，并使用各 CLI 现有必需 argv fixture 完整实现：

- `test_prepare_baseline_cli_uses_supplied_pool_manifest`
- `test_cost_cli_uses_supplied_pool_manifest`
- `test_solve_cli_uses_supplied_pool_manifest`
- `test_assemble_cli_uses_supplied_pool_manifest`
- `test_validate_cli_uses_supplied_pool_manifest`

- [ ] **Step 4: Test spawned worker args preserve manifest path**

在 `test_cost_table.py` 捕获 baseline/worker args，断言均包含同一 absolute path。

- [ ] **Step 5: Run tests and confirm custom-root CLI tests fail on old code**

- [ ] **Step 6: Implement helper and wire all CLIs**

不得增加第二个 `--candidate_pool_root` 参数；manifest 已足够确定 root。

- [ ] **Step 7: Wire spawn children**

确保 parent 和 child 使用同一 manifest hash。

- [ ] **Step 8: Update README custom-root example**

写明：candidate training 使用 `--output_root X` 后，后续阶段统一传 `--pool_manifest X/candidate_manifest.json`。

- [ ] **Step 9: Run focused tests**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_cli_pool_manifest.py \
  mix_bit/tests/test_cost_table.py \
  mix_bit/tests/test_checkpoint_pool.py -q
```

- [ ] **Step 10: Commit Task 8 files**

```bash
git add \
  mix_bit/checkpoint_pool.py \
  mix_bit/cost_table.py \
  mix_bit/cli/prepare_uniform_baseline.py \
  mix_bit/cli/compute_cost_table.py \
  mix_bit/cli/solve_allocation.py \
  mix_bit/cli/assemble_mixed_model.py \
  mix_bit/cli/validate_mixed_model.py \
  mix_bit/tests/test_cli_pool_manifest.py \
  mix_bit/tests/test_cost_table.py \
  mix_bit/README.md
git commit -m "fix: honor supplied candidate pool manifests"
```

---

