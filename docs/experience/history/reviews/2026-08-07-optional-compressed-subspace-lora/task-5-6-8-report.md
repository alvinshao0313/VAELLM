> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-5-6-8-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5 / 6 / 8 Report

## Status

**DONE**

## Files changed

### Task 5 — checkpoint schema
- `train_utils/model_checkpoint_io.py`
  - Import `normalize_low_rank_scope` / `LOW_RANK_SCOPE_FULL`.
  - Loader: `low_rank_scope = normalize_low_rank_scope(spec.get("low_rank_scope", "full"))`，传入 `VAELinear(...)`.
  - `_collect_vae_linear_specs()`：仅当 resolved scope != `full` 时写入 `"low_rank_scope"`；不写 `"full"` / null。
  - 未 bump meta version；shape contract 仍只在 `VAELinear.__init__`。

### Task 6 — category CLI
- `train_utils/cat_train_args.py`
  - `NormalizedCatArgs.compressed_lora_scope`
  - `--compressed_lora_scope` choices=`sorted(VALID_LOW_RANK_SCOPES)`，default=`LOW_RANK_SCOPE_FULL`
  - normalize 时再调用 `normalize_low_rank_scope(...)`
- `scripts/catlora_distill_4gpu_res0.sh`：显式 `--compressed_lora_scope "full"`
- `scripts/catlora_distill_from_checkpoint.sh`：显式 `--compressed_lora_scope "full"`

### Task 8 — full export scope
- `e2e_common/peft_proxy.py`
  - `export_peft_proxy_lora_to_low_rank`：写 A/B 前设置 `base_layer.low_rank_scope = LOW_RANK_SCOPE_FULL`
  - `detach_and_clear_vae_low_rank_payloads`：不重置 scope（原实现已满足，加测试锁死）
- `e2e_common/low_rank_lora.py`
  - `write_low_rank_payloads_to_compressed_model(..., *, expected_scope=None)`：可选 scope 校验，永不改 module scope

### Tests
- `tests/test_compressed_subspace_lora.py`：§15 checkpoint + Task 8 export/detach
- `tests/test_cat_compressed_lora_scope.py`：§16 parser（route 留给 Task 7）

## Test summary

```bash
python -m pytest -q \
  tests/test_compressed_subspace_lora.py \
  tests/test_e2e_checkpoint_io_legacy.py \
  tests/test_cat_compressed_lora_scope.py
```

**36 passed, 4 skipped**（skip 为既有 Task 7 deferred stubs）

定向覆盖：
- old/new full missing scope key → load `full`
- subspace input-protection roundtrip
- corrupted incompatible shape/scope → fail
- parser default/full/subspace/illegal
- full export sets `scope=full`；detach 保留 scope

## Concerns

1. Route / continuation guard tests（§16 后半）刻意未做，等 Task 7。
2. `e2e_common/checkpoint_io.py` 的 `_collect_single_vae_linear_spec` 未改；本任务按计划只改 `model_checkpoint_io.py`。若 E2E save 也需 omit/write scope key，需在后续 E2E 任务对齐。
3. 未跑完整训练 smoke；仅单元/checkpoint IO 测试。
