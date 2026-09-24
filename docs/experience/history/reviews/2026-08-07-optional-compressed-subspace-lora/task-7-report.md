> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-7-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 7 Report

## Status

**DONE**

## Files changed

### `train_utils/cat_after_category_distill.py`
- Import scope constants + subspace helpers；新增 `_validate_existing_model_low_rank_scope` / `_validate_category_targets_low_rank_scope`。
- `_resolve_base_layer` 接受 `CompressedSubspacePeftProxy`。
- 新增 `_enable_subspace_compressed_trainable_params`（未改动原 `_enable_compressed_trainable_params`）。
- `_run_compressed_category_distill`：`mode in {compressed_lora, both}` 时先全局 scope guard，再 presence/skip；按 `compressed_lora_scope` 分支 full PEFT 原路径 vs subspace wrap/inject/init/export；no-trainable cleanup 与 post-train export 按 scope 收尾；subspace 不调用 materialize / ensure_peft_vae_proxy_adapter / `_set_proxy_decoder_adapter_mode`；subspace 每 category 汇总一次参数量日志。

### `train_utils/hif4_act.py`
- Lazy-import `CompressedSubspacePeftProxy`；`_is_hif4_wrapped_module` 将其视为单个逻辑 Linear。

### `e2e_common/temporary_mode.py`
- Skip `CompressedSubspacePeftProxy` 的 `base_layer` 与 carrier descendants。

### `e2e_common/proxy_trainables.py`
- `iter_named_vae_module_refs` 顶层 yield subspace proxy；跳过 base/carrier。

### `train_utils/cat_train_pipeline.py`
- after-category / final save leftover 检查同时覆盖 `iter_named_compressed_subspace_peft_proxies`。

### Tests
- `tests/test_compressed_subspace_lora.py`：实现 4 个原先 skip 的 HiF4 / temporary / vae_module_refs 测试。
- `tests/test_cat_compressed_lora_scope.py`：§16 route（full/subspace）+ continuation/global scope guard + subspace export 恢复 bare VAELinear。

## Test summary

```bash
conda activate bitvae
python -m pytest -q \
  tests/test_compressed_subspace_lora.py \
  tests/test_cat_compressed_lora_scope.py
# 40 passed

python -m pytest -q \
  tests/test_temporary_switch_residency.py \
  tests/test_cat_eval_adapter_match.py
# 23 passed
```

## Concerns

1. Route 测试通过 monkeypatch 在 enable-trainable 处返回空列表以跳过真实 Trainer；wrap/inject 调用链与 carrier `numel==1` 已断言，但未跑真实一步 subspace 训练。
2. `collect_compressed_category_targets` 未新增对 `CompressedSubspacePeftProxy` 的 skip（与计划一致；正常路径 export 后即为 bare VAELinear，save guard 拦截残留 proxy）。
3. 未跑完整 E2E / 真实模型 smoke（属后续任务范围）。
