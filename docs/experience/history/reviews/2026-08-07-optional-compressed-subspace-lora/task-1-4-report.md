> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-1-4-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Tasks 1–4 Report — optional compressed subspace LoRA

**Status:** DONE  
**Date:** 2026-08-07  
**Workdir:** `/home/shaoyuantian/program/VAELLM`  
**Env:** `bitvae`, `peft==0.10.0`

## Summary

Implemented Tasks 1–4 exactly per plan §3 / §6 / §7 / §9 / §13 / §14:

1. Scope truth source `litebsq/low_rank_scope.py`
2. O(1) `PeftZeroLinearCarrier` + full `CompressedSubspacePeftProxy` lifecycle in `e2e_common/compressed_subspace_lora.py`
3. `VAELinear` scope-aware shape validator + subspace/full finalize order
4. Core numerical + PEFT gate tests in `tests/test_compressed_subspace_lora.py`

No git commits / worktrees / branch switches.  
No changes to `block_vae_lora` / `remaining_lora` / PEFT library.  
No mask-after-merge, no dense `[Oc,Ic]` carrier, no custom LoRA math, no PEFT upgrade.

## PEFT 0.10.0 gates (ran first)

Both gates **PASS** under `bitvae` / `peft==0.10.0` before completing proxy/lifecycle work:

| Gate | Result |
|------|--------|
| `inject_adapter_in_model` on `PeftZeroLinearCarrier` | PASS |
| `get_peft_model` CausalLM path on `PeftZeroLinearCarrier` | PASS |

Observed after injection:

- `base_layer.weight.shape == [1,1]`, `numel()==1`, value 0, `requires_grad=False`
- `lora_A=[r,Ic]`, `lora_B=[Oc,r]`
- initial forward == 0; nonzero A/B forward matches `B(A(x))*scaling`
- backward grads on A/B finite; sentinel weight has no grad

## Files changed

### Created

| File | Purpose |
|------|---------|
| `litebsq/low_rank_scope.py` | Scope constants + `normalize_low_rank_scope()` only; no `VAELinear` / `e2e_common` imports |
| `e2e_common/compressed_subspace_lora.py` | Carrier, proxy, inject/init/extract/wrap/export/unwrap helpers |
| `tests/test_compressed_subspace_lora.py` | Tasks 1–4 core tests + PEFT gates; Task 7 traversal tests skipped |
| `.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-1-4-report.md` | This report |

### Modified

| File | Change |
|------|--------|
| `litebsq/vae_linear.py` | Added `low_rank_scope` ctor arg; `_expected_low_rank_shape_for_scope` / `_validate_low_rank_payload_tensors`; `_apply_low_rank_patch_to_weight`; scope-aware `_finalize_decoded_weight_from_compressed` order; ctor uses validator |

## Implementation notes

### `litebsq/low_rank_scope.py`

Exact plan §3 content:

- `LOW_RANK_SCOPE_FULL = "full"`
- `LOW_RANK_SCOPE_COMPRESSED_SUBSPACE = "compressed_subspace"`
- `normalize_low_rank_scope()`

### `e2e_common/compressed_subspace_lora.py`

Implemented all required symbols:

- `_resolve_proxy_device_dtype`
- `PeftZeroLinearCarrier` (1×1 sentinel, O(1) storage, identically-zero forward)
- `_build_compressed_indices` (`keep_mask -> nonzero` order)
- `CompressedSubspacePeftProxy` (`forward` / `set_temporary` per §7.4)
- `inject_compressed_subspace_peft_lora`
- `initialize_subspace_peft_lora_from_low_rank`
- `extract_subspace_peft_low_rank_payloads`
- `_subspace_proxy_root`, `iter_named_compressed_subspace_peft_proxies`, `_select_subspace_proxy_refs`
- `wrap_vae_linears_with_compressed_subspace_peft_proxy`
- `export_compressed_subspace_peft_lora_to_vae_low_rank` (validate candidate payload before mutating scope/A/B; replace via `set_module_by_name(root, ...)`)
- `unwrap_compressed_subspace_peft_proxies`

Reused from `e2e_common.peft_proxy`:

- `_get_default_adapter_name`
- `_adapter_uses_dora`
- `is_peft_lora_linear`
- `is_peft_adalora_linear`

`set_module_by_name` imported from `litebsq.misc` (same source used by `peft_proxy`).

### `litebsq/vae_linear.py`

- Default `low_rank_scope=LOW_RANK_SCOPE_FULL`
- Normalize/store scope before A/B registration
- Constructor shape checks replaced by `_validate_low_rank_payload_tensors`
- Finalize order:
  1. subspace low-rank on compressed weight (if scope=subspace)
  2. materialize full
  3. protected residual
  4. full low-rank on full weight (if scope=full) — original position preserved
  5. sparse residual

## Tests

### Commands

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bitvae
python -m pytest tests/test_compressed_subspace_lora.py -q --tb=short
python -m pytest tests/test_e2e_checkpoint_io_legacy.py tests/test_temporary_switch_residency.py -q --tb=line
```

### Results

| Suite | Result |
|-------|--------|
| `tests/test_compressed_subspace_lora.py` | **15 passed, 4 skipped** |
| legacy checkpoint + temporary residency | **15 passed** |

### Covered (Tasks 1–4)

- `test_full_low_rank_scope_is_default`
- `test_full_low_rank_scope_keeps_existing_full_shape_contract`
- `test_subspace_input_protection_keeps_protected_columns_unchanged`
- `test_subspace_output_protection_keeps_protected_rows_unchanged`
- `test_subspace_no_protection_matches_full_numerically`
- `test_full_scope_still_allows_delta_on_protected_coordinates`
- `test_invalid_subspace_low_rank_shape_fails`
- `test_peft_zero_carrier_has_constant_base_storage`
- `test_peft_zero_carrier_inject_adapter_gate`
- `test_peft_zero_carrier_get_peft_model_gate`
- `test_subspace_peft_initial_delta_is_zero`
- `test_subspace_peft_forward_matches_exported_vae_linear`
- `test_subspace_peft_payload_restore_export_roundtrip`
- `test_subspace_proxy_root_resolves_original_names_under_root_peft_model`
- `test_subspace_proxy_uses_base_device_and_dtype`

### Skipped until Task 7

- `test_hif4_collect_treats_subspace_proxy_as_one_logical_linear`
- `test_hif4_collect_does_not_hook_subspace_proxy_descendants`
- `test_temporary_mode_visits_subspace_proxy_once`
- `test_vae_module_refs_yield_subspace_proxy_not_nested_base_layer`

## Concerns

None blocking.

Minor notes (non-blocking):

1. Task 7 traversal wiring intentionally not implemented; four tests are `pytest.mark.skip` stubs.
2. Checkpoint schema / category CLI / E2E routing remain for later tasks; old checkpoints without `low_rank_scope` are not handled yet (Task 5).
3. Tiny VAELinear factories in the new test file use real VQ bits + Decoder (same pattern as legacy checkpoint tests), plus `_finalize_decoded_weight_from_compressed` for weight-level invariance checks.

## Explicit non-goals confirmed

- No dense `[Oc,Ic]` zero Linear carrier fallback
- No PEFT upgrade
- No custom LoRA A/B/dropout/scaling math
- No mask-after-merge
- No edits to `block_vae_lora` / `remaining_lora` / PEFT package
- No git commit
