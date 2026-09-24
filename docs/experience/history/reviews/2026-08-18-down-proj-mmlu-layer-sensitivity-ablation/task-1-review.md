> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-1-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1 Review: Isolated Core and Layer-State Semantics

**Reviewer scope:** Task-scoped gate (requirements + build quality).  
**Evidence:** `task-1-brief.md`, `task-1-report.md`, `task-1-review-package.diff` only. No git re-run, no working-tree mutation, no test re-run.

---

## Spec Compliance: ✅

Implementation matches the brief on all binding requirements. No Missing, Extra (beyond allowed scope), or Misunderstood semantics found in the diff.

### Requirements checklist (verified against diff)

| Requirement | Verdict | Evidence |
|-------------|---------|----------|
| All new code under `experiments/down_layer_sensitivity/` | ✅ | `core.py`, `__init__.py`, `tests/test_core.py`, `tests/__init__.py` |
| `experiments/__init__.py` only if needed for imports | ✅ | Empty file; enables `from experiments.down_layer_sensitivity...` |
| No production file modifications | ✅ | Diff contains only `experiments/**` |
| Exact public API (9 symbols) | ✅ | All exported from `__init__.py` and implemented in `core.py` |
| Reuse listed interfaces only | ✅ | `VAELinear`, `load_model_checkpoint`, `iter_named_vae_module_refs`, `NamedVAELinearTarget`, `prime_named_vae_linear_cache` |
| `DownLayerRef` + `_DOWN_RE` parsing | ✅ | `core.py:65-73` |
| `discover_down_layers` 10-step contract | ✅ | `core.py:76-107` — 36 layers, contiguous 0..35, `VAELinear`, `original_weight`, `always_use_original=False` |
| `set_temporary(True)` = compressed, `False` = original | ✅ | `set_down_restore_set` uses `ref.layer_idx not in restore_layers` as temporary flag |
| `reset_all_vae_to_compressed` rejects `always_use_original` | ✅ | `core.py:110-115` |
| `set_down_restore_set` unknown-index guard | ✅ | `core.py:122-125` |
| `assert_down_restore_set` strict check | ✅ | `core.py:131-145` |
| `unload_non_down_original_weights` count dict + branching | ✅ | `core.py:148-194` — five exact keys, protected/already-unloaded/success paths |
| `prewarm_compressed_weights` via grouped prewarm, `failed == 0` | ✅ | `core.py:197-213` |
| Weight metrics from `_cached_weight` only (no re-decode) | ✅ | `core.py:216-250` |
| `load_worker_model` fixed 9-step order | ✅ | `core.py:253-272` |
| Six mandatory discovery tests (separate, not merged) | ✅ | `test_core.py:395-458` — all six cases present |
| Mandatory state-leakage sequence | ✅ | `test_core.py:462-482` — exact four-step sequence |
| Unload compressed-forward stability test | ✅ | `test_core.py:491-498` |
| Synthetic tests use real small `VAELinear`, no prod branches | ✅ | `_build_vae_linear` helper |

### Notes on report claims (independently verified)

- **13 tests:** Diff shows 13 test methods; count matches report. Pass count not re-run (implementer reported 13/13).
- **No production edits:** Confirmed from diff file list.
- **`load_worker_model` order:** Matches brief step-for-step; return dict keys `{model, meta, down_layers, prewarm_stats}` correct.
- **Duplicate-layer discovery case:** Report concern is accurate — true duplicate indexes are impossible with `_DOWN_RE` on a normal module tree. Brief lists “duplicate/non-contiguous” as one combined case; test covers non-contiguous omission (layer 0 missing). Acceptable for spec.

### Allowed extras (not spec violations)

- `experiments/down_layer_sensitivity/tests/__init__.py` — standard test package marker.
- Additional tests beyond brief minimum: unknown restore index, unload count branches, protected/unprotected runtime error, weight-metrics cache tests — all aligned with specified semantics.

---

## ⚠️ Cannot verify from diff alone

1. **Test execution:** 13/13 pass claimed in report; not re-run in this review.
2. **`load_worker_model` end-to-end:** No integration test against real multi-GB checkpoint (brief Step 11 only requires unit tests; deferral is reasonable).
3. **Real Qwen3-8B model tree:** `discover_down_layers` regex and 36-layer assumption not exercised on production checkpoint in this task.
4. **`prime_named_vae_linear_cache` runtime behavior:** Import path and call shape match production re-exports; actual decode/cache population on full model not shown in diff.
5. **Review package hygiene:** Diff listing includes `__pycache__/*.pyc` artifacts — not source deliverables; should not be committed later, but out of scope for semantics review.

---

## Code Quality Findings

### Critical

None.

### Important

None.

### Minor

1. **`core.py:148-186` — Non-down `temporary=True` invariant not asserted in unload helper**  
   Brief Step 6 states non-down VAE must remain on compressed forward path regardless of protected originals. Behavior is preserved in practice (`reset_all_vae_to_compressed` precedes unload in `load_worker_model`; successful unload sets `temporary=True` in `VAELinear.unload_original_linear`), and one test asserts `temporary is True` post-unload (`test_core.py:524-527`). Production helper does not validate or re-enforce the invariant. Low risk given call order; optional hardening for later tasks.

2. **`core.py:197-200` — Redundant work when called from `load_worker_model`**  
   `prewarm_compressed_weights` re-runs `reset_all_vae_to_compressed`, `model.to(device)`, and `model.eval()` after `load_worker_model` already did equivalent steps. Correct per brief (prewarm must include those steps); slightly redundant in the composed path.

3. **`test_core.py:411-414` — “Duplicate” half of combined discovery case not isolated**  
   Non-contiguous coverage is sufficient per brief wording; a literal duplicate-index failure mode is not constructible without corrupting the module tree. Documented limitation only.

4. **No unit test for `discover_down_layers` when `num_hidden_layers != 36`**  
   Logic exists (`core.py:78-81`); not among the six mandated discovery cases. Optional coverage gap.

5. **No unit test for `reset_all_vae_to_compressed` raising on stray `always_use_original`**  
   Symmetric with discovery validation; not required by brief Step 2/5. Optional coverage gap.

6. **Review package includes bytecode artifacts**  
   `__pycache__` entries in review diff are noise for reviewers and should be excluded from any future commit.

---

## Task Quality Verdict

**Approved**

Core semantics, interface surface, file isolation, and mandatory test matrix match the brief. Implementation is direct, reuses existing production hooks without modification, and tests use minimal synthetic trees with real `VAELinear` instances as required. Residual items are minor hygiene/optional-coverage notes, not blockers for downstream experiment tasks.
