> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-2-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2 Report: Enforce the Contract During Export, Pool Indexing and Resume

## Status

DONE

## Commits

none — working-tree changes only, no `git add` / `git commit` run.

## Files Changed

| File | Change |
| --- | --- |
| `mix_bit/candidate_artifact.py` | Wired shared contract into export. |
| `mix_bit/checkpoint_pool.py` | Wired shared contract into pool indexing. |
| `mix_bit/candidate_pool.py` | Replaced weak resume detection with full trial validation. |
| `mix_bit/tests/test_candidate_artifact.py` | Updated fixtures to be contract-consistent; added 5 negative export tests. |
| `mix_bit/tests/test_checkpoint_pool.py` | Updated fixtures to be contract-consistent; added 6 pool-index negative tests. |
| `mix_bit/tests/test_candidate_pool.py` | Updated `_toy_modes` for contract consistency; added 5 resume tests; updated one legacy weak-detection test. |
| `tests/test_cat_train_candidate_artifact_hook.py` | No source change required; existing hook tests still pass. |

## What Changed and Why

### Export contract (`candidate_artifact.py`)

- Imported `candidate_mode_from_payload` and `validate_module_spec_mode_contract` from `mix_bit.candidate_contract` (Task 1 helpers reused, no validation logic duplicated).
- `save_candidate_artifact_from_model` now:
  1. Parses `trial_spec["mode"]` with `candidate_mode_from_payload(...)` up front, before any output file is touched.
  2. Deletes any pre-existing `completed.json` before entering the export block, so a failed re-export never leaves a stale seemingly-complete marker. Old `module_state.pt` / `candidate_meta.json` are intentionally kept.
  3. After `_collect_vae_linear_specs` and the `has_original_weight` check, and before `torch.save`, calls `validate_module_spec_mode_contract(spec, mode, label=...)` for every selected spec. Any failure raises before state/meta/completed are written.
  4. Writes `candidate_meta.mode` as the parsed five-field canonical dict (`name/nominal_bit/codebook_bits/codebook_dim/residual_stages`) instead of copying the raw trial JSON blob.

### Pool index contract (`checkpoint_pool.py`)

- Imported `validate_mode_payload` and `validate_module_spec_mode_contract`.
- In `_load_and_validate_artifact`:
  - Replaced the old "only compare `mode.name`" check with `validate_mode_payload(meta["mode"], mode, label=...)`, which requires all five mode fields to match.
  - After `_validate_module_spec_against_target`, calls `validate_module_spec_mode_contract(spec, mode, label=f"{label}/{module_name}")` for every spec. The label encodes `category/mode/module_name` so error messages carry all three identifiers.
  - The mode-contract loop completes before any `ModuleCandidate` is constructed.

### Resume contract (`candidate_pool.py`)

- Added `validate_trial_completion(trial: TrialSpec) -> None` which raises unless the artifact exactly belongs to the trial. It verifies:
  - the three artifact files exist;
  - `completed` and `meta` `format` are both `vaellm_candidate_modules_v1`;
  - state SHA matches both `meta.module_state_sha256` and `completed.module_state_sha256`; meta SHA matches `completed.candidate_meta_sha256`;
  - the five hash fields (`run_config_sha256`, `candidate_space_sha256`, `training_recipe_sha256`, `model_profile_sha256`, `model_inventory_fingerprint`), `category_name`, the five mode fields, and `expected_module_names` (order + no duplicates) all match the `TrialSpec`;
  - the module-spec name set equals `expected_module_names`;
  - every module spec passes `validate_module_spec_mode_contract`;
  - `completed.module_count == len(expected_module_names)`.
- `is_trial_complete(trial)` now delegates to `validate_trial_completion` and only catches `(FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError)` — no bare `except Exception`.
- `_run_one_trial` now calls `validate_trial_completion` after a subprocess exit code of 0; on failure it writes the full `type(exc).__name__: exc` to the trial log and forces `exit_code = 1`. The public `is_trial_complete(trial)` name is preserved so unrelated callers are unaffected.

### Test fixture updates

The pre-existing test fixtures built `VAELinear` modules and `CandidateMode` values whose `nominal_bit` did not satisfy the Task 1 invariant `nominal_bit == residual_stages * codebook_bits / codebook_dim`, and whose decoder `in_dim` / VQ `logical_shape[-1]` did not equal `codebook_bits`. Those fixtures predate the contract, so once the contract is enforced they would fail export/indexing. They were corrected (not weakened) so the modules genuinely match their trial modes:

- `test_candidate_artifact.py`: `_make_decoder`/`_make_vae_linear` now take `codebook_bits` and `codebook_dim` separately; VQ `logical_shape[-1]` is `codebook_bits`, decoder `in_dim` is `codebook_bits`, `out_dim` is `codebook_dim`. `_trial_spec` accepts an optional mode; default mode `b16d4s2` now has `nominal_bit=8.0` (2*16/4).
- `test_checkpoint_pool.py` / `test_candidate_pool.py`: `_toy_modes` now sets `nominal_bit = residual_stages * codebook_bits / codebook_dim`. `_module_spec_for(target, mode)` now emits the full contract fields (`residual_stages`, `codebook_dim`, `stage_codebook_dims`, `parallel_parts`, `stage_vq_weights`, `stage_decoders`, legacy `vq_weights`/`decoders`) so valid artifacts pass the contract.

### New tests

- `test_candidate_artifact.py`: `test_export_rejects_trial_s2_when_actual_module_is_s1`, `test_export_rejects_trial_mode_when_actual_codebook_dim_differs`, `test_export_rejects_trial_mode_when_actual_vq_logical_bits_differ`, `test_export_rejects_trial_mode_when_decoder_in_dim_differs`, `test_export_does_not_write_completed_on_contract_failure`. Each constructs a real `VAELinear` (via the test doubles) whose structure disagrees with the trial mode, and asserts the export raises and (for the last test) that a pre-existing `completed.json` is removed.
- `test_checkpoint_pool.py`: `test_pool_rejects_same_mode_name_with_wrong_nominal_bit`, `..._wrong_codebook_bits`, `..._wrong_codebook_dim`, `..._wrong_residual_stages`, `test_pool_rejects_mislabeled_s2_artifact_with_s1_module_spec`, `test_pool_rejects_mislabeled_artifact_with_wrong_vq_logical_bits`. The mislabeled-s2 test reproduces the audit-confirmed hole (old name-only check accepted an s1 module spec under an s2 label) and confirms the fix raises `ValueError`.
- `test_candidate_pool.py`: `test_resume_retrains_when_mode_metadata_differs`, `..._when_inventory_fingerprint_differs`, `..._when_expected_module_order_differs`, `..._when_module_spec_mode_contract_fails`, `test_resume_accepts_exact_valid_artifact`. Each asserts `is_trial_complete` returns `False`/`True` and directly calls `validate_trial_completion` to check the error message.

One pre-existing weak-detection test (`test_completed_trial_requires_compact_artifact_and_hashes`) had its final assertion updated: matching hashes alone no longer mark a trial complete, because the strict validator also requires the full meta contract. The old "matching hashes ⇒ True" expectation was the exact weak behavior this task replaces.

## Tests Run

All commands run in the `bitvae` conda environment (`/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`).

1. Focused suite (Step 8):
   ```
   python -m pytest mix_bit/tests/test_candidate_artifact.py mix_bit/tests/test_checkpoint_pool.py mix_bit/tests/test_candidate_pool.py -q
   -> 56 passed
   ```
2. Candidate hook regression (Step 9):
   ```
   python -m pytest tests/test_cat_train_candidate_artifact_hook.py mix_bit/tests/test_candidate_artifact.py mix_bit/tests/test_checkpoint_pool.py mix_bit/tests/test_candidate_pool.py -q
   -> 61 passed
   ```
3. Task 1 contract regression (sanity):
   ```
   python -m pytest mix_bit/tests/test_candidate_contract.py -q
   -> 20 passed
   ```

No linter errors were introduced on the modified production or test files.

## Scope Adherence

- Only the Task 2 files listed in the brief were modified; `train_utils/cat_train_pipeline.py` was not touched.
- Task 1 validation logic is imported and reused; nothing was duplicated into export/pool/resume.
- No git commit was performed; changes remain in the working tree.

## Concerns

None.

---

## Round 1/5 Fix: `.gitignore` overly-broad `tests/` rule

### Finding

The repo-root `.gitignore` contained the rules `tests/` (line 52) and `test/` (line 53). Because gitignore patterns without a leading slash match any directory of that name at any depth, `tests/` matched both `mix_bit/tests/` and the repo-root `tests/`. Verified with `git check-ignore -v`:

```
.gitignore:52:tests/    mix_bit/tests/test_candidate_artifact.py
.gitignore:52:tests/    tests/test_cat_train_candidate_artifact_hook.py
.gitignore:52:tests/    mix_bit/tests/test_checkpoint_pool.py
.gitignore:52:tests/    mix_bit/tests/test_candidate_pool.py
```

As a result the Task 2 test files existed on disk and passed, but were not git-trackable, so Plan Step 10 `git add` of those tests would silently skip them.

### What changed

`.gitignore`: removed the two overly-broad lines `tests/` and `test/`. A repo-wide audit (`find . -type d -name tests` / `-name test`) confirmed the only directories those rules matched are `./mix_bit/tests` and `./tests`, both of which contain real project pytest sources. There are no `test/` directories anywhere in the tree, and no other `tests/` directories that should remain ignored (other noisy locations such as `lightning_logs/`, `wandb/`, `.result`, `**/__pycache__/` are already covered by their own rules). No test files were deleted; no other rules were touched; scope was limited to making plan-required tests trackable.

### Covering tests

The fix un-ignores the Task 2 test files plus the Task 1 contract test and previously-untracked repo-root tests:

- `mix_bit/tests/test_candidate_artifact.py`
- `mix_bit/tests/test_checkpoint_pool.py`
- `mix_bit/tests/test_candidate_pool.py`
- `mix_bit/tests/test_candidate_contract.py` (Task 1)
- `tests/test_cat_train_candidate_artifact_hook.py`
- `tests/test_model_utils_auto_loader.py`
- `tests/test_temporary_switch_residency.py`

### Verification command

```
git check-ignore -v \
  mix_bit/tests/test_candidate_artifact.py \
  tests/test_cat_train_candidate_artifact_hook.py \
  mix_bit/tests/test_checkpoint_pool.py \
  mix_bit/tests/test_candidate_pool.py \
  mix_bit/tests/test_candidate_contract.py
```

Output: exit code 1 (no matches) — none of the paths are ignored anymore.

`git status --short -- mix_bit/tests/ tests/ .gitignore` now reports them as untracked candidates:

```
MM .gitignore
?? mix_bit/tests/
?? tests/test_cat_train_candidate_artifact_hook.py
?? tests/test_model_utils_auto_loader.py
?? tests/test_temporary_switch_residency.py
```

### Test re-run command

```
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  tests/test_cat_train_candidate_artifact_hook.py \
  mix_bit/tests/test_candidate_artifact.py \
  mix_bit/tests/test_checkpoint_pool.py \
  mix_bit/tests/test_candidate_pool.py -q
```

Output:

```
.............................................................            [100%]
61 passed in 6.67s
```

### Commit

None — working-tree change only, per instructions.
