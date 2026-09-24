> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-9-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 9 Report — Tokenizer Fingerprint v2 and Self-Contained Final Tokenizer

## Status

Complete. All Task 9 focused tests pass; full `mix_bit` suite (299 tests) green.

## Commits

None (per workspace rule — only working-tree changes retained).

## Changes

### `mix_bit/calibration.py`
- Added `TOKENIZER_FINGERPRINT_VERSION = 2` and `TOKENIZER_INIT_KWARGS_EXCLUDED` constant set.
- Added recursive JSON normalizer (`_normalize_json_value`), `_stable_init_kwargs`, `_core_tokenizer_bytes` (backend JSON preferred, `sorted_vocab` fallback, hard fail otherwise), `_special_tokens_map`, `_added_vocab`.
- Added `build_tokenizer_fingerprint_payload` (provenance + content split) and rewrote `compute_tokenizer_config_sha256` to hash only `version` + `content` (path is provenance-only). Kept the old function name per brief.
- Added `tokenizer_fingerprint_version` to `CalibrationDatasetManifest`.
- `_assert_resume_compatible` now requires `tokenizer_fingerprint_version=2`; legacy manifests missing the field fail with a "regenerate calibration" hint instead of being silently accepted.
- `prepare_calibration_dataset` routes user-supplied tokenizers through `normalize_tokenizer_for_mix_bit` and writes the version field into the manifest.

### `mix_bit/model_adapter.py`
- Added `normalize_tokenizer_for_mix_bit(tokenizer, *, source_label)` with the fixed order: `padding_side="right"` → reset flag → `pad_token_id=eos_token_id` if missing. `GenericDecoderAdapter.load_tokenizer` now calls it instead of inlining the logic.

### `mix_bit/assembler.py`
- `assemble_optimal_mixed_checkpoint` and `save_full_checkpoint_from_assignments` gained keyword-only `access_token`.
- `save_full_checkpoint_from_assignments` now: loads source tokenizer via adapter → computes v2 fingerprint → enriches `extra_meta.mix_bit` with `tokenizer_fingerprint_version` / `tokenizer_fingerprint_sha256` / `source_tokenizer_reported_name_or_path` → passes tokenizer to `save_model_checkpoint` → local-only reloads from `output_dir` (`local_files_only=True`, `trust_remote_code=False`) → normalizes → recomputes fingerprint and compares; on mismatch fails and retains state/meta/tokenizer files (no fabricated/deleted marker). Return payload adds `tokenizer_fingerprint_sha256` and `tokenizer_reported_name_or_path`.
- Skip path: after Task 6 state fingerprint check, added `_verify_existing_tokenizer_fingerprint` (local-only reload, version=2 check, SHA compare). Provenance equality now excludes the 3 tokenizer fields (verified separately). Skip only succeeds when both state and tokenizer checks pass.
- Did not modify `save_model_checkpoint` API (already accepted `tokenizer`).

### `mix_bit/validation.py`
- Added `_load_tokenizer_from_final_dir` and `_verify_final_tokenizer_fingerprint` (local-only reload, version=2 check, recompute + compare, wrapped as `ValueError`).
- `validate_mixed_model` verifies the tokenizer fingerprint before KL/downstream work and adds a `tokenizer` section to the report (`fingerprint_version`, `fingerprint_sha256`, `reported_name_or_path`, `local_reload_passed`).
- `_run_downstream_eval` now takes the tokenizer from the final dir instead of reloading via the profile adapter.

### `mix_bit/cli/assemble_mixed_model.py`
- Added `--access_token` and forwarded it to `assemble_optimal_mixed_checkpoint`; prints `tokenizer_fingerprint_sha256`.

### `mix_bit/README.md`
- Documented the fingerprint v2 contract, normalization helper, calibration manifest gate, assembler save/reload flow, skip contract, validation report, and CLI token.

### Tests
- `test_calibration.py`: upgraded `FakeTokenizer` for v2 (vocab/chat_template/added_vocab/special_tokens_map); replaced the name_or_path-based resume test with a content-based one; added 8 fingerprint v2 unit tests + 3 calibration resume tests (records v2, rejects legacy, rejects same-vocab-size changed core).
- `test_assembler.py` / `test_validation.py` / `test_tiny_integration.py`: added a `_TinyTokenizer` save/reload fixture + `_patch_tiny_tokenizer` monkeypatch (raises on `local_files_only` when the marker is missing, mimicking real HF) wired into the existing `assembled_world` / `validation_world` fixtures and the integration test. Added Step 3 tests (final save/reload fingerprint match, skip rejects missing tokenizer files, skip rejects v1 meta, skip rejects tampered content) and Step 4 validation negative tests (tokenizer section reported, tampered file fails on fingerprint mismatch not KL, v1 meta fails, missing file fails).

## Test summary

```
mix_bit/tests/test_calibration.py + test_assembler.py + test_validation.py + test_tiny_integration.py: 71 passed
mix_bit/tests/ (full suite): 299 passed
```

Old-code-failure confirmation (Step 5): the new fingerprint-v2 and resume-v2 tests were added before the implementation was wired in; they failed against the legacy `_tokenizer_config_payload` (no `get_vocab`/`backend_tokenizer`, name_or_path in digest, no version field) and only passed after the calibration module was upgraded.

## Concerns

- The tiny-tokenizer fixture is duplicated across `test_assembler.py` / `test_validation.py` / `test_tiny_integration.py` to keep scope to Task 9 files; a future task could fold it into a shared `conftest.py`.
- `save_model_checkpoint` already accepted `tokenizer`; no API change was needed there, matching the brief's "不得修改 `save_model_checkpoint` API" constraint.
- The skip-path provenance comparison now excludes the 3 tokenizer fields (verified separately by `_verify_existing_tokenizer_fingerprint`) so the base provenance equality check stays meaningful.

## Report path

`/home/shaoyuantian/program/VAELLM/.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-9-report.md`

---

## Fix round 1/5

### Finding addressed
Brief Step 1 requires the `backend_tokenizer.to_str()` priority path to be covered. All prior fingerprint unit tests used `FakeTokenizer` without `backend_tokenizer`, so only the `sorted_vocab` fallback was exercised — production Qwen fast tokenizers use `backend_tokenizer_json`.

### Changes
- `mix_bit/tests/test_calibration.py`: added `_FakeBackend` (mimics `backend_tokenizer.to_str()`) and `_BackendTokenizer` (subclass exposing `backend_tokenizer`). Added two focused tests:
  - `test_tokenizer_fingerprint_v2_backend_tokenizer_json_path_is_used_and_stable`: asserts `core_kind == "backend_tokenizer_json"`, identical backend JSON → identical hash, different backend JSON → different hash.
  - `test_tokenizer_fingerprint_v2_backend_path_wins_over_get_vocab`: two tokenizers with the same backend JSON but different `get_vocab` content must hash equally (backend path wins).
- Cheap cleanups (trivial, in-scope):
  - `mix_bit/assembler.py`: removed dead `source_payload = build_tokenizer_fingerprint_payload(...)` line and the now-unused `build_tokenizer_fingerprint_payload` import (kept `TOKENIZER_FINGERPRINT_VERSION` and `compute_tokenizer_config_sha256`).
  - `mix_bit/calibration.py`: removed the redundant inner `from pathlib import Path` inside `_normalize_json_value` (Path is already imported at module top).

### Test summary
```
mix_bit/tests/test_calibration.py: 24 passed
focused (calibration + assembler + validation + tiny_integration): 73 passed
```
No linter errors. No git commit.

### Concerns
None.
