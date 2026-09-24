> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-6-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6 Report: Replace Full State Cloning with Streaming State Fingerprints

## Status
Complete. All Task 6 tests pass; no regressions in existing assembler/tiny-integration suites.

## Commits
None (per workspace rule: no auto git commit).

## Files changed
- Created `mix_bit/state_fingerprint.py` — streaming SHA256 fingerprint module:
  - `STATE_FINGERPRINT_KIND`, `STATE_FINGERPRINT_CHUNK_BYTES`, `STATE_FINGERPRINT_FILENAME` constants.
  - `fingerprint_tensor` — rejects non-strided / non-contiguous tensors, hashes a canonical header (dtype/shape/numel) plus raw bytes in bounded CPU chunks; reinterprets each chunk as `uint8` so bfloat16 is handled without `.numpy()` casting; never calls `.clone()`.
  - `fingerprint_model_state` — iterates `model.state_dict()` once, fingerprints each tensor, returns a manifest holding only strings/integers.
  - `compare_state_fingerprints` — raises `ValueError` on kind/key-count/key-set/dtype/shape/numel/sha256 mismatch.
  - `write_state_fingerprint_manifest` — atomic canonical-JSON write, returns absolute path.
  - `verify_saved_checkpoint_state` — loads the base model via the profile adapter, applies the saved checkpoint with `strict=True`, fingerprints the reloaded state, compares against the expected manifest, and verifies every expected converted module is a `VAELinear` with `original_weight is None`; finally deletes the model, `gc.collect()`, and `torch.cuda.empty_cache()` when CUDA is available.
- Created `mix_bit/tests/test_state_fingerprint.py` — 17 tests covering float32/bfloat16/uint8/bool/int64/zero-length tensors, non-contiguous + sparse rejection, bounded chunk equivalence, single-value/dtype/shape/key-missing/extra-key/kind mutation detection, identical-model round trip, atomic manifest write, and a no-clone regression (monkeypatching `torch.Tensor.clone` to raise).
- Modified `mix_bit/assembler.py`:
  - `save_full_checkpoint_from_assignments` now fingerprints the source model after `save_model_checkpoint` returns (after the temporary decoder pack/unpack context exits), writes `<final_model>/state_fingerprint.json`, deletes the model + gc/cuda cache, then calls `verify_saved_checkpoint_state`. Removed the `reference_state` dict, the per-key `torch.testing.assert_close` loop, and the manual reload block. Return payload adds `state_fingerprint`.
  - `assemble_optimal_mixed_checkpoint` skip path now requires `pytorch_model.bin`, `checkpoint_meta.json`, `reference_logits.pt`, and `state_fingerprint.json` to all exist, reads + validates the fingerprint manifest kind, derives expected converted module names from allocation entries, and calls `verify_saved_checkpoint_state`; any missing file / bad manifest / strict-load failure / fingerprint mismatch raises `ValueError` with an `overwrite=True` hint. Skip return payload adds `state_fingerprint`.
  - Dropped the now-unused `load_checkpoint_into_model` import.
- Modified `mix_bit/tests/test_assembler.py` — added 4 tests: fingerprint manifest written + payload path + key-count equals reload state key count + source inspection (no `reference_state`, no `.cpu().clone()`, no `assert_close`); skip rejected when `state_fingerprint.json` is missing; skip rejected when manifest kind is wrong or an entry sha256 is corrupted; skip returns `skipped_identical=True` with the fingerprint path when all four files are valid and the strict reload hash matches.
- Modified `mix_bit/tests/test_tiny_integration.py` — asserts `state_fingerprint.json` exists and `assembled["state_fingerprint"]` points at it.

## Test summary
```
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_state_fingerprint.py \
  mix_bit/tests/test_assembler.py \
  mix_bit/tests/test_tiny_integration.py -q
46 passed in 12.19s
```
Breakdown: 17 new fingerprint tests + 25 existing assembler/tiny tests + 4 new assembler skip/manifest tests = 46.

## Concerns
- The fingerprint algorithm requires strided + contiguous tensors; `model.state_dict()` for the toy/VAELinear modules returned contiguous tensors in every test. If a future model exposes non-contiguous state_dict views, `fingerprint_tensor` will raise rather than silently materialize a contiguous copy (intentional per the brief).
- `verify_saved_checkpoint_state` loads the base model through the profile adapter, so it depends on the same `get_model` patch surface the existing reload tests already use (`train_utils.model_checkpoint_io.get_model` and `rotation.model_utils.get_model`); no new mock surface was introduced.
- The skip path now performs a full strict reload + fingerprint comparison on every identical-provenance hit, which is more expensive than the previous metadata-only skip; this is the contract the brief specifies (Task 9 will additionally add tokenizer verification on the same path).

## Report path
/home/shaoyuantian/program/VAELLM/.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-6-report.md
