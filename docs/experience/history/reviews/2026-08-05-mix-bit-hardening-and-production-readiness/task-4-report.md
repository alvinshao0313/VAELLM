> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-4-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4 Report: Build Teacher Top-k Cache Without Full-Logits CPU Transfer

## Status
Complete. Device-side top-k implemented; full-logits CPU transfer removed.

## Commits
None (per user override — no git commit performed).

## Files Changed
- `mix_bit/teacher_cache.py`
  - `build_teacher_topk_chunk`: rewrote extraction to run `torch.topk` on the
    original device, then move only compact `[N_valid, K]` indices/probs and
    `[B+1]` offsets to CPU. Preserved all existing input validation
    (shape/dtype/sample_ids/K/dtype/zero-valid checks) and the cache schema
    (fields, shapes, order, no tail fields).
  - `build_teacher_topk_cache`: call site now passes
    `shifted_teacher_logits=shifted.detach()` instead of
    `shifted.detach().cpu()`, so teacher logits stay on GPU until compact
    tensors are extracted.
- `mix_bit/tests/test_kl_metric.py`
  - Added test-only `dense_teacher_topk_reference` helper (float32 full-row
    top-k over valid positions).
  - `test_teacher_topk_chunk_matches_dense_reference_float32`: batch>1, varied
    valid lengths, float32 cache dtype, asserts CPU compact output and
    numerical equivalence to dense reference.
  - `test_teacher_topk_chunk_matches_dense_reference_bfloat16`: bfloat16
    cache dtype, indices match, probs within bf16 quantization tolerance.
  - `test_teacher_topk_only_returns_compact_cpu_tensors`: CUDA transfer
    boundary test (skipped without CUDA); asserts input on CUDA, outputs on
    CPU, `[N_valid, K]` shapes, equivalence to dense reference.
  - `test_teacher_cache_source_does_not_transfer_full_logits_to_cpu`: source
    guard using `inspect.getsource(build_teacher_topk_cache)`; asserts
    `shifted.detach().cpu()` and
    `shifted_teacher_logits=shifted.detach().cpu()` are absent.
- `mix_bit/tests/test_tiny_integration.py`
  - Added `test_tiny_teacher_topk_chunk_stays_compact_offline`: runs the toy
    model, builds a teacher top-k chunk, and asserts compact `[N_valid, K]`
    CPU tensors (integration guard for the device-side path).

## Test Summary
- Step 5 (red): source guard failed on old code
  (`shifted.detach().cpu()` present in `build_teacher_topk_cache`).
- Step 7 (green): focused teacher-cache tests all pass.
  - `mix_bit/tests/test_kl_metric.py`: 23 passed.
  - `mix_bit/tests/test_tiny_integration.py::test_tiny_teacher_topk_chunk_stays_compact_offline`: passed.
  - Combined run: 24 passed, 1 pre-existing unrelated failure.
- Pre-existing failure (NOT caused by Task 4):
  `test_tiny_offline_end_to_end_integration` fails in
  `candidate_contract.candidate_mode_from_payload` with
  `nominal_bit mismatch: actual=1.0 expected=8.0`. This is a candidate-mode
  fixture validation issue in `candidate_artifact.py` /
  `candidate_contract.py`, unrelated to teacher cache. Task 4 only touched
  `teacher_cache.py` and the two test files; the failing path does not import
  or use `build_teacher_topk_chunk` / `build_teacher_topk_cache`.
- Linter: no errors on the three edited files.

## Concerns
- The pre-existing `test_tiny_offline_end_to_end_integration` failure is out
  of scope for Task 4 and was not touched. It should be triaged separately
  (candidate mode `nominal_bit` vs `residual_stages * codebook_bits /
  codebook_dim` derivation).
- Numerical equivalence relies on no ties in teacher logits; tests use
  `torch.randn` float32 inputs as required by the brief. For bf16 model
  logits, `torch.topk` runs on bf16 (per brief spec), which could in
  principle differ from float32 topk on near-ties — this is the documented
  contract, not a regression.
- CUDA transfer-boundary test only runs when CUDA is available; on CPU-only
  machines it is skipped.

## Report Path
/home/shaoyuantian/program/VAELLM/.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-4-report.md

## Cross-task fixture fix

Status: Complete. Tiny-integration and three sibling fixtures now satisfy the
candidate-contract `nominal_bit == residual_stages * codebook_bits / codebook_dim`.

What changed (minimal fixture/test changes only; candidate_contract untouched):
- `mix_bit/tests/test_tiny_integration.py`: `_three_modes()` now uses
  codebook_bits=4, codebook_dim=4 with residual_stages 1/2/3 → nominal_bit
  1.0/2.0/3.0 (low/baseline/high). Mode name constants updated to
  `b4d4s1`/`b4d4s2`/`b4d4s3`. `_export_pool` passes
  `cdim=trial.mode.codebook_dim, residual_stages=trial.mode.residual_stages`
  to `_make_vae_linear` so the built VAE matches each trial's mode.
- `mix_bit/tests/test_validation.py`, `mix_bit/tests/test_assembler.py`,
  `mix_bit/tests/test_cost_table.py`: same pattern — modes switched to
  codebook_bits=4/codebook_dim=4 with residual_stages 1/2, names to
  `b4d4s1`/`b4d4s2`, and `_export_pool` passes the trial mode's structure.
  Hardcoded `"b16d4s2"` literals in test_assembler updated to `"b4d4s1"`.
- Scanned remaining fixtures: test_candidate_pool, test_checkpoint_pool,
  test_candidate_artifact, test_candidate_contract, test_solver,
  test_calibration already satisfy the contract (or intentionally violate it
  for rejection tests) — left untouched.

Test summary:
- Verify command (5 files): 84 passed.
- Verify + sibling export-path files (8 files): 147 passed.
- Linter: no errors on the four edited files.

