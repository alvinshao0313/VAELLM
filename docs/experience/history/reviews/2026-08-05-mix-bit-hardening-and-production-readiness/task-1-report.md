> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-1-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1 Report: Shared Candidate Mode Contract

## Status

**DONE**

## Summary

Implemented dependency-light `mix_bit/candidate_contract.py` with three public functions and a full TDD test suite in `mix_bit/tests/test_candidate_contract.py`. All 20 focused tests pass; regression with `test_candidate_space.py` passes (32 total).

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `mix_bit/candidate_contract.py` | Created | Mode payload parsing, metadata validation, module spec contract |
| `mix_bit/tests/test_candidate_contract.py` | Created | 20 tests per brief (7 parsing + 13 module contract) |

## Implementation Notes

### `candidate_mode_from_payload`

- Requires `Mapping` with exactly five keys: `name`, `nominal_bit`, `codebook_bits`, `codebook_dim`, `residual_stages`.
- Converts via `str`/`float`/`int`; rejects empty name, non-finite nominal bit, values `< 1`.
- Enforces `abs(nominal_bit - residual_stages * codebook_bits / codebook_dim) <= 1e-12`.
- Returns existing `CandidateMode`; ignores unrelated metadata keys.

### `validate_mode_payload`

- Compares raw payload fields against `expected` with 1e-12 nominal bit tolerance.
- Error format: `{label}: {field} mismatch: actual=... expected=...`

### `validate_module_spec_mode_contract`

- Validates `parallel_parts` (default 1), `residual_stages`, `codebook_dim`, `stage_codebook_dims` (no single-element replication).
- **S2 (stages > 1):** requires `stage_vq_weights` / `stage_decoders` with correct stage and part nesting.
- **S1 (stages == 1):** validates legacy `vq_weights` / `decoders`; optional stage fields must match stage-0 legacy when present.
- Each VQ spec passed through `validate_bitpack_u8_spec`; checks `logical_shape[-1] == mode.codebook_bits`.
- Each decoder spec checks `in_dim == codebook_bits`, `out_dim == codebook_dim`.

### Dependencies

- Imports: `litebsq.bitpack.validate_bitpack_u8_spec`, `mix_bit.schema.CandidateMode`
- Does **not** import `candidate_pool.py` or `checkpoint_pool.py`.

## TDD Evidence

### RED (Step 4) — module missing

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest mix_bit/tests/test_candidate_contract.py -q
```

```
==================================== ERRORS ====================================
__________ ERROR collecting mix_bit/tests/test_candidate_contract.py ___________
ImportError while importing test module '.../mix_bit/tests/test_candidate_contract.py'.
...
E   ModuleNotFoundError: No module named 'mix_bit.candidate_contract'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.08s
```

### GREEN (Step 6) — all contract tests pass

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest mix_bit/tests/test_candidate_contract.py -q
```

```
....................                                                     [100%]
20 passed in 1.19s
```

### Regression (Step 7)

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_candidate_space.py \
  mix_bit/tests/test_candidate_contract.py -q
```

```
................................                                         [100%]
32 passed in 1.50s
```

## Self-Review

| Check | Result |
|-------|--------|
| All 20 named tests implemented verbatim | ✓ |
| Type annotations on new functions | ✓ |
| Error messages include label, field, actual, expected | ✓ |
| No candidate_pool/checkpoint_pool imports | ✓ |
| S1 legacy + S2 single/parallel paths covered | ✓ |
| `validate_bitpack_u8_spec` used for VQ specs | ✓ |
| No git commit (per AGENTS.md) | ✓ |

### Minor deviation

- `validate_mode_payload` compares raw payload fields directly (not via `candidate_mode_from_payload`) so per-field mismatch tests can target individual fields without tripping the derived nominal-bit check first. This matches the brief's listed comparison semantics.

## Concerns

None blocking. Module is ready for Task 2 wiring into export/pool/resume.

## Commit

Skipped intentionally per AGENTS.md / task overrides.
