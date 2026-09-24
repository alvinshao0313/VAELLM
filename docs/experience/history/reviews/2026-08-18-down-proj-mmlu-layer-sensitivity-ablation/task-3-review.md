> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-3-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3 Review: Per-GPU Worker and Self-Describing Jobs

**Reviewer scope:** Task-scoped gate (spec compliance + build quality).  
**Evidence:** `task-3-brief.md`, `task-3-report.md`, `task-3-review-package.diff` only. No working-tree mutation, no test re-run.

---

## Spec Compliance: ✅

Implementation matches the Task 3 brief on all binding requirements. No Missing, Extra (beyond allowed scope), or Misunderstood semantics found in the diff.

### Requirements checklist (verified against diff)

| Requirement | Verdict | Evidence |
|-------------|---------|----------|
| New code only under `experiments/down_layer_sensitivity/` | ✅ | `worker.py`, `tests/test_job_manifest.py` |
| No production file modifications | ✅ | Diff contains only experiment files |
| Fixed CLI (6 flags only) | ✅ | `parse_args`: checkpoint_dir, manifest_path, jobs_dir, worker_meta_path, worker_id, physical_gpu_id |
| Manifest + CLI identity check | ✅ | `validate_manifest` lines 49–57 |
| Validation before model load | ✅ | `main`: validate → seeds → load model |
| Reject duplicate `job_id` | ✅ | lines 72–74 |
| Reject restore layer outside 0..35 | ✅ | lines 97–106 |
| Reject duplicate layer in restore list | ✅ | lines 101–108 |
| Reject mode not in {smoke, formal} | ✅ | manifest + per-job mode checks |
| Reject formal job with `lm_limit is not None` | ✅ | lines 84–86 |
| Reject smoke job with `lm_limit != 2` | ✅ | lines 87–88 |
| Seed 31 once before model/tokenizer load | ✅ | `_set_inference_seeds` after validation, before `load_worker_model` |
| No deterministic-algorithm / cuDNN / TF32 changes | ✅ | Only four seed calls; no `use_deterministic_algorithms` |
| Load model exactly once | ✅ | Single `load_worker_model` call |
| Build tokenizer once | ✅ | Single `build_tokenizer` call |
| Weight metrics only when `write_weight_metrics=true` | ✅ | lines 244–250 |
| Canonical filename `weight_metrics_worker.json` | ✅ | `WEIGHT_METRICS_FILENAME` constant |
| Worker metadata recorded (brief field list) | ✅ | `_collect_worker_metadata` |
| Runtime sequence: load → tokenizer → metrics? → meta → jobs | ✅ | `main` order matches brief Step 3 |
| Per-job reset / assert / eval / finally-reset | ✅ | `_execute_job` matches brief Step 4 exactly |
| try/finally only for final reset; eval exception not swallowed | ✅ | `finally` in `_execute_job`; outer loop re-raises via `SystemExit(1)` |
| Exact per-job JSON fields | ✅ | lines 191–205 — all 14 fields present |
| No model state / raw logits in job JSON | ✅ | Only eval-derived scalars and subject_metrics |
| Fail worker on first failed job; traceback + non-zero exit | ✅ | lines 277–279 |
| Unit tests for listed invalid schemas + valid manifests | ✅ | 20 parametrized cases in diff |
| No git commit | ✅ | Report + diff scope confirm |

### Allowed extras (not spec violations)

- **Job mode must match manifest mode** — aligns with global “mode mismatch” constraint; fail-loud, not silent fallback.
- **Strict `write_weight_metrics` bool** — rejects JSON `1`/`"true"`; appropriate for manifest contract.
- **`restore_layers` must be list; entries must be `int`** — prevents ambiguous restore sets.
- **`PREWARM_GROUP_SIZE = 8`** — not spelled out in Task 3 brief text, but matches Task 4 experiment constant; correct for this run family.

### Notes on report claims (independently verified)

- **20 tests:** Parametrize expansion in diff yields 20 cases; count matches report. Pass count not re-run (implementer reported 20/20).
- **TDD narrative:** Plausible; tests import `validate_manifest` without requiring GPU.
- **No production edits:** Confirmed from diff file list.

---

## Artifact Path Verification (implementer concerns)

### 1. `weight_metrics_worker.json` path

**Brief:** Step 3 names the file `weight_metrics_worker.json` but does **not** specify a directory.

**Implementation:** `os.path.join(dirname(abspath(jobs_dir)), "weight_metrics_worker.json")`.

**Verdict: ✅ Spec-compliant.**

With Task 4 launch shape `--jobs_dir <phase_dir>/jobs`, the resolved path is `<phase_dir>/weight_metrics_worker.json`. This is the correct placement:

- Keeps the artifact out of `jobs/`, so Task 5 job inventory (38+W job IDs under `jobs/`) will not treat it as an MMLU result.
- Filename matches the brief’s canonical name exactly.
- Task 5 “Merge canonical weight metrics by `layer_idx`” should read this fixed path (document in Task 5, not a Task 3 defect).

### 2. Per-job filename `{job_id}.json`

**Brief:** “one JSON result per job under `phase*/jobs/`” — no explicit filename pattern.

**Implementation:** `jobs_dir/{job_id}.json`.

**Verdict: ✅ Spec-compliant and downstream-compatible.**

Task 5 inventory is keyed by `job_id` strings (`restore_L00`, `compressed_baseline_worker00`, etc.). Stem-equals-`job_id` naming is the natural convention; Task 5 can glob `jobs/*.json` and validate by embedded `job_id` or filename stem. No brief conflict.

---

## ⚠️ Cannot verify from diff alone

1. **Test execution:** 20/20 pass claimed in report; not re-run in this review.
2. **End-to-end worker on GPU:** Job loop, MMLU eval, weight-metrics write, and metadata collection require CUDA + checkpoint; out of Task 3 unit-test scope (brief Step 7 is manifest-only).
3. **`load_worker_model(..., prewarm_group_size=8)` on real checkpoint:** Integration behavior deferred to formal run.
4. **Cross-task contract:** Task 4/5 must adopt the de-facto paths above; brief silence is a documentation gap, not an implementation error in Task 3.

---

## Code Quality Findings

### Critical

None.

### Important

None.

### Minor

1. **Extra manifest validations untested** — Job/manifest mode mismatch, non-bool `write_weight_metrics`, non-list `restore_layers`, non-int layer entries are implemented but not covered by `test_job_manifest.py`. Brief Step 7 lists only the six invalid-schema families; optional coverage for Task 4 manifest builders.

2. **`test_mode_not_smoke_or_formal_raises` couples manifest and job mode** — Valid for the parametrized cases; does not isolate “valid manifest mode + invalid job mode”. Low risk given symmetric check at lines 79–88.

3. **`physical_gpu_id` type not enforced on manifest value** — Comparison is strict equality; Task 4 manifests must use string GPU IDs (as in brief example `"0"`). Coordinator responsibility.

4. **Unknown extra manifest/job fields not rejected** — Not listed in Task 3 brief Step 1; user global “unknown fields” constraint is “per brief”. Acceptable for this task; optional hardening if later tasks require strict schemas.

5. **`PREWARM_GROUP_SIZE` duplicated in worker** — Will also live in `run.py` (Task 4). Duplication is consistent today; future drift risk is a cross-task maintenance note only.

---

## Task Quality Verdict

**Approved**

Worker CLI, manifest validation gate, seed-once semantics, single load path, per-job state reset contract, result JSON schema, and fail-fast job loop all match the brief. Weight-metrics and job output paths are reasonable, spec-consistent interpretations where the brief is silent; downstream tasks should treat `<phase_dir>/weight_metrics_worker.json` and `<phase_dir>/jobs/{job_id}.json` as the canonical contracts. Residual items are minor test/documentation notes, not blockers.

---

## Summary

| Gate | Result |
|------|--------|
| **Spec** | ✅ |
| **Verdict** | **Approved** |
