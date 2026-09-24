> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/final-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Final Whole-Branch Review — Prompt-Weighted KD

**Reviewer:** Senior Code Reviewer (read-only)
**Base:** `b41153cb59bd6ce894c80e493a67d9d4ab7a4fa5`
**Head:** uncommitted working tree (commits forbidden by plan / AGENTS.md)
**Diff:** `.superpowers/sdd/2026-08-08-add-prompt-kd-weight/final-review-package.diff`
**Scope:** 17 tracked files, +852 / −117; plus 1 gitignored local-only test fix.

## Verdict

**Merge-ready.** No Critical or Important issues. One Minor issue worth a conscious decision (gitignored regression fix). All Acceptance Criteria satisfied per `task-10-report.md` and re-verified by this review against the live tree.

---

## Strengths

1. **Minimal, surgical core change.** `build_distill_token_mask` extends the binary causal mask to a float32 weighted mask in one place. At `prompt_kd_weight == 0.0` the labels branch returns `labels.ne(-100).to(float32)` with no attention-mask check — byte-for-byte the legacy path. Exact-backward-compat requirement is met by construction, not by approximation.

2. **Single source of truth per trainer.**
   - `train_utils/lora_training.py`: one local `build_token_mask(reference_logits)` closure; all 16 tokenwise KD branches call it (verified via `rg` — 16 call sites, no surviving direct `build_distill_token_mask` call in a KD branch).
   - `compressed_e2e_fintuning/trainer.py`: one private `_build_distill_token_mask`; dense, CPU-student, and CPU-teacher-gamma paths all route through it (3 call sites, lines 460/607/698).
   - No `labels.ne(-100)` / `labels.eq(-100)` KD-mask construction remains in either trainer. The only `shift_labels.ne(-100)` left in `trainer.py:172` is the MCQA **choice-score** path (not a tokenwise KD mask), which is correctly untouched.

3. **Padding precedence is correct at p>0.** `response_validity` and `prompt_validity` are both AND-ed with `attention_validity`, so padding is forced to 0 regardless of label content. At p=0 the attention mask is deliberately not consulted, preserving legacy semantics (documented trade-off, not a bug).

4. **EAKLD gamma and KL share the same weighted mask.** `_build_cpu_teacher_targets` builds `gamma_mask` via the same private helper that the KL path uses, so entropy/gamma and the KL term see identical fractional weights. Fractional-mask dense-vs-CPU value+gradient smoke + unit tests pass.

5. **Strong TDD evidence.** +497 lines in `tests/test_distill_losses.py` covering: p=0 exact-equiv, fractional shift `[0.1,0.1,1,1,1,0]`, padding exclusion, p=1 == shifted attention validity, interleaved/multi-turn prompts, `labels=None` fallback (attention-only and no-metadata), negative rejection, >1 acceptance, gradient isolation (prompt-only grad zero at p=0, nonzero at p>0, padding/final always zero), manual weighted-mean numerics for forward KL, EAKLD fractional telemetry = `mask.sum()`, EAKLD-topK fractional dense match, CPU-teacher EAKLD/EAKLD-topK fractional dense-vs-chunk match.

6. **CE / hidden / pre-MLP untouched.** Trainer CE branches keep original response-only labels; hidden and pre-MLP alignment continue to use attention masks, not the weighted KD mask. `kd` / `kd_top*` / `dual_kd*` / `eakld_kd` CE terms keep the original `alpha` mix; only the KD term is reweighted.

7. **CLI chains complete and validated.**
   - Category: `--distill_prompt_kd_weight` (OverrideTable, `default=0.0`) → `NormalizedCatArgs` → `resolve_distill_runtime_config` → `_ResolvedDistillStageConfig` → `_build_lora_trainer` → `CustomSFTTrainer.__init__` (rejects `<0`). Negative rejected at parse; 2.0 accepted; after-category override resolves.
   - E2E: `--prompt_kd_weight` (float, `default=0.0`) → `validate_args` (rejects `<0`; rejects `!=0` for `mcqa`) → runtime log + `prompt_kd_weight=float(...)` into `VAEDecoderE2ETrainer.__init__` (rejects `<0`).

8. **Defaults preserved in scripts.** `scripts/catlora_distill_4gpu_res0.sh` adds `--distill_prompt_kd_weight "default=0.0"`; `e2e_decoder.sh` adds `--prompt_kd_weight 0.0` to both `dp` and `layer_mp` branches. No shell-variable wrapping of the hyperparam (AGENTS.md compliant).

9. **Docs honest.** `docs/cat_train_args.md` §6.11.1 and `compressed_e2e_fintuning/README.md` both state 0.0 = current behavior, 0.05/0.1 are experimental examples only (not recommended/validated), EOS target stays 1.0, padding/final logits stay 0, EAKLD gamma+KL share the mask, CE/hidden unchanged.

10. **No commits.** Policy respected.

---

## Issues by Severity

### Critical
None.

### Important
None.

### Minor

**M1. Gitignored regression fix is not committable (conscious decision needed).**
`tests/test_cat_compressed_lora_scope.py` is untracked and matches `.gitignore:62` (`tests/`). The patch made `prompt_kd_weight` a **required** field of `_ResolvedDistillStageConfig` (no default), which breaks `_fake_cfg()` in that file with `TypeError: missing 1 required positional argument`. The implementer correctly patched it locally (`prompt_kd_weight=0.0`), and the file is gitignored by repo convention, so the fix will **not** appear in any future commit. Anyone who locally has that file and runs `pytest tests -q` will pass; anyone without it is unaffected. The other test files in the diff are tracked (force-added before the `tests/` rule), so they *do* show as modified — the asymmetry is a pre-existing repo convention, not introduced here.
- Risk: low (file isn't part of the tracked suite).
- Suggested resolutions (pick one, none are blocking):
  1. Give `_ResolvedDistillStageConfig.prompt_kd_weight` a dataclass default of `0.0` so future untracked test fixtures don't break. Cleanest, but slightly weakens "required" intent.
  2. `git add -f tests/test_cat_compressed_lora_scope.py` to track it. Changes repo convention.
  3. Accept as local-only and move on (current state).

**M2. `valid_tokens` telemetry becomes fractional at p>0.**
The plan explicitly permits this ("effective token-weight mass, 即 `mask.sum()`, 允许非整数"), and the telemetry key name and checkpoint schema are unchanged. Any downstream consumer that casts `valid_tokens` to int (logging, dashboards, LR schedulers keyed on token count) could be surprised. No such consumer was found in-scope, but a quick audit of telemetry readers would be prudent before a non-zero `prompt_kd_weight` is used in a real run. Not a blocker for merge; flag for the first non-zero experiment.

**M3. `prompt_kd_weight == 0.0` does not enforce attention-mask padding.**
Intentional for exact legacy parity (a padding position with a leaked non-`-100` label still gets weight 1.0 at p=0). At p>0 the attention mask is enforced. This asymmetry is documented in the plan but not in user-facing docs. Consider adding a one-liner to `docs/cat_train_args.md` §6.11.1 noting p=0 preserves legacy label-only semantics and only p>0 enforces padding via attention mask. Cosmetic.

**M4. E2E runtime log uses `getattr(args, "prompt_kd_weight", 0.0)`.**
The attribute is always set by argparse, so the `getattr` fallback is defensive dead code. It mirrors the existing `hidden_loss_weight` log line style directly above, so it's consistent with local convention. No action needed; noted for completeness.

---

## Plan Alignment

| Plan section | Status |
|---|---|
| Global Constraints | All met: only prompt KD added; no truncation/filter/data-ratio/loss-type/ckpt/inference changes; both CLIs default 0.0; `>=0` enforced, no upper bound; response fixed 1.0; CE/hidden/pre-MLP untouched; EOS stays response; TDD followed; no worktree/branch/commit; tests in `bitvae`. |
| Required Mathematical Semantics | Verified against tests: target-position weights → left-shift to logits → last logit 0; `KD = sum(w * per_token_KD) / max(sum(w),1)`; `labels=None` fallback unchanged. |
| Required examples | `[-100,-100,-100,A,B,EOS]` p=0→`[0,0,1,1,1,0]`, p=0.1→`[0.1,0.1,1,1,1,0]`, p=1→`[1,1,1,1,1,0]` all covered by tests. Padding example covered. |
| Tasks 1–10 | All marked complete in ledger; re-verified branch coverage and chain audit against live tree. |
| Acceptance Criteria | All 14 criteria PASS per `task-10-report.md`; this review independently re-confirmed the structural ones (CLI presence/defaults, mask equiv, branch coverage, EAKLD gamma+KL mask sharing, CE/hidden isolation, script defaults, diff scope). |

## Verification Evidence (re-checked)

- `rg build_distill_token_mask train_utils/lora_training.py` → 1 import + 1 closure def + 16 call sites. ✓
- `rg build_distill_token_mask compressed_e2e_fintuning/trainer.py` → 1 import + 1 private helper def + 3 call sites (dense, CPU student, CPU gamma). ✓
- No surviving `labels.ne(-100)` / `labels.eq(-100)` KD-mask construction in either trainer. ✓
- `tests/test_cat_compressed_lora_scope.py` gitignored & untracked; local `_fake_cfg` fix present at line 128. ✓
- Other test files tracked (show as `M`), so their additions will be committed when the user chooses to commit. ✓
- Test results (from `task-10-report.md`, not re-run per instructions): 51 / 23 / 39(+9 known) / 3 / 5; combined 121 passed + same 9 known fail; full suite after local fix 246 passed + same 9 known fail. The 9 `test_e2e_dataset_mix.py` failures are pre-existing (`FileNotFoundError: dummy.txt` + eval_before_save), unrelated to this patch.

## Merge Readiness

**Ready to merge.** No blocking issues. The only follow-up is a decision on M1 (gitignored test file) — acceptable to defer since it doesn't affect the tracked suite. Recommend the user, when they choose to commit, also decide whether to give `prompt_kd_weight` a dataclass default (M1 option 1) to harden against future untracked fixtures. No commits made by this review.
