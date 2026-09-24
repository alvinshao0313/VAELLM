> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-8-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 8 Review: Update Experiment Scripts, Preserve Defaults

**Reviewer:** Codex (subagent)  
**Date:** 2026-08-10  
**Scope:** `task-8-review-package.diff` vs `task-8-brief.md`

---

## Spec Compliance

| # | Requirement | Verdict | Evidence |
|---|-------------|---------|----------|
| 1 | `scripts/catlora_distill_4gpu_res0.sh` 在 distill loss 参数附近显式增加 `--distill_prompt_kd_weight "default=0.0"` | ✅ | Line 55, immediately after `--distill_loss_alpha "default=0.5"`, same override-table style as sibling distill flags |
| 2 | `compressed_e2e_fintuning/scripts/e2e_decoder.sh` 的 DP 和 layer_mp 两个分支都显式增加 `--prompt_kd_weight 0.0` | ✅ | DP branch line 77; layer_mp branch line 134; both after `--distill_alpha 0.5` |
| 3 | 不默认改成 0.05/0.1；只增加能力，不改变现有实验 | ✅ | Both scripts use `0.0`; no non-zero prompt-KD weight introduced |
| 4 | 按 `AGENTS.md` 不新增 shell 中间变量包装该超参数 | ✅ | Values are literals on the `python`/`torchrun` command line; no `PROMPT_KD_*` or similar vars added |

**Spec: ✅ (4/4)**

---

## Behavior Change Check

Explicit CLI values match existing Python defaults, so runtime behavior is unchanged:

| Script | CLI | Python default | Equivalent? |
|--------|-----|----------------|---------------|
| `catlora_distill_4gpu_res0.sh` | `--distill_prompt_kd_weight "default=0.0"` | `cat_train_args.py` argparse `default="default=0.0"` | ✅ |
| `e2e_decoder.sh` (both branches) | `--prompt_kd_weight 0.0` | `compressed_e2e_fintuning/args.py` `default=0.0` | ✅ |

With `prompt_kd_weight == 0.0`, `distill_losses.py` skips prompt-region KD weighting (existing no-op path). Passing `0.0` explicitly does not enable new loss terms.

**Behavior change: None observed.**

---

## Quality

| Aspect | Verdict | Notes |
|--------|---------|-------|
| Diff scope | ✅ | Only the two files named in the brief; +3 lines each file |
| Placement | ✅ | Adjacent to related distill loss flags in both scripts |
| Convention match | ✅ | catlora uses `"default=…"` override syntax; e2e uses plain float like `--distill_alpha` |
| AGENTS.md shell rules | ✅ | No new hyperparameter shell vars; no `conda activate` / `conda run` added |
| Pre-existing shell vars in e2e | ✅ (N/A) | `SEED`, `MAX_STEPS`, etc. predate this task; new flag correctly not wrapped |
| Report accuracy | ✅ | `task-8-report.md` matches the diff and verification claims |

**Quality: ✅ Good** — minimal, convention-aligned wiring with no scope creep.

---

## Concerns

None blocking. Optional note for operators: to experiment with prompt KD, change the literal on the command line (e.g. `"default=0.05"` or `--prompt_kd_weight 0.05`); defaults remain off.

---

## Verification Performed

- Read `task-8-brief.md`, `task-8-report.md`, `task-8-review-package.diff`
- Read current `scripts/catlora_distill_4gpu_res0.sh` and `compressed_e2e_fintuning/scripts/e2e_decoder.sh`
- Cross-checked Python defaults in `train_utils/cat_train_args.py` and `compressed_e2e_fintuning/args.py`
- Confirmed loss no-op at `0.0` in `train_utils/distill_losses.py`

No runtime execution (shell-only CLI exposure; defaults unchanged).

---

## Verdict

**Approve.** Task 8 meets all brief requirements: explicit `0.0` defaults in both scripts, both e2e branches covered, no shell variable wrapping for the new hyperparameter, and no effective behavior change at default settings.
