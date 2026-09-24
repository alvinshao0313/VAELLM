> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-8-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 8 Review: Thin Smoke/Formal Shell Scripts

**Reviewer:** spec + quality  
**Date:** 2026-08-19  
**Verdict:** **Spec ✅** · **Approved**

---

## Spec Checklist

| Requirement | Result | Evidence |
|-------------|--------|----------|
| `run_smoke.sh` matches brief Step 1 verbatim | ✅ | Byte-identical to brief lines 14–27: `GPUS="${GPUS:-0}"`, `--mode smoke`, fixed `CHECKPOINT_DIR` / `OUTPUT_DIR` |
| `run_formal.sh` matches brief Step 2 verbatim | ✅ | Byte-identical to brief lines 37–50: `GPUS="${GPUS:-0,1,2,3}"`, `--mode formal`; only intentional diff from smoke is GPU default + mode |
| No `conda activate` / `conda run` inside `.sh` | ✅ | Grep over `scripts/` — no matches |
| Path vars only (no scientific CLI shell vars) | ✅ | Only `CHECKPOINT_DIR`, `OUTPUT_DIR`, `GPUS`; all other args inline on `python` command |
| `GPUS` defaults correct | ✅ | smoke → `0`; formal → `0,1,2,3`; both overridable via env |
| README exact usage block | ✅ | Lines 9–12 match brief Step 3 exactly (including `conda activate bitvae` in README, not in shell) |
| README explains GPU = independent worker, not DDP | ✅ | Line 15: 每 GPU 对应独立 MMLU worker，非 DDP |

---

## Quality Checklist

| Check | Result | Notes |
|-------|--------|-------|
| Shell logic minimal | ✅ | No job scheduling, parsing, or result handling |
| `set -euo pipefail` | ✅ | Both scripts |
| `export PYTHONPATH=.` | ✅ | Both scripts |
| Executable bit | ✅ | `-rwxrwxr-x` on both scripts |
| `bash -n` syntax | ✅ | Both pass |
| AGENTS.md shell rules | ✅ | No conda in `.sh`; fixed hyperparams on CLI; path vars only |

---

## Files Reviewed

- `experiments/down_layer_sensitivity/scripts/run_smoke.sh` — create, matches brief
- `experiments/down_layer_sensitivity/scripts/run_formal.sh` — create, matches brief
- `experiments/down_layer_sensitivity/README.md` — updated「运行方式」section

---

## Non-blocking Notes

- End-to-end smoke/formal runs not executed (acceptable; not required by brief).
- Operator must ensure `.result/catlora/res0-bf16-protect-channel-vae/final_model` exists before running.

---

## Outcome

**Spec ✅**  
**Approved** — no changes requested.
