> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-9-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 9: Closed-Loop Verification Before the Expensive Formal Run

**Files:**
- No new production files. If verification exposes a real problem in an existing repository interface that blocks this experiment, stop and report the blocker; do not repair production code as part of this plan.

- [ ] **Step 1: Verify environment**

Run:

```bash
conda activate bitvae
which python
python -V
python -c "import torch, transformers, lm_eval; print(torch.__version__); print(transformers.__version__); print(getattr(lm_eval, '__version__', 'unknown'))"
```

Must use `bitvae` Python.

- [ ] **Step 2: Run all isolated unit tests**

```bash
pytest -q experiments/down_layer_sensitivity/tests
```

All pass before smoke.

- [ ] **Step 3: Run smoke on one GPU**

```bash
GPUS=0 bash experiments/down_layer_sensitivity/scripts/run_smoke.sh
```

Smoke uses `lm_limit=2`, requires exactly one GPU, and exactly four jobs in this order:

```text
compressed_baseline_worker00
compressed_baseline_worker00_repeat
restore_L00
all_down_original
```

- [ ] **Step 4: Smoke acceptance gates**

Require:

```text
- checkpoint loads successfully
- tokenizer loads from the same final checkpoint directory
- exactly 36 down VAELinear discovered
- all 36 down original weights present
- non-down unload statistics are recorded; modules protected from original-weight unloading are allowed
- all non-down VAELinear remain on temporary=True compressed path
- prewarm failed=0
- baseline worker00/repeat identical on smoke subset
- n_samples_total is derived from subject sample counts, not the top-level lm_eval n_samples object
- restore_L00 state assertion passes before/after evaluation
- all-down-original state assertion passes
- smoke outputs written under a `_smoke` run ID
- no formal CSV/report generated from smoke
```

Do not interpret smoke accuracy numerically.

- [ ] **Step 5: Inspect git diff before formal run**

Run:

```bash
git status --short
git diff -- experiments/down_layer_sensitivity docs/superpowers/plans/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation.md
```

Confirm production files are untouched. Existing unrelated `scripts/catlora_simple.sh` modification must remain untouched.

---

