> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-8-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 8: Add Thin Smoke/Formal Shell Scripts

**Files:**
- Create: `experiments/down_layer_sensitivity/scripts/run_smoke.sh`
- Create: `experiments/down_layer_sensitivity/scripts/run_formal.sh`

## `run_smoke.sh`

- [ ] **Step 1: Keep shell logic minimal**

Use:

```bash
#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

CHECKPOINT_DIR=".result/catlora/res0-bf16-protect-channel-vae/final_model"
OUTPUT_DIR=".result/experiments/down_layer_sensitivity"
GPUS="${GPUS:-0}"

python experiments/down_layer_sensitivity/run.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --gpus "${GPUS}" \
  --mode smoke
```

No conda activation inside shell, per project rules.

## `run_formal.sh`

- [ ] **Step 2: Formal script differs only in mode/default GPU list**

```bash
#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

CHECKPOINT_DIR=".result/catlora/res0-bf16-protect-channel-vae/final_model"
OUTPUT_DIR=".result/experiments/down_layer_sensitivity"
GPUS="${GPUS:-0,1,2,3}"

python experiments/down_layer_sensitivity/run.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --gpus "${GPUS}" \
  --mode formal
```

`GPUS` is intentionally overridable because GPU availability is machine-specific; scientific settings are not shell-overridable.

- [ ] **Step 3: Document usage in README**

README must include exactly:

```bash
conda activate bitvae
GPUS=0 bash experiments/down_layer_sensitivity/scripts/run_smoke.sh
GPUS=0,1,2,3 bash experiments/down_layer_sensitivity/scripts/run_formal.sh
```

Also explain one GPU per independent MMLU job worker, not DDP.

---

