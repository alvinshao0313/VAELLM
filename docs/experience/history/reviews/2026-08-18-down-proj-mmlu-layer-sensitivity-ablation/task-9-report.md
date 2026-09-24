> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-9-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 9 Report — BLOCKED (GPU memory)

## Status
BLOCKED on smoke/formal GPU availability. Code/unit-test portion passed.

## Step 1 Environment — PASS
- `which python`: `/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`
- Python 3.11.13
- torch 2.6.0+cu124
- transformers 4.51.0
- lm_eval version attr: unknown (import OK)

## Step 2 Unit tests — PASS
`pytest -q experiments/down_layer_sensitivity/tests` → **77 passed** in 6.11s

## Step 3 Smoke attempt — FAIL (resource)
- Command: `GPUS=5 bash experiments/down_layer_sensitivity/scripts/run_smoke.sh`
- Run ID: `20260819_021415_smoke` (`mode=smoke`)
- Failed in `prewarm_compressed_weights` / grouped `up_proj` decode
- OOM: process ~45.5 GiB + co-tenant pid 171577 ~33.4 GiB on physical GPU 5
- CUDA binding (`CUDA_VISIBLE_DEVICES=5` + logical `cuda:0`) appears correct; not a code bug

## Step 4 Smoke gates — NOT MET (run failed before jobs)

## Step 5 Git inspection — PASS
- `?? experiments/` — expected new package only
- `M scripts/catlora_simple.sh` — pre-existing unrelated user change; untouched
- No production files under litebsq/train_utils/tools modified

## GPU occupancy at blocker
| GPU | Occupant | ~Used |
|-----|----------|-------|
| 0–3 | lizhangming s1 python | 69–74 GiB |
| 4–7 | xiezhinan dreamtorch python | 34–53 GiB |

No empty 80GB GPU. Formal needs ≥1 empty GPU (W=1) or preferably 4 empty GPUs (default `GPUS=0,1,2,3`).

## Ask human
Free exclusive GPUs (or designate which ones we may use), then resume Task 9 smoke → Task 10 formal.
