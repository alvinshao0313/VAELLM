#!/usr/bin/env bash
# Qwen3-8B mixed-bit 全流程（GPU 0,1,2,3）。
# 用法（必须已激活 bitvae）：
#   conda activate bitvae
#   bash mix_bit/scripts/run_qwen3_8b_mix_bit_gpus0123.sh
#
# 说明：
# - 候选训练 35 job、Cost 搜索很长；中断后重跑同一脚本可 resume。
# - 当前工作区若存在旧 calibration（无 tokenizer_fingerprint_version=2），
#   prepare_calibration 带 --overwrite；v2 写好后若不想每次重建，可删掉该参数。
# - 不要用 conda run / 裸 PATH 乱指的 python；本脚本直接调用 python。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_CONFIG="${ROOT_DIR}/mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/model_inventory.json"
RUN_ROOT="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit"
POOL_MANIFEST="${RUN_ROOT}/candidate_pool/candidate_manifest.json"
BASELINE_OVERLAY="${RUN_ROOT}/baseline/b32d32s2/baseline_overlay.json"
DATASET="${RUN_ROOT}/calibration/dataset.pt"
DATASET_MANIFEST="${RUN_ROOT}/calibration/dataset_manifest.json"
TEACHER_CACHE="${RUN_ROOT}/calibration/teacher_topk/k256"
COST_TABLE="${RUN_ROOT}/costs/topk_k256/cost_table.jsonl"
COST_TABLE_META="${RUN_ROOT}/costs/topk_k256/cost_table_meta.json"
ALLOCATION="${RUN_ROOT}/allocation/topk_k256/optimal_2bit.json"
MIXED_MODEL_DIR="${RUN_ROOT}/mixed_model/topk_k256/optimal_2bit/final_model"

echo "[mix_bit] step 1/10 build_model_inventory"
python -m mix_bit.cli.build_model_inventory \
  --run_config "${RUN_CONFIG}" \
  --output "${INVENTORY}"

echo "[mix_bit] step 2/10 train_candidate_pool (35 jobs, GPUs 0,1,2,3)"
python -m mix_bit.cli.train_candidate_pool \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --gpus 0,1,2,3

echo "[mix_bit] step 3/10 inventory_candidate_pool"
python -m mix_bit.cli.inventory_candidate_pool \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}"

echo "[mix_bit] step 4/10 prepare_uniform_baseline"
python -m mix_bit.cli.prepare_uniform_baseline \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --device cuda:1

echo "[mix_bit] step 5/10 prepare_calibration (overwrite for tokenizer fingerprint v2)"
python -m mix_bit.cli.prepare_calibration \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --overwrite

echo "[mix_bit] step 6/10 build_teacher_cache (K=256)"
python -m mix_bit.cli.build_teacher_cache \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --dataset "${DATASET}" \
  --dataset_manifest "${DATASET_MANIFEST}" \
  --teacher_topk 256 \
  --cache_prob_dtype bfloat16 \
  --chunk_samples 16 \
  --batch_size 1 \
  --device cuda:1 \
  --output_dir "${TEACHER_CACHE}"

echo "[mix_bit] step 7/10 compute_cost_table (GPUs 0,1,2,3)"
python -m mix_bit.cli.compute_cost_table \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --baseline_overlay "${BASELINE_OVERLAY}" \
  --dataset "${DATASET}" \
  --dataset_manifest "${DATASET_MANIFEST}" \
  --kl_mode teacher_topk \
  --teacher_topk 256 \
  --teacher_cache "${TEACHER_CACHE}" \
  --gpus 0,1,2,3 \
  --batch_size 1

echo "[mix_bit] step 8/10 solve_allocation"
python -m mix_bit.cli.solve_allocation \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --cost_table "${COST_TABLE}" \
  --cost_table_meta "${COST_TABLE_META}"

echo "[mix_bit] step 9/10 assemble_mixed_model"
python -m mix_bit.cli.assemble_mixed_model \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --allocation "${ALLOCATION}" \
  --device cuda:1

echo "[mix_bit] step 10/10 validate_mixed_model"
python -m mix_bit.cli.validate_mixed_model \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --cost_table "${COST_TABLE}" \
  --cost_table_meta "${COST_TABLE_META}" \
  --allocation "${ALLOCATION}" \
  --baseline_overlay "${BASELINE_OVERLAY}" \
  --mixed_model_dir "${MIXED_MODEL_DIR}" \
  --dataset "${DATASET}" \
  --dataset_manifest "${DATASET_MANIFEST}" \
  --teacher_cache "${TEACHER_CACHE}" \
  --device cuda:1 \
  --lm_batch_size auto

echo "[mix_bit] done"
echo "mixed_model_dir=${MIXED_MODEL_DIR}"
echo "allocation=${ALLOCATION}"
echo "cost_table=${COST_TABLE}"
