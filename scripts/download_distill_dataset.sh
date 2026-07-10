#!/usr/bin/env bash
# Cat / Block / E2E 蒸馏共用 EdgeRazor 配方数据下载（不依赖 EdgeRazor 仓库）。
#
# 产出目录（默认 data/edgerazor_qwen3）需包含：
#   ii_7M_instruct.jsonl
#   ii_gen_1.4M_instruct.jsonl
#   tulu_0.6M_instruct.jsonl
#   am_1.4M_instruct.jsonl
#   task_vaellm_eval_instruct.jsonl
#
# 通用语料：tools/prepare_edgerazor_distill_jsonl.py 从 HuggingFace 下载并转换；
# 下游任务切片：tools/prepare_vaellm_task_mix.py。
#
# 用法（在已激活 bitvae 的 shell 中）：
#   bash scripts/download_distill_dataset.sh
#
# 可选环境变量：
#   OUTPUT_DIR             默认 <repo>/data/edgerazor_qwen3
#   HF_TOKEN               BAAI/Infinity-Instruct 为 gated 数据集，需有效 token
#   MAX_SAMPLES            非空时每个 general jsonl 只取 N 条（smoke）
#   MAX_SAMPLES_PER_TASK   非空时 task jsonl 每任务只取 N 条（smoke）
#
# Infinity-Instruct 为 gated 数据集：
#   1) 在 https://huggingface.co/datasets/BAAI/Infinity-Instruct 用同一 HF 账号申请访问；
#   2) 将下方 hf_xxx 替换为有效 token（须与申请访问的账号一致）。
#
# smoke 示例：
#   MAX_SAMPLES=100 MAX_SAMPLES_PER_TASK=10 bash scripts/download_distill_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/edgerazor_qwen3}"

export PYTHONPATH="${PROJECT_ROOT}"
unset HF_ENDPOINT
# BAAI/Infinity-Instruct 为 gated 数据集；将 hf_xxx 替换为你的 token。
export HF_TOKEN="${HF_TOKEN:-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}"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

REQUIRED_FILES=(
  "ii_7M_instruct.jsonl"
  "ii_gen_1.4M_instruct.jsonl"
  "tulu_0.6M_instruct.jsonl"
  "am_1.4M_instruct.jsonl"
  "task_vaellm_eval_instruct.jsonl"
)

echo "============================================"
echo " VAELLM 蒸馏数据集下载"
echo " 输出目录: ${OUTPUT_DIR}"
echo " HF Hub: https://huggingface.co"
if [[ -n "${HF_TOKEN}" && "${HF_TOKEN}" != "hf_xxx" ]]; then
  echo " HF_TOKEN: 已配置"
else
  echo " HF_TOKEN: 未配置（请将脚本中的 hf_xxx 替换为有效 token）"
fi
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

GENERAL_ARGS=(--output_dir "${OUTPUT_DIR}" --skip_existing)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  GENERAL_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

python tools/prepare_edgerazor_distill_jsonl.py "${GENERAL_ARGS[@]}"

echo "生成下游任务切片: task_vaellm_eval_instruct.jsonl ..."
TASK_ARGS=(--output_dir "${OUTPUT_DIR}")
if [[ -n "${MAX_SAMPLES_PER_TASK:-}" ]]; then
  TASK_ARGS+=(--max_samples_per_task "${MAX_SAMPLES_PER_TASK}")
fi
if [[ -f "${OUTPUT_DIR}/task_vaellm_eval_instruct.jsonl" && -z "${MAX_SAMPLES_PER_TASK:-}" ]]; then
  echo "已存在，跳过: task_vaellm_eval_instruct.jsonl"
else
  python tools/prepare_vaellm_task_mix.py "${TASK_ARGS[@]}"
fi

missing=0
for dataset_name in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${OUTPUT_DIR}/${dataset_name}" ]]; then
    echo "缺失: ${OUTPUT_DIR}/${dataset_name}"
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "错误: 蒸馏数据不完整，请检查上方日志。"
  exit 1
fi

echo "============================================"
echo " 蒸馏数据已就绪: ${OUTPUT_DIR}"
echo " 默认 mix: edgerazor_ii_7m, edgerazor_ii_gen, edgerazor_tulu, edgerazor_am, vaellm_eval_task"
echo "============================================"
