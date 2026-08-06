#!/usr/bin/env bash
set -euo pipefail

# 单 trial 入口：GPU id 与父进程 Python 解释器由 argv[1]/argv[2] 固定传入。
# 用法: bash mix_bit/scripts/train_candidate_single.sh <CUDA_VISIBLE_DEVICES> <PYTHON_EXECUTABLE> [cat_train args...]

if [ "$#" -lt 2 ]; then
  echo "Usage: bash $0 <CUDA_VISIBLE_DEVICES> <PYTHON_EXECUTABLE> [cat_train_arguments]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID="$1"
PYTHON_EXECUTABLE="$2"
shift 2

if [ ! -x "${PYTHON_EXECUTABLE}" ]; then
  echo "Python executable is not executable: ${PYTHON_EXECUTABLE}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
exec "${PYTHON_EXECUTABLE}" tools/cat_train.py "$@"
