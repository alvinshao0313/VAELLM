#!/usr/bin/env bash
# Resume a formal run inside a detached tmux session (survives Cursor close).
set -euo pipefail

RUN_DIR="${1:?usage: resume_formal_tmux.sh <run_dir>}"
SESSION="${2:-down_sens_formal}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/.result/experiments/down_layer_sensitivity"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/resume_$(basename "${RUN_DIR}")_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "attach with: tmux attach -t ${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
source \"\$(conda info --base)/etc/profile.d/conda.sh\"
conda activate bitvae
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo \"[\$(date -Is)] resume start: ${RUN_DIR}\" | tee -a '${LOG_FILE}'
python experiments/down_layer_sensitivity/run.py \
  --resume_run_dir '${RUN_DIR}' \
  2>&1 | tee -a '${LOG_FILE}'
echo \"[\$(date -Is)] resume finished with exit=\$?\" | tee -a '${LOG_FILE}'
"

echo "started tmux session: ${SESSION}"
echo "log: ${LOG_FILE}"
echo "attach: tmux attach -t ${SESSION}"
echo "status: tmux ls"
