#!/usr/bin/env bash
# EdgeRazor 蒸馏 / E2E 统一模型与序列长度配置。
# 切换 Qwen / Llama：只改 MODEL_PATH，其余脚本自动跟随。
#
# Qwen:  MODEL_PATH=Qwen/Qwen3-8B
# Llama: MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DISTILL_MODEL_MAX_LENGTH="${DISTILL_MODEL_MAX_LENGTH:-8192}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"
VAELLM_EDGERAZOR_DATASET_MIX="${VAELLM_EDGERAZOR_DATASET_MIX:-edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009}"
