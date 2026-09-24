> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-8-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 8 Report: Thin Smoke/Formal Shell Scripts

## Status

**完成** — 已按 brief 逐字创建 shell 脚本并更新 README。

## 变更文件

| 文件 | 操作 |
|------|------|
| `experiments/down_layer_sensitivity/scripts/run_smoke.sh` | 新建 |
| `experiments/down_layer_sensitivity/scripts/run_formal.sh` | 新建 |
| `experiments/down_layer_sensitivity/README.md` | 新增「运行方式」章节 |

## 自检清单

- [x] `run_smoke.sh` 内容与 brief Step 1 完全一致（`GPUS` 默认 `0`，`--mode smoke`）
- [x] `run_formal.sh` 内容与 brief Step 2 完全一致（`GPUS` 默认 `0,1,2,3`，`--mode formal`）
- [x] 两脚本均：`set -euo pipefail`、`export PYTHONPATH=.`、路径变量 + 直接 CLI 传参
- [x] 无 `conda activate` / `conda run`
- [x] 无 job 分配、排序或结果解析逻辑
- [x] `CHECKPOINT_DIR` 固定为 `.result/catlora/res0-bf16-protect-channel-vae/final_model`
- [x] README 含 brief 要求的 exact 命令块
- [x] README 说明每 GPU 对应独立 MMLU worker，非 DDP
- [x] 脚本已 `chmod +x`
- [x] `bash -n` 语法检查通过
- [x] 与 brief 内容 `diff` 无差异

## 测试

| 测试 | 结果 |
|------|------|
| `bash -n run_smoke.sh` | 通过 |
| `bash -n run_formal.sh` | 通过 |
| brief 逐行 diff | 无差异 |
| smoke 端到端运行 | **未执行**（用户未要求；checkpoint 存在性未验证） |
| formal 端到端运行 | **未执行**（用户明确禁止） |

## Git

未提交。工作区保留改动。

## 关注点

- 运行前需确认 checkpoint 目录 `.result/catlora/res0-bf16-protect-channel-vae/final_model` 存在。
- 正式实验默认占用 4 卡；机器 GPU 数量不足时通过 `GPUS` 环境变量覆盖。
