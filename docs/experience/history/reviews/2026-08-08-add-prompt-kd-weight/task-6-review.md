> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-6-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6 Review: E2E CLI / Runtime `prompt_kd_weight` Chain

**Reviewer:** code review (Task 6)  
**Date:** 2026-08-10  
**Verdict:** **PASS** — spec satisfied; safe to proceed to Task 7.

---

## Spec Checklist

| Item | Status | Evidence |
|------|--------|----------|
| 参数测试先行：默认 `0.0` | ✅ | `VAEE2EPromptKdWeightArgsTest.test_prompt_kd_weight_defaults_to_zero` |
| 接受 `0.05` | ✅ | `test_prompt_kd_weight_accepts_fractional_value` |
| 接受 `2.0`（>1 不封顶） | ✅ | `test_prompt_kd_weight_accepts_value_above_one` |
| 负值 `SystemExit` | ✅ | `test_prompt_kd_weight_rejects_negative_weight` |
| MCQA：`0.0` 保留 | ✅ | `test_mcqa_allows_zero_prompt_kd_weight` |
| MCQA：非零 parser error | ✅ | `test_mcqa_rejects_nonzero_prompt_kd_weight` + `validate_args` MCQA guard |
| `VAEDecoderE2EArguments.prompt_kd_weight: float = 0.0` | ✅ | `compressed_e2e_fintuning/args.py:53` |
| CLI `--prompt_kd_weight` `type=float` `default=0.0` | ✅ | `args.py:124` |
| parse validation 拒绝 `<0` | ✅ | `args.py:289-290` |
| runtime 独立日志：resolved prompt weight + response 固定 `1.0` | ✅ | `runtime.py:979-982` |
| runtime 构造 trainer 时传入 `prompt_kd_weight` | ✅ | `runtime.py:1031` |
| 运行 `pytest tests/test_e2e_dataset_mix.py -q` | ⚠️ | 已运行；**9 failed, 38 passed**（见下方） |

**Full-file pytest 说明（⚠️，非 Task 6 回归）：**

- `DatasetMixArgsTest.test_parse_args_eval_before_save_requires_tasks_and_save_steps` — 仍使用已移除的 `--eval_before_save`（现为 `--eval_after_save`）。
- 8 个 `DatasetMixBuilderTest` — lazy mix / `dummy.txt` 路径问题，与 prompt KD 无关。
- `VAEE2EPromptKdWeightArgsTest`：**6/6 passed**（本地复验一致）。

---

## Verification Run

```text
conda activate bitvae
export PYTHONPATH=.
pytest tests/test_e2e_dataset_mix.py::VAEE2EPromptKdWeightArgsTest -q
# 6 passed in ~6.5s

pytest tests/test_e2e_dataset_mix.py -q
# 9 failed, 38 passed (pre-existing; unrelated to Task 6)
```

---

## Implementation Notes

### `compressed_e2e_fintuning/args.py`

- Dataclass 字段与 CLI 参数位置合理（紧跟 `hidden_loss_weight`，与 category 侧语义对齐）。
- 通用校验 `prompt_kd_weight >= 0` 与 MCQA 专用 `!= 0.0` 分工清晰；MCQA 非零会给出明确 error（含 “choice KD has no token mask”），避免 silent no-op。
- MCQA 下负值会先命中 MCQA 分支（`-0.1 != 0.0`），error 文案是 MCQA 专用而非 `>= 0`；行为上仍被拒绝，可接受。

### `compressed_e2e_fintuning/runtime.py`

- 日志格式与现有 `Hidden alignment config` 块一致，独立一行记录 `prompt_kd_weight` 与固定 `response_kd_weight=1.0`。
- Trainer 构造处显式 `float(args.prompt_kd_weight)`，与 `hidden_loss_weight` 处理方式一致。
- 日志行使用 `getattr(args, "prompt_kd_weight", 0.0)`，而同函数其他处直接用 `args.*`；字段已在 dataclass 中定义，略冗余但不影响行为。

### `compressed_e2e_fintuning/trainer.py`（stub，Task 6 允许）

- 增加 `prompt_kd_weight` 参数、存储、`<0` 拒绝；未接入 dense/CPU/gamma mask（属 Task 7 范围）。
- 与 brief 文件列表相比多改了一个文件，但符合约束 “pass to trainer (stub OK)”。

### `tests/test_e2e_dataset_mix.py`

- `VAEE2EPromptKdWeightArgsTest` 覆盖 brief 全部参数场景；helper `_parse_with_checkpoint` 与同类测试风格一致。
- RED → GREEN 证据在 report 中可信；本地复验通过。

---

## Quality

| 维度 | 评价 |
|------|------|
| **范围控制** | 好。未改 loss/mask/数据路径；trainer 仅 stub。 |
| **TDD** | 好。先 6 fail 后 6 pass，RED/GREEN 有记录。 |
| **与 plan 一致性** | 好。默认 0.0、无上限、MCQA 禁非零、response 固定 1.0 均符合 Global Constraints。 |
| **错误信息** | 好。MCQA 与负值均有明确 `parser.error` 文案。 |
| **测试缺口（非阻塞）** | runtime 日志内容、runtime→trainer kwargs 无单测；category 侧在 `test_cat_eval_adapter_match.py` 有类似 kwargs 测试，E2E 可留给 Task 7 smoke/helper 测试。 |
| **技术债** | Task 7 必须完成 mask 三路统一；当前 stub 存储值但不影响训练语义（默认 0.0 行为不变）。 |

**无 blocking defect。**

---

## Summary

Task 6 将 `--prompt_kd_weight` 从 CLI 经 `validate_args`、runtime 日志与 trainer 构造完整贯通；参数校验与 MCQA  guard 符合 spec；新增 6 个测试全部通过。全文件 pytest 仍有 9 个既有失败，与本次改动无关。

**Recommendation:** merge Task 6 work as-is; continue with Task 7 (E2E dense / CPU-offload / teacher gamma mask wiring).
