> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-7-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 7 Review: Unify E2E Dense / CPU / Gamma Masks

**Reviewer:** code review (Task 7)  
**Date:** 2026-08-10  
**Verdict:** **PASS** — hard requirements met; safe to proceed to Task 8.

---

## Spec Checklist

| Item | Status | Evidence |
|------|--------|----------|
| `__init__` 接受 `prompt_kd_weight=0.0`，保存并拒绝负值 | ✅ | `trainer.py` 构造参数 + `self.prompt_kd_weight` + `<0` → `ValueError`（与 Task 6 stub 语义一致） |
| private `_build_distill_token_mask` 唯一调用共享 helper 并传 `self.prompt_kd_weight` | ✅ | `trainer.py:402-412`；共享调用仅此一处 |
| `_compute_legacy_dense_loss` 走 private helper | ✅ | `token_mask = self._build_distill_token_mask(inputs, logits)` |
| `_compute_teacher_first_cpu_loss` 走 private helper | ✅ | 同上模式于 CPU student KL 分支 |
| `_build_cpu_teacher_targets` 的 `gamma_mask` 走同一 private helper | ✅ | `gamma_mask = self._build_distill_token_mask(inputs, teacher_logits)` |
| 轻量单测验证 0.1 转发到共享 mask | ✅ | `VAEE2ETrainerPromptKdMaskHelperTest`（`__new__` + mock） |
| `rg`：1 import + 1 真实共享调用；三路径走 private | ✅ | 见下方 Verification；行号 28 / 402 / 407 / 460 / 607 / 698 |
| smoke：`prompt_kd_weight=0.1` fractional mask + 全 `DENSE_LOSS_TYPES` | ✅ | `test_dense_dispatcher_loss_pipeline_smoke` |
| smoke：EAKLD dense vs CPU-offload loss/telemetry/grad 一致 | ✅ | `test_offload_cpu_eakld_dispatcher_smoke`（含 `dense_loss.backward` + grad `allclose`） |
| one-step：builder 支持 weight；dense + CPU EAKLD 各一组 0.1 | ✅ | `_build_e2e_trainer(..., prompt_kd_weight=)`；两测试均传 `0.1`；CPU 仍断言 entropy 调用 1 次 |
| 未改 `e2e_common/dense_loss.py` / lazy_datasets / ckpt schema | ✅ | review package 仅 4 文件：`trainer.py` + 3 个测试文件 |
| 指定 pytest 通过 | ✅ | 本地复验 9 passed |

**Hard requirements（用户核对项）**

1. Single private helper；仅此处调用共享 `build_distill_token_mask` — ✅  
2. legacy dense + teacher-first CPU + `gamma_mask` 均使用该 helper — ✅  
3. Smoke 覆盖 fractional 0.1、EAKLD dense/CPU 一致、one-step 0.1 — ✅  
4. 未动 dense_loss 公式 / lazy_datasets / ckpt schema — ✅  

---

## Verification Run

```text
conda activate bitvae
which python   # .../envs/bitvae/bin/python
python -V      # Python 3.11.13
export PYTHONPATH=.

rg -n "build_distill_token_mask" compressed_e2e_fintuning/trainer.py
# 28: import
# 402/407: private helper + sole shared call (prompt_kd_weight=self.prompt_kd_weight)
# 460: gamma_mask
# 607: legacy dense
# 698: teacher-first CPU

pytest tests/test_e2e_dataset_mix.py::VAEE2ETrainerPromptKdMaskHelperTest \
  tests/smoke/test_loss_pipeline_smoke.py \
  tests/smoke/test_one_step_train_smoke.py -q
# .........
# 9 passed in 5.59s
```

与 report 中 GREEN 证据一致。

---

## Implementation Notes

### `compressed_e2e_fintuning/trainer.py`

- Private helper 接口干净：`(inputs, reference_logits)` → 从 `inputs` 取 `labels` / `attention_mask`，权重固定来自 `self.prompt_kd_weight`，避免各分支各自拼参。
- 三路 reference 选择正确：gamma 用 `teacher_logits`；dense/CPU student KL 用 student `logits`。共享 `build_distill_token_mask` 仅用 reference 的 shape/device，labels 相同则权重一致，满足「gamma 与 KL 共用同一 weighted mask」。
- 旧代码 `labels=labels`（局部 `inputs.get("labels")`）改为 `inputs.get("labels")`，语义等价。
- 未触及 choice KD、SFT/origin CE、hidden alignment 路径。

### Tests

- Helper 单测只验证转发与 kwargs，符合「轻量、不启大模型」。
- Loss-pipeline smoke 在 dispatcher 层直接构造 fractional mask，覆盖全 dense loss 类型；offload 路径补上 dense backward 与 grad 对齐，正好钉住硬性要求 3。
- One-step 经真实 `VAEDecoderE2ETrainer.training_step` 行使 0.1；CPU 路径保留 entropy 单次计数，说明 gamma 仍在 target build 阶段计算一次。

### Report / package 小记（非代码缺陷）

- Report 写「`__init__` 已由 Task 6 提供、本任务未改语义」；review package diff 仍包含该参数块——终态满足 brief，属 diff 打包边界，不构成 Spec 失败。
- `VAEE2EPromptKdWeightArgsTest` 属于 Task 6 范围，出现在本 package 的 `test_e2e_dataset_mix.py` hunk 中；Task 7 新增关键是 `VAEE2ETrainerPromptKdMaskHelperTest`。

---

## Quality

| 维度 | 评价 |
|------|------|
| **范围控制** | 好。生产改动只在 E2E trainer mask 接线；公式/数据/ckpt 未动。 |
| **硬性统一** | 好。dense / CPU KL / teacher gamma 收敛到同一 private helper，消除「gamma 仍用默认 0.0、KL 用 0.1」类分裂风险。 |
| **TDD** | 好。RED（缺 private helper）→ GREEN 有记录；本地复验通过。 |
| **测试深度** | 够用。缺「直接断言 CPU gamma 数值随 0.1 变化」的专项单测，但 private 转发 + one-step CPU + dispatcher 对齐已覆盖 brief 要求。 |
| **可维护性** | 好。后续改权重来源只改 private helper 一处。 |

**无 blocking defect。**

---

## Summary

Task 7 将 E2E dense、CPU-offload student KL 与 teacher gamma 的 token mask 统一到 `_build_distill_token_mask`，且该处是共享 `build_distill_token_mask` 的唯一生产调用点并始终传入 `self.prompt_kd_weight`。Smoke 覆盖 fractional 0.1、dense/CPU 对齐与 one-step 0.1；未改 loss 公式或数据/ckpt 路径。

**Recommendation:** accept Task 7; proceed to Task 8 (experiment scripts defaults).
