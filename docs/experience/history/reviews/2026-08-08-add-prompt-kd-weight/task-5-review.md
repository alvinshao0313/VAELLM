> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-5-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5 Review: Integrate Prompt Weight into Every Category KD Branch

**Reviewer:** read-only code review  
**Date:** 2026-08-10  
**Scope:** `task-5-brief.md`, `task-5-report.md`, `task-5-review-package.diff`；并对照工作区 `train_utils/lora_training.py`、`tests/test_cat_eval_adapter_match.py`

---

## Verdict

| Dimension | Result |
|-----------|--------|
| **Spec** | ✅ |
| **Quality** | **Approved** |

---

## Spec Checklist (brief + global constraints)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `CustomSFTTrainer.__init__` 有 `prompt_kd_weight: float = 0.0`，保存并拒绝负值 | ✅ | `lora_training.py` 保存 `float(...)`；`< 0` 抛 `ValueError`（Task 4 已落地，本任务保留） |
| `compute_loss()` 内局部 `build_token_mask(reference_logits)` | ✅ | 闭包调用共享 `build_distill_token_mask(...)`，传入 `full_inputs` 的 labels/attention 与 `self.prompt_kd_weight` |
| 所有 tokenwise KD 分支走该 helper，禁止分支内自拼参数 | ✅ | 16 处均为 `token_mask = build_token_mask(logits)`；无分支内直接拼 `build_distill_token_mask(...)` |
| 覆盖所列 KD 变体 | ✅ | `rkl` / `dual_rkl` / `kl` / `r_kl_top*` / `dual_r_kl_top*` / `kl_top*` / `kd_top*` / `mse` / `kd` / `dual_kl` / `dual_kl_top*` / `dual_kd_top*` / `dual_kd` / `eakld_top*`（`is_eakld_top_loss`）/ `eakld` / `eakld_kd` |
| SFT/origin 不变 | ✅ | `{"origin","sft"}` 分支无 `build_token_mask`；仍走原 CE / hidden 路径 |
| `kd` / `kd_top` / `dual_kd*` / `eakld_kd`：CE 仍用 response-only labels + 原 alpha；weight 只作用 KD 项 | ✅ | 这些分支 `student_forward(full_inputs)`，`ori_loss = outputs["loss"]`；`token_mask` 仅传入 distill loss；混合仍为 `ori_loss*(1-alpha)+distill_loss*alpha` |
| hidden / pre-MLP 继续用 attention mask，不传 weighted KD mask | ✅ | `add_hidden_alignment_loss` 仍只传 `attention_mask=full_inputs.get("attention_mask")` |
| `rg build_distill_token_mask` → 1 import + 1 helper 内调用 | ✅ | 行 10 import；行 751 helper 内唯一真实调用；无遗留分支调用 |
| 无 ad-hoc 二值 KD mask | ✅ | 未发现 `labels != -100` / 手写 shifted attention 等替代路径；KD mask 全部经 helper |
| 要求的 pytest | ✅ | 复跑：`74 passed`（`bitvae`，`PYTHONPATH=.`） |
| No git commit | ✅ | Report 写明 none |

---

## Findings

No findings.

逐分支核对：`compute_loss` 中全部 tokenwise distill 路径均经局部 `build_token_mask`；生产侧 `build_distill_token_mask` 调用已收敛到 helper 一处。CE 混合分支未把 weighted mask 注入 `ori_loss`；hidden/pre-MLP 仍只用 attention mask。未发现仍自行构造 binary KD mask 的分支。

---

## Quality Notes (non-blocking)

1. **Review package 不完整：** `task-5-review-package.diff` 仅含 `lora_training.py`；报告中的 `_build_pre_mlp_trainer` 对 `prompt_kd_weight=0.0` 的补丁在工作区 `tests/test_cat_eval_adapter_match.py:439` 存在，但未打进 review package。不影响实现正确性，后续打包宜一并包含。

2. **Task 4 / Task 5 边界：** `__init__` store/validate 已在 Task 4 完成；本任务实质增量是 `compute_loss` helper + 16 处替换。与 brief 要求一致，无重复逻辑冲突。

3. **测试缺口（可选加固，非 spec 失败）：** 无针对 `compute_loss` 的轻量断言验证 `prompt_kd_weight=0.1` 经 helper 进入 KD mask。共享 mask 语义已在 Task 1–3 覆盖；类别路径目前依赖接线正确性 + fixture 不 AttributeError。

4. **报告已注明的残余风险：** `tests/smoke/test_one_step_train_smoke.py` 若用 `__new__` 构造 trainer 且走到 KD mask，需同样设置 `prompt_kd_weight`；不在本任务 pytest 集合内。

---

## Files Reviewed

| File | Role |
|------|------|
| `train_utils/lora_training.py` | local helper + 全部 KD 分支接线；CE/hidden 保持 |
| `tests/test_cat_eval_adapter_match.py` | `__new__` fixture 补 `prompt_kd_weight`（工作区；未入 review package） |

**Diff size (review package):** 1 file, +28 / −80 lines。
