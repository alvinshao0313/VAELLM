> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-5-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5 Report: Add Trainer-Level Coverage for Every Dense Loss

**Status:** DONE  
**Commits created:** none（禁止提交）

## What I implemented

只改了 `tests/test_e2e_teacher_first.py`。未改 production 代码。未从 `tests/test_distill_losses.py` import。

### Step 1: 替换过时 unsupported-KL 测试

删除 `test_unsupported_kl_raises_before_student_forward()`。新增 `test_cpu_kl_top_1000_teacher_before_student_and_backward(tmp_path)`，按现有 `test_cpu_eakld_teacher_before_student_and_backward` 建模，`loss_type="kl_top_1000"`。

断言：`events == ["teacher_forward", "student_forward"]`；`_active_teacher_targets` 在 backward/release 前存在；`logits_cpu` 在 CPU；response/prompt EAKLD metadata 全 None；loss finite；backward 成功；至少一个 trainable student grad 非 None 且 finite；finally `_release_active_teacher_targets()` 后为 None。

### Step 2: 全 dense-loss 参数化 trainer 测试

在本文件复制 Task 1 的 exact tuple `CPU_OFFLOAD_DENSE_DISTILL_LOSS_TYPES`（26 种）。新增 `test_cpu_offload_all_dense_distillation_losses_backward(tmp_path, loss_type)`。

固定参数：`hidden_loss_weight=0.0`、`prompt_kd_weight=0.03`、`teacher_output_offload="cpu"`；`teacher_output_chunk_tokens=2` 仍由 `_build_trainer` 硬编码。每个 case：teacher first、student second、loss finite、backward 成功、student grad finite、finally release。

### Step 3: 非 EAKLD metadata skip 回归

`test_cpu_kl_top_1000_skips_eakld_metadata_and_copies_logits_once`：`loss_type="kl_top_1000"`、`prompt_kd_weight=0.03`、`hidden_loss_weight=0`。同时 counting monkeypatch `compute_teacher_entropy_mean_and_gamma` 和 trainer 侧 `copy_detached_tensor_to_cpu`。断言 entropy/gamma helper 0 次、完整 logits CPU copy 恰好 1 次、`logits_cpu` 存在、EAKLD metadata 全 None。输入用带 prompt prefix 的 labels，覆盖 prompt 区域存在时仍跳过 metadata。

### Step 4: 保留现有 EAKLD metadata-count 语义

本文件没有直接调用 `_build_cpu_teacher_targets()` 的站点，无需改调用方式。现有 zero prompt weight → helper 1 次、positive prompt weight → helper 2 次的断言未改，测试通过。

### Step 5: 非 EAKLD + hidden alignment

`test_cpu_kl_top_1000_adaptive_top_2_hidden_collectors`：`kl_top_1000`、`hidden_loss_weight=0.1`、`hidden_layer_weighting="adaptive_top_2"`、`prompt_kd_weight=0.03`。teacher/student 均 `output_hidden_states=False`；selected hidden count == 2；teacher CPU logits 存在；EAKLD metadata None；总 loss backward 成功。

### Step 6: training_step lifecycle

`test_training_step_clears_active_teacher_targets_kl_top_1000`：走 `trainer.training_step(student, inputs)`，step loss finite，返回后 `_active_teacher_targets is None`。

### Step 7: Trainer-level legacy vs CPU parity

`_build_trainer` 增加可选关键字 `teacher_output_offload: str = "cpu"`，默认仍是 `"cpu"`，现有测试语义不变。

`test_cpu_kl_top_1000_legacy_vs_cpu_trainer_parity`：相同 seed `20260817` 分别构造 `cpu`/`none` 两套 teacher/student；构造后逐参数确认初始参数相同；`loss_type="kl_top_1000"`、`prompt_kd_weight=0.03`、默认 temperature/alpha、`hidden_loss_weight=0`。分别调用 `compute_loss()`（不直接调 dense-loss helper），比较 scalar loss（`atol=2e-6, rtol=2e-5`）；backward 后比较 `student.lm_head.weight.grad`（`atol=3e-6, rtol=3e-5`）。finally 清理 CPU trainer `_active_teacher_targets`。

tiny vocab=17 的 `kl_top_1000` 未改 loss matrix；production clamp K，测试通过。

## Tests / results

环境：`/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`，Python 3.11.13。`PYTHONPATH=.`。

```bash
pytest -q tests/test_e2e_teacher_first.py
# 44 passed in 13.24s
```

包括：原有 EAKLD teacher-first / sft skip / entropy helper 1 次与 2 次 / 单次 logits copy / telemetry / training_step cleanup；新增 `kl_top_1000` teacher-first、26 种 dense loss backward、metadata skip、hidden alignment、kl_top_1000 training_step cleanup、legacy-vs-CPU trainer parity。

未提交。

## Self-review

- 只改测试文件；函数名、tuple、trainer 固定参数、容差与 brief 一致。
- 过时 unsupported-KL 测试已删除，无残留引用。
- 现有 `loss_type="eakld"` 测试断言未改，仍然通过。
- `_patch_tiny_get_layers` 仍是 autouse fixture，新测试自动覆盖。
- 非 EAKLD 路径确认 0 次 entropy/gamma、1 次完整 logits copy。
- parity 走完整 Trainer `compute_loss()`，不是底层 dense-loss helper。

## Concerns

无。
