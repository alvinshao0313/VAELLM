> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-1-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1 Report: Write Failing Low-Level Parity Tests First

## Status

**DONE**

## What you implemented

只改测试，未改生产代码。

在 `tests/test_distill_losses.py` 中：

- 加入完整 `CPU_OFFLOAD_DENSE_DISTILL_LOSS_TYPES`（26 项，未缩减）。
- 加入确定性 fixture：`B=2, L=6, V=17`，float32，`sequence_chunk_size=2`，`temperature=1.3`，`alpha=0.4`，`prompt_kd_weight=0.03`，`eakld_confidence_k=16`。response/prompt mask 都非空；chunk 0 的 response mask 全零，chunk 1 的 prompt mask 全零。
- CE-blended loss（`kd` / `kd_top*` / `dual_kd` / `dual_kd_top*` / `eakld_kd`）各自从对应 student clone 生成可微 CE surrogate（该 clone 的平方均值），不共享 CE tensor。
- EAKLD-family 判定为 `loss_type in {"eakld", "eakld_kd"} or is_eakld_top_loss(loss_type)`。只有这些 case 调用 `compute_teacher_entropy_mean_and_gamma()`；non-EAKLD 把全部 6 个 EAKLD metadata 显式传 `None`。
- 新增参数化对照 `test_cpu_offload_dense_loss_matches_legacy_value_and_gradient`：同一 base student → 两个独立 `requires_grad=True` clone；legacy 调 `compute_dense_loss_from_logits()`，offload 调 `compute_dense_loss_from_offloaded_teacher()`，`sequence_chunk_size=2`；断言 loss `atol=2e-6, rtol=2e-5`，grad `atol=3e-6, rtol=3e-5`。
- 删除 `test_offloaded_teacher_dense_loss_rejects_non_eakld()`，替换为 `test_offloaded_teacher_dense_loss_supports_kl_top_1000_without_eakld_metadata()`：`loss_type="kl_top_1000"`，完全省略 EAKLD metadata keyword。

## What you tested and test results

环境：`bitvae`（`/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`，Python 3.11.13）。收集测试需要项目惯例 `PYTHONPATH=.`。

验证命令：

```bash
which python
python -V
pytest -q tests/test_distill_losses.py -k "offloaded_teacher or cpu_offload_dense"
```

**加测试前（基线）：** `5 passed, 62 deselected`。4 个现有 EAKLD `test_offloaded_teacher_dense_loss_finite_backward` + 1 个旧的 non-EAKLD 拒绝测试。

**加测试后：** `21 failed, 10 passed, 62 deselected`。

通过（10）：

- 现有 `test_offloaded_teacher_dense_loss_finite_backward`：`eakld` / `eakld_kd` / `eakld_top_7` / `eakld_topk_7`
- 新参数化对照中的 6 个 EAKLD：`eakld` / `eakld_kd` / `eakld_top` / `eakld_top_7` / `eakld_topk` / `eakld_topk_7`

失败（21，均为预期 RED）：

- 20 个 non-EAKLD 参数化对照
- `test_offloaded_teacher_dense_loss_supports_kl_top_1000_without_eakld_metadata`

## TDD Evidence

### RED

加测试前：

```
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python
Python 3.11.13
.....                                                                    [100%]
5 passed, 62 deselected in 10.49s
```

加测试后：

```
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python
Python 3.11.13
....FFFFFFFFFFF......FFFFFFFFFF                                          [100%]
21 failed, 10 passed, 62 deselected in 5.69s
```

non-EAKLD 参数化对照失败原因（以 `kl` 为代表）：当前 `compute_dense_loss_from_offloaded_teacher()` 在 `prompt_kd_weight > 0` 时无条件要求 prompt-region EAKLD gamma/entropy/count。测试按合同把这些字段传 `None`，因此在 EAKLD-only dispatcher 之前就被 gamma requirement 拦住：

```
ValueError: prompt_kd_weight > 0 requires teacher_prompt_gamma_cpu, teacher_entropy_mean_cpu, and teacher_prompt_valid_token_count_cpu.
```

位置：`e2e_common/dense_loss.py:483`。这是当前 EAKLD-only gamma requirement，不是测试写错。

`kl_top_1000` 无 metadata 测试失败原因：完全省略 EAKLD keyword 后，当前签名仍把 `teacher_gamma_cpu` / `teacher_entropy_mean_cpu` / `teacher_valid_token_count_cpu` 当作必填：

```
TypeError: compute_dense_loss_from_offloaded_teacher() missing 3 required keyword-only arguments: 'teacher_gamma_cpu', 'teacher_entropy_mean_cpu', and 'teacher_valid_token_count_cpu'
```

这正是接口尚未与 EAKLD 参数解耦的证据。

失败不是 import 错误、typo 或 legacy 路径缺 CE。CE-blended case 已为每个 student clone 单独构造 CE surrogate；legacy `compute_dense_loss_from_logits()` 能算出 finite loss，失败发生在 offload helper。

### GREEN

对本任务：N/A（生产代码不改）。例外：现有 EAKLD offload 以及新对照里的 6 个 EAKLD case 必须保持绿，实测均通过。

## Files changed

- `tests/test_distill_losses.py`：新增 CPU offload 全矩阵对照与 `kl_top_1000` 无 metadata 测试；删除 `test_offloaded_teacher_dense_loss_rejects_non_eakld()`。

未改生产代码。未 git add / commit。

## Self-review findings

- 26 项 loss tuple 与 brief 逐字一致，未缩成 family representative。
- CE surrogate 按 clone 独立生成，不共享。
- EAKLD detector 与 brief 一致；non-EAKLD 六个 metadata 全部显式 `None`；`kl_top_1000` 替换测试省略全部 EAKLD metadata keyword。
- 空 chunk 覆盖：`sequence_chunk_size=2` 时 response 的 chunk 0、prompt 的 chunk 1 全零。
- 现有 EAKLD finite-backward 与新 EAKLD 对照都绿。
- 未发现测试自身导致的非预期失败（例如忘传 `ce_loss`）。

## Any issues or concerns

- 简报里的 pytest 命令未写 `PYTHONPATH=.`；本仓库测试收集需要它，已按项目惯例加上。
- non-EAKLD 当前先撞上 `prompt_kd_weight > 0` 的 prompt-gamma 检查，而不是 `"supports only EAKLD-family losses."` 这句 dispatcher 文本。原因是 brief 固定 `prompt_kd_weight=0.03`。这仍属于 EAKLD-only gamma requirement，后续 generic dispatcher 应只对 EAKLD-family 校验这些字段。
