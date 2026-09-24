> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-3-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3 Report: Generalize `compute_dense_loss_from_offloaded_teacher()`

**Status:** DONE  
**Commits created:** none（禁止提交）

## What I implemented

只改了 `e2e_common/dense_loss.py`。未改 `train_utils/distill_losses.py`、`compressed_e2e_fintuning/trainer.py`，也未改 Task 1 测试。

### 1. Import

新增 `compute_chunked_token_mean_from_cpu_teacher_logits`。没有为每个 loss 增加 `*_from_cpu_teacher_logits`。

### 2–4. 三个精确 helper

按 brief 原文加入：

- `_is_eakld_family_loss()`：`eakld` / `eakld_kd` / `is_eakld_top_loss()`
- `_offloaded_region_loss_type()`：只把 CE-blended KD 映射成 region primitive（`kd→kl`、`kd_top*→kl_top*`、`dual_kd→dual_kl`、`dual_kd_top*→dual_kl_top*`）；普通 `kl_top*` 保持不变
- `_is_ce_blended_dense_loss()`：只识别 `kd` / `kd_top*` / `dual_kd` / `dual_kd_top*` / `eakld_kd`，不把 `kl_top*` 判成 KD

### 5. 一个 generic non-EAKLD region helper

`_compute_offloaded_non_eakld_region_loss()`：

1. `region_loss_type = _offloaded_region_loss_type(loss_type)`
2. `chunk_loss_fn` 调用现有 `compute_dense_loss_from_logits()`，固定 `ce_loss=None`、当前 chunk mask、原 temperature、`prompt_mask=None`、`prompt_kd_weight=0`、`telemetry_out=None`
3. 交给 `compute_chunked_token_mean_from_cpu_teacher_logits()`

没有在 helper 里重写 softmax / KL / MSE / dual / top-k 公式。

### 6. EAKLD metadata 校验改为条件执行

`teacher_gamma_cpu` / `teacher_entropy_mean_cpu` / `teacher_valid_token_count_cpu` 改为 `Optional`，默认 `None`。

只有 EAKLD-family：

- 要求 response `teacher_gamma_cpu` 非 None
- `telemetry_out` 非 None 时要求 response entropy/count
- `prompt_kd_weight > 0` 时要求 prompt gamma/entropy/count
- 校验 `eakld_confidence_k >= 2`

非 EAKLD 即使 `prompt_kd_weight > 0` 也不要求任何 EAKLD metadata；metadata 可全部省略，不只是传 `None`。

### 7. 保留现有 EAKLD specialized path

`eakld` / `eakld_kd` / `eakld_top*` / `eakld_topk*` 继续走 `compute_eakld_from_cpu_teacher_logits()` / `compute_eakld_topk_from_cpu_teacher_logits()`。response telemetry 仍写入，prompt telemetry 不写。`eakld_kd` 仍按 `(1-alpha)*CE + alpha*distill` 混合，CE 只算一次。

### 8. 非 EAKLD response/prompt 聚合

- response：generic region helper
- prompt：`_combine_region_loss()`，仅 `prompt_kd_weight>0` 时第二次 region helper
- `_is_ce_blended_dense_loss(norm)` 为真时要求 `ce_loss`，返回 `(1-alpha)*CE + alpha*distill`
- 普通 distillation 直接返回 region loss

### 9. 删除 EAKLD-only 白名单报错

已删除 `teacher_output_offload=cpu supports only EAKLD-family losses.`。未知 loss 落到 `compute_dense_loss_from_logits()` 的统一 `Unsupported dense loss type`。

## Tests / results

环境：`/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`，Python 3.11.13。`PYTHONPATH=.`。

```bash
pytest -q tests/test_distill_losses.py -k "offloaded_teacher or cpu_offload_dense"
# 31 passed, 62 deselected in 4.83s
```

包含 Task 1 全矩阵 value/gradient parity，以及 `kl_top_1000` 完全省略 EAKLD metadata 的接口测试。现有 EAKLD prompt-weight / CE mix 回归也通过。

未跑 trainer / smoke；本任务范围不包含它们。未提交。

## Self-review

- KD 映射只发生在 chunk 内 region primitive；CE 仍在完整 student logits 上混合一次。
- `_is_ce_blended_dense_loss` 含 `eakld_kd`，但 generic CE-blend 路径不会执行到它：EAKLD 分支先返回。这与 brief 一致。
- 没有改 Task 2 executor，没有改 `trainer.py`。
- 未发现需要降级或兼容补丁的问题。
