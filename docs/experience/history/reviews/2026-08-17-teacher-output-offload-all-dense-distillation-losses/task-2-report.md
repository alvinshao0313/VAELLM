> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-17-teacher-output-offload-all-dense-distillation-losses/task-2-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2 Report: Generic Sequence-Chunk Executor

**Status:** DONE  
**Commits created:** none（禁止提交）

## What I implemented

只改了 `train_utils/distill_losses.py`。未改 `e2e_common/dense_loss.py`、`compressed_e2e_fintuning/trainer.py`，也未改写 Task 1 测试。

### 1. Import

- `from typing import Callable, MutableMapping, Optional`
- 未增加新依赖

### 2. Generic validator

把原 `_validate_cpu_teacher_distill_inputs()` 拆成 `_validate_cpu_teacher_logits_inputs()`，只检查：

- student/teacher `ndim == 3`
- teacher 必须在 CPU
- student/teacher shape 完全一致
- `sequence_chunk_size >= 1`

错误文案保持原样。已删除旧的合并 validator。

### 3. EAKLD 仍独立

`_compute_eakld_from_cpu_teacher_logits_impl()` 只改校验入口：

1. 调 `_validate_cpu_teacher_logits_inputs()`
2. 再检查 `gamma` 必须是 scalar tensor

EAKLD 仍按 chunk 累计 forward/reverse numerator，再用 region-global gamma 合成。没有改写成新的 token-mean executor。

### 4. Fixed-binding factory

新增 `_make_checkpointed_token_mean_chunk_forward()`，签名与计划完全一致：

- `fixed_start = int(start)` / `fixed_end = int(end)`，避免 checkpoint backward late-binding
- teacher H2D 只走 `copy_teacher_logit_chunk_to_device()`
- 调 `chunk_loss_fn`，要求返回 scalar，再乘 `valid_count`

### 5. Global token-weighted executor

新增 `compute_chunked_token_mean_from_cpu_teacher_logits()`。执行顺序固定为：

generic validate → `_default_token_mask` → global denominator → `iter_token_chunk_ranges` → factory → checkpoint/direct → stack/sum numerator → divide global denominator

行为约束：

- 返回全局 token-weighted mean 标量
- 需要梯度且 student chunk `requires_grad` 时用 `torch_checkpoint.checkpoint(..., use_reentrant=False, preserve_rng_state=False)`，否则直接调用
- 不平均各 chunk mean
- 不沿 vocab 分块
- 不把完整 CPU teacher logits 搬到 GPU
- 不对 student chunk detach
- 全局分母使用 `resolved_mask.sum().clamp_min(1.0)`（与现有 EAKLD 一致，避免全零 mask 除零）
- 空 chunk（`mask_chunk.sum()==0`）在 executor 里用 `torch.where` 把 numerator 置 0，避免 `NaN * 0 = NaN`

## Tests / results

环境：`/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`，Python 3.11.13。命令前设置 `PYTHONPATH=.`。

```bash
pytest -q tests/test_teacher_target_offload.py
# 18 passed
```

```bash
pytest -q tests/test_distill_losses.py -k "eakld and offloaded"
# 11 passed, 1 failed, 81 deselected
```

失败项：

`test_offloaded_teacher_dense_loss_supports_kl_top_1000_without_eakld_metadata`

原因：pytest `-k "eakld and offloaded"` 会匹配到函数名里的 `without_eakld`。这是 Task 1 的 non-EAKLD 接口测试，当前 `compute_dense_loss_from_offloaded_teacher()` 仍要求 EAKLD metadata。按任务约束，这属于 Task 3，本任务不修。

现有 EAKLD offload 用例（`eakld` / `eakld_kd` / `eakld_top_7` / `eakld_topk_7` 以及 prompt-weight / CE mix）全部通过。

额外确认（简报未要求，用于校验拆分 validator）：

```bash
pytest -q tests/test_distill_losses.py -k "cpu_teacher_eakld"
# 14 passed, 79 deselected
```

本地 smoke（未落地测试文件）：新 executor 与全量 token-mean 对齐；空 chunk 即使 `chunk_loss_fn` 返回 NaN，numerator 仍有限。

未跑 Task 1 全量 non-EAKLD parity：按约束留给 Task 3。

## Files changed

| File | Change |
| --- | --- |
| `train_utils/distill_losses.py` | +84 / −7。新增 generic validator、factory、executor；EAKLD 复用 generic validator 并保留 gamma 检查 |
| `tests/test_distill_losses.py` | **本任务未改**。工作区里的修改来自 Task 1 |

未 commit。

## Self-review

- 计划中的签名、factory 主体、算法顺序、H2D 路径、checkpoint 条件、EAKLD 独立性均按原文落地。
- 空 chunk 的 `torch.where` 放在 factory 之外、checkpoint 之后，是为了让 factory 与计划逐字一致，同时满足 “empty chunk numerator 必须为 0，不能 NaN”。
- 未把 EAKLD 特化路径合并进 token-mean executor。
- 未引入新依赖，未改 dense dispatcher / trainer。

## Concerns

1. 简报中的 `-k "eakld and offloaded"` 会误选 Task 1 的 `kl_top_1000` 测试，该测试在 Task 3 完成前会失败。这不是本任务回归。
2. 空 chunk 的 NaN 防护在 checkpoint 之后。对会返回 NaN 的 `chunk_loss_fn`，forward 已被 `torch.where` 清成 0；若后续发现 backward 被 NaN 污染，应把该防护移进 factory / checkpoint 内部。
3. 本任务没有给新 executor 增加正式 pytest。正确性目前靠 EAKLD 回归 + 一次性 smoke。Task 3 接入 `compute_dense_loss_from_logits()` 后才会有完整 parity。
