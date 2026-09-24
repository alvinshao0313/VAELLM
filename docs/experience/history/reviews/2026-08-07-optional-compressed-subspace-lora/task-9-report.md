> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-9-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 9 Report

## Status

**DONE**

## Files changed

### `compressed_e2e_fintuning/trainables.py`
- 新增 `validate_selected_low_rank_scope(selected_modules) -> str`（§12.1）：要求完整 low-rank payload、normalize scope、拒绝 mixed scope、返回唯一 scope。

### `compressed_e2e_fintuning/runtime.py`
- 新增 `_build_subspace_low_rank_peft_model(...)`（§12.4）：clear payload → wrap proxy → `get_peft_model`（`target_modules=[CARRIER_NAME]`，`alpha=rank`，`dropout=0.0`，`TaskType.CAUSAL_LM`）→ 校验 carrier → `initialize_subspace_peft_lora_from_low_rank` → 构造 `VAEDecoderTrainableSelection`。
- 新增 `_prepare_compressed_lora_train_model(...)`：按 scope 路由 full（materialize + `_build_low_rank_peft_model`）vs subspace（上述 helper）。
- `run()`：`compressed_lora` 保留 uniform-rank + scope 校验；`both` 只做 scope 校验、不新增 uniform-rank、仍走 `select_vae_decoder_trainables`（不建 proxy）。
- 日志：`Resolved compressed low-rank scope: ...`；subspace builder 记录 `compressed_lora_scope=...`。
- final export：full → `extract_low_rank_payloads_from_lora`；subspace → `extract_subspace_peft_low_rank_payloads`；统一 reload source + `write_low_rank_payloads_to_compressed_model(..., expected_scope=...)`。
- `using_root_peft` / `_peft_base_model` / resume 逻辑仍按 `train_mode == "compressed_lora"` 共用，不按 scope 拆开。

### `tests/test_e2e_compressed_lora_scope.py`
- uniform validator（full / subspace / mixed）。
- full route regression（monkeypatch 断言 materialize + full builder）。
- subspace route（proxy + carrier + root `PeftModel`；不走 materialize/full builder；carrier `numel==1`；trainable params）。
- final export roundtrip（payload 更新、protected 不变、无 proxy/PEFT）。
- `both`：decoder + subspace low_rank 可训，无 proxy。
- tiny resume：step1 save → fresh reconstruct → `_load_from_checkpoint` 校验 A/B=checkpoint ≠ source init → resume 到 step2 → extract/writeback。

## Test summary

```bash
conda activate bitvae
which python  # .../envs/bitvae/bin/python
python -V     # Python 3.11.13
python -c "import peft; print(peft.__version__)"  # 0.10.0

python -m pytest -q tests/test_e2e_compressed_lora_scope.py
# 8 passed
```

## Concerns

1. Resume 单测用 Transformers `Trainer` + `use_cpu=True` + `lr_scheduler_type=constant`；未覆盖真实多卡 / streaming offload 下的 resume。
2. full route regression 通过 monkeypatch 断言调用链，未在本任务内跑真实 dense materialize + 完整 `run()`。
3. 未跑真实 Qwen / GPU E2E 短跑（属 Task 11 / §24）。
