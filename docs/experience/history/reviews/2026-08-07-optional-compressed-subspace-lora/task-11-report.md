> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-07-optional-compressed-subspace-lora/task-11-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 11 Report（partial: smoke + remaining verification）

## Status

**DONE（partial）** — tiny one-step smoke 已落地并验证；§22 定向测试全部通过。  
未做 §24 真实 Qwen/GPU 短跑（约束：不下载真实模型、不跑真实 GPU 短实验）。

## Files changed

### `tests/test_compressed_subspace_lora.py`
- 新增 `CompressedSubspaceLoRATests.test_tiny_one_step_subspace_peft_smoke`（计划 §23）。
- 复用现有 tiny factories：`_build_vae_linear` / `TinyProxyHost` / wrap+inject helpers。
- 使用真实 PEFT 0.10.0（`_require_peft_010()`），不 mock LoRA 数学。
- 闭环：channel protection → VAELinear → `CompressedSubspacePeftProxy` + `PeftZeroLinearCarrier` → `inject_compressed_subspace_peft_lora` → finite loss → backward → one SGD step → `extract_subspace_peft_low_rank_payloads` → `export_compressed_subspace_peft_lora_to_vae_low_rank` → checkpoint save/load → final forward。
- 硬性检查覆盖：
  - loss / `lora_A`/`lora_B` grad finite
  - optimizer step 后至少一个 PEFT LoRA 参数变化
  - carrier sentinel `weight.numel()==1` 且始终 frozen/zero
  - protected input 坐标 LoRA weight delta == 0；non-protected 至少一处变化
  - export 后无 subspace proxy / carrier / PEFT LoRA linear
  - save/load forward 与 export 前 proxy forward 一致（atol/rtol）

## Environment

```bash
conda activate bitvae
which python  # .../envs/bitvae/bin/python
python -V     # Python 3.11.13
python -c "import peft; print(peft.__version__)"  # 0.10.0
```

## Test results

### Smoke + compressed subspace unit suite

```bash
python -m pytest -q tests/test_compressed_subspace_lora.py::CompressedSubspaceLoRATests::test_tiny_one_step_subspace_peft_smoke --tb=short
# 1 passed

python -m pytest -q tests/test_compressed_subspace_lora.py --tb=line
# 25 passed
```

### teacher_first（用户指定）

```bash
python -m pytest -q tests/test_e2e_teacher_first.py --tb=line
# 12 passed
```

无 pre-existing unrelated failures。

### Remaining §22 verification

```bash
python -m pytest -q \
  tests/test_cat_compressed_lora_scope.py \
  tests/test_e2e_compressed_lora_scope.py \
  tests/test_e2e_checkpoint_io_legacy.py \
  tests/test_temporary_switch_residency.py \
  tests/test_cat_eval_adapter_match.py \
  --tb=line
# 58 passed, 2 warnings
```

Warning：`test_trainer_resume_continues_from_checkpoint_not_source_init` 触发 PEFT `Could not find a config file`（既有 resume 路径行为，非本次引入失败）。

## §24 真实短跑

**未跑。** 原因：本次任务约束明确禁止下载真实模型 / 跑真实 GPU 短实验。属可选算法实验，不阻塞单元测试闭环。

## Concerns

1. Smoke 为了保证第一步有有效梯度，在 PEFT 默认 `init_lora_weights=True`（B≈0）之后手动写入非零 `lora_A`/`lora_B`；仍走真实 PEFT 前向/反传与 SGD step，未 mock LoRA 数学。
2. Smoke 只覆盖 input channel protection（与现有 tiny factory 主路径一致）；output protection 的同类闭环未在本 smoke 重复覆盖（已有独立 unit tests）。
3. §24 A/B（`full` vs `compressed_subspace`）真实短跑与下游 PPL/task 指标仍待有 GPU/checkpoint 条件时补做。
