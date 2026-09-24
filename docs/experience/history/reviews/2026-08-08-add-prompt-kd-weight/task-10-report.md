> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-10-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 10 Report: Closed-Loop Verification

**Date:** 2026-08-10  
**Workspace:** `/home/shaoyuantian/program/VAELLM`  
**Mode:** verification only（发现本 patch 回归后做了最小测试修复）  
**Commits:** none

## Environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate bitvae
export PYTHONPATH=.
which python  # /home/shaoyuantian/anaconda3/envs/bitvae/bin/python
python -V     # Python 3.11.13
CONDA_DEFAULT_ENV=bitvae
```

说明：未设置 `PYTHONPATH=.` 时，`tests/test_distill_losses.py` 在 collection 阶段会因 `ModuleNotFoundError: e2e_common` 失败。后续命令均在 `PYTHONPATH=.` 下执行。

## Sequential Suite Results

| Suite | Result | Notes |
|---|---|---|
| `pytest tests/test_distill_losses.py -q` | **51 passed** (5.30s) | PASS |
| `pytest tests/test_cat_eval_adapter_match.py -q` | **23 passed** (5.44s) | PASS |
| `pytest tests/test_e2e_dataset_mix.py -q` | **9 failed, 39 passed** (11.52s) | 与 brief 已知失败一致 |
| `pytest tests/smoke/test_loss_pipeline_smoke.py -q` | **3 passed** (4.82s) | PASS |
| `pytest tests/smoke/test_one_step_train_smoke.py -q` | **5 passed** (5.70s) | PASS |

### Known pre-existing failures in `test_e2e_dataset_mix.py`

与 brief 声明一致（eval_before_save + lazy mix），共 9 个：

1. `DatasetMixArgsTest::test_parse_args_eval_before_save_requires_tasks_and_save_steps`
2. `DatasetMixBuilderTest::test_build_datasets_mix_changes_with_different_seed`
3. `DatasetMixBuilderTest::test_build_datasets_mix_interleaves_and_resizes_sources`
4. `DatasetMixBuilderTest::test_build_datasets_mix_is_deterministic_for_same_seed`
5. `DatasetMixBuilderTest::test_build_datasets_mix_limits_train_preprocessing`
6. `DatasetMixBuilderTest::test_build_datasets_mix_repeats_short_source_to_target`
7. `DatasetMixBuilderTest::test_build_datasets_mix_skips_eval_when_eval_strategy_is_no`
8. `DatasetMixBuilderTest::test_build_datasets_mix_supports_long_sources_without_eval`
9. `DatasetMixBuilderTest::test_build_datasets_single_skips_eval_when_eval_strategy_is_no`

失败症状：`FileNotFoundError: Unable to find '.../dummy.txt'`（datasets lazy 路径）以及 eval_before_save 相关断言。失败路径与 `prompt_kd_weight` / mask 无关；本轮未扩展修复。

## Combined Five-Suite Run

```bash
pytest tests/test_distill_losses.py \
  tests/test_cat_eval_adapter_match.py \
  tests/test_e2e_dataset_mix.py \
  tests/smoke/test_loss_pipeline_smoke.py \
  tests/smoke/test_one_step_train_smoke.py -q
```

**Result:** `9 failed, 121 passed`（12.12s）

失败集合与单独跑 `test_e2e_dataset_mix.py` 完全相同；其余四文件合计 `51+23+3+5=82` 全部通过。**未见跨文件状态污染**（无新增失败、无顺序依赖型变红）。

## Full Suite (`pytest tests -q`)

### Before local fix

`11 failed, 244 passed`：

- 上述 9 个 known `test_e2e_dataset_mix` 失败；
- **本 patch 引入的 2 个失败：**
  - `tests/test_cat_compressed_lora_scope.py::test_full_route_calls_materialize_not_subspace`
  - `tests/test_cat_compressed_lora_scope.py::test_subspace_route_calls_wrap_inject_not_materialize`
  - 根因：`_ResolvedDistillStageConfig` 新增必填字段 `prompt_kd_weight`，但 `_fake_cfg()` 未传入 → `TypeError: missing 1 required positional argument: 'prompt_kd_weight'`。
  - 对照：`git stash` 后 clean HEAD 上同文件 **16 passed**；因此判定为 **本 patch 回归**。

### Local fix applied

在 `tests/test_cat_compressed_lora_scope.py` 的 `_fake_cfg()` 中增加 `prompt_kd_weight=0.0`。

注意：`.gitignore` 含 `tests/`，且该文件 **未被 git 跟踪**（`git ls-files` 为空；`git check-ignore` 命中）。修复存在于本地工作区文件，但 **不会出现在 `git status` / `git diff`**。生产路径 `_resolve_distill_stage_config()` 已正确传入 `prompt_kd_weight`。

### After local fix

```text
9 failed, 246 passed, 2 warnings
```

仅剩已知 9 个 `test_e2e_dataset_mix` 失败；`test_cat_compressed_lora_scope.py` 16 passed。  
`pytest tests -q --ignore=tests/test_e2e_dataset_mix.py` 在修复后为全绿（此前仅上述 2 个 scope 失败）。

## Manual Chain Audit

### Category: CLI → OverrideTable → runtime → stage → CustomSFTTrainer → shared mask

| Step | Evidence | Status |
|---|---|---|
| CLI | `train_utils/cat_train_args.py`：`--distill_prompt_kd_weight`，default `"default=0.0"`；`_DISTILL_PROMPT_KD_WEIGHT_SPEC` + `_parse_nonnegative_float_text` | PASS |
| OverrideTable | `NormalizedCatArgs.distill_prompt_kd_weight`；`_normalize_cat_train_script_args` 解析；`cat_train_pipeline.py` 把该 table 列入 distill override 校验 | PASS |
| Runtime | `resolve_distill_runtime_config` → `ResolvedDistillRuntimeConfig.prompt_kd_weight` | PASS（手工：default→0.0；`after:q_proj=0.05`→对应类别 0.05） |
| Stage | `lora_utils._resolve_distill_stage_config` / `_ResolvedDistillStageConfig.prompt_kd_weight`；`_build_lora_trainer(... prompt_kd_weight=...)` | PASS |
| Trainer | `CustomSFTTrainer.__init__` 保存并拒绝 `<0`；本地 `build_token_mask` 调 `build_distill_token_mask(..., prompt_kd_weight=self.prompt_kd_weight)` | PASS |
| Shared mask | 全部 tokenwise KD 分支（16 处 `token_mask = build_token_mask(...)`）统一走该 helper；`sft/origin` 不建 KD mask（正确） | PASS |

相关单测：`test_distill_prompt_kd_weight_*`（defaults / override / negative / >1）均在 `test_cat_eval_adapter_match.py` 通过。

### E2E: CLI → args → runtime → trainer → dense / CPU student / gamma

| Step | Evidence | Status |
|---|---|---|
| CLI/args | `compressed_e2e_fintuning/args.py`：`--prompt_kd_weight` default `0.0`；`<0` 拒绝；`mcqa` 且 `!=0` 拒绝 | PASS |
| Runtime | `runtime.py` 日志 + `prompt_kd_weight=float(args.prompt_kd_weight)` 传入 trainer | PASS |
| Trainer helper | `_build_distill_token_mask` → shared `build_distill_token_mask(..., prompt_kd_weight=self.prompt_kd_weight)` | PASS |
| Dense path | `_compute_*` dense：`token_mask = self._build_distill_token_mask(inputs, logits)` → `compute_dense_loss_from_logits` | PASS |
| CPU student loss | teacher-first CPU：`token_mask = self._build_distill_token_mask(inputs, logits)` → `compute_dense_loss_from_offloaded_teacher` | PASS |
| CPU teacher gamma | `_build_cpu_teacher_targets`：`gamma_mask = self._build_distill_token_mask(inputs, teacher_logits)` → `compute_teacher_entropy_mean_and_gamma` | PASS |

相关单测：fractional mask 下 EAKLD dense/offload value+gradient（`test_distill_losses.py` / smoke）均通过。

## `build_distill_token_mask` Branch Coverage

- 定义唯一位置：`train_utils/distill_losses.py::build_distill_token_mask`。
- `train_utils/lora_training.py`：仅通过本地闭包 `build_token_mask` 调用；所有 KD loss 分支使用它。
- `compressed_e2e_fintuning/trainer.py`：仅通过 `_build_distill_token_mask` 调用；覆盖 dense、CPU student、CPU gamma 三处。
- **未发现** trainer 内绕过 shared helper 的遗留 `labels.ne(-100)` KD mask 分支。

## Diff Scope Check

`git diff --stat`（17 files, +852 / -117）：

```
compressed_e2e_fintuning/{README.md,args.py,runtime.py,scripts/e2e_decoder.sh,trainer.py}
docs/cat_train_args.md
scripts/catlora_distill_4gpu_res0.sh
tests/smoke/{test_loss_pipeline_smoke.py,test_one_step_train_smoke.py}
tests/{test_cat_eval_adapter_match.py,test_distill_losses.py,test_e2e_dataset_mix.py}
train_utils/{cat_train_args.py,cat_train_pipeline.py,distill_losses.py,lora_training.py,lora_utils.py}
```

扫描结论：

- **无** truncation / max_length / cutoff 逻辑改动；
- **无** 样本过滤逻辑改动；
- **无** 新 loss type / loss 公式改动（只换 mask 构建与传参）；
- **无** checkpoint 格式改动；
- **无** 数据配比 / dataset mix ratio 改动（`test_e2e_dataset_mix.py` 仅增 prompt_kd 相关用例）；
- **无** optimizer / LR 改动；
- 默认脚本显式写入 `0.0`：`scripts/catlora_distill_4gpu_res0.sh`、`e2e_decoder.sh`（两处），保持默认行为。

本地额外改动（不在 git diff）：`tests/test_cat_compressed_lora_scope.py`（gitignore / untracked）补 `prompt_kd_weight=0.0`。

## Acceptance Criteria Checklist

| Criterion | Result | Evidence |
|---|---|---|
| 两套 CLI 均存在且默认 0.0 | **PASS** | category `"default=0.0"`；E2E `default=0.0`；resolve/parse 验证 |
| 不传参数与显式 0.0 的 mask 完全相同 | **PASS** | `test_distill_mask_prompt_weight_zero_is_exact_current_behavior` |
| p=0 时现有 response-target KD 数值/梯度语义不变 | **PASS** | 上项 + forward KL / EAKLD 对照测试 |
| p>0 时非 padding prompt target 得到指定权重 | **PASS** | `test_distill_mask_assigns_fractional_prompt_weights_after_causal_shift` 等 |
| response target 始终 1.0 | **PASS** | mask 构造 + 多轮/交错 prompt 测试 |
| negative weight 被拒绝；>1.0 被允许 | **PASS** | mask/CLI：`rejects_negative_*` / `accepts_*_above_one` |
| padding 和 final logits 始终 0 | **PASS** | padding / final-logit / causal-shift 测试 |
| 预测 EOS 的 logits 仍为 1.0 | **PASS** | `test_distill_mask_exactly_matches_next_label_validity` 与 plan 示例语义覆盖（EOS 作为 response target） |
| CE、hidden loss、pre-MLP hidden loss 均未改变 | **PASS** | trainer 中 CE/hidden/pre-MLP 路径不吃 prompt mask；相关 adapter/smoke 测试通过 |
| 所有 category tokenwise KD 分支都使用统一 weighted mask | **PASS** | 16 处 `build_token_mask`；无遗漏分支 |
| E2E dense、CPU student loss、CPU teacher gamma 使用统一 weighted mask | **PASS** | trainer 三处均走 `_build_distill_token_mask` |
| fractional mask 下 EAKLD dense/offload loss 与 gradient 测试通过 | **PASS** | `test_eakld_*fractional*` / `test_cpu_teacher_eakld_*fractional*` |
| 默认脚本行为保持不变 | **PASS** | 脚本显式 `0.0`；默认 resolve=0.0 |
| checkpoint、推理、数据截断和数据配比未改变 | **PASS** | diff 范围审查；无相关逻辑改动 |

## Bugs Found

1. **本 patch 回归（已本地修复，文件 gitignored/untracked）**  
   `tests/test_cat_compressed_lora_scope.py::_fake_cfg` 缺少 `prompt_kd_weight`，导致全量套件 2 失败。  
   修复：补 `prompt_kd_weight=0.0`。  
   验证：修复后该文件 16 passed；全量仅剩已知 9 个 mix 失败。

2. **已知预存在（未修）**  
   `tests/test_e2e_dataset_mix.py` 9 失败（eval_before_save + lazy mix / `dummy.txt`），与本 patch 无关。

## Completion Summary (facts)

新增了可配置 prompt-weighted logit KD；response 固定 1.0，prompt 默认 0.0；0.0 精确保留当前 response-target-only 行为；CE/hidden/pre-MLP 不变；EAKLD teacher entropy/gamma 与 KL 使用同一 weighted mask。

实际测试命令与结果：

- 逐文件：51 / 23 / 39(+9 known fail) / 3 / 5 通过；
- 五文件组合：121 passed + 同 9 known fail，无污染；
- 全量：修复 scope helper 后 246 passed + 同 9 known fail。

未做 downstream accuracy 控制实验，不宣称精度提升。未执行 git commit。
