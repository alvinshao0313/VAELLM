> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-2-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 2: Enforce the Contract During Export, Pool Indexing and Resume

**Files:**
- Modify: `mix_bit/candidate_artifact.py`
- Modify: `mix_bit/checkpoint_pool.py`
- Modify: `mix_bit/candidate_pool.py`
- Modify: `mix_bit/tests/test_candidate_artifact.py`
- Modify: `mix_bit/tests/test_checkpoint_pool.py`
- Modify: `mix_bit/tests/test_candidate_pool.py`
- Modify: `tests/test_cat_train_candidate_artifact_hook.py`

**Interfaces:**

Add in `candidate_pool.py`:

```python
def validate_trial_completion(trial: TrialSpec) -> None:
    """Raise ValueError unless the existing artifact exactly belongs to this trial."""


def is_trial_complete(trial: TrialSpec) -> bool:
    """Return True only if validate_trial_completion succeeds."""
```

### Export contract

In `save_candidate_artifact_from_model`:

1. Parse `trial_spec["mode"]` with `candidate_mode_from_payload` before creating output files。
2. After `_collect_vae_linear_specs` and before `torch.save`，对每个 selected spec 调用 `validate_module_spec_mode_contract`。
3. 若任一模块不匹配，不得创建或覆盖：
   - `module_state.pt`
   - `candidate_meta.json`
   - `completed.json`
4. 若 output directory 已存在旧 `completed.json`，验证失败时不得保留一个看似 complete 的旧文件；进入导出前先只删除旧 `completed.json`，不删除旧 state/meta，最终成功后再原子重写 completed。
5. `candidate_meta.mode` 保存 parsed mode 的五字段 canonical dict，不直接原样复制任意 trial JSON。

### Pool index contract

在 `_load_and_validate_artifact`：

1. 用 `validate_mode_payload(meta["mode"], expected_mode)` 替换当前只比较 `name` 的逻辑。
2. 在 `_validate_module_spec_against_target` 后，对每个 spec 调用 `validate_module_spec_mode_contract(spec, mode)`。
3. mode contract 必须在创建 `ModuleCandidate` 前完成。
4. 错误信息必须包含 `category/mode/module_name`。

### Resume contract

`validate_trial_completion` 必须验证：

- 三个 artifact 文件存在；
- completed 和 meta 的 `format` 均为 `vaellm_candidate_modules_v1`；`module_state.pt` 是 tensor state 文件，不存在独立 format 字段，只验证文件 hash 和 payload summaries；
- state SHA 与 meta、completed 同时一致；
- meta SHA 与 completed 一致；
- 以下值与 TrialSpec 完全一致：
  - `run_config_sha256`
  - `candidate_space_sha256`
  - `training_recipe_sha256`
  - `model_profile_sha256`
  - `model_inventory_fingerprint`
  - `category_name`
  - mode 五字段
  - `expected_module_names`，包括顺序和无重复；
- module spec name set 与 expected names 完全一致；
- 每个 module spec 通过 Task 1 mode contract；
- completed `module_count == len(expected_module_names)`。

`is_trial_complete` 只能捕获：

```python
(FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError)
```

不得用裸 `except Exception`。

`_run_one_trial` 在 subprocess exit code 0 后必须调用 `validate_trial_completion`；失败时把完整错误写入 trial log 并把 exit code 改为 1。

- [ ] **Step 1: Add the exact negative export tests**

在 `test_candidate_artifact.py` 完整实现：

- `test_export_rejects_trial_s2_when_actual_module_is_s1`
- `test_export_rejects_trial_mode_when_actual_codebook_dim_differs`
- `test_export_rejects_trial_mode_when_actual_vq_logical_bits_differ`
- `test_export_rejects_trial_mode_when_decoder_in_dim_differs`
- `test_export_does_not_write_completed_on_contract_failure`

测试必须构造真实或测试替身 `VAELinear` module spec，不允许只测试 Task 1 helper。

- [ ] **Step 2: Add the exact pool-index negative tests**

在 `test_checkpoint_pool.py` 完整实现：

- `test_pool_rejects_same_mode_name_with_wrong_nominal_bit`
- `test_pool_rejects_same_mode_name_with_wrong_codebook_bits`
- `test_pool_rejects_same_mode_name_with_wrong_codebook_dim`
- `test_pool_rejects_same_mode_name_with_wrong_residual_stages`
- `test_pool_rejects_mislabeled_s2_artifact_with_s1_module_spec`
- `test_pool_rejects_mislabeled_artifact_with_wrong_vq_logical_bits`

其中 `test_pool_rejects_mislabeled_s2_artifact_with_s1_module_spec` 必须复现审查中已经确认的漏洞：旧实现会接受，修复后必须抛 `ValueError`。

- [ ] **Step 3: Add resume-invalidating tests**

在 `test_candidate_pool.py` 完整实现：

- `test_resume_retrains_when_mode_metadata_differs`
- `test_resume_retrains_when_inventory_fingerprint_differs`
- `test_resume_retrains_when_expected_module_order_differs`
- `test_resume_retrains_when_module_spec_mode_contract_fails`
- `test_resume_accepts_exact_valid_artifact`

断言 `is_trial_complete` 返回 False/True；同时直接调用 `validate_trial_completion` 检查错误信息。

- [ ] **Step 4: Run the focused tests and confirm at least the mislabeled-artifact test fails on old code**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  mix_bit/tests/test_candidate_artifact.py \
  mix_bit/tests/test_checkpoint_pool.py \
  mix_bit/tests/test_candidate_pool.py -q
```

- [ ] **Step 5: Wire the shared contract into export**

不要复制 Task 1 验证逻辑到 `candidate_artifact.py`。

- [ ] **Step 6: Wire the shared contract into pool indexing**

删除旧的“只比较 mode name”条件。

- [ ] **Step 7: Replace weak resume detection with full trial validation**

保持 public `is_trial_complete(trial)` 名称，避免无关调用方改动。

- [ ] **Step 8: Run focused tests**

Expected: all PASS。

- [ ] **Step 9: Run candidate hook regression**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  tests/test_cat_train_candidate_artifact_hook.py \
  mix_bit/tests/test_candidate_artifact.py \
  mix_bit/tests/test_checkpoint_pool.py \
  mix_bit/tests/test_candidate_pool.py -q
```

- [ ] **Step 10: Commit Task 2 files only**

```bash
git add \
  mix_bit/candidate_artifact.py \
  mix_bit/checkpoint_pool.py \
  mix_bit/candidate_pool.py \
  mix_bit/tests/test_candidate_artifact.py \
  mix_bit/tests/test_checkpoint_pool.py \
  mix_bit/tests/test_candidate_pool.py \
  tests/test_cat_train_candidate_artifact_hook.py
git commit -m "fix: validate candidate artifacts before resume and indexing"
```

---

