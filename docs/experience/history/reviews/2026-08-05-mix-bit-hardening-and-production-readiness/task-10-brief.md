> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-05-mix-bit-hardening-and-production-readiness/task-10-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 10: Full Regression, Static Guards and Production Smoke Gates

**Files:**
- Modify: `mix_bit/README.md`
- Modify tests only if a test exposes a real implementation defect;不得为让测试通过而放宽 contract。

### Static search gates

以下搜索必须无结果：

```bash
! grep -En 'exec python tools/cat_train.py|python tools/cat_train.py' mix_bit/scripts/train_candidate_single.sh
! grep -En 'shifted\.detach\(\)\.cpu\(\)' mix_bit/teacher_cache.py mix_bit/cost_search.py
! grep -En 'reference_state[[:space:]]*=|cpu\(\)\.clone\(\)' mix_bit/assembler.py
```

以下无 timeout queue get 必须不存在于 baseline/ready 路径。人工检查 `mix_bit/cost_table.py` 中每个 `result_queue.get`：

- startup/ready 必须带 timeout；
- 只允许测试 helper 或明确已经有 timeout 的 job loop。

### Unit and regression commands

- [ ] **Step 1: Run all mixed-bit tests**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest mix_bit/tests -q
```

Expected: all pass, zero unexpected skip；CUDA-specific tests may skip only when CUDA unavailable。

- [ ] **Step 2: Run repository integration regressions**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m pytest \
  tests/test_cat_train_candidate_artifact_hook.py \
  tests/test_model_utils_auto_loader.py \
  tests/test_e2e_checkpoint_io_legacy.py \
  tests/test_temporary_switch_residency.py \
  tests/test_distill_losses.py -q
```

- [ ] **Step 3: Run all CLI help commands**

每个命令必须 exit 0：

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.build_model_inventory --help
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.train_candidate_pool --help
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.inventory_candidate_pool --help
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.prepare_uniform_baseline --help
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.prepare_calibration --help
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.build_teacher_cache --help
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.compute_cost_table --help
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.solve_allocation --help
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.assemble_mixed_model --help
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.validate_mixed_model --help
```

- [ ] **Step 4: Build real Qwen3-8B inventory**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.build_model_inventory \
  --run_config mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json \
  --output .result/mix_bit/qwen3_8b/model_inventory.json
```

Required output：

```text
C=7
L=252
block_count=36
```

- [ ] **Step 5: Run Qwen candidate dry-run with the pinned interpreter**

```bash
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python -m mix_bit.cli.train_candidate_pool \
  --run_config mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json \
  --inventory .result/mix_bit/qwen3_8b/model_inventory.json \
  --gpus 4,5,6,7 \
  --dry_run
```

Required output：

```text
total_trials=35
dry_run_unique_commands=35
```

人工检查每条 command 的第二个位置参数是 `/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`。

- [ ] **Step 6: Run the tiny end-to-end integration in both KL modes**

`test_tiny_integration.py` 必须覆盖：

- exact valid candidate mode contract；
- tensor-free baseline；
- teacher cache compact outputs；
- student K-way gather；
- complete atomic cost rows；
- solver；
- final tokenizer save；
- state fingerprint；
- strict reload/validation。

- [ ] **Step 7: Run one deliberate Qwen 1-step candidate smoke only after all tests pass**

该步骤是人工 production smoke，不写入正式 35-job pool。使用临时 run/profile/recipe/space 放在 `.result/mix_bit/_smoke_config/`，固定：

```text
model = Qwen/Qwen3-8B
category = q_proj only
mode = b16d32s2 only
steps_per_category = 1
batch_size = 128
gpu = 4
output_root = .result/mix_bit/_smoke_candidate_pool
```

验收：

- subprocess 使用 `bitvae/bin/python`；
- exit 0；
- artifact 有且只有 `module_state.pt`、`candidate_meta.json`、`completed.json`；
- module count 36；
- 每个 module contract 为 16 bits、32 dim、2 stages；
- pool index 成功；
- 从 Qwen backbone 安装 36 个 q_proj 后，4-token forward logits 全 finite；
- 完成后删除 `.result/mix_bit/_smoke_config` 和 `.result/mix_bit/_smoke_candidate_pool`。

不得把临时 smoke config 加入 git。

- [ ] **Step 8: Verify no full production workload was accidentally launched**

确认没有新增 35 个 long-running process，没有创建正式 Cost rows，没有覆盖用户已有 `.result` 生产实验。

- [ ] **Step 9: Update README failure gates**

README 必须明确列出：

- mode/payload mismatch；
- wrong Python executable；
- top-k full-logits CPU transfer regression；
- worker startup/runtime death；
- tokenizer fingerprint mismatch；
- custom manifest root mismatch；
- final state fingerprint mismatch。

- [ ] **Step 10: Final git diff scope review**

只允许本计划文件表中的源码、测试和 README 变化。不得删除或改写用户已有实验脚本和 distillation 代码。

- [ ] **Step 11: Final commit**

```bash
git add mix_bit tests/test_cat_train_candidate_artifact_hook.py docs/superpowers/plans/2026-08-05-mix-bit-hardening-and-production-readiness.md
git commit -m "test: harden mixed-bit production workflow"
```

提交前使用 `git diff --cached --name-only`，若出现本计划以外路径，必须取消暂存这些文件，不得一起提交。

---

## Final Acceptance Matrix

| Gate | Required result |
|---|---|
| Candidate mode metadata | 五字段与 candidate space 完全一致 |
| Actual candidate structure | stages/dim/logical bits/decoder dims 完全一致 |
| Candidate resume | stale/mislabeled artifact 必须重新训练 |
| Candidate subprocess | 使用父进程 absolute `sys.executable` |
| Teacher top-k | 只搬 `[N_valid,K]` 到 CPU |
| Student top-k | 直接 gather `[B,T,K]`，不生成/搬运 `[N_valid,V]` |
| Exact KL | 数学定义与旧实现一致 |
| Final state verification | 流式 16 MiB SHA，无完整 CPU clone |
| Worker startup | 900 秒 timeout，child death 立即失败 |
| Worker runtime | 任一 worker 非预期死亡立即失败 |
| Custom pool root | `--pool_manifest.parent` 是唯一真实 root |
| Calibration tokenizer | fingerprint v2，core/chat template/added vocab 均受保护 |
| Final tokenizer | final dir local-only 可重载且 fingerprint 相同 |
| Qwen inventory | 36 blocks、7 categories、252 linears |
| Candidate planner | 35 unique s2 jobs |
| Full tests | mixed-bit + selected repository regressions 全通过 |

---

## Cursor Stop Conditions

Cursor 遇到以下任一情况必须停止当前 Task、保留失败日志并报告，不得自行改设计：

1. 真实 `VAELinear` spec 无法用本计划列出的字段确定 codebook bits、dim 或 stages。
2. `validate_bitpack_u8_spec` 返回结构与当前 checkpoint metadata 不一致。
3. top-k direct gather 与现有 dense reference 数值误差超过 `rtol=1e-5, atol=1e-6` 的 float32 范围。
4. bfloat16 cache 经过 float32 renormalization 后与当前 metric 单测不一致。
5. `save_model_checkpoint` 显式传入 tokenizer 后仍无法生成 local-only 可加载 tokenizer。
6. streaming fingerprint 对保存前/重载后模型产生不同 hash，但逐 tensor 值检查表明相同。
7. 自定义 pool manifest 的 parent 不是实际 artifact root。
8. worker liveness 修改导致正常 worker 被误判死亡。
9. Qwen smoke 需要修改正式训练 recipe 或候选空间文件。
10. 任何修复需要改动 `mix_bit/solver.py`、VAE loss、distillation 或 BSQ 核心。

停止报告必须包含：失败命令、完整 traceback、涉及文件、actual/expected 值和当前 Task 编号。不得用放宽测试、提高容差、删除门禁或 fallback 继续收口。
