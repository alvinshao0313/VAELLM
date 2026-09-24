> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.ai-bridge/TRAINING_STACK_GO_MODE_FINAL_AUDIT.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../README.md) 查阅。

# VAELLM Training Stack Go Mode Final Audit

审计时间：2026-09-06T05:53:33Z  
执行计划：`.ai-bridge/current-plan.md` 及其指定的 2026-09-06 Go Mode 收尾计划  
最终结论：`READY`

## 1. Phase 完成状态

| Phase | 状态 | 结果 |
| --- | --- | --- |
| Phase 0 | 完成 | 修复 torch init 全局污染；分离 v6 training-step 与 stable LoRA 语义；建立 distributed guarded-main；CAT 入口接入 common config；删除旧 CLI contract；修复 mix-bit metric renderer。 |
| Phase 1 | 完成 | checkpoint-distill 使用 common config 和 v6，只支持 `current_lora`、`current_decoder`、`current_lora_decoder`。 |
| Phase 2 | 完成 | 建立唯一 migration-only legacy reader 和 `tools/migrate_checkpoint_v6.py`，固定拒绝 legacy residual/subspace/DoRA/RSLoRA/AdaLoRA。 |
| Phase 3 | 完成 | surviving training/eval/mix-bit consumer 全部迁移到 v6；specialized converter 保留明确 legacy 边界。 |
| Phase 4 | 完成 | 删除 compressed-subspace、独立 block-VAE/block-distill、旧 checkpoint wrapper、旧 loss/CLI 分支及对应废弃测试。 |
| Phase 5 | 完成 | 正式 CAT、checkpoint-distill、E2E 脚本迁移并通过语法及 parser 验证；stage1/stage2 脚本保持删除。 |
| Phase 6 | 完成 | README 和当前训练文档更新为 common config、五种 loss、v6 checkpoint 与迁移边界。 |
| Phase 7 | 完成 | 默认全量 pytest、静态扫描、入口检查、metadata/schema audit 与交付物全部完成。 |

## 2. 最终架构状态

- CAT、E2E、checkpoint-distill 共用 `train_utils/config` 的 public config/CLI truth；CAT 只通过集中 runtime adapter 映射内部调用。
- 模型级蒸馏只保留 `sft`、`kl`、`kl_top`、`kd`、`kd_top` 五种 loss。
- 模型级 LoRA 只保留 plain full-space 方案；existing heterogeneous ranks 由 PEFT `rank_pattern` 精确表达，显式 rank 冲突直接报错，不做 padding 或 SVD。
- v6 `training_step` 保存 exact PEFT adapter topology、optimizer/scheduler/RNG、round base 引用，可精确续训。
- v6 stable `category_boundary` / `final_model` 独立可加载，顶层 `lora_config=null`；finalized low-rank topology 由每个 target payload shape 定义。
- stable save 会拒绝 live PEFT wrapper、LoRA layer、proxy 和非空 `lora_config`；final artifact 不保留可训练 Sparse score，hard bits 已提交。
- rank0-only I/O/probe/save/prune 通过 guarded-main 广播成功或异常，避免其它 rank 卡在 barrier。
- legacy schema 只可通过 `train_utils.legacy_checkpoint_io` 进入迁移或专用 converter；active training/eval 不再直接读取旧主 checkpoint。
- checkpoint-distill progress 与 online CAT `completed_categories` 分离，不污染 category prefix。

## 3. 删除文件清单

- `compressed_e2e_fintuning/runtime.py`
- `compressed_e2e_fintuning/scripts/e2e_stage1_pretrain.sh`
- `compressed_e2e_fintuning/scripts/e2e_stage2_instruct.sh`
- `compressed_e2e_fintuning/trainables.py`
- `docs/block_vae_lora.md`
- `docs/cat_distill_test.md`
- `e2e_common/checkpoint_io.py`
- `e2e_common/compressed_checkpoint.py`
- `e2e_common/compressed_subspace_lora.py`
- `e2e_common/low_rank_lora.py`
- `e2e_common/peft_proxy.py`
- `e2e_common/temporary_mode.py`
- `litebsq/low_rank_scope.py`
- `scripts/block_prefix_eval.sh`
- `scripts/block_vae_lora_simple.sh`
- `tests/test_block_attention_distill_losses.py`
- `tests/test_cat_compressed_lora_scope.py`
- `tests/test_cat_distill_args_contract.py`
- `tests/test_cat_eval_adapter_match.py`
- `tests/test_cat_inline_remaining_lora.py`
- `tests/test_cat_inline_resume_progress.py`
- `tests/test_compressed_subspace_lora.py`
- `tests/test_e2e_checkpoint_io_legacy.py`
- `tests/test_e2e_compressed_lora_scope.py`
- `tests/test_remaining_lora_dataset_seed_cache.py`
- `tools/block_prefix_eval.py`
- `tools/block_vae_lora_train.py`
- `tools/convert_legacy_checkpoint.py`
- `tools/plot_layerwise_recovery_hparam_curves.py`
- `train_utils/block_distill.py`
- `train_utils/block_vae_cache.py`
- `train_utils/block_vae_lora_args.py`
- `train_utils/block_vae_lora_checkpoint.py`
- `train_utils/cat_train_args.py`
- `train_utils/model_checkpoint_io.py`

## 4. Surviving consumer 迁移清单

- `experiments/down_layer_sensitivity/core.py`
- `mix_bit/assembler.py`
- `mix_bit/candidate_artifact.py`
- `mix_bit/module_swap.py`
- `mix_bit/state_fingerprint.py`
- `mix_bit/validation.py`
- `tools/bench_parallel_decode_opt.py`
- `tools/cat_category_prefix_eval.py`
- `tools/cat_eval.py`
- `tools/opencompass_vaellm_model.py`
- `train_utils/cat_residual_from_base.py`
- Specialized converters：`tools/convert_cat_checkpoint_to_bitpack.py`、`tools/extract_down_transfer_artifact.py`

## 5. 正式脚本清单

- CAT：`scripts/catlora_simple.sh`、`scripts/catlora_simple2.sh`、`scripts/catlora_codebook_ab_single.sh`、`scripts/catlora_codebook_ab_down_channel_single.sh`
- checkpoint-distill：`scripts/catlora_distill_4gpu_res0.sh`、`scripts/catlora_distill_from_checkpoint.sh`
- E2E：`compressed_e2e_fintuning/scripts/e2e_decoder.sh`、`scripts/compressed_e2e_simple.sh`
- 八个脚本全部通过 `bash -n`；CAT 六个正式调用和 E2E DP/layer_mp/simple parser-only 验证均通过。

## 6. 测试结果

所有 Python 命令均在 `bitvae` conda 环境执行：

```text
/home/shaoyuantian/anaconda3/envs/bitvae/bin/python
Python 3.11.13
```

- 早期 common/CAT/E2E focused battery：`206 passed`
- v6/migration/mix-bit focused battery：`378 passed`
- metadata/rank/distributed focused battery：`93 passed, 24 warnings`
- heterogeneous rank、stable topology、exact training-step 新增定向测试：`3 passed`
- 最终默认全量：`1048 passed, 24 warnings in 105.87s`，`0 failed`
- 警告为 TRL 默认 `max_seq_length`、Transformers tokenizer deprecation 和 tiny smoke collator 性能提示，不影响 contract 或结果。

## 7. 静态扫描与 metadata/schema audit

- `git diff --check`：通过。
- Active production deleted-symbol scan：通过。
- Active formal script deleted-symbol scan：通过。
- Current docs public deleted-CLI scan：通过；历史 plans/results 和专用 residual 文档按计划排除。
- Deleted-name tests 只存在于显式 rejection/contract allowlist：通过。
- 旧主 loader `train_utils.model_checkpoint_io`、`e2e_common.checkpoint_io` 的 active production import：0。
- `train_utils.legacy_checkpoint_io` active import 仅在 `tools/migrate_checkpoint_v6.py` 和 `tools/convert_cat_checkpoint_to_bitpack.py`：通过。
- Production entry：E2E 仅进入 `runtime_v6`；CAT 进入 common runtime adapter；checkpoint-distill 进入 common config + v6：通过。
- Tiny v6 metadata audit：`format=vaellm_model_checkpoint_v6`、`schema_version=6`、UUID id、round-base id、stable `lora_config=null`、exact `rank_pattern`、prefix/inventory、stable load/forward parity 全部通过。
- 两进程 Gloo guarded-main 成功与 rank0 异常传播测试通过；不存在 rank0 exception 后其它 rank 卡 barrier 的测试失败。

## 8. Hardware-deferred

未在真实大模型、多 GPU CUDA、完整外部数据集上执行端到端训练实验；这些需要计划外硬件、模型和数据。默认测试已覆盖 tiny CPU 模型、两进程 Gloo、step-exact resume、stable roundtrip 和 parser/script contract，不构成 READY blocker。

## 9. Verdict

`READY`

默认全量 pytest 为 0 failed，active legacy/dead reference scan、checkpoint I/O 边界、production entry 和 v6 metadata/schema audit 均通过。工作区修改按要求保留，未执行 `git add`、`git commit`、切分支或创建 worktree。
