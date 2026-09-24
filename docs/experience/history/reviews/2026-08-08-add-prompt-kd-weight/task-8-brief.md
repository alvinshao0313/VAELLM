> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-8-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 8: Update Experiment Scripts, Preserve Defaults

**Files:** two shell scripts

- [ ] `scripts/catlora_distill_4gpu_res0.sh` 在 distill loss 参数附近显式增加 `--distill_prompt_kd_weight "default=0.0"`。
- [ ] `compressed_e2e_fintuning/scripts/e2e_decoder.sh` 的 DP 和 layer_mp 两个分支都显式增加 `--prompt_kd_weight 0.0`。
- [ ] 不默认改成 0.05/0.1；本次只增加能力，不改变现有实验。
- [ ] 按 `AGENTS.md` 不新增 shell 中间变量包装该超参数。

---

