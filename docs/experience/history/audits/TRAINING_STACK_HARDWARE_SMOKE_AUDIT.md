> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.ai-bridge/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../README.md) 查阅。

# VAELLM training-stack 真实硬件 smoke 审计

日期：2026-09-06

最终判定：`HARDWARE_BLOCKED`

## 判定说明

代码级 READY 已在真实 Qwen3-8B、真实数据、真实 v6 checkpoint 和 A800 GPU 上完成 CAT、E2E `decoder_sparse_bit`、E2E `decoder_lora` 的单卡/单 rank 生命周期验证。

真实多 GPU 部分为 `hardware-deferred`：本机有 8 张 NVIDIA A800 80GB PCIe，但验收期间只有物理 GPU 3 可以安全使用；GPU 0、1、2、7 被其他用户训练占用，GPU 4、5、6 被其他用户的 VLLM 作业占用。最终复查时 GPU 3 空闲，其余 7 张仍在使用。没有权限也没有理由中断其他用户作业，因此无法安全启动两卡以上 DP collective 或真正跨 GPU 的 layer_mp。

这不是 PASS：多卡目标尚缺真实硬件证据，所以总判定必须是 `HARDWARE_BLOCKED`。

## 验收基线

- 已读取 `.ai-bridge/TRAINING_STACK_GO_MODE_FINAL_AUDIT.md`。
- 已读取 `docs/superpowers/plans/2026-09-06-vaellm-training-stack-go-mode-completion-plan-cn.md`，未重新执行 Go plan，也未继续架构重构。
- CAT 参数语义来自 `scripts/catlora_simple2.sh`。
- E2E DP 参数语义来自 `compressed_e2e_fintuning/scripts/e2e_decoder.sh`。
- Python 环境：`/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`，Python 3.11.13。
- 真实基础模型：`/home/shaoyuantian/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`。
- 正式脚本引用的 `/root/data/...` checkpoint 因目录权限不可读；CAT smoke 使用同一真实 Qwen3-8B 生成合法 v6 student checkpoint，后续 E2E 全部从该真实 v6 checkpoint 启动。
- 真实训练数据使用正式 EdgeRazor mix：`edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009`。
- 所有大 artifact 位于 `/tmp/vaellm_hardware_smoke_20260906`，未写入仓库。

## 1. CAT 主链真实 smoke

状态：单 GPU PASS；多 GPU CAT 不在本次明确验收项中。

### 配置与生命周期

- 使用 common CAT CLI：`python tools/cat_train.py`。
- 保留正式 recipe 的 BSQ、双 residual stage、32 bit/codebook dim、channel protection、`remaining_lora_prefix_decoder`、真实数据 mix、KL top-k、LoRA/decoder/aux 参数语义。
- smoke 缩短项：模型长度 128、样本 2、q_proj 每 stage 1 个 VAE optimizer step、after-category recovery 2 个 optimizer step；随后从 q boundary 恢复并完成 k_proj（每 stage 100 个短 VAE step）与 2 个 recovery step。
- q_proj：36 个真实 linear 完成双 stage compression；recovery 有 19,612,928 个 trainable parameter，真实执行 backward 和 2 次 optimizer step。
- q_proj round-base 保存约 29.4 秒，category-boundary 保存约 21.8 秒。
- q_proj boundary 保存后，fresh process 独立加载并 forward 成功：load 8.785 秒，36 个 VAELinear，logits shape `(1, 4, 151936)`，forward peak allocation 16087.1 MiB。
- resume 明确读取 `completed_categories=q_proj`，跳过已完成 q_proj，进入 k_proj。
- k_proj recovery 有 18,020,864 个 trainable parameter，真实执行 2 次 optimizer step；完整 resume 进程耗时 2:00.34，退出码 0。
- k_proj round-base 保存约 29.5 秒，category-boundary 保存约 18.9 秒，final_model 保存约 19.8 秒。

### v6 结果

q boundary：

- `format=vaellm_model_checkpoint_v6`，`schema_version=6`，`checkpoint_kind=category_boundary`。
- `completed_categories=[q_proj]`。
- `compressed_targets=36`，`pending_dense_targets=36`，`skip_targets=0`。
- 顶层 `lora_config=null`。
- state_dict 中 PEFT/FullCompressedPeftProxy key 均为 0。

k boundary / CAT final：

- `completed_categories=[q_proj,k_proj]`。
- `compressed_targets=72`，`pending_dense_targets=0`，`skip_targets=0`。
- 顶层 `lora_config=null`。
- final_model 文件约 15.1 GB。

### 资源与异常

- q 进程 GPU 3 观测峰值 33645 MiB；resume 进程峰值 33433 MiB。
- recovery 实测约 3.2–4.3 秒/step。
- 修复后没有 CUDA OOM、distributed hang、device mismatch、decoder pack 错误或 PEFT/proxy residue。
- 关键 artifact：
  - q boundary：`/tmp/vaellm_hardware_smoke_20260906/cat_smoke6/home_shaoyuantian_.cache_huggingface_hub_models--Qwen--Qwen3-8B_snapshots_b968826d9c46dd6066d109eabc6255188de91218_20260906_125451/after_q_proj`
  - resume final：`/tmp/vaellm_hardware_smoke_20260906/cat_resume2/home_shaoyuantian_.cache_huggingface_hub_models--Qwen--Qwen3-8B_snapshots_b968826d9c46dd6066d109eabc6255188de91218_20260906_130026/final_model`
  - 日志：`/tmp/vaellm_hardware_smoke_20260906/cat_smoke6/first_run.log`、`/tmp/vaellm_hardware_smoke_20260906/cat_resume2/resume_run.log`。

## 2. E2E DP `decoder_sparse_bit` 真实 smoke

状态：单 rank DP 路径 PASS；真实多 GPU DP 为 `hardware-deferred`。

### 配置与中断恢复链

- 从上述 CAT v6 final_model 启动。
- 使用 `torchrun --standalone --nproc_per_node=1`；`train_mode=decoder_sparse_bit`，target layers 0–35，target modules all，正式 KL/hidden loss、Sparse Bit `rms_sgd`、decoder/norm/lm-head 参数语义。
- smoke 缩短为 20 optimizer steps，batch size 1，sequence length 128，`save_steps=5`。
- 第一次进程执行到 step 13 后人工中断；此前已产生 `checkpoint-5`、`checkpoint-10`。
- fresh process 从 `checkpoint-10` 恢复，第一条 resumed training 记录为 global_step 11，随后连续运行到 global_step 20。resume 进程创建到首个 teacher/step 统计约 28.6 秒。
- 该恢复链最初在 finalization 暴露真实数值收口缺陷；修复后从同一恢复链留下的 `checkpoint-20` 再由 fresh process exact-load，日志明确记录 `global_step=20 max_steps=20`，没有额外执行 step 21，直接 finalization，耗时 59.89 秒，退出码 0。
- 恢复链最终 `run_meta.global_step=20`。

### checkpoint 内容与 exact state

保留的 clean `checkpoint-15` / `checkpoint-20` 各约 386 MB（约 369 MiB），均包含：

- `optimizer.pt`
- `scheduler.pt`
- `rng_state.pth`
- `trainer_state.json`
- `training_model_state.pt`
- `sparse_bit_tuning/exact_state.pt`
- `checkpoint_meta.json`

checkpoint 中没有完整 8B model；最大文件是约 277 MB 的 Sparse Bit exact sidecar。

exact state 证据：

- global bit round 从 15 连续到 20。
- 144 个 banks、144 个 packed banks、6 个 score chunks、144 个 sampler states 均存在。
- checkpoint-15 到 checkpoint-20 有 5/144 个 packed banks 实际变化。
- `training_model_state.pt` 有 434 个 mutable key，其中 240 个在 step 15→20 发生变化；238 个是 decoder，2 个是 aux 参数。
- HF optimizer 中 `_sparse_bit_main_optimizer` 保存 3 个 param group、434 个 state entry，decoder/aux optimizer moments 被恢复并继续更新。
- 正式 `bit_optimizer=rms_sgd` 本身是无 moment 的 Sparse Bit optimizer；其 exact sidecar 仍保存 optimizer 类型、LR、score、packed bits、round/sampler 状态。Sparse Bit Adam/AdamW moment 在本配置中不适用。

### finalization 与 fresh load

- v6 final_model：`lora_config=null`，`completed_categories=[q_proj,k_proj]`，`compressed_targets=72`，pending/skip 均为 0。
- `finalized_status`：Sparse Bit committed、decoder finalized、aux finalized、runtime clean、inference parity 全部为 true。
- 从恢复链生成的 final_model 由 fresh process 独立加载并 forward 成功：load 8.953 秒，72 个 VAELinear，72 个 packed decoder，PEFT/proxy 均为 0，logits shape `(1, 5, 151936)`，forward peak allocation 16156.2 MiB。

### 资源与异常

- 相对空闲时约 1.9–3.3 秒/step；共享 GPU 受其他作业竞争时约 4.4–6.2 秒/step。
- checkpoint 文件发布区间约 1.3–1.6 秒，训练可见 pause 约 2 秒。
- clean 20-step 进程 GPU 3 观测峰值 47156 MiB，启动基线 8715 MiB，增量约 38441 MiB；teacher 首步内部 peak allocation 34225511424 bytes（约 31.9 GiB）。
- finalization + final save 约 34–41 秒。
- 没有 CUDA OOM、NCCL hang 或 device mismatch。torchrun 正常退出时曾提示 process group 未显式销毁；E2E main 已增加 finally teardown，并有成功/异常双路径回归。
- 关键 artifact：
  - 恢复链 checkpoint-20：`/tmp/vaellm_hardware_smoke_20260906/e2e_dp_sparse_resume_final/home_shaoyuantian_.cache_huggingface_hub_models--Qwen--Qwen3-8B_snapshots_b968826d9c46dd6066d109eabc6255188de91218_20260906_131053/trainer_state/checkpoint-20`
  - 恢复链 final_model：`/tmp/vaellm_hardware_smoke_20260906/e2e_dp_sparse_resume_final/home_shaoyuantian_.cache_huggingface_hub_models--Qwen--Qwen3-8B_snapshots_b968826d9c46dd6066d109eabc6255188de91218_20260906_131053/final_model`
  - clean 20-step checkpoint 对照：`/tmp/vaellm_hardware_smoke_20260906/e2e_dp_sparse_final_clean2/home_shaoyuantian_.cache_huggingface_hub_models--Qwen--Qwen3-8B_snapshots_b968826d9c46dd6066d109eabc6255188de91218_20260906_132430/trainer_state`
  - 中断/恢复日志：`/tmp/vaellm_hardware_smoke_20260906/e2e_dp_sparse_retry/initial_run.log`、`/tmp/vaellm_hardware_smoke_20260906/e2e_dp_sparse_retry/resume_run.log`。

## 3. E2E layer_mp + decoder/LoRA 更新 smoke

状态：单 GPU layer_mp 路径 PASS；真实跨 GPU layer_mp 为 `hardware-deferred`。

- `train_mode=decoder_lora`，target layers 0–35，72 个 q/k compressed target；decoder LR 与 LoRA LR 均为 `3e-6`，2 optimizer steps。
- 以 plain `python` 启动，未建立 DP collective；`layer_device_map=auto` 在唯一可见 GPU 下解析为 36 层全部 `cuda:0`。
- fresh process 从真实 `checkpoint-1` 精确恢复并完成 global_step 2、finalization 与 final save，退出码 0。
- checkpoint-1→checkpoint-2：576 个 mutable key 中 338 个发生变化，其中 266 个 decoder key、72 个 LoRA key；optimizer 有 2 个 param group、576 个 state entry。因此 decoder 与 LoRA 均有真实更新。
- finalization 前后 BF16 core、structural、runtime-cleanup、end-to-end probe 均为 `max_abs=0`、`relative_l2=0`。
- final_model：v6/schema 6、`lora_config=null`、72 compressed target、pending/skip 为 0，decoder 与 LoRA finalized、runtime clean。
- fresh process 独立加载并 forward 成功：load 9.843 秒，72 个 VAELinear、72 个 packed decoder、72 个 low-rank payload，PEFT/proxy 均为 0，logits shape `(1, 7, 151936)`，forward peak allocation 16167.4 MiB。
- resume/final 进程 GPU 3 观测峰值 40070 MiB，启动基线 6702 MiB，增量约 33368 MiB；单步约 4–8 秒（同卡有其他用户短作业竞争），finalization + save 约 25.7 秒。
- 没有 CUDA OOM、hang、device mismatch 或 decoder pack 错误。
- 关键 artifact：`/tmp/vaellm_hardware_smoke_20260906/e2e_layer_mp_lora_resume2/home_shaoyuantian_.cache_huggingface_hub_models--Qwen--Qwen3-8B_snapshots_b968826d9c46dd6066d109eabc6255188de91218_20260906_133345`。

## smoke 暴露并修复的问题

仅做了验收所需的最小正确性修复，没有继续架构重构：

1. common CLI 在 `argv=None` 时没有读取真实 `sys.argv`。
2. CAT target collection 使用了错误关键字参数。
3. CAT runtime snapshot 不能稳定序列化 frozenset/Namespace 私有 callable。
4. CAT boundary tokenizer 保存路径包含硬编码未定义 token 变量。
5. Qwen3 context-sensitive ChatML 多轮 response mask 不能依赖 prefix tokenization；改为使用 fast-tokenizer offsets 对齐真实 role/end 边界。
6. decoder finalization 曾重新 unpack 已训练 packed decoder，导致 BF16 数值变化；现在保留训练后的 packed payload。
7. Accelerate FP32 output wrapper 未在结构 finalization 前移除，造成伪 dtype mismatch。
8. LoRA finalization 原先把两段低秩支路预先合成整权重，真实 Qwen3-8B 上 relative-L2 达 0.853%；现在保留 payload dtype，并在稳定 VAELinear 中执行独立低秩支路，真实 finalization parity 为 0。
9. 已完成 checkpoint resume 会让 IterableDataset 多跑 step 21；现在 exact-load 后直接 finalization，并拒绝 `global_step > max_steps`。
10. E2E torchrun 入口补充 process-group finally teardown。
11. lm_head 大矩阵线性融合在 BF16 下存在预期的累加顺序舍入；只对这个明确重结合使用 `atol=0.25、rtol=1e-3、relative_l2<=0.005`，decoder/LoRA core parity 保持严格。

## 回归测试

在 `bitvae` 环境运行受影响的完整测试文件集合：134 passed；新增 E2E main teardown 测试：2 passed。合计 136 passed。

`git diff --check` 通过。

## 最终矩阵

| 验收项 | 结果 |
|---|---|
| CAT 真实模型 compression/recovery/boundary save | PASS |
| CAT boundary fresh load/forward | PASS |
| CAT boundary resume 进入下一 category | PASS |
| v6 inventory / `lora_config=null` / 无 residue | PASS |
| E2E DP 单 rank 20-step decoder + Sparse Bit | PASS |
| E2E exact interruption/resume/finalization/fresh load | PASS |
| E2E checkpoint 非完整 8B + HF/exact sidecars | PASS |
| E2E layer_mp 单 GPU路径 + decoder/LoRA 实际更新 | PASS |
| 真实两卡以上 DP collective | `hardware-deferred` |
| 真实跨 GPU layer_mp 通信与放置 | `hardware-deferred` |

最终判定：`HARDWARE_BLOCKED`
