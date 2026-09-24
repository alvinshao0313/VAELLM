# LiftQuant-style VAELLM recovery

## 当前入口与历史边界

现有 checkpoint 恢复使用 `recover.py`；范围和命令见 [checkpoint 恢复说明](../../docs/experience/records/experiments/liftquant_recovery/CHECKPOINT_RECOVERY.md)。具体机制和结果状态从 [编码恢复经验](../../docs/experience/lessons/liftquant_recovery.md) 进入，不由本首页重复维护。

下面保留早期 `smoke.py` / `run_smoke.sh` 的两阶段冒烟设计，仅说明那条历史路径。它不能代表当前恢复流程，也不构成正式运行授权。

## 早期两阶段冒烟说明

This directory contains an independent two-stage smoke harness. It deliberately starts stage A from the base model path and never consumes an existing VAELLM checkpoint.

- Stage A calls the existing CAT VAE initializer with all seven decoder projection categories, after_category_mode=none, no LoRA, and a fresh output root.
- Stage B loads only the fresh stage-A final model, visits two adjacent blocks, and trains the seven VAELinear modules in the current block jointly. It uses the existing packed decoder and SparseBitTuningManager for continuous all-bit score proxies, hard packed bits with STE, and decoder gradients.
- The smoke keeps the project transpose and protection settings; the smoke uses codebook bits/dim=64 and one residual stage but disables optional two-sided weight rotation to avoid a CPU-bound Hadamard preprocessing path under a shared GPU; the final configuration must restore the recorded rotation setting.
- The block target is full hidden-state MSE with the teacher FP-prefix hidden state injected into the student block. Attention-mask filtering is retained even though the smoke samples are unpadded.
- Each block writes an experiment sidecar containing packed bits and decoder parameters, then a fresh stage-A load is rebuilt and checked against the saved hard forward.

The smoke harness is intentionally tiny (two RedPajama samples, short sequence, one VAE step and one recovery optimizer step per block). It is environment and path validation only; it is not the requested final 4096-sample/2-epoch experiment and does not run downstream lm-eval tasks.

The implementation follows the public LiftQuant reference at commit 72b3875c770e4579639931fed89dc95e4067edac: same FP-prefix input, current-block objective, joint current-block linears, hard repack after optimization, and reload validation. The official ALL-group weight-proxy schedule remains a final-run configuration item; this smoke path validates the VAELLM-native packed-bit score proxy plus trainable decoder path first.

## 全层两组学习率对照（2026-09-24）

本轮具体配置、验证与运行状态见[两组学习率记录](../../docs/experience/records/experiments/liftquant_recovery/2026-09-24_full_layers_lr.md)。`run_full_decoder_1p25e5.sh` / `run_full_decoder_6p25e6.sh` 分别对应本轮授权 GPU 0 / 1，输出目录固定且必须未存在；在已激活 bitvae 的项目根目录调用。不要重复运行覆盖旧输出。

`run_experiment.py` 顺序执行现有 recovery 和相同八任务评测，恢复/保存契约失败则停止。其 `--evaluate-baseline` 额外重评初始模型，其余训练参数原样传给 `recover.py`；正式模式不接受 `--eval-limit`。`--purpose short_validation --eval-limit 1` 仅用于验证衔接，不能当正式分数。

`evaluate_pair.py --residency resident` 在充足显存时将模型和一次预热的 packed BF16 解码权重驻留 GPU；默认仍为 `streamed`。`--only A` 或 `--only B` 可只评对应模型，省略仍评两者；两份输入 metadata 均检查比较范围。`run_lm_eval` 的任务、指标、0-shot 与 batch1 口径不变，正式评测省略 `--limit`。
