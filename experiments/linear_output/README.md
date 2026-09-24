# Independent Linear Output W2

This directory is isolated from CAT/E2E entry points. The current first-stage
algorithm has one VAE per target Linear, keeps the original
[out_features, in_features] layout, splits the input axis into 32-element blocks,
uses one residual-free stage, and optimizes full Linear output alignment. The formal W2 setting uses codebook_bits=64 and codebook_dim=32 (64 coded bits per 32 weights).

For valid teacher tokens X and original weight W, the target is Y = X W^T. The
decoded blocks are accumulated before the loss is taken:

  Y_hat = sum_k X_k W_hat_k^T
  L_out = mean((Y_hat - Y)^2)

The BSQ auxiliary loss is unchanged and is evaluated over the complete weight-block
set used by the update. The reference implementation is in output_kernel.py;
the Triton implementation is used on CUDA and has the same backward derivative.
The first comparison mode uses the frozen original teacher input. A later, separate
student-input/teacher-target mode is an error-compensation ablation.

## 实验记录

阶段交付、取消的对照及历史输出的保留情况，以 [结果总结](../../docs/experience/records/experiments/linear_output/EXPERIMENT_SUMMARY.md) 为准；方法经验见 [输出损失经验](../../docs/experience/lessons/output_losses.md)。本页的命令是实现检查示例，不表示历史整组实验仍在运行。

## Checks

Run in the bitvae environment:

```bash
python -m pytest -q tests/test_linear_output.py
python -m compileall -q experiments/linear_output
```

The q_proj smoke validates a real one-stage VAE update, Triton output loss, native
packed export, and v6 reload:

```bash
CUDA_VISIBLE_DEVICES=4 python -m experiments.linear_output.run \
  --model_path Qwen/Qwen3-8B \
  --output_dir result/linear_output/smoke_single_stage_q0 \
  --target_linears model.layers.0.self_attn.q_proj \
  --objective linear_output_mse \
  --steps 2 --batch_size 8 --calibration_microbatch_size 1 \
  --dataset_mix "edgerazor_ii_7m=0.614,edgerazor_ii_gen=0.121,edgerazor_tulu=0.050,edgerazor_am=0.115,vaellm_eval_task=0.100" \
  --dataset_task sft --model_max_length 1024 --dynamic_padding true
```

The q_proj smoke is an implementation check, not a W2 quality result. Full
efficacy requires running all seven projection categories (excluding lm_head),
then evaluating both the teacher-input A/B and the separate student-input
error-compensation ablation with the same downstream recipe.
