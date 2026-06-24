# cat_residual_from_base

`tools/cat_residual_from_base.py` 用于在已训练好的 base VAE checkpoint 上追加一次 residual/outlier protection。

它不会重新训练 base VAE stage1/stage2，也不会做 grid search。每次运行只接受一组明确参数，输出一个新的 checkpoint。

## 输入和输出

输入：

- `--model_path`：原始 dense model，用于恢复 `original_weight`。
- `--base_vae_checkpoint`：已训练 base VAE checkpoint。可传 run 目录、`final_model` 目录或 `checkpoint_meta.json`。
- `--target_categories`：要处理的 linear category。

输出目录：

```text
output_dir/
  config.json
  metrics.json
  payload_summary.json
  residual_from_base.log
  checkpoint/
  completed.json
```

`checkpoint/` 是新的 residual-augmented checkpoint，不会覆盖 `--base_vae_checkpoint`。

如果 `output_dir` 已存在，默认直接报错；需要复用目录时显式传 `--overwrite`。

## 支持模式

`--outlier_protect_mode none`

只加载 base checkpoint 并另存一份 checkpoint。不要传 residual/channel/sparse 参数。

`--outlier_protect_mode residual_sparse`

必须传：

- `--outlier_rank_metric sparse_residual_abs|sparse_residual_actmax_abs|sparse_weight_abs|sparse_weight_actmax_abs`
- `--sparse_residual_ratio`

如果 metric 带 `actmax`，会根据 `wa_mse_calib_*` 参数现场采集 activation stats。

`--outlier_protect_mode channel_residual_vae`

必须传：

- `--outlier_rank_metric channel_weight_abs|channel_weight_actmax_abs|channel_residual_abs|channel_residual_actmax_abs|channel_residual_actrms_abs`
- `--outlier_protect_axis input|output`
- `--outlier_channel_scope layer|category`
- `--outlier_protect_count`
- `--outlier_residual_vae_decoder_share_scope none|category`
- `--outlier_residual_vae_batch_multiplier`
- `--outlier_residual_vae_steps`
- `--outlier_residual_vae_lr`

如果 metric 带 `actmax` 或 `actrms`，会根据 `wa_mse_calib_*` 参数现场采集 activation stats。

## Activation Stats

`cat_residual_from_base.py` 不支持读取提前保存的 activation stats。

如果 `--outlier_rank_metric` 需要 activation，例如带 `actmax` 或 `actrms` 的 metric，每次运行都会按下面这些参数重新采集目标 linear 的 activation max / second moment / RMS：

- `--wa_mse_calib_dataset`
- `--wa_mse_calib_nsamples`
- `--wa_mse_calib_seqlen`
- `--wa_mse_calib_seed`
- `--wa_mse_calib_device`
- `--wa_mse_calib_log_every`

示例：

```bash
python tools/cat_residual_from_base.py \
  ... \
  --outlier_rank_metric "channel_residual_actrms_abs" \
  --wa_mse_calib_dataset "alpaca=1" \
  --wa_mse_calib_nsamples "128" \
  --wa_mse_calib_seqlen "8192" \
  --wa_mse_calib_seed "31" \
  --wa_mse_calib_device "" \
  --wa_mse_calib_log_every "0"
```

## 示例

见 [scripts/catlora_residual_from_base.sh](../scripts/catlora_residual_from_base.sh)。

## Residual 前后评估

入口支持在每个 category 的 residual protection 前后评估完整 LLM：

- `--eval_before_residual`：当前 category 的 base VAE cache 清理并预热后评估一次，此时当前 category 还没有新 residual protection。
- `--eval_after_residual`：当前 category 的 residual protection 训练、挂载、cache 清理并重新预热后评估一次。
- `--eval_ppl`：是否跑 PPL。
- `--eval_tasks`：下游任务列表，例如 `boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu`。
- `--ppl_limit`：PPL limit，沿用现有 cat_train 语义。
- `--eval_hif4_act`：沿用现有 eval 语义。

评估对象始终是完整模型，不是单个 category 或单层指标。例如 `--target_categories "o_proj"` 时，`o_proj/before_residual` 和 `o_proj/after_residual` 都是完整 LLM 的结果。

多 category 是逐类别累积激活评估，不是一开始让所有 VAELinear 都走 VAE 分支。未激活 category 会显式切到 `original_weight` 分支：

- `q_proj/before_residual`：只激活 `q_proj` base VAE。
- `q_proj/after_residual`：只激活 `q_proj` base VAE + `q_proj` residual VAE。
- `v_proj/before_residual`：激活 `q_proj` residual VAE 和 `v_proj` base VAE。
- `v_proj/after_residual`：激活 `q_proj` residual VAE 和 `v_proj` residual VAE。

例如 `--target_categories "q_proj,v_proj,o_proj,down_proj"` 时，`o_proj/before_residual` 只会让 `q_proj,v_proj,o_proj` 走 VAE 分支，其中 `q_proj,v_proj` 已带 residual VAE，`o_proj` 只有 base VAE，`down_proj` 和其他未激活类别仍走 original weight。日志中的 `active_categories=...` 会记录每次评估实际激活的类别集合。

运行前需要确认当前 shell 已激活 `bitvae` 环境。
