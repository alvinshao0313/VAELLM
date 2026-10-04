# LongBench v1 paired evaluation

This isolated evaluator compares the original BF16 Qwen3-8B with a native v6
compressed model. It never trains, changes a checkpoint, or enables YaRN.
The root experiment directory is `result/qwen3_8b_longbench_20261004`.

Official source: THUDM/LongBench commit
`2e00731f8d0bff23dc4325161044d0ed8af94c1e`; unchanged metrics, prompt and output
budget files, their exact SHA256 values and MIT license are in `upstream/`.
Data: `THUDM/LongBench` revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`, `data.zip` SHA256
`cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`.
The original dataset source/licensing remains that of THUDM and its constituent
datasets; the included MIT file licenses the upstream code.

## Fixed protocol

- All 21 ordinary-v1 tasks, 4750 examples; no LongBench-E/v2 rows.
- One shared little-endian uint32 input archive, with exact per-example hashes.
- Total input plus output window 32768. Official task output budgets are 32–512.
  Final token sequences, including chat overhead, are truncated in the middle
  to preserve both ends. There is no decode/re-encode after truncation. This
  explicitly adapts the old upstream truncate-before-chat procedure to Qwen3.
- Qwen3 official chat template with `enable_thinking=False`; the six upstream
  few-shot/code exceptions remain bare continuations. No extra demonstrations.
- B1, seed42, greedy, one beam, KV cache. Both models receive the same explicit
  generation configuration. Original HF EOS `[151645,151643]`, pad151643;
  SAMSum instead uses the official EOS/newline and minimum-length special case.
  `use_model_defaults=False` is explicit: Transformers 4.51 otherwise replaces
  default-valued fields of a custom GenerationConfig with model sampling defaults.
- Original HF BF16/FA2 loader, without casting FP32 RoPE buffers. Compressed
  loading uses fresh strict v6, actual bit-budget audit and existing group8
  decoded-weight cache. There is no checkpoint or architecture conversion.
- Unchanged official task metrics, reference maximum and first-line rules.
  Report task scores, six categories, task macro and category macro separately.
  Neither macro is uniformly an accuracy metric.

## Commands

Run from the repository root using the activated `bitvae` environment. Data
preparation and CPU tests run with `CUDA_VISIBLE_DEVICES=''`; GPU execution
requires exactly one GPU assigned by the root agent.

```bash
python -m experiments.longbench_v1.prepare_data --output result/qwen3_8b_longbench_20261004/data
python -m experiments.longbench_v1.configure \
  --manifest result/qwen3_8b_longbench_20261004/data/manifest.json \
  --hf-model /home/star/models/Qwen/Qwen3-8B \
  --compressed-model result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model \
  --output result/qwen3_8b_longbench_20261004/preparation/configs
```

`--resume-download` explicitly continues an existing `.part` with an exact HTTP
byte range and still requires the original full SHA. No automatic retry runs.

Before full evaluation, execute `validate --config .../dense_validation.json
--output .../validation/dense`, then the compressed configuration in a separate
process. This evaluates only three real cases selected by input geometry:
longest input, longest GovReport (512-token budget), and longest SAMSum. Actual
prefill and cached decoding are recorded; these outputs cannot be scored as the
benchmark. It is a real resource/workflow test, not a quality claim.

```bash
python -m experiments.longbench_v1.predict \
  --config result/qwen3_8b_longbench_20261004/preparation/configs/dense_full.json \
  --output result/qwen3_8b_longbench_20261004/dense
python -m experiments.longbench_v1.score \
  --manifest result/qwen3_8b_longbench_20261004/data/manifest.json \
  --prediction-dir result/qwen3_8b_longbench_20261004/dense \
  --output result/qwen3_8b_longbench_20261004/dense/scores.json
```

Use the compressed configuration/output for the second model. Formal prediction
has no sample-limit option. `--datasets` can assign disjoint complete task
partitions; full scoring rejects missing/extra rows. Explicit `--resume` checks
the same configuration and every existing row before continuing. A
`--max-runtime-seconds` resource budget is checked between complete samples,
never by reducing a task's generation budget. It is not a scheduler or retry
service; the caller must reserve time for an in-flight generation and loading.
Per-example predictions include original scoring fields, IDs, input hashes,
generated token IDs, latency and GPU peaks. Each execution records its actual
PID/start tick, exit status, strict load audit and before/after input pins.
