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

## Full21 throughput and task partitions (2026-10-21)

The read-only Qwen J32 audit found no repeated model loading or decoded-weight
prewarm inside the example loop. The original complete run spent
17970.392 seconds in its 4750 recorded generations and 17994.141 seconds in the
recorded native interval; its one 252-module group8 prewarm took 2.204 seconds.
Within that interval, loading, per-row fsync and other work outside recorded
generations together accounted for only 23.749 seconds. The raw source is
[the original execution report](../../result/qwen3_8b_mmlu65_20261004/qwen_c1_eta_retention128_20261020/eta001_R05/step_32/evaluation/longbench/groups/part_01/runs/1792471691752660139_652368.json)
and the neighboring `predictions/` files. Do not change B1, output budgets or
stopping rules to claim protocol-preserving acceleration. Generation already
uses KV cache, and installed Transformers selects `logits_to_keep=1` for Qwen3.

For a **new evaluation**, assign complete, disjoint tasks to one process and
independent output directory per GPU. The native `--datasets` interface supports
this without changing generation or scoring. It has no row-index shard option.
Each worker locks all of its selected prediction files before loading its model;
an empty file in a running all21 job is therefore not an available work unit.
Do not add workers to such a job or launch a duplicate evaluation beside a
previously scheduled single-card chain.

CPU replay of the original per-task elapsed sums gives these fixed, largest-task
first assignments. Entries below preserve canonical task order within each
worker. These are generation-only estimates, excluding each worker's loading,
final merge and candidate-specific output-length changes; no new GPU run was
performed.

| GPUs / worker | Estimated generation seconds | Complete tasks |
| --- | ---: | --- |
| 2 / 1 | 9042.546 | narrativeqa, multifieldqa_zh, 2wikimqa, gov_report, qmsum, triviaqa, samsum, passage_count, passage_retrieval_en, passage_retrieval_zh, repobench-p |
| 2 / 2 | 8927.846 | qasper, multifieldqa_en, hotpotqa, musique, dureader, multi_news, vcsum, trec, lsht, lcc |
| 3 / 1 | 6018.509 | multifieldqa_zh, hotpotqa, 2wikimqa, gov_report, triviaqa, samsum, lsht |
| 3 / 2 | 5980.044 | narrativeqa, musique, dureader, multi_news, passage_retrieval_en, passage_retrieval_zh, lcc |
| 3 / 3 | 5971.839 | qasper, multifieldqa_en, qmsum, vcsum, trec, passage_count, repobench-p |

The estimated longest partition is 2h30m43s on two GPUs or 1h40m19s on three,
versus 4h59m30s of serial generation. The existing
[night evaluation planner](../../result/qwen3_8b_mmlu65_20261004/qwen_mlp_night_eval/evaluate.py)
already supports two fixed task groups (historical longest 9236.476 seconds);
its `make_plan` accepts only one or two groups. A three-group plan needs a small
new candidate-specific caller; the prediction, binding and merge interfaces do
not require two groups. The retention and current channel-scale chains instead
assert one assigned GPU and one all21 unit. Their pinned callers must remain
unchanged; these assignments are for a future explicitly scheduled candidate.

Reuse `start_group(unit, config, config_sha)` for independent logs, launch
identity and terminal exit records. After every worker exits successfully, call
[bind_terminal.py](../../result/qwen3_8b_mmlu65_20261004/llama_fullqk_qkonly2000_longbench21/bind_terminal.py)
sequentially with `--plan`, `--unit-index`, `--native-report`, `--outer-launch`,
`--outer-exit`, and `--outer-log`. It checks exactly one owner per task, canonical
row identity/input bytes, explicit generation options, source/config identity
and retired processes. Only after all 4750 rows bind, set the plan to
`prepared_not_scored` and call
[merge_full21.py --plan](../../result/qwen3_8b_mmlu65_20261004/llama_fullqk_qkonly2000_longbench21/merge_full21.py)
in the recorded public CPU environment. It copies the 21 complete task files,
verifies their hashes, runs unchanged official scoring and then the original
nine-task command. There is no concurrent plan writer or shared prediction file.

For ordinary native checkpoints, retain the strict checkpoint UUID/config
contract. For a channel-scale composite, the native UUID identifies only its
base: every worker must use the candidate's explicit bundle consumer and bind
the same composite identity, sidecar SHA and joint bit budget. The current
[scale consumer](../../result/qwen3_8b_mmlu65_20261004/qwen_j32_l9_channel_scale_20261021/cs_consumer.py)
passes `--datasets` through and records actual scale consumption per prediction
and per execution, but its witness covers its three declared projections only.
A future wider scale candidate must check its own complete projection scope.
The ordinary bind/merge checks do not replace these composite-consumption checks.
