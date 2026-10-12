"""Full-eight BF16 category restoration diagnostics on the fixed Llama CAT source."""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.longbench_v1.common import check_hashes, check_stat, sha256
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import leaf_metrics, _group_agrees
from experiments.qwen3_w25.summarize_results import TASK_METRICS

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'result/llama31_8b_w25_20261004'
SOURCE_ID = '35776715-2707-45a9-9fb4-2da1efef3f0e'
ARMS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
SHARDS = (('mmlu', 'mmlu', 64),
          ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', 32))
PROTOCOL = {'num_fewshot': 0, 'limit': None, 'mmlu_batch_size': 64, 'seven_batch_size': 32,
    'leaf_count': 64, 'sample_count': 24742, 'mmlu_sample_count': 14042,
    'mmlu_aggregation': 'Official test-size weighted57 subjects.',
    'prompt': 'Unmodified native bare causal likelihood; no chat template or system instruction.',
    'diagnostic_only': True, 'model_exported': False, 'goal_acceptance': False,
    'boundary': 'Restore exactly one entire category to original HF BF16 while all other CAT3577 state stays fixed. Exceeds2.5bpw; neither a deployable candidate nor a prediction of re-quantization gain.'}


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    require(record['schema'] == 'llama_single_category_full8_oracle_v1'
            and record['source_checkpoint_id'] == SOURCE_ID and record['protocol'] == PROTOCOL
            and record['arm'] in ARMS and not record['queued'] and not record['automatic_retry'],
            'Unexpected fixed-source/category/protocol contract.')
    check_hashes(record['source_sha256']); check_hashes(record['input_evidence_sha256'])
    for item in record['protected_stats']:
        check_stat(item)
    meta = read(Path(record['source_checkpoint']) / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID and meta['base_model_path'] == record['hf_path'],
            'The fixed compressed/HF identities changed.')
    expected = [name for name in record['architecture']['projection_shapes']
                if name.rsplit('.', 1)[1] == record['arm']]
    require(record['selected_targets'] == expected and len(expected) == 32,
            'Restore exactly one category across all32 layers.')


def verify_shard(metrics, reference, expected_batch):
    require(metrics['num_fewshot'] == 0 and metrics['limit'] is None
            and metrics['batch_size'] == expected_batch, 'Require full native64/32 zero-shot evaluation.')
    for field in ('configs', 'versions', 'n_shot', 'n_samples'):
        require(metrics[field] == reference[field], 'Native full8 reference differs: ' + field)
    scores, subjects, batch = leaf_metrics(metrics['raw_results'], metrics['n_samples'],
                                         metrics['n_shot'], metrics['configs'])
    _group_agrees(scores, metrics['raw_results'], metrics['group_results'])
    require(batch == expected_batch and set(metrics['tasks']) == set(scores)
            and set(metrics['task_metrics']) == set(scores)
            and set(metrics['task_metric_keys']) == set(scores), 'Native shard inventory differs.')
    for task, score in scores.items():
        require(metrics['task_metric_keys'][task] == TASK_METRICS[task]
                and math.isclose(metrics['task_metrics'][task], score, rel_tol=0, abs_tol=1e-12),
                'Native summary differs from complete leaf scores.')
    return scores, subjects


def summarize(output, record):
    scores, subjects, leaves, samples = {}, {}, 0, 0
    for shard, _tasks, batch in SHARDS:
        metrics = read(Path(output) / ('lm_eval_results_' + shard + '.json'))
        reference = read(record['control_result_paths'][shard])['evals']['lm_eval']
        current, sub = verify_shard(metrics, reference, batch)
        require(not set(scores) & set(current), 'Duplicate full8 task.')
        scores.update(current); subjects.update(sub)
        leaves += len(metrics['n_samples'])
        samples += sum(row['effective'] for row in metrics['n_samples'].values())
    require(set(scores) == set(TASK_METRICS) and len(subjects) == 57
            and leaves == 64 and samples == 24742, 'Incomplete full-eight evaluation.')
    mean = math.fsum(scores.values()) * 100 / 8
    return {'task_metrics': scores, 'task_metric_keys': TASK_METRICS, 'mean_percent': mean,
        'mmlu_subject_metrics': subjects, 'leaf_count': leaves, 'sample_count': samples,
        'mean_delta_vs_3577_percentage_points': mean - record['control_mean_percent'],
        'mean_delta_vs_bf16_percentage_points': mean - record['baseline_mean_percent'],
        'task_delta_vs_3577_percentage_points': {task: 100 * (scores[task] - record['control_task_metrics'][task])
                                              for task in scores},
        'result_sha256': {shard: sha256(Path(output) / ('lm_eval_results_' + shard + '.json'))
                          for shard, _tasks, _batch in SHARDS}}
