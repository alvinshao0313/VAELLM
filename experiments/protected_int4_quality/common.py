"""Fixed-source isolated INT4 protection simulations, never native low-bit exports."""
from pathlib import Path
import json
import math

from experiments.longbench_v1.common import check_hashes, check_stat, sha256
from experiments.long_context_recovery.contract import require
from experiments.protected_int4_probe.measure import SOURCES
from experiments.llama_category_full8.common import SHARDS, verify_shard

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'
PROOF = RESULT / 'protected_int4_probe_cpu_r1/report.json'
ARMS = ('output_scale_int4', 'rht_output_scale_int4')


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    require(record['schema'] == 'protected_int4_expanded_bf16_simulation_v1'
            and record['model_label'] in SOURCES and record['arm'] in ARMS
            and record['source_checkpoint_id'] == SOURCES[record['model_label']][1]
            and not record['queued'] and not record['automatic_retry'], 'Unexpected simulation identity.')
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)
    require(read(Path(record['source_checkpoint']) / 'checkpoint_meta.json')['checkpoint_id']
            == record['source_checkpoint_id'], 'Original compressed source changed.')


def summarize(output, record):
    scores, subjects = {}, {}; samples = leaves = 0
    for shard, _tasks, batch in SHARDS:
        metrics = read(Path(output) / ('lm_eval_results_' + shard + '.json'))
        reference = read(record['control_result_paths'][shard])['evals']['lm_eval']
        current, sub = verify_shard(metrics, reference, batch)
        scores.update(current); subjects.update(sub)
        leaves += len(metrics['n_samples']); samples += sum(x['effective'] for x in metrics['n_samples'].values())
    require(len(scores) == 8 and len(subjects) == 57 and leaves == 64 and samples == 24742,
            'Require complete native eight-task coverage.')
    mean = math.fsum(scores.values()) * 100 / 8
    return {'task_metrics': scores, 'mmlu_subject_metrics': subjects, 'mean_percent': mean,
        'mean_delta_vs_source_pp': mean - record['control_mean_percent'],
        'task_delta_vs_source_pp': {task: 100 * (score - record['control_task_metrics'][task]) for task, score in scores.items()},
        'leaf_count': leaves, 'sample_count': samples,
        'raw_result_sha256': {shard: sha256(Path(output) / ('lm_eval_results_' + shard + '.json'))
                              for shard, _tasks, _batch in SHARDS}}
