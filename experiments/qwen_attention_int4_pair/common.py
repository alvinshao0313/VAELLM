"""Fixed fresh-attention pair; all outputs remain isolated BF16 simulations."""
from pathlib import Path
import json
import math

from experiments.longbench_v1.common import check_hashes, check_stat, sha256
from experiments.qwen3_w25.assemble_categories import _require
from experiments.llama_category_full8.common import SHARDS, verify_shard

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = Path('/home/star/VAELLM')
RESULT = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
CATEGORIES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
BITS = {'control': dict(zip(CATEGORIES, (32, 32, 32, 48))),
        'candidate': dict(zip(CATEGORIES, (40, 40, 40, 56)))}
DEPENDENCIES = ('experiments/protected_int4_probe/codec.py',
                'experiments/protected_int4_scale_search/search.py',
                'experiments/protected_int4_quality/protection.py',
                'experiments/llama_category_full8/common.py',
                'experiments/longbench_attribution/restore.py',
                'experiments/long_context_recovery/contract.py')


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    _require(ROOT != PUBLIC and Path(record['cwd']) == ROOT and record['schema'] == 'fresh_attention_int4_pair_v1'
             and record['arm'] in BITS and record['codebook_bits'] == BITS[record['arm']]
             and record['categories'] == CATEGORIES and record['base_checkpoint_id'] == SOURCE_ID
             and not record['queued'] and not record['automatic_retry'], 'Unexpected isolated pair contract.')
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)


def summarize(output, record):
    scores, subjects = {}, {}; leaves = samples = 0
    for shard, _tasks, batch in SHARDS:
        raw = read(Path(output) / ('lm_eval_results_' + shard + '.json'))
        native = read(record['control_result_paths'][shard])['evals']['lm_eval']
        current, sub = verify_shard(raw, native, batch)
        verify_shard(raw, read(record['source_scale_control']['raw_result_paths'][shard]), batch)
        scores.update(current); subjects.update(sub)
        leaves += len(raw['n_samples']); samples += sum(v['effective'] for v in raw['n_samples'].values())
    _require(len(scores) == 8 and len(subjects) == 57 and leaves == 64 and samples == 24742,
             'Require the complete fixed-protocol eight tasks.')
    mean = math.fsum(scores.values()) * 100 / 8
    return {'task_metrics': scores, 'mean_percent': mean, 'mmlu_subject_metrics': subjects,
            'leaf_count': leaves, 'sample_count': samples,
            'delta_vs_native_f927_pp': mean - record['source_native_mean_percent'],
            'delta_vs_recovered_f927_scale_pp': mean - record['source_scale_control']['mean_percent'],
            'primary_comparison': 'candidate versus this pair fresh control; requires both completed arms',
            'raw_result_sha256': {s: sha256(Path(output) / ('lm_eval_results_' + s + '.json')) for s, _, _ in SHARDS}}
