"""Bind the two reviewed Gen800 decoder unions to the existing official retrieval400 protocol."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

from experiments.llama_gen_category_pair.run import assembly, check
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
PAIR = R2 / 'llama_gen800_category_pair'
GEN = R2 / 'llama_gen800_retrieval400'
OUTPUT = R2 / 'llama_gen800_category_retrieval400'
TASKS = ('passage_retrieval_en', 'passage_retrieval_zh')
IDS = {'attention': '1f1623e4-dd13-418b-ba89-6a901fa5f8e6',
       'mlp': 'aa36262f-1d9f-42ad-a0e3-0731edc5ecfa'}


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def candidate(arm, comparison):
    name = 'gen_' + arm; directory = PAIR / name
    reviewed = comparison['arms'][name]; launch = read(directory / 'launch.json')
    require(launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['checkpoint_id'] == reviewed['result']['checkpoint_id'] == IDS[arm]
            and set(reviewed['execution']['exits'].values()) == {0}
            and all((directory / (s + '.exit')).read_text().strip() == '0'
                    for s in ('assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
            'Require the actual completed union and its independent complete-eight review.')
    check(launch)
    proof = assembly(launch)
    require(proof['bit_budget'] == reviewed['bit_budget']
            and reviewed['native_proof']['source_union_save_fresh_tensors'] == 2979
            and reviewed['native_proof']['frozen_nondecoder_tensors'] == 1635
            and reviewed['native_proof']['frozen_packed_banks'] == 448,
            'Reuse the existing native union and frozen-complement proof without reloading large weights.')
    inputs = {str(PAIR / 'comparison.json'): sha256(PAIR / 'comparison.json')}
    for name in ('launch.prepared.json', 'full8_summary.json', 'mmlu.json', 'seven.json',
                 'final_model/checkpoint_meta.json', 'final_model/assembly_report.json'):
        path = directory / name
        require(reviewed['evidence_sha256'][str(path)] == sha256(path), 'Reviewed endpoint artifact changed.')
        inputs[str(path)] = sha256(path)
    path = directory / 'final_model/config.json'; inputs[str(path)] = sha256(path)
    return directory, launch, reviewed, inputs


def prepare():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(),
            'Use CUDA-hidden public-source bitvae and a new output root.')
    comparison = read(PAIR / 'comparison.json')
    require(comparison['status'] == 'independent_cpu_verified'
            and sha256(PAIR / 'comparison.json') ==
            '205e47cf415d563c755e6059444434ef038518039c4517d16ed20996eec28456',
            'Require the actual completed two-union independent comparison.')
    old, config0, review = (read(GEN / name) for name in ('preparation.json', 'config.json', 'review.json'))
    require(sha256(GEN / 'review.json') == '424cb0699c86c1d573bcc56e5462a4f560bc8bbdb1b315de300be0bb143f71bc'
            and review['status'] == 'verified' and review['samples'] == 400
            and review['checkpoint_id'] == '4cdf6c5c-f6cf-44cf-aa39-e76d65102c1e'
            and review['execution_exit_code'] == 0 and review['actual_input_tokens_and_original_ids_exact']
            and review['generation_options_effective_and_decoded_tokens_verified']
            and old['config_sha256'] == sha256(GEN / 'config.json')
            and review['scores_sha256'] == sha256(GEN / 'scores.json'),
            'Reuse verified full Gen800 native400 predictions and their exact protocol.')
    common_inputs = dict(old['evidence_sha256'])
    for path in (GEN / 'preparation.json', GEN / 'config.json', GEN / 'scores.json', GEN / 'review.json'):
        common_inputs[str(path)] = sha256(path)
    controls = {'e767': old['controls']['e767'], 'gen800': {
        'directory': str(GEN), 'prediction_root': str(GEN / 'run'),
        'checkpoint_id': review['checkpoint_id'],
        'tasks': {task: {'score_percent': review['task_results'][task]['score_percent'], 'samples': 200}
                  for task in TASKS}}}
    for control in controls.values():
        directory = Path(control['directory']); old_config = read(directory / 'config.json')
        require(all(old_config[k] == config0[k] for k in ('protocol', 'archive', 'package_versions',
                'model_policy', 'prewarm_group_size', 'device')), 'Historical400 controls use different protocols.')
        for task in TASKS:
            path = Path(control['prediction_root']) / 'predictions' / (task + '.jsonl')
            require(review['evidence_sha256'][str(path)] == sha256(path), 'Verified original400 predictions changed.')
            common_inputs[str(path)] = sha256(path)
    _, rows = load_archive(config0['archive']['path'], check_token_file=False)
    require(all([r['row_index'] for r in rows if r['dataset'] == task] == list(range(200)) for task in TASKS),
            'Use all original200 rows per retrieval task without filtering or resampling.')
    bound = old['nine_item_necessary_bound']; records = []
    for arm in IDS:
        directory, launch, reviewed, evidence = candidate(arm, comparison)
        output = OUTPUT / arm
        config = make_config(config0['archive']['path'], kind='compressed',
            hf_path=config0['model']['hf_path'], checkpoint=directory / 'final_model', mode='full')
        require(config['model']['checkpoint_id'] == IDS[arm]
                and all(config[k] == config0[k] for k in ('schema', 'mode', 'protocol', 'archive',
                    'package_versions', 'model_policy', 'prewarm_group_size', 'device')),
                'Only the reviewed candidate model identity may differ from original native retrieval400.')
        check_config(config)
        sources = dict(launch['source_sha256'])
        for path, digest in config['source_sha256'].items():
            require(path not in sources or sources[path] == digest, 'Native implementation identity drifted.')
            sources[path] = digest
        sources[str(Path(__file__).resolve())] = sha256(__file__)
        inputs = {**common_inputs, **evidence}; check_hashes(sources); check_hashes(inputs)
        for stat in (*launch['protected_stats'], config['model']['weight_stat']):
            check_stat(stat)
        mean = reviewed['result']['mean_percent']; maximum_nine = (8 * mean + 100) / 9
        require(maximum_nine < bound['minimum_nine_mean'], 'This fixed endpoint is a diagnosis, not an acceptance candidate.')
        record = {'schema': 'llama_gen800_category_retrieval400_v1', 'status': 'prepared_unbound', 'gpu': None,
            'arm': arm, 'source_root': str(ROOT), 'python_executable': sys.executable, 'checkpoint_id': IDS[arm],
            'config_path': str(output / 'config.json'), 'output_dir': str(output / 'run'),
            'datasets': list(TASKS), 'sample_count': 400, 'task_samples': {task: 200 for task in TASKS},
            'source_sha256': sources, 'evidence_sha256': inputs, 'source_bit_budget': reviewed['bit_budget'],
            'completed_candidate': {'directory': str(directory), 'checkpoint_dir': str(directory / 'final_model'),
                'checkpoint_id': IDS[arm], 'full8_mean_percent': mean,
                'native_proof': reviewed['native_proof'], 'weight_stat': config['model']['weight_stat'],
                'review_path': str(PAIR / 'comparison.json'), 'selected_categories': launch['categories']},
            'controls': controls, 'environment': old['environment'], 'input_geometry': old['input_geometry'],
            'launcher_argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict',
                '--config', str(output / 'config.json'), '--output', str(output / 'run'), '--datasets', *TASKS],
            'timeout_seconds': 900, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False,
            'resume': False, 'resource_evidence': {'wall_seconds_estimate': [500, 650], 'outer_cap_seconds': 900,
                'gen800_actual_seconds': review['duration_seconds'],
                'gen800_peak_allocated_bytes': review['generation_peak_allocated_bytes'],
                'gen800_peak_reserved_bytes': review['generation_peak_reserved_bytes'],
                'boundary': 'Same original Llama32K/SDPA/FP32RoPE/native EOS/group8, all400. No extra GPU short.'},
            'selection': 'Original archived EN/ZH rows0..199 and token slices; no sample selection.',
            'scoring': old['scoring'], 'generation_default_boundary': old['generation_default_boundary'],
            'validation_reuse': 'New union2979 native save/fresh/full8 plus exact Gen8001635 nondecoder freeze. '
                'Existing Gen800/e767400 controls reused; native predictor still loads fresh strict and checks actual storage.',
            'comparison_boundary': 'Complementary Gen800 attention or MLP decoder source unions relative to e767 and fullGen800. '
                'Only retrieve400 to diagnose the observed retrieval/knowledge tradeoff; conditional effects need not add.',
            'nine_item_necessary_bound': {'bf16_nine_mean': bound['bf16_nine_mean'],
                'minimum_nine_mean': bound['minimum_nine_mean'], 'candidate_eight_mean': mean,
                'required_longbench21_mean': 9 * bound['minimum_nine_mean'] - 8 * mean,
                'maximum_nine_if_longbench21_100': maximum_nine, 'cannot_meet_nine_target_at_this_endpoint': True,
                'post_retrieval_upper_bound_formula': bound['post_retrieval_upper_bound_formula'],
                'interpretation': 'This exact endpoint cannot pass even at LB21=100. No19-task expansion or acceptance claim.'},
            'scope': 'Only2/21 complete official tasks; no remaining19 tasks scheduled.',
            'full_longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
            'model_mutations': False, 'model_exported': False, 'cuda_initialized': False}
        records.append((output, config, record))
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    for output, config, record in records:
        output.mkdir(parents=True); write_json(output / 'config.json', config)
        record['config_sha256'] = sha256(output / 'config.json'); write_json(output / 'preparation.json', record)
        print({'arm': record['arm'], 'status': record['status'], 'source_count': len(record['source_sha256']),
               'input_count': len(record['evidence_sha256']), 'preparation_sha256': sha256(output / 'preparation.json'),
               'maximum_nine_if_longbench21_100': record['nine_item_necessary_bound']['maximum_nine_if_longbench21_100']}, flush=True)


if __name__ == '__main__':
    prepare()
