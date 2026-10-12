"""Prepare reviewed Gen800 retrieval400 against the original e767 and BF16 controls."""
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_candidate_probe.prepare_llama_decoder1200_pair import (
    ROOT, R2, TASKS, E767_ID, controls_and_inputs, merge, read, require,
)
from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.longbench_v1.models import check_config, make_config

GEN = R2 / 'general_lm_llama_gen_s2000'
GEN_ID = '4cdf6c5c-f6cf-44cf-aa39-e76d65102c1e'
GEN_REVIEW_SHA = 'd2a4202f24c557d18b67eb82b99d43dadf855b3e93177ca366cde051d6589c7a'
OUTPUT = R2 / 'llama_gen800_retrieval400'


def endpoint():
    directory = GEN / 'step_0800'
    required = ['independent_review.json', 'audit.json', 'full8_summary.json', 'step_export.json',
                'final_model/checkpoint_meta.json', 'final_model/export_record.json']
    require(all((directory / name).is_file() for name in required),
            'Require actual Gen800 export, full8, and independent review; no inferred readiness.')
    review, audit, summary, callback, meta = (read(directory / name) for name in
        ('independent_review.json', 'audit.json', 'full8_summary.json', 'step_export.json', 'final_model/checkpoint_meta.json'))
    uid = meta['checkpoint_id']
    require({r['checkpoint_id'] for r in (review, audit, summary, callback)} == {uid}
            and review['source_checkpoint_id'] == audit['source_checkpoint_id'] == E767_ID
            and review['global_step'] == audit['selected_global_step'] == callback['global_step'] == 800
            and callback['status'] == 'evaluated' and audit['status'] == 'verified'
            and summary['status'] == 'full8_evaluation_verified' and summary['leaf_count'] == 64
            and summary['sample_count'] == 24742, 'Actual800 identity/complete evaluation differs.')
    require(uid == GEN_ID and review['status'] == 'independent_cpu_verified'
            and sha256(directory / 'independent_review.json') == GEN_REVIEW_SHA,
            'Use the exact independently reviewed Gen800 endpoint.')
    score = review['candidate']
    require(summary['mean_percent'] == score['mean_percent'] and summary['task_metrics'] == score['task_metrics']
            and callback['full8_summary_sha256'] == sha256(directory / 'full8_summary.json')
            and callback['audit_sha256'] == sha256(directory / 'audit.json'), 'Independent scores/callback pins differ.')
    require(audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 2979
            and audit['fresh_strict_reload_state_exact'] and audit['frozen_tensors_compared'] == 1635
            and audit['selected_decoder_tensor_count'] == 1344 and audit['packed_codes_unchanged']
            and audit['changed_code_bits'] == 0 and audit['rank_zero'], 'Require unchanged native D-only export contract.')
    budget = audit['bit_budget']
    require(budget['budget_passed'] and not budget['contract_violations']
            and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17095180288
            and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0
            and budget['components']['decoder']['dtypes'] == ['bfloat16'], 'Wrong actual saved storage.')
    stages = ['export', 'audit', 'mmlu', 'seven']
    require(all((directory / (name + '.exit')).read_text().strip() == '0' for name in stages), 'A native800 stage failed.')
    # Generation uses the archive's original HF tokenizer, pinned by make_config/check_config below.
    paths = [directory / name for name in required + ['final_model/config.json',
              'mmlu.json', 'seven.json', *[s + '.exit' for s in stages]]]
    paths.append(GEN / 'launch.prepared.json')
    evidence = {str(path): sha256(path) for path in paths}
    for path, digest in review['evidence_sha256'].items():
        if path in evidence:
            require(evidence[path] == digest, 'Reviewed endpoint evidence changed: ' + path)
    launch = read(GEN / 'launch.prepared.json')
    require(launch['source_checkpoint_id'] == E767_ID
            and launch['schema'] == 'llama_gen_native_lm_fresh2000_v1',
            'The Gen2000 trajectory must start at e767.')
    return {'directory': str(directory), 'checkpoint_dir': str(directory / 'final_model'), 'checkpoint_id': uid,
        'full8_mean_percent': summary['mean_percent'], 'bit_budget': budget,
        'weight_stat': file_stat(directory / 'final_model' / meta['state_dict_file']),
        'review_path': str(directory / 'independent_review.json'),
        'native_proof': {'persistent_tensors': 2979, 'decoder_tensors': 1344, 'frozen_tensors': 1635,
                         'packed_code_xor': 0, 'strict_fresh_reload': True, 'full8_group8_completed': True}}, launch, evidence


def prepare():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only preparation in the public repository.')
    require(not OUTPUT.exists(), 'Use a new independent retrieval output.')
    torch.set_num_threads(4)
    candidate, launch, evidence = endpoint()
    dense, controls, common_evidence, geometry, reference, bf16 = controls_and_inputs()
    helper_files = [Path(__file__).resolve(), ROOT / 'experiments/longbench_candidate_probe/prepare_llama_decoder1200_pair.py',
                    ROOT / 'experiments/longbench_candidate_probe/prepare_llama_best1600_d16.py']
    config = make_config(dense['archive']['path'], kind='compressed', hf_path=dense['model']['hf_path'],
                         checkpoint=candidate['checkpoint_dir'], mode='full')
    require(all(config[k] == dense[k] for k in ('schema', 'mode', 'protocol', 'archive', 'package_versions',
            'model_policy', 'prewarm_group_size', 'device'))
            and config['model']['checkpoint_id'] == candidate['checkpoint_id']
            and config['model']['weight_stat'] == candidate['weight_stat'], 'Actual official model/config identity differs.')
    check_config(config)
    sources = merge(launch['source_sha256'], dense['source_sha256'], config['source_sha256'],
                    {str(p): sha256(p) for p in helper_files})
    evidence = merge(evidence, common_evidence)
    check_hashes(sources); check_hashes(evidence)
    required_lb = 9 * bf16['required_compressed_nine_minimum'] - 8 * candidate['full8_mean_percent']
    maximum_nine = (8 * candidate['full8_mean_percent'] + 100) / 9
    record = {'schema': 'llama_gen800_retrieval400_v1', 'status': 'prepared_unbound', 'gpu': None,
        'arm': 'gen', 'source_root': str(ROOT), 'python_executable': sys.executable, 'checkpoint_id': candidate['checkpoint_id'],
        'config_path': str(OUTPUT / 'config.json'), 'output_dir': str(OUTPUT / 'run'), 'datasets': list(TASKS),
        'sample_count': 400, 'task_samples': {t: 200 for t in TASKS}, 'source_sha256': sources, 'evidence_sha256': evidence,
        'source_bit_budget': candidate['bit_budget'], 'completed_candidate': candidate, 'controls': controls,
        'initial_checkpoint_id': E767_ID, 'global_step': 800, 'training_schedule_max_steps': 2000,
        'environment': {k: v for k, v in launch['environment'].items() if k != 'CUDA_VISIBLE_DEVICES'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict', '--config', str(OUTPUT / 'config.json'),
                          '--output', str(OUTPUT / 'run'), '--datasets', *TASKS],
        'timeout_seconds': 900, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False, 'resume': False,
        'resource_evidence': {'reference_duration_seconds': reference['duration_seconds'], 'wall_seconds_estimate': [550, 750],
            'reference_generation_peak_allocated_bytes': reference['generation_peak_allocated_bytes'],
            'reference_generation_peak_reserved_bytes': reference['generation_peak_reserved_bytes'], 'outer_cap_seconds': 900,
            'boundary': 'Same Llama2979/SDPA/group8 path; D1200 actual400687.43s also supports estimate. No new GPU short.'},
        'input_geometry': geometry, 'selection': 'Original official EN/ZH rows0..199 with identical archived tokens, no filtering.',
        'scoring': 'check_prediction/score_example official fractional200+200; no full21 scorer.',
        'comparison_boundary': 'Gen800 versus its initial e767 checkpoint and original BF16 on the same official400. '
            'No FineWeb800 retrieval comparison: its weights and exact training state were retired. '
            'The completed full8 Gen800-FineWeb800 review remains valid separately; this retrieval is not a data-only intervention.',
        'generation_default_boundary': 'Original HF tokenizer and explicit GenerationConfig with use_model_defaults=False, '
            'pad/EOS/use_cache and all-one masks; source checkpoint defaults are not inherited by generation.',
        'validation_reuse': 'Pinned native800 export/fresh/full8 independent review plus actual original e767 official400 generation.',
        'nine_item_necessary_bound': {'bf16_nine_mean': bf16['baseline_nine_mean'],
            'minimum_nine_mean': bf16['required_compressed_nine_minimum'], 'candidate_eight_mean': candidate['full8_mean_percent'],
            'required_longbench21_mean': required_lb, 'maximum_nine_if_longbench21_100': maximum_nine,
            'cannot_meet_nine_target_at_this_endpoint': maximum_nine < bf16['required_compressed_nine_minimum'],
            'post_retrieval_upper_bound_formula': 'LB21_max=(1900+EN+ZH)/21; nine_max=(8*eight_mean+LB21_max)/9.',
            'interpretation': 'Data recovery direction diagnostic only; necessary ceiling is not a measured full21 or nine score.'},
        'scope': 'Only2/21 complete official tasks.', 'full_longbench_status': 'not_evaluated',
        'nine_item_goal_status': 'not_evaluated', 'model_mutations': False, 'model_exported': False, 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    OUTPUT.mkdir(); write_json(OUTPUT / 'config.json', config)
    record['config_sha256'] = sha256(OUTPUT / 'config.json'); write_json(OUTPUT / 'preparation.json', record)
    print(json.dumps({'arm': record['arm'], 'checkpoint_id': record['checkpoint_id'], 'status': record['status'],
        'preparation_sha256': sha256(OUTPUT / 'preparation.json'), 'source_count': len(sources), 'input_count': len(evidence),
        'required_longbench21_mean': required_lb, 'maximum_nine_if_longbench21_100': maximum_nine}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    prepare()
