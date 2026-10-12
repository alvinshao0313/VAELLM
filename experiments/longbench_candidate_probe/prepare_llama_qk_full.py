"""Prepare the completed Llama Q/K-full endpoint for official EN200/ZH200."""
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.longbench_v1.models import check_config, make_config

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
ROOT = R2 / 'rotation_factor_probe/source_snapshot_20261005_01'
TASKS = ('passage_retrieval_en', 'passage_retrieval_zh')
E767_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'

CANDIDATE = R2 / 'llama_qk_rotation_pair/full'
CANDIDATE_ID = '4e1cec35-be20-42d5-91e3-dcaebc1d0552'
REVIEW_SHA = '4d919b55f285d8d25119f2c90b9f5654e3de3aba39375ee7464f47701c2f5554'
OUTPUT = R2 / 'llama_qk_full_retrieval400'


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def merge(*groups):
    result = {}
    for group in groups:
        for path, digest in group.items():
            require(path not in result or result[path] == digest, 'Conflicting source/evidence: ' + path)
            result[path] = digest
    check_hashes(result)
    return result


def controls_and_inputs():
    # Reuse the already executed400-token/generation/fractional-score CPU check.
    # Its public inference configuration was never GPU-launched and is superseded.
    archive = OUTPUT / 'superseded_public_prepare'
    old = read(archive / 'preparation.json'); dense = read(Path(old['controls']['bf16']['directory']) / 'config.json')
    bf16 = read(Path(old['controls']['bf16']['directory']) / 'score_verification.json')
    reference = read(Path(old['controls']['e767']['directory']) / 'review.json')
    require(sha256(archive / 'preparation.json') == 'f1b6ecb12e4f39a2d6d6820f11af5e382e7dbd9bf656b6eebc6eb8bc12275373'
            and (archive / 'preparation.exit').read_text().strip() == '0'
            and old['checkpoint_id'] == CANDIDATE_ID and old['sample_count'] == 400
            and bf16['status'] == reference['status'] == 'verified', 'Original official400 CPU proof differs.')
    evidence = merge(old['evidence_sha256'], {str(archive / p): sha256(archive / p)
        for p in ('preparation.json', 'config.json', 'prepare_source.py', 'preparation.exit', 'reason.json')})
    return dense, old['controls'], evidence, old['input_geometry'], reference, bf16


def endpoint():
    directory = CANDIDATE; final = directory / 'final_model'
    launch, review, audit, summary, meta, assembled = (read(directory / p) for p in
        ('launch.json', 'independent_review.json', 'audit.json', 'full8_summary.json',
         'final_model/checkpoint_meta.json', 'final_model/assembly_report.json'))
    require(sha256(directory / 'independent_review.json') == REVIEW_SHA
            and review['status'] == 'independent_cpu_verified' and launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['arm'] == 'full' and launch['block_size'] == 0 and launch['categories'] == ['q_proj', 'k_proj']
            and all(r['checkpoint_id'] == CANDIDATE_ID for r in (launch, review, audit, summary, meta, assembled))
            and audit['status'] == assembled['status'] == 'verified'
            and launch['base_checkpoint_id'] == audit['base_checkpoint_id'] == assembled['sources']['base_checkpoint_id'] == E767_ID,
            'Require the actual completed independently reviewed Llama full-QK endpoint.')
    require(audit['source_union_save_fresh_tensors'] == assembled['serialized_tensors_compared']
            == assembled['reloaded_tensors_compared'] == 2979 and audit['packed_bank_count'] == 448
            and audit['replaced_projection_count'] == 64 and audit['preserved_projection_count'] == 160
            and audit['replaced_local_tensor_count'] == 832 and audit['frozen_complement_tensor_count'] == 2147
            and audit['assembly_report_sha256'] == sha256(final / 'assembly_report.json')
            and launch['full8_summary_sha256'] == sha256(directory / 'full8_summary.json')
            and summary['leaf_count'] == 64 and summary['sample_count'] == 24742
            and summary['mean_percent'] == review['candidate']['mean_percent'] == launch['mean_percent']
            and summary['task_metrics'] == review['candidate']['task_metrics'], 'Native union/full8 proof changed.')
    stages = ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')
    require(all((directory / (s + '.exit')).read_text().strip() == '0' for s in stages)
            and all(s['exit_code'] == 0 for s in launch['stages'].values()), 'Incomplete source terminal stages.')
    budget = audit['bit_budget']
    require(budget == assembled['bit_budget'] == review['bit_budget'] and budget['budget_passed']
            and not budget['contract_violations'] and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17095180288
            and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0,
            'Native same-rate source budget changed.')
    paths = [directory / p for p in ('launch.json', 'launch.prepared.json', 'independent_review.json', 'audit.json',
        'full8_summary.json', 'mmlu.json', 'seven.json', 'final_model/checkpoint_meta.json',
        'final_model/assembly_report.json', 'final_model/config.json', *(s + '.exit' for s in stages))]
    evidence = {str(p): sha256(p) for p in paths}
    for p, digest in review['evidence_sha256'].items():
        if p in evidence:
            require(evidence[p] == digest, 'Reviewed source evidence changed: ' + p)
    return {'directory': str(directory), 'checkpoint_dir': str(final), 'checkpoint_id': CANDIDATE_ID,
        'full8_mean_percent': summary['mean_percent'], 'bit_budget': budget,
        'weight_stat': file_stat(final / meta['state_dict_file']), 'review_path': str(directory / 'independent_review.json'),
        'native_proof': {'persistent_tensors': 2979, 'packed_banks': 448, 'replaced_qk_projections': 64,
            'preserved_e767_projections': 160, 'strict_fresh_reload': True, 'full8_group8_completed': True}}, launch, evidence


def prepare():
    require(Path.cwd() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
            and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only preparation in the actual H252 snapshot.')
    require(OUTPUT.is_dir() and not (OUTPUT / 'preparation.json').exists() and not (OUTPUT / 'run').exists(),
            'Only the nonexecuted public preparation archive may already exist.')
    torch.set_num_threads(4)
    candidate, launch, evidence = endpoint()
    dense, controls, common_evidence, geometry, reference, bf16 = controls_and_inputs()
    helper_files = [Path(__file__).resolve(), ROOT / 'rotation/hadamard_utils.py', ROOT / 'rotation/hadamard_extended.py']
    from rotation.hadamard_utils import get_hadK
    had, factor = get_hadK(4032)
    require(factor == 252 and tuple(had.shape) == (252, 252), 'Actual snapshot H252 geometry is unavailable.')
    config = make_config(dense['archive']['path'], kind='compressed', hf_path=dense['model']['hf_path'],
                         checkpoint=candidate['checkpoint_dir'], mode='full')
    require(all(config[k] == dense[k] for k in ('schema', 'mode', 'protocol', 'archive', 'package_versions',
            'model_policy', 'prewarm_group_size', 'device'))
            and config['model']['checkpoint_id'] == candidate['checkpoint_id']
            and config['model']['weight_stat'] == candidate['weight_stat'], 'Actual official model/config identity differs.')
    check_config(config)
    sources = merge(launch['source_sha256'], config['source_sha256'],
                    {str(p): sha256(p) for p in helper_files})
    evidence = merge(evidence, common_evidence)
    check_hashes(sources); check_hashes(evidence)
    required_lb = 9 * bf16['required_compressed_nine_minimum'] - 8 * candidate['full8_mean_percent']
    maximum_nine = (8 * candidate['full8_mean_percent'] + 100) / 9
    record = {'schema': 'llama_qk_full_retrieval400_v1', 'status': 'prepared_unbound', 'gpu': None,
        'arm': 'full', 'source_root': str(ROOT), 'python_executable': sys.executable, 'checkpoint_id': candidate['checkpoint_id'],
        'config_path': str(OUTPUT / 'config.json'), 'output_dir': str(OUTPUT / 'run'), 'datasets': list(TASKS),
        'sample_count': 400, 'task_samples': {t: 200 for t in TASKS}, 'source_sha256': sources, 'evidence_sha256': evidence,
        'source_bit_budget': candidate['bit_budget'], 'completed_candidate': candidate, 'controls': controls,
        'initial_checkpoint_id': E767_ID, 'matched_block32': {'status': 'not_available', 'checkpoint_id': None,
            'boundary': 'Prepare the matched arm only after real native assembly/full8; no inferred endpoint or automatic queue.'},
        'environment': {**{k: v for k, v in launch['environment'].items() if k != 'CUDA_VISIBLE_DEVICES'}, 'PYTHONPATH': str(ROOT)},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict', '--config', str(OUTPUT / 'config.json'),
                          '--output', str(OUTPUT / 'run'), '--datasets', *TASKS],
        'timeout_seconds': 900, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False, 'resume': False,
        'resource_evidence': {'reference_duration_seconds': reference['duration_seconds'], 'wall_seconds_estimate': [550, 750],
            'reference_generation_peak_allocated_bytes': reference['generation_peak_allocated_bytes'],
            'reference_generation_peak_reserved_bytes': reference['generation_peak_reserved_bytes'], 'outer_cap_seconds': 900,
            'boundary': 'Same Llama2979/SDPA/group8 path; D1200 actual400687.43s also supports estimate. No new GPU short.'},
        'input_geometry': geometry, 'selection': 'Original official EN/ZH rows0..199 with identical archived tokens, no filtering.',
        'scoring': 'check_prediction/score_example official fractional200+200; no full21 scorer.',
        'comparison_boundary': 'Full Q/K versus e767 and original BF16 on identical official400. '
            'The primary full-minus-fresh-block32 rotation comparison waits for the actual block32 endpoint and predictions. '
            'Both arms require retrieval regardless of full8; fresh re-encoding and missing recovery history confound comparisons to e767.',
        'generation_default_boundary': 'Original HF tokenizer and explicit GenerationConfig with use_model_defaults=False, '
            'pad/EOS/use_cache and all-one masks; source checkpoint defaults are not inherited by generation.',
        'validation_reuse': 'Reviewed native64QK/160complement save/fresh/full8 plus original e767 official400 generation.',
        'nine_item_necessary_bound': {'bf16_nine_mean': bf16['baseline_nine_mean'],
            'minimum_nine_mean': bf16['required_compressed_nine_minimum'], 'candidate_eight_mean': candidate['full8_mean_percent'],
            'required_longbench21_mean': required_lb, 'maximum_nine_if_longbench21_100': maximum_nine,
            'cannot_meet_nine_target_at_this_endpoint': maximum_nine < bf16['required_compressed_nine_minimum'],
            'post_retrieval_upper_bound_formula': 'LB21_max=(1900+EN+ZH)/21; nine_max=(8*eight_mean+LB21_max)/9.',
            'interpretation': 'Rotation direction diagnostic only; necessary ceiling is not a measured full21 or nine score.'},
        'scope': 'Only2/21 complete official tasks.', 'full_longbench_status': 'not_evaluated',
        'nine_item_goal_status': 'not_evaluated', 'model_mutations': False, 'model_exported': False, 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    write_json(OUTPUT / 'config.json', config)
    record['config_sha256'] = sha256(OUTPUT / 'config.json'); write_json(OUTPUT / 'preparation.json', record)
    print(json.dumps({'arm': record['arm'], 'checkpoint_id': record['checkpoint_id'], 'status': record['status'],
        'preparation_sha256': sha256(OUTPUT / 'preparation.json'), 'source_count': len(sources), 'input_count': len(evidence),
        'required_longbench21_mean': required_lb, 'maximum_nine_if_longbench21_100': maximum_nine}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    prepare()
