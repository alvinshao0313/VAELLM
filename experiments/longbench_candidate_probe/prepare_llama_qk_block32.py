"""Bind the completed block32 endpoint to the identical S official400 protocol."""
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_candidate_probe.prepare_llama_qk_full import (
    PUBLIC, ROOT, R2, TASKS, E767_ID, read, require, merge)
from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.longbench_v1.models import make_config, check_config

FULL = R2 / 'llama_qk_full_retrieval400'
SOURCE = R2 / 'llama_qk_rotation_pair/block32'
OUTPUT = R2 / 'llama_qk_block32_retrieval400'
UID = '328e0390-0597-4c65-aec4-a13772466ec3'
PAIR = SOURCE.parent / 'independent_review.json'
PAIR_SHA = '7410ae70fb1051018cb96ce489614df680ee9deda91dd98823bd8f2ac56285ba'


def prepare():
    require(Path.cwd() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
            and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and not OUTPUT.exists(), 'One fresh CPU-only S preparation is required.')
    original = read(FULL / 'preparation.json'); full_config = read(FULL / 'config.json')
    require(sha256(FULL / 'preparation.json') == '4f7f6d8adfca2e64ffb5d069ca5b8c3fdf48c3e7325ce80b53f85759904446c1'
            and sha256(FULL / 'config.json') == original['config_sha256']
            and original['datasets'] == list(TASKS) and original['sample_count'] == 400
            and original['source_root'] == str(ROOT), 'Require the corrected S full-arm protocol proof.')
    check_hashes(original['source_sha256']); check_hashes(original['evidence_sha256'])
    pair, launch, audit, summary, meta, assembled = (read(p) for p in (PAIR, SOURCE / 'launch.json', SOURCE / 'audit.json',
        SOURCE / 'full8_summary.json', SOURCE / 'final_model/checkpoint_meta.json', SOURCE / 'final_model/assembly_report.json'))
    candidate = pair['arms']['block32']; stages = ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')
    require(sha256(PAIR) == PAIR_SHA and pair['status'] == 'independent_cpu_verified'
            and pair['primary_comparison'] == 'full_minus_new_fresh_block32'
            and all(x['checkpoint_id'] == UID for x in (candidate, launch, audit, summary, meta, assembled))
            and launch['status'] == 'completed' and launch['exit_code'] == 0 and launch['block_size'] == 32
            and launch['categories'] == ['q_proj', 'k_proj'] and all((SOURCE / (s + '.exit')).read_text().strip() == '0' for s in stages)
            and audit['status'] == assembled['status'] == 'verified' and audit['base_checkpoint_id'] == E767_ID,
            'Require the actual completed and independently reviewed block32 endpoint.')
    require(candidate['leaf_count'] == 64 and candidate['sample_count'] == 24742
            and candidate['mean_percent'] == summary['mean_percent'] == launch['mean_percent']
            and candidate['task_metrics'] == summary['task_metrics']
            and audit['source_union_save_fresh_tensors'] == assembled['serialized_tensors_compared']
            == assembled['reloaded_tensors_compared'] == 2979 and audit['packed_bank_count'] == 448
            and audit['replaced_projection_count'] == 64 and audit['preserved_projection_count'] == 160
            and audit['replaced_local_tensor_count'] == 832 and audit['frozen_complement_tensor_count'] == 2147
            and audit['assembly_report_sha256'] == sha256(SOURCE / 'final_model/assembly_report.json')
            and launch['full8_summary_sha256'] == sha256(SOURCE / 'full8_summary.json'), 'Source union/full8 proof differs.')
    budget = audit['bit_budget']
    require(budget['user_linear_bits'] == budget['packed_linear_bits'] == 17095180288
            and budget['code_padding_bits'] == 0 and budget['budget_passed'] and not budget['contract_violations']
            and budget['original_linear_weights'] == 6979321856, 'Wrong actual source budget.')
    final = SOURCE / 'final_model'
    config = make_config(full_config['archive']['path'], kind='compressed', hf_path=full_config['model']['hf_path'],
                         checkpoint=str(final), mode='full')
    model = deepcopy(full_config['model'])
    for key in ('checkpoint', 'checkpoint_id', 'metadata_sha256', 'weight_stat'):
        model[key] = config['model'][key]
    require(config['model'] == model and config['model']['checkpoint_id'] == UID
            and all(config[k] == full_config[k] for k in full_config if k != 'model'),
            'Only the real endpoint may differ from the full-arm generation configuration.')
    check_config(config)
    source_files = {str(Path(__file__).resolve()): sha256(__file__)}
    require(Path(__file__).read_bytes() == (PUBLIC / Path(__file__).resolve().relative_to(ROOT)).read_bytes(), 'Public/S entry differs.')
    evidence_paths = [PAIR, FULL / 'preparation.json', FULL / 'config.json', *[SOURCE / p for p in (
        'launch.json', 'launch.prepared.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json',
        'final_model/checkpoint_meta.json', 'final_model/config.json', 'final_model/assembly_report.json', *(s + '.exit' for s in stages))]]
    record = deepcopy(original); record.pop('matched_block32')
    eight = candidate['mean_percent']; bound = record['nine_item_necessary_bound']
    bound.update(candidate_eight_mean=eight, required_longbench21_mean=9 * bound['minimum_nine_mean'] - 8 * eight,
                 maximum_nine_if_longbench21_100=(8 * eight + 100) / 9,
                 cannot_meet_nine_target_at_this_endpoint=(8 * eight + 100) / 9 < bound['minimum_nine_mean'])
    record.update(schema='llama_qk_block32_retrieval400_v1', arm='block32', checkpoint_id=UID,
        config_path=str(OUTPUT / 'config.json'), output_dir=str(OUTPUT / 'run'), source_bit_budget=budget,
        source_sha256=merge(original['source_sha256'], launch['source_sha256'], config['source_sha256'], source_files),
        evidence_sha256=merge(original['evidence_sha256'], {str(p): sha256(p) for p in evidence_paths}),
        completed_candidate={'directory': str(SOURCE), 'checkpoint_dir': str(final), 'checkpoint_id': UID,
            'full8_mean_percent': eight, 'bit_budget': budget, 'weight_stat': file_stat(final / meta['state_dict_file']),
            'review_path': str(PAIR), 'native_proof': deepcopy(original['completed_candidate']['native_proof'])},
        matched_full={'directory': str(FULL), 'checkpoint_id': original['checkpoint_id'], 'pair_review': str(PAIR)},
        comparison_boundary='Primary full-minus-fresh-block32 on identical official400; both actual native endpoints are independently reviewed. Original e767/BF16 are secondary controls. Only2/21 tasks; fresh QK re-encoding/history confounds comparisons to e767.',
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.longbench_v1.predict', '--config', str(OUTPUT / 'config.json'),
                       '--output', str(OUTPUT / 'run'), '--datasets', *TASKS])
    require(record['gpu'] is None and record['status'] == 'prepared_unbound' and record['timeout_seconds'] == 900
            and record['termination_grace_seconds'] == 20 and not torch.cuda.is_initialized(), 'Unbound900+20 CPU contract differs.')
    OUTPUT.mkdir(); write_json(OUTPUT / 'config.json', config)
    record['config_sha256'] = sha256(OUTPUT / 'config.json'); write_json(OUTPUT / 'preparation.json', record)
    print({'status': record['status'], 'sha256': sha256(OUTPUT / 'preparation.json'), 'sources': len(record['source_sha256']),
           'evidence': len(record['evidence_sha256']), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    prepare()
