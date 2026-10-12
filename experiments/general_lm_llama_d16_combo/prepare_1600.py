"""Bind the unchanged fresh FineWeb2000 recipe to the verified1600+D16 source."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.general_lm_llama_d16_combo.prepare import D16_PROOF
from experiments.general_lm_recovery.prepare import R2, data_contract
from experiments.general_lm_recovery.train import ROOT, check_inputs, configuration, read, training_arguments
from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable

TEMPLATE = R2 / 'general_lm_llama_d16_combo_s2000'
COMBO = R2 / 'llama_fineweb1600_down_d16_combo'
CHECKPOINT_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Use a fresh CPU-only preparation directory.')
    free = shutil.disk_usage(output.parent).free
    require(free >= 30_000_000_000, 'Reserve30GB for five full independent exports.')
    manifest, stats, train_file = data_contract()
    template_path = TEMPLATE / 'launch.json'; template = read(template_path)
    require(template['status'] == 'prepared_unbound' and template['source_checkpoint_id']
            == 'e985bada-3960-4a5a-82af-4652ef94ca6c', 'Keep the original800-source preparation unchanged and unbound.')
    check_inputs(template); configuration(template)
    record = deepcopy(template)
    review, terminal, summary = (read(COMBO / name) for name in ('review.json', 'launch.json', 'full8_summary.json'))
    source = COMBO / 'final_model'; meta = read(source / 'checkpoint_meta.json'); union = read(source / 'assembly_report.json')
    require(review['status'] == union['status'] == 'verified' and terminal['status'] == 'completed'
            and terminal['exit_code'] == 0 and all(item['checkpoint_id'] == CHECKPOINT_ID
            for item in (review, terminal, summary, meta, union)), 'Require the real completed1600+D16 source identity.')
    check_hashes(review['evidence_sha256'])
    for name in ('assembly', 'mmlu', 'seven', 'dispatcher', 'outer'):
        require((COMBO / (name + '.exit')).read_text().strip() == '0', 'Incomplete original combination: ' + name)
    proof = review['native_proof']; sources = union['sources']
    require(proof['reused_union_save_fresh_evidence'] and proof['metadata_specs_exact_source_union']
            and proof['donor_down_count'] == 32 and proof['preserved_base_projection_count'] == 192
            and proof['serialized_tensors_compared'] == proof['reloaded_tensors_compared']
            == union['serialized_tensors_compared'] == union['reloaded_tensors_compared'] == 2979
            and sources['categories'] == ['down_proj'] and len(sources['replaced_targets']) == 32,
            'Require the complete mixed-geometry native source union.')
    base, donor = Path(sources['base_checkpoint_dir']), Path(sources['donor_checkpoint_dir'])
    base_audit, donor_audit = read(base.parent / 'audit.json'), read(donor.parent / 'audit.json')
    require(base_audit['status'] == donor_audit['status'] == 'verified'
            and base_audit['checkpoint_id'] == sources['base_checkpoint_id']
            and donor_audit['checkpoint_id'] == sources['donor_checkpoint_id'] == '8a4633a7-93bd-429a-8a0e-a519a2ae30ae'
            and base_audit['source_checkpoint_id'] == '35776715-2707-45a9-9fb4-2da1efef3f0e'
            and base_audit['selected_global_step'] == 1600 and base_audit['frozen_tensors_compared'] == 1635
            and base_audit['changed_code_bits'] == 0 and base_audit['rank_zero'] and base_audit['frozen_bytes_unchanged'],
            'Bind the actual1600 decoder history and unchanged original backbone.')
    old_meta = read(Path(template['source_checkpoint']) / 'checkpoint_meta.json')
    require(meta['converted_modules'] == old_meta['converted_modules']
            and meta['norm_train_mode'] == meta['lm_head_train_mode'] == 'none' and meta['lora_config'] is None,
            'Keep the actual32 D16/K24 and192 D32 native geometries, protection/rotation schema and rank0.')
    budget = summary['bit_budget']
    require(budget == union['bit_budget'] and budget['budget_passed'] and budget['code_padding_bits'] == 0
            and budget['user_linear_bits'] == 17095180288 and budget['original_linear_weights'] == 6979321856,
            'Actual combined source storage differs.')
    for key in ('user_linear_bits', 'original_linear_weights', 'components'):
        require(budget[key] == template['source_bit_budget'][key], 'Only the source history may change: ' + key)
    kernel = read(D16_PROOF)
    require(kernel['status'] == 'passed' and kernel['exit_code'] == 0 and kernel['case'] ==
            {'K': 24, 'D': 16, 'H': 128, 'stages': 2, 'parameter_dtype': 'float32',
             'compute_dtype': 'bfloat16', 'group_size': 8}, 'Reuse the actual D16 FP32 native mechanism proof.')
    check_hashes({str(ROOT / key): value for key, value in kernel['source_sha256'].items()})
    inventory = expected_inventory(source, record['architecture'], tuple(record['architecture']['projection_shapes']))
    require(inventory == template['expected_inventory'], 'Mixed-geometry trainable/frozen inventory changed.')
    paths = [COMBO / (name + '.json') for name in ('mmlu', 'seven')]
    current = collect_results(paths, record['architecture'], checkpoint=(source, meta))
    require(current['mean_percent'] == review['mean_percent'] == summary['mean_percent']
            and current['leaf_count'] == 64 and current['sample_count'] == 24742, 'Complete primary control differs.')
    evidence = [template_path, COMBO / 'review.json', base.parent / 'audit.json', donor.parent / 'audit.json']
    record['input_evidence_sha256'].update(review['evidence_sha256'])
    record['input_evidence_sha256'].update({str(path): sha256(path) for path in evidence})
    path = Path(__file__).resolve(); ast.parse(path.read_text()); record['source_sha256'][str(path)] = sha256(path)
    record.update(schema='general_lm_llama_d16_combo1600_fresh2000_v1', status='prepared_unbound', gpu=None, export_gpu=None,
        output_dir=str(output), python_executable=sys.executable, source_checkpoint=str(source), source_checkpoint_id=CHECKPOINT_ID,
        source_weight_stat=file_stat(source / 'pytorch_model.bin'), source_bit_budget=budget, expected_inventory=inventory,
        control_result_paths=list(map(str, paths)), control_mean_percent=current['mean_percent'],
        training_argv=training_arguments(source, record['architecture']['base_model_path'], output, train_file),
        argv=[sys.executable, '-u', '-m', 'experiments.general_lm_recovery.train', '--launch_record', str(output / 'launch.json')],
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled'},
        initialization={'source_union': sources, 'new_optimizer': True, 'new_scheduler': True,
            'local_step_range': [0, 2000], 'exact_resume': False,
            'data_replay': 'Replay seed0/epoch0 first8000 native sampler documents.192 projections already carry originalFineWeb1600 updates over first6400 documents;32 D16down come from freshHF CAT and have no FineWeb recovery history.',
            'comparison_boundary': 'New independent mixed-geometry segment from e767, not1600 exact-resume. Same data and training recipe as the unrun800-source preparation; source history differs.'})
    expected_argv = list(template['training_argv'])
    expected_argv[expected_argv.index('--student_checkpoint_dir') + 1] = str(source)
    expected_argv[expected_argv.index('--run_root_dir') + 1] = str(output)
    require(record['training_argv'] == expected_argv, 'Only source/output may change from the original fresh2000 recipe.')
    record['validation_reuse']['mixed_geometry_save_reload_full8'] = str(COMBO / 'review.json')
    record['estimate']['disk_free_at_prepare'] = free
    cfg, hf, args, names, architecture = configuration(record)
    canonical = to_runtime_jsonable(cfg); expected = deepcopy(template['cpu_configuration']['canonical_config'])
    expected['student_checkpoint_dir'], expected['run_root_dir'] = str(source), str(output)
    require(canonical == expected and canonical['data'] == stats['data_config']
            and record['data']['jsonl_sha256'] == manifest['jsonl_sha256'], 'Data or native science differs beyond fresh source/output.')
    record['cpu_configuration'] = {'canonical_config': canonical, 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'all_projection_count': len(names),
        'model_type': architecture['model_type'], 'native_data_config_matches_statistics': True}
    checkpoint = output / 'NATIVE_RUN/trainer_state/checkpoint-400'
    record['stage_templates'] = [{'name': name, 'argv': argv, 'allowed_exit_codes': allowed}
        for name, argv, allowed in stage_commands(checkpoint=checkpoint, source_dir=source, output_dir=output / 'step_0400')]
    check_inputs(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['readiness']['cpu'] = 'passed'; record['cuda_initialized'] = False
    output.mkdir(); write_json(output / 'launch.json', record)
    prepared = {'status': 'cpu_prepared_unbound', 'cuda_initialized': False, 'source_checkpoint_id': CHECKPOINT_ID,
        'launch_sha256': sha256(output / 'launch.json'), 'source_pin_count': len(record['source_sha256']),
        'input_pin_count': len(record['input_evidence_sha256']), 'source_bpw': budget['user_linear_bpw'],
        'expected_inventory': inventory, 'primary_control_mean_percent': current['mean_percent'],
        'disk_free_bytes': free, 'scope': 'Fresh2000/new optimizer/cosine schedule from1600+D16; no resume, GPU binding or queue.'}
    write_json(output / 'preparation.json', prepared); print(prepared, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=R2 / 'general_lm_llama_d16_combo1600_s2000')
    prepare(parser.parse_args().output)
