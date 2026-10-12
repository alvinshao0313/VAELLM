"""Prepare the existing FineWeb decoder recipe from the completed alternate-order Llama."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.general_lm_recovery.prepare import R2, data_contract
from experiments.general_lm_recovery.train import ROOT, SAVE_STEPS, configuration, training_arguments
from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import file_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import collect_results, validate_checkpoint
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable

CONTROL = R2 / 'general_lm_recovery_s2000/llama'
SOURCE_RUN = R2 / 'llama_order_kvqodgu_s500'
SOURCE_ID = '5581a1fd-9a37-4471-b112-b745b5db6d75'


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and os.environ.get('ACCELERATE_GRADIENT_ACCUMULATION_STEPS', '1') == '1'
            and Path.cwd().resolve() == ROOT, 'Prepare on CPU in the repository with CUDA hidden.')
    require(not output.exists(), 'Use a fresh output directory.')
    free = shutil.disk_usage(output.parent).free
    require(free >= 30_000_000_000, 'Insufficient storage for the existing decoder trajectory.')
    control = read(CONTROL / 'launch.json')
    require(control['status'] == 'completed' and control['exit_code'] == 0
            and control['source_checkpoint_id'] == '35776715-2707-45a9-9fb4-2da1efef3f0e'
            and control['training_result']['global_step'] == 2000
            and control['completed_steps'] == SAVE_STEPS, 'Require the completed original-order FineWeb control.')
    check_inputs(control)
    native_paths = list(CONTROL.glob('home_*/normalized_e2e_runtime_args.json'))
    require(len(native_paths) == 1, 'Require one actual native control configuration.')
    snapshot = read(native_paths[0])
    require(snapshot['canonical_config'] == control['cpu_configuration']['canonical_config'],
            'The actual original-order control differs from its prepared scientific configuration.')
    _, statistics, train_file = data_contract()
    require(str(train_file) == control['data']['train_file']
            and statistics['data_config'] == snapshot['canonical_config']['data'],
            'Keep the actual FineWeb documents, encoding and seeded native sampler.')

    source_run = read(SOURCE_RUN / 'launch.json')
    review = read(SOURCE_RUN / 'independent_review.json')
    summary = read(SOURCE_RUN / 'full8_summary.json')
    require(source_run['status'] == 'completed' and source_run['exit_code'] == 0
            and review['status'] == 'independently_verified'
            and source_run['checkpoint_id'] == review['checkpoint_id'] == summary['checkpoint_id'] == SOURCE_ID
            and source_run['full8_summary_sha256'] == sha256(SOURCE_RUN / 'full8_summary.json'),
            'Require the completed and independently reviewed alternate-order final.')
    source, meta, budget = validate_checkpoint(source_run['final_checkpoint_dir'], control['architecture'])
    require(meta['checkpoint_id'] == SOURCE_ID and source.name == 'final_model'
            and budget['budget_passed'] and budget['user_linear_bpw'] < 2.5
            and budget == summary['bit_budget'], 'Actual alternate-order source or storage differs from its completed audit.')
    for key in ('components', 'user_linear_bits', 'packed_linear_bits', 'original_linear_weights'):
        require(budget[key] == control['source_bit_budget'][key], 'The matched source geometry/budget differs: ' + key)
    inventory = expected_inventory(source, control['architecture'], tuple(control['architecture']['projection_shapes']))
    require(inventory == control['expected_inventory']
            and inventory['selected_decoder_tensors'] == 1344 and inventory['serialized_tensors'] == 2979
            and inventory['frozen_tensors'] == 1635 and inventory['all_code_banks'] == 448
            and inventory['selected_code_banks'] == inventory['active_score_elements'] == 0,
            'Keep exactly all native decoders trainable and the same frozen complement.')
    original_paths = [SOURCE_RUN / (name + '.json') for name in ('mmlu', 'seven')]
    original = collect_results(original_paths, control['architecture'], checkpoint=(source, meta))
    require(original['mean_percent'] == summary['mean_percent'] == review['mean_percent'],
            'The alternate-order initial raw score differs from its independent review.')
    for candidate_path, control_path in zip(original_paths, control['control_result_paths'], strict=True):
        candidate_raw, control_raw = (read(p)['evals']['lm_eval'] for p in (candidate_path, control_path))
        for field in ('configs', 'versions', 'n_shot', 'n_samples'):
            require(candidate_raw[field] == control_raw[field], 'Initial paired benchmark protocol differs: ' + field)

    sources, inputs = deepcopy(control['source_sha256']), deepcopy(control['input_evidence_sha256'])
    for path in [ROOT / 'experiments/general_lm_recovery/prepare.py', *sorted(Path(__file__).parent.glob('*.py'))]:
        ast.parse(path.read_text()); digest = sha256(path)
        require(str(path) not in sources or sources[str(path)] == digest, 'A reused source changed: ' + str(path))
        sources[str(path)] = digest
    evidence = [CONTROL / 'launch.json', CONTROL / 'training.json', native_paths[0],
        SOURCE_RUN / 'launch.json', SOURCE_RUN / 'full8_summary.json', SOURCE_RUN / 'independent_review.json',
        source.parent / 'normalized_cat_runtime_args.json', *original_paths]
    evidence += [source / name for name in ('checkpoint_meta.json', 'config.json', 'tokenizer.json',
                                           'tokenizer_config.json', 'special_tokens_map.json')]
    matched_trajectory = {}
    for step in SAVE_STEPS:
        path = CONTROL / f'step_{step:04d}'
        result, audit = read(path / 'full8_summary.json'), read(path / 'audit.json')
        require(result['status'] == 'full8_evaluation_verified' and result['global_step'] == step
                and result['audit_sha256'] == sha256(path / 'audit.json')
                and audit['status'] == 'verified' and audit['selected_global_step'] == step
                and audit['source_checkpoint_id'] == control['source_checkpoint_id']
                and audit['changed_code_bits'] == 0 and audit['frozen_bytes_unchanged'],
                'The matched original-order trajectory evidence is incomplete.')
        matched_trajectory[str(step)] = {'mean_percent': result['mean_percent'], 'checkpoint_id': result['checkpoint_id']}
        evidence += [path / name for name in ('full8_summary.json', 'audit.json', 'independent_review.json')]
    inputs.update({str(path): sha256(path) for path in evidence})
    require(not any(Path(path).is_relative_to(SOURCE_RUN) and Path(path).suffix in ('.bin', '.pt', '.pth')
                    for path in inputs), 'Do not pin retired category/round/trainer payloads.')

    record = {key: deepcopy(control[key]) for key in ('kind', 'architecture', 'hf_source', 'input_data_stat',
        'data', 'environment', 'schedule', 'protocol', 'validation_reuse', 'resource_budget_seconds', 'termination_grace_seconds')}
    record.update(schema='general_lm_alternate_order_decoder_2000_v1', status='prepared_unbound',
        gpu=None, export_gpu=None, queued=False, automatic_retry=False, cwd=str(ROOT), python_executable=sys.executable,
        output_dir=str(output), source_checkpoint=str(source), source_checkpoint_id=SOURCE_ID,
        source_weight_stat=file_stat(source / 'pytorch_model.bin'), source_bit_budget=budget,
        expected_inventory=inventory, source_sha256=sources, input_evidence_sha256=inputs,
        control_result_paths=[str(p) for p in original_paths], control_mean_percent=original['mean_percent'],
        training_argv=training_arguments(source, control['architecture']['base_model_path'], output, train_file),
        argv=[sys.executable, '-u', '-m', 'experiments.general_lm_recovery.train', '--launch_record', str(output / 'launch.json')],
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled'},
        comparison={'primary': str(CONTROL), 'matched_control_source_checkpoint_id': control['source_checkpoint_id'],
            'primary_mode': 'Same-step fresh FineWeb all-decoder trajectory from alternate versus original CAT-order sources.',
            'scientific_change': 'Only the initial student checkpoint differs; original native training/export/evaluation code is reused.',
            'matched_control_trajectory': matched_trajectory,
            'interpretation': 'CAT order also changes prior category RNG, fused dense targets and recovery context. Do not attribute the full trajectory difference solely to moving down, or compare e767 as a matched control.',
            'secondary': 'The alternate-order source itself; LongBench and the nine-item goal require separate evaluation.'},
        estimate={'storage_bytes_per_branch': 30_000_000_000, 'disk_free_at_prepare': free,
            'matched_control_total_seconds': control['finished_unix'] - control['started_unix'],
            'training_hours_estimate': [9, 15], 'estimate_not_measurement': True},
        experience={'references': ['docs/experience/lessons/experiment_design.md', 'docs/experience/lessons/kd_and_data.md'],
            'difference': 'The alternate-order native CAT source is 60.0389%, versus original-order 59.3577%; this does not predict the recovery difference.',
            'action': 'Keep the full original2000/warm20 schedule and all five complete evaluations; no head, norm, LoRA or code learning.',
            'hypothesis': 'The source-order improvement may persist or change after the identical existing recovery.'})
    expected_argv = list(control['training_argv'])
    for flag, value in (('--student_checkpoint_dir', str(source)), ('--run_root_dir', str(output))):
        require(expected_argv.count(flag) == 1, 'Ambiguous original native argument: ' + flag)
        expected_argv[expected_argv.index(flag) + 1] = value
    require(record['training_argv'] == expected_argv, 'Only source and output CLI values may change.')
    cfg, hf, args, names, architecture = configuration(record)
    canonical = to_runtime_jsonable(cfg)
    expected_canonical = deepcopy(snapshot['canonical_config'])
    expected_canonical.update(student_checkpoint_dir=str(source), run_root_dir=str(output))
    require(canonical == expected_canonical and to_runtime_jsonable(hf) == control['cpu_configuration']['hf_args'],
            'Actual canonical configuration differs beyond source and output.')
    training = to_runtime_jsonable(args)
    expected_training = deepcopy(control['cpu_configuration']['training_args'])
    expected_training['logging_dir'] = training['logging_dir']
    require(training == expected_training, 'Training arguments differ beyond the generated logging timestamp.')
    record['cpu_configuration'] = {'canonical_config': canonical, 'hf_args': to_runtime_jsonable(hf),
        'training_args': training, 'all_projection_count': len(names), 'model_type': architecture['model_type'],
        'native_data_config_matches_statistics': True, 'only_source_and_output_changed': True,
        'training_args_difference': 'Only generated logging_dir timestamp, with identical original scientific CLI.'}
    checkpoint = output / 'NATIVE_RUN/trainer_state/checkpoint-400'
    record['stage_templates'] = [{'name': name, 'argv': command, 'allowed_exit_codes': allowed}
        for name, command, allowed in stage_commands(checkpoint=checkpoint, source_dir=source, output_dir=output / 'step_0400')]
    check_inputs(record)
    require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA.')
    record.update(cuda_initialized=False); record['readiness']['cpu'] = 'passed'
    output.mkdir(parents=True)
    write_json(output / 'launch.prepared.json', record); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'launch_sha256': sha256(output / 'launch.prepared.json'),
        'sources': len(sources), 'inputs': len(inputs), 'source_id': SOURCE_ID,
        'decoder_tensors': inventory['selected_decoder_tensors'], 'frozen_tensors': inventory['frozen_tensors'],
        'bpw': budget['user_linear_bpw'], 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=R2 / 'general_lm_llama_old_order_s2000')
    prepare(parser.parse_args().output_dir)
