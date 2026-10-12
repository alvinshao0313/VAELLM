"""CPU preparation of fresh FineWeb decoder/all-norm controls and four-step validation."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import torch
from experiments.general_lm_recovery.prepare import R2, SOURCES, data_contract
from experiments.general_lm_norm_recovery.train import ROOT, SAVE_STEPS, configuration, training_arguments
from experiments.general_lm_norm_recovery.callback import stage_commands
from experiments.general_lm_norm_recovery.state import expected_inventory
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import validate_checkpoint
from train_utils.runtime_snapshot import to_runtime_jsonable


# These copied summaries are provenance, not live launch-file hash dependencies.
# The original formal controls remain active and their launch files keep changing.
def prepare(output_root):
    output_root = Path(output_root).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU preparation requires hidden CUDA.')
    require(not output_root.exists(), 'Use a new preparation directory.')
    free = shutil.disk_usage(output_root.parent).free
    require(free >= 80_000_000_000, 'Reserve two formal branches and both short validation outputs.')
    manifest, statistics, train_file = data_contract()
    baseline_root = R2 / 'general_lm_recovery_s2000'
    baseline_preparation = baseline_root / 'preparation.json'
    baseline_records = {name: read(baseline_root / name / 'launch.json') for name in SOURCES}
    sources = {}
    for base in baseline_records.values():
        check_inputs(base)
        for key, digest in base['source_sha256'].items():
            require(key not in sources or sources[key] == digest, 'Baseline native source versions disagree.')
            sources[key] = digest
    extras = ['experiments/w25_eval/norm_state.py', 'experiments/w25_eval/validate_wide_subset_training.py',
              'experiments/qwen3_w25/subset_step_export_callback.py']
    for path in [*(ROOT / item for item in extras), *sorted(Path(__file__).parent.glob('*.py'))]:
        ast.parse(path.read_text())
        digest = sha256(path)
        require(str(path) not in sources or sources[str(path)] == digest, 'Previously validated source drift: ' + str(path))
        sources[str(path)] = digest
    records = {}
    for model_type, base in baseline_records.items():
        source, source_id, _ = SOURCES[model_type]
        architecture = base['architecture']
        _path, meta, budget = validate_checkpoint(source, architecture)
        require(meta['checkpoint_id'] == base['source_checkpoint_id'] == source_id and budget == base['source_bit_budget'],
                'Original source or persistent budget differs from the active decoder-only control.')
        inventory = expected_inventory(source, architecture)
        require(inventory['selected_projections'] == architecture['projection_count']
                and inventory['selected_decoder_tensors'] == inventory['all_decoder_tensors']
                and inventory['active_score_elements'] == 0 and inventory['norm_tensors'] > 0,
                'The union must include every decoder and backbone norm, no active code scores.')
        for diagnostic in (False, True):
            name = model_type if not diagnostic else 'validate_' + model_type + '_s4'
            output = output_root / name
            record = {key: deepcopy(base[key]) for key in (
                'source_checkpoint', 'source_checkpoint_id', 'source_weight_stat', 'source_bit_budget', 'architecture',
                'hf_source', 'input_data_stat', 'input_evidence_sha256', 'data', 'control_result_paths', 'control_mean_percent',
                'environment', 'protocol')}
            record['input_evidence_sha256'][str(baseline_preparation)] = sha256(baseline_preparation)
            record.update(schema='fineweb_all_decoder_all_norm_v1', status='prepared_unbound',
                kind='validate_s4' if diagnostic else 'formal_s2000', gpu=None, export_gpu=None,
                queued=False, automatic_retry=False, cwd=str(ROOT), python_executable=sys.executable,
                output_dir=str(output), expected_inventory=inventory, source_sha256=sources,
                validation_proof=None, validation_expected_path=str(output_root / ('validate_' + model_type + '_s4') / 'training.json'),
                resource_budget_seconds=1800 if diagnostic else 64800, termination_grace_seconds=20,
                training_argv=training_arguments(source, architecture['base_model_path'], output, train_file, diagnostic=diagnostic),
                argv=[sys.executable, '-u', '-m', 'experiments.general_lm_norm_recovery.train', '--launch_record', str(output / 'launch.json')],
                schedule={'max_steps': 2000, 'warmup_steps': 20, 'scheduler': 'cosine', 'save_steps': 2 if diagnostic else 400,
                          'actual_stop_step': 4 if diagnostic else 2000, 'full8_steps': [] if diagnostic else SAVE_STEPS},
                comparison={'decoder_only_control_output': str(baseline_root / model_type),
                    'baseline_preparation_sha256': sha256(baseline_preparation),
                    'scientific_delta': {'norm_train_mode': ['none', 'all'], 'norm_lr': [None, 1e-5],
                                         'distill_fp32_components': [['decoder'], ['decoder', 'norm']]},
                    'no_other_scientific_change': True, 'fresh_source_for_formal': True},
                readiness={'cpu': 'pending', 'gpu': 'not_scheduled', 'formal_requires_successful_short_proof': not diagnostic},
                experience={'reference': 'docs/experience/lessons/kd_and_data.md',
                    'difference': 'All backbone norms on the unchanged FineWeb decoder control; earlier final-norm/task or category-CAT runs are different conditions.',
                    'claim': 'Existing native union; no quality improvement presumed. BF16 persistent budget unchanged.'},
                estimate={'storage_bytes_per_formal_branch': 30_000_000_000, 'disk_free_at_prepare': free})
            if diagnostic:
                record['protocol'].update(full8=False, scope='Independent export/audit at steps2+4 and resumed training; no downstream tasks.')
            cfg, hf, args, names, actual_architecture = configuration(record)
            canonical = to_runtime_jsonable(cfg)
            original = deepcopy(base['cpu_configuration']['canonical_config'])
            adjusted = deepcopy(canonical)
            require(set(adjusted['explicit_cli_flags']) - set(original['explicit_cli_flags']) == {'--norm_lr'}
                    and not set(original['explicit_cli_flags']) - set(adjusted['explicit_cli_flags']), 'Only norm_lr is a new native CLI option.')
            adjusted['explicit_cli_flags'].remove('--norm_lr')
            if diagnostic:
                remaining = adjusted['remaining_argv']
                require(remaining.count('--save_steps') == 1 and remaining[remaining.index('--save_steps') + 1] == '2',
                        'Short save interval mirror differs.')
                remaining[remaining.index('--save_steps') + 1] = '400'
            for item in (original, adjusted):
                item.pop('run_root_dir')
                item['opt'].pop('logging_steps')
                item['opt'].pop('distill_fp32_components')
                item['aux'].pop('norm_train_mode')
                item['aux'].pop('norm_lr')
            require(original == adjusted, 'Canonical runtime differs beyond the declared norm union/output/short logging.')
            require(canonical['data'] == statistics['data_config'], 'Native LM data contract or seeded sampler configuration changed.')
            record['cpu_configuration'] = {'canonical_config': canonical, 'hf_args': to_runtime_jsonable(hf),
                'training_args': to_runtime_jsonable(args), 'all_projection_count': len(names),
                'model_type': actual_architecture['model_type'], 'native_data_config_matches_statistics': True,
                'canonical_difference_limited_to_declared_union': True}
            step = 2 if diagnostic else 400
            checkpoint = output / f'NATIVE_RUN/trainer_state/checkpoint-{step}'
            record['stage_templates'] = [{'name': key, 'argv': command, 'allowed_exit_codes': allowed}
                for key, command, allowed in stage_commands(checkpoint=checkpoint, source_dir=source,
                     output_dir=output / f'step_{step:04d}', diagnostic=diagnostic)]
            check_inputs(record)
            record.update(cuda_initialized=False)
            record['readiness']['cpu'] = 'passed'
            records[name] = record
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output_root.mkdir()
    for name, record in records.items():
        (output_root / name).mkdir()
        write_json(output_root / name / 'launch.json', record)
    summary = {'status': 'cpu_prepared_unbound', 'cuda_initialized': False, 'source_pin_count': len(sources),
        'launch_sha256': {name: sha256(output_root / name / 'launch.json') for name in records},
        'inventories': {name: {'decoders': record['expected_inventory']['selected_decoder_tensors'],
                               'norms': record['expected_inventory']['norm_tensors'],
                               'frozen_tensors': record['expected_inventory']['frozen_tensors'],
                               'bpw': record['source_bit_budget']['user_linear_bpw']} for name, record in records.items()},
        'scope': 'Two fresh formal2000 and two independent4-update validation records; no GPU execution.'}
    write_json(output_root / 'preparation.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=R2 / 'general_lm_norm_recovery_s2000')
    prepare(parser.parse_args().output_root)
