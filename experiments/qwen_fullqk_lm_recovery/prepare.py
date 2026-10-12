"""Prepare the existing2000-step recipe; proof completion remains a separate dependency."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.general_lm_recovery.train import configuration, training_arguments
from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen_fullqk_lm_validation.config import (
    CONTROL, OUTPUT as SHORT, PUBLIC, ROOT, R2, SOURCE, SOURCE_ID, check_inputs, read)
from experiments.qwen_fullqk_lm_recovery.train import completed_proof
from train_utils.runtime_snapshot import to_runtime_jsonable

OUTPUT = R2 / 'general_lm_qwen_fullqk_s2000'


def prepare(complete_proof=False):
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Use CPU-only bitvae in actual S cwd/PYTHONPATH.')
    if complete_proof:
        record = read(OUTPUT / 'launch.prepared.json')
        require(record['status'] == 'prepared_dependency_pending'
                and not (OUTPUT / 'launch.verified.prepared.json').exists(), 'Complete only the untouched pending preparation.')
        check_inputs(record)
        record['resource_validation'] = completed_proof()
        record['input_evidence_sha256'].update(record['resource_validation']['evidence_sha256'])
        record['status'] = 'prepared_unbound'
        record['readiness']['fullqk_backward_resource'] = 'verified'
        target = OUTPUT / 'launch.verified.prepared.json'
    else:
        require(not OUTPUT.exists(), 'Use a new formal output; no overwrite or resume.')
        short, control = read(SHORT / 'launch.prepared.json'), read(CONTROL)
        require(short['readiness']['cpu'] == 'passed' and short['source_checkpoint_id'] == SOURCE_ID,
                'Require the actual short CPU source binding, without assuming its GPU result.')
        record = deepcopy(short)
        for key in ('dispatch_argv', 'cpu_configuration', 'resource_cap_enforcement', 'scope', 'validation_reuse'):
            record.pop(key)
        record.update(schema='qwen_fullqk_all_decoder_lm_2000_v1', kind='formal_s2000',
            status='prepared_dependency_pending', output_dir=str(OUTPUT), export_gpu=None,
            training_argv=training_arguments(SOURCE, short['architecture']['base_model_path'], OUTPUT, short['data']['train_file']),
            argv=[sys.executable, '-u', '-m', 'experiments.qwen_fullqk_lm_recovery.train',
                  '--launch_record', str(OUTPUT / 'launch.json')],
            schedule=deepcopy(control['schedule']), resource_budget_seconds=control['resource_budget_seconds'],
            estimate=deepcopy(control['estimate']), protocol=deepcopy(control['protocol']),
            control_result_paths=deepcopy(control['control_result_paths']), control_mean_percent=control['control_mean_percent'],
            resource_validation={'status': 'pending', 'path': str(SHORT / 'verification.json'), 'verification_sha256': None},
            readiness={'cpu': 'pending', 'gpu': 'not_scheduled', 'fullqk_backward_resource': 'pending'},
            scientific_boundary='Fresh43b0 full-QK source retains its original codes/decoders; no quality improvement is assumed and this is not a pure-rotation contrast.',
            validation_reuse='Original synchronous every400 export/full8 callback; full-QK backward must pass the separate actual four-step proof.',
            comparison_boundary='Primary trajectory starts from43b0. Original f927 FineWeb recipe/control is retained for context; different starting codes/decoders preclude a pure-rotation causal claim.')
        record['input_evidence_sha256'][str(SHORT / 'launch.prepared.json')] = sha256(SHORT / 'launch.prepared.json')
        copied = ROOT / 'experiments/general_lm_recovery/train.py'
        original = PUBLIC / 'experiments/general_lm_recovery/train.py'
        require(copied.read_bytes() == original.read_bytes(), 'Only a byte-identical addition of the missing native entry is permitted.')
        record['added_native_entry_copy'] = {'source': str(original), 'copy': str(copied), 'sha256': sha256(copied)}
        for path in [copied, *sorted(Path(__file__).parent.glob('*.py'))]:
            ast.parse(path.read_text())
            record['source_sha256'][str(path)] = sha256(path)
        expected = list(control['training_argv'])
        for option, value in (('--student_checkpoint_dir', str(SOURCE)), ('--run_root_dir', str(OUTPUT))):
            expected[expected.index(option) + 1] = value
        require(record['training_argv'] == expected, 'Formal native argv may change only source/output.')
        free = shutil.disk_usage(OUTPUT.parent).free
        require(free >= 35_000_000_000, 'Reserve35GB for five independent endpoints and native final/step state.')
        record['estimate'].update(disk_free_at_prepare=free, storage_bytes_per_branch=35000000000,
            fullqk_throughput='Pending actual short validation; original block32 wall-time is only a reference.')
        target = OUTPUT / 'launch.prepared.json'
    cfg, hf, args, names, _ = configuration(record)
    canonical = to_runtime_jsonable(cfg)
    expected_canonical = deepcopy(read(CONTROL)['cpu_configuration']['canonical_config'])
    expected_canonical.update(student_checkpoint_dir=str(SOURCE), run_root_dir=str(OUTPUT))
    require(canonical == expected_canonical and len(names) == 252, 'Actual parser differs beyond fresh source/output.')
    record['cpu_configuration'] = {'canonical_config': canonical, 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'canonical_difference_from_control': ['student_checkpoint_dir', 'run_root_dir']}
    record['stage_templates'] = [{'name': name, 'argv': argv, 'allowed_exit_codes': allowed}
        for name, argv, allowed in stage_commands(checkpoint=OUTPUT / 'NATIVE_RUN/trainer_state/checkpoint-400',
            source_dir=SOURCE, output_dir=OUTPUT / 'step_0400')]
    check_inputs(record)
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['readiness']['cpu'] = 'passed'
    if not complete_proof:
        OUTPUT.mkdir()
    write_json(target, record)
    print({'status': record['status'], 'path': str(target), 'sha256': sha256(target),
        'source_pins': len(record['source_sha256']), 'inputs': len(record['input_evidence_sha256']),
        'resource_validation': record['resource_validation']['status'], 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete-proof', action='store_true')
    prepare(parser.parse_args().complete_proof)
