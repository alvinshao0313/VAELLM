"""CPU-only matched QK-scope preparation; retain the all-decoder arm unchanged."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen_fullqk_lm_recovery.train import completed_proof
from experiments.qwen_fullqk_lm_validation.config import ROOT, R2, SOURCE, check_inputs, read
from experiments.qwen_fullqk_qkonly_lm.train import CONTROL, OUTPUT, configuration, training_arguments
from experiments.w25_eval.subset_state import selected_names
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Prepare in CPU-only bitvae with actual S cwd/PYTHONPATH.')
    require(not OUTPUT.exists(), 'Use a fresh QK-only output.')
    control = read(CONTROL)
    require(control['status'] == 'prepared_unbound' and completed_proof() == control['resource_validation'],
            'Keep the existing all-decoder preparation and its actual H252 proof.')
    record = deepcopy(control)
    names = selected_names('all', 'q_proj,k_proj', control['architecture'])
    inventory = expected_inventory(SOURCE, control['architecture'], names)
    require(inventory['selected_projections'] == 72 and inventory['selected_decoder_tensors'] == 432
            and inventory['selected_decoder_elements'] == 1239552 and inventory['frozen_tensors'] == 2991
            and inventory['all_code_banks'] == 504 and inventory['serialized_tensors'] == 3423,
            'Actual source key/shape inventory differs from the native QK-only contract.')
    record.update(schema='qwen_fullqk_qk_scope_lm_2000_v1', status='prepared_unbound', output_dir=str(OUTPUT),
        matched_all_decoder_output=control['output_dir'], matched_all_decoder_prepared=str(CONTROL),
        selected_targets=list(names), expected_inventory=inventory, training_argv=training_arguments(OUTPUT),
        argv=[sys.executable, '-u', '-m', 'experiments.qwen_fullqk_qkonly_lm.train',
              '--launch_record', str(OUTPUT / 'launch.json')],
        comparison_boundary='Same original43b0/FineWeb/2000 schedule as the all-decoder arm. Only update scope252 to72 and output change; f927 and its LongAlpaca QK8K run are not matched quality controls.',
        validation_reuse='Actual43b0 allD H252144-stage backward plus originalf927 QK72 frozen180-cache/independent export proof. No repeated GPU diagnostic.',
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled', 'fullqk_backward_resource': 'verified', 'qk_cached_subset': 'verified_reuse'})
    record['input_evidence_sha256'][str(CONTROL)] = sha256(CONTROL)
    cached = R2 / 'qwen_qk_longalpaca8k_validate_s4'
    proof, trained, bound = (read(cached / f) for f in ('verification.json', 'training.json', 'launch.json'))
    require(proof['status'] == bound['status'] == 'verified' and bound['dispatcher_exit_code'] == 0
            and proof['native_vs_independent_tensors_compared'] == 3423
            and proof['source_checkpoint_id'] == 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
            and trained['selected_targets'] == list(names) and bound['expected_inventory']['frozen_tensors'] == 2991
            and trained['observations']['inventory']['selected_projections'] == 72
            and trained['observations']['inventory']['decoder_tensors'] == 432,
            'Require the already completed real QK72/cache/native-export evidence.')
    for name in ('verification.json', 'training.json', 'launch.json'):
        path = cached / name
        record['input_evidence_sha256'][str(path)] = sha256(path)
    for path in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(path.read_text())
        record['source_sha256'][str(path)] = sha256(path)
    record['estimate'].update(frozen_BF16_dense_cache_bytes=12381585408,
        all_decoder_measured_peak_bytes=35621856256, simple_peak_plus_cache_bytes=48003441664,
        training_peak_gib_estimate=[45, 55],
        boundary='11.53125GiB frozen180 full-weight cache is additional persistent runtime storage. Peak estimate is not a measured bound; subset backward workspace decreases, and allD H252 plus QK8K cache evidence are reused.')
    cfg, hf, args, actual_names = configuration(record)
    require(tuple(actual_names) == names and record['source_checkpoint'] == control['source_checkpoint'], 'Wrong source/scope.')
    record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'canonical_difference_from_matched_control': ['target_modules', 'run_root_dir']}
    record['stage_templates'] = [{'name': name, 'argv': argv, 'allowed_exit_codes': allowed}
        for name, argv, allowed in stage_commands(checkpoint=OUTPUT / 'NATIVE_RUN/trainer_state/checkpoint-400',
            source_dir=SOURCE, output_dir=OUTPUT / 'step_0400')]
    check_inputs(record)
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['readiness']['cpu'] = 'passed'
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.prepared.json', record)
    print({'status': 'prepared_unbound', 'path': str(OUTPUT / 'launch.prepared.json'),
        'sha256': sha256(OUTPUT / 'launch.prepared.json'), 'sources': len(record['source_sha256']),
        'inputs': len(record['input_evidence_sha256']), 'inventory': inventory, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    prepare()
