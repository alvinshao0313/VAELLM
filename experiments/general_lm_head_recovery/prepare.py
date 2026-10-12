"""CPU-only native Llama decoder/full-head contract from the active e767 FineWeb control."""
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
from experiments.general_lm_head_recovery.train import ROOT, SAVE_STEPS, configuration, training_arguments
from experiments.general_lm_head_recovery.callback import stage_commands
from experiments.general_lm_head_recovery.export import CODE_FILES
from experiments.general_lm_head_recovery.state import expected_inventory
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import validate_checkpoint
from train_utils.runtime_snapshot import to_runtime_jsonable

CONTROL = R2 / 'general_lm_llama_d16_combo1600_s2000'
SOURCE_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'


def prepare(output, *, kind='validate_s4', validation_proof=None):
    output = Path(output).resolve()
    require(kind in ('validate_s4', 'formal_s2000'), 'Select the independent four-update validation or fresh formal2000.')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and os.environ.get('ACCELERATE_GRADIENT_ACCUMULATION_STEPS', '1') == '1', 'CPU preparation requires hidden CUDA.')
    require(not output.exists(), 'Use an independent fresh output directory.')
    disk_parent = output.parent
    while not disk_parent.exists(): disk_parent = disk_parent.parent
    free = shutil.disk_usage(disk_parent).free
    require(free >= (25_000_000_000 if kind == 'validate_s4' else 40_000_000_000), 'Insufficient native/head/optimizer/export storage reserve.')
    control_path = CONTROL / 'launch.prepared.json'
    control = read(control_path)
    check_inputs(control)
    normalized = list(CONTROL.glob('home_*/normalized_e2e_runtime_args.json'))
    require(len(normalized) == 1, 'Require the actual immutable native decoder-only control snapshot.')
    snapshot = read(normalized[0])
    require(snapshot['canonical_config'] == control['cpu_configuration']['canonical_config'], 'Actual decoder-control science differs from its preparation.')
    manifest, statistics, train_file = data_contract()
    require(str(train_file) == control['data']['train_file']
            and snapshot['canonical_config']['data'] == statistics['data_config'], 'Keep the identical native FineWeb data/sampler contract.')
    source, meta, budget = validate_checkpoint(control['source_checkpoint'], control['architecture'])
    require(meta['checkpoint_id'] == control['source_checkpoint_id'] == SOURCE_ID
            and control['architecture']['model_type'] == 'llama'
            and budget == control['source_bit_budget'] and budget['user_linear_bits'] == 17095180288
            and budget['original_linear_weights'] == 6979321856, 'Require the actual e767/D16 source and unchanged storage budget.')
    inventory = expected_inventory(source, control['architecture'])
    require(inventory['selected_projections'] == control['architecture']['projection_count']
            and inventory['selected_decoder_tensors'] == inventory['all_decoder_tensors']
            and inventory['head_tensors'] == 1 and inventory['active_score_elements'] == 0,
            'Use all native decoder leaves plus exactly the original dense head.')
    sources, inputs = deepcopy(control['source_sha256']), deepcopy(control['input_evidence_sha256'])
    extras = ('experiments/general_lm_recovery/prepare.py', 'experiments/general_lm_recovery/train.py',
              'experiments/general_lm_full_kl_probe/train.py', 'experiments/general_lm_full_kl_probe/run.py',
              'experiments/w25_eval/validate_wide_subset_training.py',
              'experiments/w25_eval/validate_decoder_subset_training.py',
              'experiments/qwen3_w25/subset_step_export_callback.py',
              'experiments/qwen3_w25/e2e_fullhead_audit.py',
              'train_utils/model_level_trainables.py', 'train_utils/model_level_optimizer.py',
              'train_utils/distill_precision.py')
    for path in [*(ROOT / name for name in (*CODE_FILES, *extras)), *sorted(Path(__file__).parent.glob('*.py'))]:
        ast.parse(path.read_text()); digest = sha256(path)
        require(str(path) not in sources or sources[str(path)] == digest, 'A reused source changed: ' + str(path))
        sources[str(path)] = digest
    for path in (control_path, normalized[0]): inputs[str(path)] = sha256(path)
    proof_path = None
    if kind == 'formal_s2000':
        require(validation_proof is not None, 'Formal preparation requires the actual successful head validation proof.')
        proof_path = Path(validation_proof).resolve(); proof = read(proof_path)
        require(proof['status'] == 'verified' and proof['source_checkpoint_id'] == SOURCE_ID
                and proof['source_sha256'] == sources and proof['expected_inventory'] == inventory
                and proof['native_vs_step4_tensors_compared'] == inventory['serialized_tensors']
                and proof['changed_head_fp32_elements'] > 0 and proof['changed_code_bits'] == 0,
                'The selected short proof does not cover this exact source/code/head inventory.')
        inputs[str(proof_path)] = sha256(proof_path)
    else:
        require(validation_proof is None, 'The short starts fresh and does not consume another validation state.')
    record = {key: deepcopy(control[key]) for key in ('source_checkpoint', 'source_checkpoint_id',
        'source_weight_stat', 'architecture', 'hf_source', 'input_data_stat', 'data', 'environment',
        'control_result_paths', 'control_mean_percent')}
    diagnostic = kind == 'validate_s4'
    record.update(schema='llama_fineweb_decoder_full_dense_head_v1', status='prepared_unbound', kind=kind,
        gpu=None, export_gpu=None, queued=False, automatic_retry=False, cwd=str(ROOT), python_executable=sys.executable,
        output_dir=str(output), source_bit_budget=budget, expected_inventory=inventory,
        source_sha256=sources, input_evidence_sha256=inputs, control_normalized=str(normalized[0]),
        control_dir=str(CONTROL), validation_proof=str(proof_path) if proof_path else None,
        source_control_note='Pin immutable prepared/normalized evidence; never the live decoder-control launch or rotating checkpoint.',
        selected_targets=list(control['architecture']['projection_shapes']),
        resource_budget_seconds=1200 if diagnostic else 64800,
        timeout_seconds=1200 if diagnostic else 64800, termination_grace_seconds=20,
        schedule={'max_steps': 2000, 'warmup_steps': 20, 'scheduler': 'cosine', 'save_steps': 400,
                  'actual_stop_step': 4 if diagnostic else 2000,
                  'forced_native_save_steps': [4] if diagnostic else [], 'full8_steps': [] if diagnostic else SAVE_STEPS},
        comparison={'primary': str(CONTROL), 'primary_mode': 'Same-step FineWeb decoder-only trajectory from the identical e767 source.',
                    'secondary': 'Existing source full8; full LongBench/nine-item outcome remains unmeasured.',
                    'scientific_delta': {'lm_head_train_mode': ['none', 'full'], 'lm_head_lr': [None, 3e-6],
                                         'distill_fp32_components': [['decoder'], ['decoder', 'lm_head']]}},
        protocol={'full8': not diagnostic, 'mmlu_batch_size': 64, 'seven_batch_size': 32, 'limit': None,
            'short': 'Four actual native updates; one native save at4, train process exits, then same-GPU independent export/audit. No full8/resume.',
            'formal': 'Fresh e767, native2000/save400 with synchronous independent export/audit/full8 every400; no validation-state continuation.',
            'precision': 'FP32 decoder/head parameters, gradients and Adam; BF16 native compute and final dense head. Frozen norms/codes/embedding.',
            'teacher': 'Original native Llama SDPA loader/no offload; no whole-model dtype override; preserve FP32 RoPE.',
            'initialization': deepcopy(control['initialization']), 'no_external_lora': True,
            'no_automatic_queue_retry_or_followup': True, 'longbench': 'not_evaluated', 'nine_item_goal': 'not_evaluated'},
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled', 'formal_requires_successful_short_proof': not diagnostic},
        experience={'reference': 'docs/experience/lessons/memory_precision.md',
            'previous_negative': 'Qwen552 task50 decoder+fullhead2000 was negative; its source/data/headLR differ from this Llama e767 FineWeb control.',
            'boundary': 'No quality gain presumed. This is native full-head support adapted to actual Llama/D16 inventories.'},
        estimate={'disk_free_at_prepare': free, 'head_fp32_parameter_bytes': inventory['head_elements'] * 4,
            'head_fp32_gradient_bytes': inventory['head_elements'] * 4, 'head_adam_two_moments_bytes': inventory['head_elements'] * 8,
            'additional_bytes_over_frozen_bf16_head': inventory['head_elements'] * 14,
            'possible_bf16_compute_cast_bytes': inventory['head_elements'] * 2,
            'training_peak_estimate_gib': [40, 45], 'estimate_not_measurement': True,
            'reference': 'Prior Llama FineWeb decoder peak30.76GiB plus head parameter/gradient/moments; Qwen old head short41.30GiB is different source.',
            'short_seconds_estimate': [400, 1000], 'formal_time_requires_actual_short_throughput': True})
    record['training_argv'] = training_arguments(record)
    record['argv'] = [sys.executable, '-u', '-m', 'experiments.general_lm_head_recovery.train', '--launch-record', str(output / 'launch.json')]
    if diagnostic: record['argv'].append('--short-training-child')
    record['launcher_argv'] = ([sys.executable, '-u', '-m', 'experiments.general_lm_head_recovery.run',
                               '--launch-record', str(output / 'launch.json')] if diagnostic else record['argv'])
    cfg, hf, args, names, architecture = configuration(record)
    record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'all_projection_count': len(names), 'model_type': architecture['model_type'],
        'only_scientific_difference': record['comparison']['scientific_delta'], 'native_data_config_matches_statistics': True}
    step = 4 if diagnostic else 400
    checkpoint = output / f'NATIVE_RUN/trainer_state/checkpoint-{step}'
    record['stage_templates'] = [{'name': name, 'argv': command, 'allowed_exit_codes': allowed}
        for name, command, allowed in stage_commands(checkpoint=checkpoint, source_dir=source,
            output_dir=output / f'step_{step:04d}', diagnostic=diagnostic)]
    check_inputs(record)
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['cuda_initialized'] = False; record['readiness']['cpu'] = 'passed'
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print({'launch': str(output / 'launch.json'), 'sha256': sha256(output / 'launch.json'),
           'sources': len(sources), 'inputs': len(inputs), 'kind': kind, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=('validate_s4', 'formal_s2000'), default='validate_s4')
    parser.add_argument('--validation-proof', type=Path)
    cli = parser.parse_args()
    prepare(cli.output_dir, kind=cli.kind, validation_proof=cli.validation_proof)
