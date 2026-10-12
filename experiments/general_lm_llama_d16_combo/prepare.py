"""Bind a fresh FineWeb2000 decoder segment to the verified Llama D16 combination."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.general_lm_recovery.prepare import DATA, R1, R2, data_contract
from experiments.general_lm_recovery.train import ROOT, check_inputs, configuration, read, training_arguments
from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable

COMBO = R2 / 'llama_fineweb800_down_d16_combo'
CHECKPOINT_ID = 'e985bada-3960-4a5a-82af-4652ef94ca6c'
ORIGINAL = R2 / 'general_lm_recovery_s2000/llama'
D16_PROOF = R1 / 'validation/d16_fp32_packed_parity/run.json'


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU preparation requires CUDA hidden and uninitialized.')
    require(not output.exists(), 'Use a fresh output directory.')
    usage = shutil.disk_usage(output.parent)
    require(usage.free >= 30_000_000_000, 'Reserve 30GB for this independent formal branch.')
    manifest, stats, train_file = data_contract()
    template = read(ORIGINAL / 'launch.json')
    check_inputs(template)
    configuration(template)
    review, completed = read(COMBO / 'review.json'), read(COMBO / 'launch.json')
    summary = read(COMBO / 'full8_summary.json')
    source = COMBO / 'final_model'
    meta, assembly = read(source / 'checkpoint_meta.json'), read(source / 'assembly_report.json')
    require(review['status'] == assembly['status'] == 'verified' and completed['status'] == 'completed'
            and review['checkpoint_id'] == completed['checkpoint_id'] == summary['checkpoint_id']
            == meta['checkpoint_id'] == assembly['checkpoint_id'] == CHECKPOINT_ID
            and all(value == 0 for value in review['exit_status'].values()), 'The complete combination proof differs.')
    check_hashes(review['evidence_sha256'])
    union = review['source_union']
    require(union['replaced_down'] == 32 and union['preserved_attention_gate_up'] == 192
            and union['all224_metadata_specs_exact_union'] and union['in_memory_serialized_fresh_state_exact_reused']
            and union['serialized_tensors_compared'] == union['fresh_reloaded_tensors_compared'] == 2979,
            'The source must retain the verified mixed-geometry source union.')
    budget = summary['bit_budget']
    require(budget == assembly['bit_budget'] and budget['user_linear_bits'] == 17095180288
            and budget['original_linear_weights'] == 6979321856 and budget['budget_passed']
            and budget['user_linear_bpw'] <= 2.5 and budget['code_padding_bits'] == 0
            and budget['components'] == review['actual_storage']['components'], 'Actual source storage differs.')
    specs = meta['converted_modules']
    downs = [item for item in specs if item['name'].endswith('.down_proj')]
    require(len(specs) == 224 and len(downs) == 32 and all(
        item['codebook_dim'] == 16 and item['stage_codebook_dims'] == [16, 16]
        and item['residual_stages'] == 2 and item['parallel_parts'] == 1 and item['transpose']
        and item['compressed_in_features'] == 14080 and item['compressed_out_features'] == 4096
        and item['protected_input_indices'] == {'shape': [256], 'dtype': 'int64'}
        and item['protected_channel_quant_format'] == 'int8' and item['weight_rotation']['block_size'] == 32
        and all(stage['in_dim'] == 24 and stage['out_dim'] == 16 and stage['hidden_dim'] == 128
                and stage['num_res_blocks'] == 0 for stage in item['stage_decoders'])
        and all(bank['logical_shape'] == [3604480, 1, 24] and bank['shape'] == [3604480, 1, 3]
                and bank['dtype'] == 'uint8' for bank in item['stage_vq_weights']) for item in downs)
        and all(item['stage_codebook_dims'] == [32, 32] for item in specs if item not in downs),
        'Expected32 D16/K24 down projections and192 original D32 projections.')
    proof = read(D16_PROOF)
    require(proof['status'] == 'passed' and proof['exit_code'] == 0
            and proof['case'] == {'K': 24, 'D': 16, 'H': 128, 'stages': 2,
                'parameter_dtype': 'float32', 'compute_dtype': 'bfloat16', 'group_size': 8},
            'Reuse the actual FP32-master/BF16-compute D16 native forward/backward proof.')
    check_hashes({str(ROOT / key): value for key, value in proof['source_sha256'].items()})
    source_pins = deepcopy(template['source_sha256'])
    source_pins[str(Path(__file__).resolve())] = sha256(__file__)
    ast.parse(Path(__file__).read_text())
    inputs = deepcopy(template['input_evidence_sha256'])
    inputs.update(review['evidence_sha256'])
    for path in (COMBO / 'review.json', D16_PROOF):
        inputs[str(path)] = sha256(path)
    # These source-lineage audits are small immutable records, not native resume payloads.
    for digest in (union['base_audit_sha256'], union['donor_audit_sha256']):
        matches = [path for path, value in completed['input_sha256'].items() if value == digest]
        require(len(matches) == 1, 'Missing unique frozen-backbone source audit.')
        inputs[matches[0]] = digest
    historical = {}
    for step in (400, 800, 1200):
        path = ORIGINAL / f'step_{step:04d}/independent_review.json'
        item = read(path)
        require(item['status'] == 'verified', 'Historical original-source comparison is not verified.')
        inputs[str(path)] = sha256(path)
        historical[str(step)] = {'review': str(path), 'comparison': 'Different starting state and trajectory; reference only.'}
    keys = ('architecture', 'hf_source', 'input_data_stat', 'data', 'environment', 'schedule', 'protocol')
    record = {key: deepcopy(template[key]) for key in keys}
    architecture = record['architecture']
    inventory = expected_inventory(source, architecture, tuple(architecture['projection_shapes']))
    require(inventory['deployed_projections'] == inventory['selected_projections'] == 224
            and inventory['serialized_tensors'] == 2979 and inventory['selected_decoder_tensors'] == 1344
            and inventory['selected_decoder_elements'] == 3789824 and inventory['all_code_banks'] == 448
            and inventory['frozen_tensors'] == 1635 and inventory['active_score_elements'] == 0,
            'The actual CPU-mapped decoder/frozen inventory differs.')
    paths = [COMBO / (name + '.json') for name in ('mmlu', 'seven')]
    control = collect_results(paths, architecture, checkpoint=(source, meta))
    require(control['mean_percent'] == summary['mean_percent'] == review['mean_percent']
            and control['leaf_count'] == 64 and control['sample_count'] == 24742, 'Complete primary control differs.')
    record.update(schema='general_lm_llama_d16_combo_fresh2000_v1', status='prepared_unbound', kind='formal_s2000',
        gpu=None, export_gpu=None, queued=False, automatic_retry=False, cwd=str(ROOT), python_executable=sys.executable,
        output_dir=str(output), source_checkpoint=str(source), source_checkpoint_id=CHECKPOINT_ID,
        source_weight_stat=file_stat(source / 'pytorch_model.bin'), source_bit_budget=budget,
        expected_inventory=inventory, source_sha256=source_pins, input_evidence_sha256=inputs,
        control_result_paths=list(map(str, paths)), control_mean_percent=control['mean_percent'],
        training_argv=training_arguments(source, architecture['base_model_path'], output, train_file),
        argv=[sys.executable, '-u', '-m', 'experiments.general_lm_recovery.train', '--launch_record', str(output / 'launch.json')],
        resource_budget_seconds=64800, termination_grace_seconds=20,
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled'},
        initialization={'source_union': union, 'new_optimizer': True, 'new_scheduler': True,
            'local_step_range': [0, 2000], 'exact_resume': False, 'data_replay': 'Seed0/epoch0 first8000 native sampler documents are replayed.192 projections previously saw the original first3200 documents;32 D16 down projections did not.',
            'comparison_boundary': 'A new mixed-geometry starting state, not800 exact-resume and not a same-trajectory single-variable comparison to fresh3577.'},
        historical_original_source_references=historical,
        validation_reuse={'existing_runtime': template['validation_reuse'], 'd16_fp32_native_kernel': str(D16_PROOF),
            'mixed_geometry_save_reload_full8': str(COMBO / 'review.json'),
            'scope': 'Reuse actual D16/K24 FP32-master BF16 forward/gradient and strict mixed-state proofs; no repeated GPU short validation.'},
        estimate={'storage_bytes_per_branch': 30_000_000_000, 'disk_free_at_prepare': usage.free,
            'training_hours_estimate': [10, 15], 'throughput_boundary': 'Inherited estimate; this mixed-geometry segment has not run.'})
    cfg, hf, args, selected, actual_architecture = configuration(record)
    canonical = to_runtime_jsonable(cfg)
    require(canonical['data'] == stats['data_config'] and record['data']['jsonl_sha256'] == manifest['jsonl_sha256'],
            'Native data encoding/sampler contract changed.')
    record['cpu_configuration'] = {'canonical_config': canonical, 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'all_projection_count': len(selected),
        'model_type': actual_architecture['model_type'], 'native_data_config_matches_statistics': True}
    checkpoint = output / 'NATIVE_RUN/trainer_state/checkpoint-400'
    record['stage_templates'] = [{'name': name, 'argv': argv, 'allowed_exit_codes': allowed}
        for name, argv, allowed in stage_commands(checkpoint=checkpoint, source_dir=source, output_dir=output / 'step_0400')]
    check_inputs(record)
    require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA.')
    record.update(cuda_initialized=False)
    record['readiness']['cpu'] = 'passed'
    output.mkdir()
    write_json(output / 'launch.json', record)
    prepared = {'status': 'cpu_prepared_unbound', 'cuda_initialized': False, 'source_checkpoint_id': CHECKPOINT_ID,
        'launch_sha256': sha256(output / 'launch.json'), 'source_pin_count': len(source_pins),
        'input_pin_count': len(inputs), 'source_bpw': budget['user_linear_bpw'], 'expected_inventory': inventory,
        'primary_control_mean_percent': control['mean_percent'], 'disk_free_bytes': usage.free,
        'scope': 'Fresh2000 with five complete400-step full8 callbacks; no GPU binding, queue or execution.'}
    write_json(output / 'preparation.json', prepared)
    print(prepared, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=R2 / 'general_lm_llama_d16_combo_s2000')
    prepare(parser.parse_args().output)
