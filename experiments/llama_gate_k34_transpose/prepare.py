"""CPU preparation of native K34 gate transpose, against completed K34 False."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_small_rate_pair.common import ROOT, ORIGINAL, RESULT, BASE_ID, LOGICAL_BITS, PACKED_BITS, check, read, _require
from experiments.llama_gate_transpose.prepare import recipe
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import file_stat, sha256, write_json
from train_utils.checkpoint_v6 import _rebuild_converted_modules, _collect_vae_linear_specs, load_v6_meta

CONTROL = RESULT / 'llama_gate_small_rate_pair/candidate'
CONTROL_ID = '2b52921f-9f5a-402c-9f08-ec00af5dd219'


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Require one fresh CPU-only snapshot preparation.')
    template = read(CONTROL / 'launch.json'); check(template)
    _require(template['status'] == 'completed' and template['exit_code'] == 0 and template['codebook_bits'] == 34
             and template['checkpoint_id'] == CONTROL_ID and template['base_checkpoint_id'] == BASE_ID
             and all((CONTROL / (s + '.exit')).read_text().strip() == '0'
                     for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
             'Require the completed exact K34 False control.')
    audit = read(CONTROL / 'audit.json'); summary = read(CONTROL / 'full8_summary.json')
    control, _ = score([CONTROL / (s + '.json') for s in ('mmlu', 'seven')], template['architecture'], CONTROL / 'final_model')
    _require(control['checkpoint_id'] == audit['checkpoint_id'] == CONTROL_ID and audit['status'] == 'verified'
             and control['mean_percent'] == template['mean_percent'] == 61.64916821652581
             and control['integer_counts'] == summary['integer_counts']
             and summary['audit_sha256'] == sha256(CONTROL / 'audit.json')
             and template['full8_summary_sha256'] == sha256(CONTROL / 'full8_summary.json'), 'Completed raw evidence differs.')
    argv, normalized = recipe(template, output)  # Existing native parser; only output/transpose differ.
    cfg = normalized['resolved_category_runtime']['gate_proj']; budget = audit['bit_budget']
    _require(cfg['codebook_bits'] == 34 and cfg['codebook_dim'] == 32 and cfg['residual_stages'] == 2
             and cfg['base_ch'] == 128 and cfg['num_res_blocks'] == 0 and cfg['channel_protect_count'] == 64
             and budget['user_linear_bits'] == LOGICAL_BITS and budget['packed_linear_bits'] == PACKED_BITS
             and budget['code_padding_bits'] == 693633024 and budget['budget_passed'], 'Wrong K34 recipe or actual budget.')
    meta = load_v6_meta(str(CONTROL / 'final_model')); gates = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    _require(len(gates) == 32 and all(not s['transpose'] and s['compressed_in_features'] == 4032
             and s['compressed_out_features'] == 14336 and s['stage_codebook_dims'] == [32, 32]
             and s['protected_input_indices']['shape'] == [64] and s['weight_rotation']['block_size'] == 32
             for s in gates) and 4032 % 32 == 14336 % 32 == 0, 'Both native orientations must be aligned.')
    # A real full-size native module reconstruction checks the combined schema on CPU;
    # no learned weights or quality claims are supplied by this geometry-only check.
    with torch.random.fork_rng(devices=[]):
        holder = torch.nn.Module(); holder.add_module('gate_proj', torch.nn.Linear(4096, 14336, bias=False))
        expected = deepcopy(gates[0]); expected.update(name='gate_proj', transpose=True)
        _rebuild_converted_modules(holder, [expected])
        constructed = _collect_vae_linear_specs(holder)[0]
        state = holder.state_dict(); storages = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in state.values()}
        physical = sum(storages.values()) * 8
        _require(constructed == expected and physical == budget['per_target'][gates[0]['name']]['storage_bits'],
                 'Actual native transposed constructor differs from control geometry/storage.')
        construction = {'spec': constructed, 'physical_bits_per_gate': physical, 'serialized_tensors_per_gate': len(state),
                        'scope': 'CPU native geometry only; no GPU decode or substitute learned-state verification.'}
        del holder, state
    old_dir = RESULT / 'llama_gate_transpose/transpose'; old = read(old_dir / 'independent_review.json')
    short = RESULT / 'llama_gate_small_rate_pair/validate_k34_s2'; proof = read(short / 'verification.json')
    _require(old['status'] == 'independent_cpu_verified' and old['same_native_recipe_except_output_and_gate_transpose']
             and proof['status'] == 'verified_native_short' and proof['zero_padding_and_pack_replay_banks'] == 64,
             'Require existing transpose and K34 native path proofs.')
    sources = dict(template['source_sha256']); inputs = dict(template['input_sha256'])
    for p in [*sorted(Path(__file__).parent.glob('*.py')), ROOT / 'experiments/llama_gate_transpose/prepare.py']:
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
        _require(sha256(ORIGINAL / p.relative_to(ROOT)) == sha256(p), 'Public/snapshot source differs.')
    native = Path(template['donor_checkpoint_dir'])
    for p in [*[CONTROL / f for f in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json')],
              CONTROL / 'final_model/checkpoint_meta.json', CONTROL / 'final_model/assembly_report.json',
              native.parent / 'normalized_cat_runtime_args.json', native / 'checkpoint_meta.json',
              CONTROL.parent / 'comparison.json', CONTROL.parent / 'comparison.exit',
              old_dir / 'independent_review.json', old_dir / 'final_model/checkpoint_meta.json', short / 'verification.json']:
        inputs[str(p)] = sha256(p)
    base = Path(template['base_dir']); stats = [*template['protected_stats'], file_stat(CONTROL / 'final_model' / meta['state_dict_file'])]
    provision = template['resource_estimate']['disk_provision_bytes']
    record = {'schema': 'llama_gate_k34_transpose_v1', 'status': 'prepared_unbound', 'gpu': None,
        'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(output), 'model_type': 'llama', 'categories': ['gate_proj'], 'codebook_bits': 34,
        'transpose': True, 'block_size': 32, 'steps_per_stage': 10000, 'full_evaluation': True,
        'base_dir': str(base), 'base_checkpoint_id': BASE_ID, 'architecture': template['architecture'],
        'historical_control_dir': str(CONTROL), 'historical_control_checkpoint_id': CONTROL_ID,
        'control_result_paths': template['control_result_paths'], 'training_argv': [sys.executable, 'tools/cat_train.py', *argv],
        'normalized_expected': normalized, 'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': stats,
        'environment': template['environment'], 'protocol': template['protocol'], 'expected_budget_components': budget['components'],
        'expected_user_linear_bits': LOGICAL_BITS, 'expected_packed_linear_bits': PACKED_BITS, 'expected_padding_bits': 693633024,
        'native_cpu_construction': construction, 'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_k34_transpose.run', '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False, 'k34_proof': str(short / 'verification.json'),
            'transpose_proof': str(old_dir / 'independent_review.json'), 'short_weights_consumed': False,
            'scope': 'K34 padding/packed/group8 kernels and native True restoration already measured; actual combined final specs/union/payloads/full8 remain checked.'},
        'comparison_boundary': 'Primary fresh K34 True minus completed K34 False, both with e767 other192/backbone. Only gate transpose/output change; transposition changes VQ vectors after the same protection/RHT. Same seed does not imply equal errors/optimization. e767 secondary; old3577 True result is direction evidence, not this control. No LongBench or additive-gain inference.',
        'resource_estimate': {'estimated_seconds': [1200, 1500], 'seconds_limit': 2700,
            'historical_k34_false_seconds': template['finished_unix'] - template['started_unix'],
            'single_arm_disk_provision_bytes': provision, 'filesystem_free_at_preparation': shutil.disk_usage(output.parent.parent).free},
        'cuda_initialized': False}
    _require(not torch.cuda.is_initialized() and shutil.disk_usage(output.parent.parent).free > provision, 'CPU initialized CUDA or insufficient disk.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print({'prepared_sha256': sha256(output / 'launch.prepared.json'), 'source': len(sources), 'input': len(inputs), 'stat': len(stats),
           'logical_bpw': LOGICAL_BITS / 6979321856, 'packed_bpw': PACKED_BITS / 6979321856, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
