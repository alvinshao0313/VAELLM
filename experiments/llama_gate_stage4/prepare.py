"""Prepare one existing native S4 gate arm using retained S2 metrics/metadata."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_small_rate_pair.common import ROOT, ORIGINAL, RESULT, BASE_ID, DENOMINATOR, check, read, _require
from experiments.llama_gate_stage4.audit import stage4_spec, TOTAL_BITS, CONTROL_ID
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import _rebuild_converted_modules, _collect_vae_linear_specs, load_v6_meta

CONTROL = RESULT / 'llama_gate_small_rate_pair/control'


def construct(source):
    with torch.random.fork_rng(devices=[]):
        holder = torch.nn.Module(); holder.add_module('gate_proj', torch.nn.Linear(4096, 14336, bias=False))
        expected = deepcopy(source); expected['name'] = 'gate_proj'
        _rebuild_converted_modules(holder, [expected]); actual = _collect_vae_linear_specs(holder)[0]
        _require(actual == expected, 'Actual original-size native schema reconstruction differs.')
        state = holder.state_dict(); storages = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in state.values()}
        return {'spec': actual, 'physical_bits': sum(storages.values()) * 8, 'state_tensors': len(state),
                'scope': 'CPU native geometry construction only; no reconstruction of deleted learned control weights.'}


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Require fresh CPU-only preparation.')
    template = read(CONTROL / 'launch.json'); check(template)
    _require(template['status'] == 'completed' and template['exit_code'] == 0 and template['checkpoint_id'] == CONTROL_ID
             and template['base_checkpoint_id'] == BASE_ID and template['codebook_bits'] == 32
             and all((CONTROL / (s + '.exit')).read_text().strip() == '0'
                     for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')), 'Require the completed original S2 control.')
    audit = read(CONTROL / 'audit.json'); summary = read(CONTROL / 'full8_summary.json')
    control, _ = score([CONTROL / (s + '.json') for s in ('mmlu', 'seven')], template['architecture'], CONTROL / 'final_model')
    _require(control['checkpoint_id'] == audit['checkpoint_id'] == CONTROL_ID and audit['status'] == 'verified'
             and control['mean_percent'] == template['mean_percent'] == 60.70877820721448
             and control['integer_counts'] == summary['integer_counts'] and summary['audit_sha256'] == sha256(CONTROL / 'audit.json')
             and template['full8_summary_sha256'] == sha256(CONTROL / 'full8_summary.json'), 'Preserved raw control evidence differs.')
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'), '--codebook_bits': 'default=32,cat:gate_proj=16',
                       '--residual_stages': 'default=2,cat:gate_proj=4'}.items():
        _require(argv.count(key) == 1, 'Ambiguous native argument: ' + key); argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv); resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'): expected[section]['output_dir'] = str(output / 'donor')
    for section in ('canonical_config', 'vae_args'):
        expected[section]['codebook_bits'] = {'default': 32, 'by_category': {'gate_proj': 16}}
        expected[section]['residual_stages'] = {'default': 2, 'by_category': {'gate_proj': 4}}
    expected['resolved_category_runtime']['gate_proj'].update(codebook_bits=16, residual_stages=4)
    _require(normalized == expected, 'Recipe changes beyond coupled K/S factorization and output.')
    cfg = resolved['gate_proj']
    _require(cat.seed == 31 and cat.batch_size == 8192 and cat.linear_group_size == 32 and cat.target_layers == 'all'
             and cat.after_category_mode == 'none' and not cat.resume_from_checkpoint and not cat.channel_refresh_after_category
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj' and cfg.steps == 10000
             and cfg.codebook_dim == 32 and cfg.channel_protect_count == 64 and cfg.base_ch == 128
             and cfg.num_res_blocks == 0 and training.bf16 and not training.fp16, 'Fixed scientific conditions changed.')
    meta = load_v6_meta(str(CONTROL / 'final_model')); budget = audit['bit_budget']
    gates = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    _require(len(gates) == 32 and all(s['stage_decoders'] == gates[0]['stage_decoders'] for s in gates), 'Wrong control geometry.')
    before, after = construct(gates[0]), construct(stage4_spec(gates[0]))
    extra = (after['physical_bits'] - before['physical_bits']) * 32
    components = deepcopy(budget['components'])
    for key in ('budget_bits', 'storage_bits'): components['decoder'][key] += extra
    components['code']['unique_storages'] += 64
    state_count = audit['source_union_save_fresh_tensors'] + 32 * (after['state_tensors'] - before['state_tensors'])
    _require(before['physical_bits'] == budget['per_target'][gates[0]['name']]['storage_bits']
             and before['state_tensors'] == 13 and after['state_tensors'] == 15 and state_count == 3043
             and extra == 4620288 and sum(v['budget_bits'] for v in components.values()) == TOTAL_BITS
             and sum(v['storage_bits'] for v in components.values()) == TOTAL_BITS, 'Actual native S4 geometry/budget differs.')
    r1 = ORIGINAL / 'result/qwen3_8b_w25_20260930'; parity_dir = r1 / 'validation/s4_decoder_regression'
    parity = read(parity_dir / 'launch.json')['gpu_stage']; packed_dir = r1 / 'validation/s4_packed_v6_cpu'
    _require(parity['status'] == 'passed' and parity['exit_code'] == parity['pytest_exit_code'] == 0
             and (parity_dir / 'gpu.exit').read_text().strip() == '0'
             and read(packed_dir / 'launch.json')['status'] == 'passed'
             and (packed_dir / 'run.exit').read_text().strip() == '0', 'Existing S4 native/group8 proofs are incomplete.')
    sources = dict(template['source_sha256']); inputs = dict(template['input_sha256'])
    critical = [k for k in parity['source_sha256'] if k.startswith('litebsq/') or k in ('train_utils/checkpoint_v6.py', 'tests/test_four_stage_decoder.py')]
    for relative in critical:
        p = ROOT / relative; _require(sha256(p) == parity['source_sha256'][relative], 'S4 kernel/schema proof source changed: ' + relative)
        sources[str(p)] = sha256(p)
    for p in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
        _require(sha256(ORIGINAL / p.relative_to(ROOT)) == sha256(p), 'Public/snapshot source differs.')
    native = Path(template['donor_checkpoint_dir'])
    for p in [*[CONTROL / f for f in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json')],
              CONTROL / 'final_model/checkpoint_meta.json', CONTROL / 'final_model/assembly_report.json',
              native.parent / 'normalized_cat_runtime_args.json', native / 'checkpoint_meta.json',
              CONTROL.parent / 'comparison.json', parity_dir / 'launch.json', parity_dir / 'gpu.exit',
              packed_dir / 'launch.json', packed_dir / 'run.exit', r1 / 'reconstruction_s4_d32_block64_o3_down3/eval/acceptance.json']:
        inputs[str(p)] = sha256(p)
    provision = template['resource_estimate']['disk_provision_bytes'] + 3 * extra // 8
    record = {'schema': 'llama_gate_s4_v1', 'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'model_type': 'llama',
        'categories': ['gate_proj'], 'codebook_bits': 16, 'residual_stages': 4, 'transpose': False, 'block_size': 32,
        'steps_per_stage': 10000, 'total_stage_updates': 40000, 'full_evaluation': True, 'base_dir': template['base_dir'],
        'base_checkpoint_id': BASE_ID, 'architecture': template['architecture'], 'historical_control_dir': str(CONTROL),
        'historical_control_checkpoint_id': CONTROL_ID, 'control_result_paths': template['control_result_paths'],
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': template['protected_stats'],
        'environment': template['environment'], 'protocol': template['protocol'], 'expected_budget_components': components,
        'expected_user_linear_bits': TOTAL_BITS, 'expected_state_tensors': state_count, 'expected_code_banks': 512,
        'budget_estimate': {'user_linear_bits': TOTAL_BITS, 'packed_linear_bits': TOTAL_BITS, 'bpw': TOTAL_BITS / DENOMINATOR,
            'code_padding_bits': 0, 'decoder_extra_bits': extra, 'decoder_extra_parameters': extra // 16},
        'native_cpu_construction': {'before': before, 'after': after}, 'timeout_seconds': 3600, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_stage4.run', '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False, 's4_gpu_proof': str(parity_dir / 'launch.json'), 's4_packed_cpu_proof': str(packed_dir / 'launch.json'),
            'critical_sources_unchanged': critical, 'deleted_control_weights_required': False,
            'scope': 'Existing native CAT/checkpoint/group8/full8 only. S2-only E2E subset exporters are not used or extended.'},
        'comparison_boundary': 'Primary fresh K16/D32/S4 minus completed fresh K32/D32/S2 with same e767192/HF target/False/input64/block32. Main code rate remains2bit, but each stage10000 means40000 versus20000 updates, more decoder capacity and changed initialization/RNG; not equal compute or total storage. Old Qwen S4 had only+.05039pp and MMLU decline, not a positive Llama prediction. e767 secondary; no LongBench or E2E compatibility claim.',
        'resource_estimate': {'estimated_seconds': [1900, 2600], 'seconds_limit': 3600,
            'historical_s2_seconds': template['finished_unix'] - template['started_unix'], 'single_arm_disk_provision_bytes': provision,
            'filesystem_free_at_preparation': shutil.disk_usage(output.parent.parent).free,
            'boundary': 'Training stage count doubles; category initialization/calibration/assembly/eval not doubled. Full Llama S4 duration not yet measured.'},
        'cuda_initialized': False}
    _require(not torch.cuda.is_initialized() and shutil.disk_usage(output.parent.parent).free > provision, 'CUDA initialized or insufficient disk.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print({'prepared_sha256': sha256(output / 'launch.prepared.json'), 'source': len(sources), 'input': len(inputs),
           'stat': len(record['protected_stats']), 'bits': TOTAL_BITS, 'state': state_count, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
