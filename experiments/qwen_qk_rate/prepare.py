"""Prepare one native Qwen K38/full-RHT arm against completed K32/full-RHT."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.full_rht_gate_pair.common import ROOT, ORIGINAL, RESULT, EXPECTED_IDS, check, read
from experiments.full_rht_gate_pair.compare import score
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256, write_json, check_hashes
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen_qk_rate.audit import CONTROL_ID, LOGICAL_BITS, PACKED_BITS, DENOMINATOR, k38_spec
from litebsq.autoencoder import Decoder
from litebsq.bitpack import validate_bitpack_u8_spec
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state

CONTROL = RESULT / 'qwen_qk_rotation_pair/full'


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Require fresh CPU-only S preparation.')
    template = read(CONTROL / 'launch.json'); check(template)
    _require(template['status'] == 'completed' and template['exit_code'] == 0 and template['checkpoint_id'] == CONTROL_ID
             and template['base_checkpoint_id'] == EXPECTED_IDS['qwen3'] and template['arm'] == 'full'
             and all((CONTROL / (s + '.exit')).read_text().strip() == '0' for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
             'Require the complete matched K32/full-RHT control.')
    native = Path(template['donor_checkpoint_dir']); control, _ = score([CONTROL / (n + '.json') for n in ('mmlu', 'seven')], template['architecture'], CONTROL / 'final_model')
    audit = read(CONTROL / 'audit.json'); summary = read(CONTROL / 'full8_summary.json')
    _require(control['checkpoint_id'] == audit['checkpoint_id'] == CONTROL_ID and audit['status'] == 'verified'
             and control['mean_percent'] == summary['mean_percent'] == template['mean_percent']
             and sha256(CONTROL / 'full8_summary.json') == template['full8_summary_sha256'], 'Control raw identity differs.')
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'), '--codebook_bits': 'default=32,cat:q_proj=38,cat:k_proj=38'}.items():
        _require(argv.count(key) == 1, 'Ambiguous native argument: ' + key); argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv); resolved = resolve_category_runtime_configs(cat, vae, ['q_proj', 'k_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'): expected[section]['output_dir'] = str(output / 'donor')
    for section in ('canonical_config', 'vae_args'): expected[section]['codebook_bits'] = {'default': 32, 'by_category': {'q_proj': 38, 'k_proj': 38}}
    for category in ('q_proj', 'k_proj'): expected['resolved_category_runtime'][category]['codebook_bits'] = 38
    _require(normalized == expected, 'Native recipe changed beyond K and output.')
    _require(cat.compression_categories == 'q_proj,k_proj' and cat.seed == 31 and cat.batch_size == 8192 and cat.linear_group_size == 36
             and cat.after_category_mode == 'none' and not cat.resume_from_checkpoint and not cat.channel_refresh_after_category
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj' and training.bf16 and not training.fp16
             and all(c.steps == 10000 and c.codebook_dim == 32 and c.residual_stages == 2 and c.base_ch == 128
                     and c.num_res_blocks == 0 and c.channel_protect_count == 64 for c in resolved.values()), 'Fixed scientific conditions changed.')
    specs = [s for s in load_v6_meta(str(CONTROL / 'final_model'))['converted_modules'] if s['name'].endswith(('.q_proj', '.k_proj'))]
    _require(len(specs) == 72, 'Wrong source inventory.')
    code_delta = storage_delta = 0
    for spec in specs:
        target = k38_spec(spec)
        for old, new in zip(spec['stage_vq_weights'], target['stage_vq_weights'], strict=True):
            validate_bitpack_u8_spec(new); vectors = old['logical_shape'][0] * old['logical_shape'][1]
            code_delta += vectors * 6; storage_delta += vectors * 8
    params = {}
    with torch.random.fork_rng(devices=[]):
        for k in (32, 38):
            decoder = Decoder(in_dim=k, out_dim=32, hidden_dim=128, num_res_blocks=0, norm_type='layer', activation_type='swish',
                              decoder_type='symmetric', use_checkpoint=True, num_models=2).to(dtype=torch.bfloat16)
            decoder._fuse_q_scale(); params[k] = sum(p.numel() for p in decoder.parameters())
    decoder_delta = (params[38] - params[32]) * 72 * 16
    components = deepcopy(audit['bit_budget']['components']); components['code']['budget_bits'] += code_delta; components['code']['storage_bits'] += storage_delta
    for field in ('budget_bits', 'storage_bits'): components['decoder'][field] += decoder_delta
    _require(code_delta == 278691840 and storage_delta == 371589120 and decoder_delta == 1769472
             and sum(v['budget_bits'] for v in components.values()) == LOGICAL_BITS
             and sum(v['storage_bits'] for v in components.values()) == PACKED_BITS and LOGICAL_BITS / DENOMINATOR <= 2.5,
             'Native decoder/count-based budget differs.')
    runtime = load_v6_cat_runtime_state(str(native), required=True); act = runtime['activation_runtime']; cache = act['cache']
    _require(runtime['global_adaptive_plan'] is None and act['dataset'] == cache['dataset'] == 'alpaca=1'
             and act['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512 and act['seqlen'] == cache['seqlen'] == 256
             and act['seed'] == cache['seed'] == 31 and {s['name'] for s in specs} <= set(act['stats_by_linear']), 'Wrong original calibration evidence.')
    calibration = {'source_runtime_sha256': sha256(native / 'cat_runtime_state.pt'), 'cache': tensor_tree(cache), 'stats': tensor_tree(act['stats_by_linear'])}
    packed_proof = RESULT / 'llama_gate_small_rate_pair/validate_k34_s2/verification.json'; proof = read(packed_proof)
    _require(proof['status'] == 'verified_native_short' and proof['zero_padding_and_pack_replay_banks'] == 64
             and proof['native_group8_prewarm']['warmed'] == 224 and proof['native_group8_prewarm']['failed'] == 0, 'Require actual non-byte-aligned native proof.')
    k34_record = read(packed_proof.parent / 'launch.json')
    for relative in ('litebsq/fused_multistage_decoder.py', 'litebsq/bitpack.py', 'litebsq/packed_bit_linear.py', 'train_utils/checkpoint_v6.py'):
        path = ROOT / relative; _require(k34_record['source_sha256'][str(path)] == sha256(path), 'Changed K34 native proof dependency.')
    # Native save_model keeps after_q, after_k, final donor and complete assembly simultaneously.
    history_dirs = [native.parent / 'after_q_proj', native.parent / 'after_k_proj', native, CONTROL / 'final_model']
    history_bytes = [sum(p.stat().st_size for p in d.rglob('*') if p.is_file()) for d in history_dirs]
    _require(all(any(p.name == 'pytorch_model.bin' for p in d.iterdir()) for d in history_dirs), 'Missing actual historical disk basis.')
    q_vectors = sum(b['logical_shape'][0] * b['logical_shape'][1] for s in specs if s['name'].endswith('.q_proj') for b in s['stage_vq_weights'])
    q_increment = (q_vectors * 8 + decoder_delta // 2) // 8; full_increment = (storage_delta + decoder_delta) // 8
    projected_bytes = sum(history_bytes) + q_increment + 3 * full_increment
    provision = 55_000_000_000
    _require(projected_bytes + 2 * 1024**3 < provision, '55GB does not cover actual four-save peak plus2GiB margin.')
    sources = dict(template['source_sha256'])
    for p in [*(Path(__file__).parent.glob('*.py')), ROOT / 'experiments/llama_gate_up_rate_pair/audit.py',
              ROOT / 'experiments/llama_gate_up_rate_pair/common.py', ROOT / 'experiments/llama_gate_small_rate_pair/formal_run.py',
              ROOT / 'experiments/llama_gate_small_rate_pair/common.py']:
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
        _require(sha256(ORIGINAL / p.relative_to(ROOT)) == sha256(p), 'Public/S source differs.')
    inputs = dict(template['input_sha256'])
    for p in [*[CONTROL / n for n in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json', 'train.exit', 'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')],
              CONTROL / 'final_model/checkpoint_meta.json', CONTROL / 'final_model/assembly_report.json', CONTROL.parent / 'comparison.json',
              native.parent / 'normalized_cat_runtime_args.json', native / 'checkpoint_meta.json', native / 'cat_runtime_state.pt', packed_proof]:
        inputs[str(p)] = sha256(p)
    record = {'schema': 'qwen_qk_rate_v1', 'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'model_type': 'qwen3', 'categories': ['q_proj', 'k_proj'],
        'codebook_bits': {'q_proj': 38, 'k_proj': 38}, 'block_size': 0, 'steps_per_stage': 10000, 'total_stage_updates': 40000, 'full_evaluation': True,
        'base_dir': template['base_dir'], 'base_checkpoint_id': EXPECTED_IDS['qwen3'], 'architecture': template['architecture'],
        'control_dir': str(CONTROL), 'control_checkpoint_id': CONTROL_ID, 'control_result_paths': template['control_result_paths'],
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized, 'control_calibration': calibration,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': template['protected_stats'], 'environment': template['environment'],
        'protocol': {**template['protocol'], 'longbench': 'not_evaluated', 'nine_item_goal': 'not_evaluated'}, 'category_rng': template['category_rng'],
        'expected_budget_components': components, 'budget_estimate': {'user_linear_bits': LOGICAL_BITS, 'packed_linear_bits': PACKED_BITS,
            'user_linear_bpw': LOGICAL_BITS / DENOMINATOR, 'packed_linear_bpw': PACKED_BITS / DENOMINATOR, 'code_padding_bits': PACKED_BITS - LOGICAL_BITS,
            'code_increment_bits': code_delta, 'decoder_increment_bits': decoder_delta, 'actual_native_two_stage_decoder_parameters': params},
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.qwen_qk_rate.run', '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'k34_packed_group8': str(packed_proof), 'full_rht_72qk': str(CONTROL / 'audit.json'), 'new_gpu_short': False,
            'scope': 'K38 uses the existing K34/K40 masked64 input tile and five-byte storage. No new kernel, packing, H252 rotation or export algorithm.'},
        'resource_estimate': {'historical_total_seconds': template['finished_unix'] - template['started_unix'],
            'historical_stage_seconds': {k: v['finished_unix'] - v['started_unix'] for k, v in template['stages'].items()},
            'estimated_seconds': [1800, 2300], 'seconds_limit': 2700, 'single_arm_disk_provision_bytes': provision,
            'historical_four_save_directories': [str(d) for d in history_dirs], 'historical_four_save_bytes': history_bytes,
            'predicted_peak_saved_bytes': projected_bytes, 'q_only_increment_bytes': q_increment, 'qk_increment_bytes': full_increment,
            'filesystem_free_at_preparation': shutil.disk_usage(output.parent.parent).free,
            'boundary': 'Actual old QK four coexisting checkpoint trees, not single-category33GB. Original1793.75s; K38 cost estimate is not a guarantee. No optimizer/round-base because after_category none.'},
        'comparison_boundary': 'Primary K38 minus completed43b0 K32: same fresh HF QK/fullRHT/seed31/10K perstage and f927180 complement. K changes encoder/BSQ/decoder shapes and RNG consumption, not appending bits to restored f927. f927 secondary includes withdrawal of its QK recovery history. No standalone eight-task/MMLU floor; LongBench/full nine-item score unknown.', 'cuda_initialized': False}
    _require(not torch.cuda.is_initialized() and shutil.disk_usage(output.parent.parent).free > provision, 'CUDA initialized or insufficient disk.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print({'prepared_sha256': sha256(output / 'launch.prepared.json'), 'sources': len(sources), 'inputs': len(inputs), 'stats': len(record['protected_stats']),
           'budget': record['budget_estimate'], 'disk_provision': provision, 'predicted_peak': projected_bytes, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
