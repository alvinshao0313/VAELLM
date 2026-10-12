"""CPU preparation of fresh Llama D16/K24 down with symmetric H256/R0."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.llama_down_dimension.compare import arm as historical_arm
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import _build_decoder_from_spec, load_v6_meta


CONTROL = RESULT / 'llama_down_dimension/block32'
EXPECTED_BITS = 17100816384
PRIOR = ORIGINAL / 'result/qwen3_8b_w25_20260930'
H256_PROOF = PRIOR / 'validation/h256_d16/launch.json'
H256_RUN = PRIOR / 'reconstruction_d16_odown_allblock64_h256'


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'), '--base_ch': 'default=256'}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['down_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    for section in ('canonical_config', 'vae_args'):
        expected[section]['base_ch']['default'] = 256
    expected['resolved_category_runtime']['down_proj']['base_ch'] = 256
    _require(normalized == expected,
             'Actual parser changed settings beyond output and symmetric hidden width.')
    cfg = resolved['down_proj']
    _require(cat.target_layers == 'all' and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and training.bf16 and not training.fp16 and cfg.steps == 10000
             and cfg.codebook_bits == 24 and cfg.codebook_dim == 16 and cfg.base_ch == 256
             and cfg.num_res_blocks == 0 and cfg.decoder_base_ch is None and cfg.decoder_type == 'symmetric',
             'The inherited complete down-category recipe changed.')
    return argv, normalized


def estimate(meta, budget):
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.down_proj')]
    _require(meta['checkpoint_id'] == '8a4633a7-93bd-429a-8a0e-a519a2ae30ae' and len(specs) == 32,
             'Require metadata of the completed H128/D16/block32 Llama control.')
    first = specs[0]['stage_decoders'][0]
    for spec in specs:
        _require(spec['in_features'] == 14336 and spec['out_features'] == 4096
                 and spec['compressed_in_features'] == 14080 and spec['compressed_out_features'] == 4096
                 and spec['transpose'] and spec['residual_stages'] == 2 and spec['parallel_parts'] == 1
                 and spec['stage_codebook_dims'] == [16, 16]
                 and spec['weight_rotation']['block_size'] == 32
                 and spec['protected_input_indices'] == {'shape': [256], 'dtype': 'int64'}
                 and len(spec['stage_decoders']) == len(spec['stage_vq_weights']) == 2,
                 'Historical down geometry differs.')
        _require(all(d == first for d in spec['stage_decoders'])
                 and all(p['logical_shape'] == [3604480, 1, 24] and p['shape'] == [3604480, 1, 3]
                         for p in spec['stage_vq_weights']), 'Historical stage inventory differs.')
    _require(first['in_dim'] == 24 and first['out_dim'] == 16 and first['hidden_dim'] == 128
             and first['num_res_blocks'] == 0 and first['decoder_type'] == 'symmetric'
             and first['param_dtype'] == 'bfloat16', 'Unexpected original decoder specification.')
    old_count = sum(p.numel() for p in _build_decoder_from_spec(first).parameters())
    new_count = sum(p.numel() for p in _build_decoder_from_spec({**first, 'hidden_dim': 256}).parameters())
    added = (new_count - old_count) * 32 * 2 * 16
    total = budget['user_linear_bits'] + added
    _require(old_count == 5520 and new_count == 11024 and added == 5636096
             and budget['user_linear_bits'] == 17095180288 and budget['original_linear_weights'] == 6979321856
             and total == EXPECTED_BITS and total * 2 < 6979321856 * 5, 'Actual decoder budget differs.')
    return {'estimated_total_bits': total, 'original_linear_weights': 6979321856,
        'estimated_bpw': total / 6979321856, 'old_decoder_parameters_per_stage': old_count,
        'new_decoder_parameters_per_stage': new_count, 'decoder_delta_bits': added,
        'estimated_decoder_bits': budget['components']['decoder']['storage_bits'] + added,
        'other_storage_components_unchanged': True,
        'boundary': 'Actual native decoder constructors and retained control metadata; final assembled tensor storage must still be audited.'}


def prior_evidence():
    proof = read(H256_PROOF); cfg = proof['configuration']
    _require(proof['status'] == 'passed' and proof['exit_code'] == proof['pytest_exit_code'] == 0
             and int((H256_PROOF.parent / 'run.exit').read_text()) == 0
             and len(proof['checks']) == 2 and all(c['outcome'] == 'passed' for c in proof['checks'])
             and (cfg['codebook_bits'], cfg['codebook_dim'], cfg['hidden_dim'], cfg['residual_stages'],
                  cfg['num_res_blocks'], cfg['vae_batch_size'], cfg['parallel_models']) == (24, 16, 256, 2, 0, 8192, 36)
             and cfg['storage_dtype'] == cfg['autocast_dtype'] == 'bfloat16',
             'Existing H256/D16 training, packed decode and v6 proof is not applicable.')
    sources = {}
    for relative, digest in proof['source_sha256'].items():
        if relative == 'rotation/hadamard_utils.py':
            continue  # The snapshot's additive factors are covered by its separately pinned rotation proof.
        path = ROOT / relative
        _require(sha256(path) == digest, 'Existing H256 proof core source changed: ' + relative)
        sources[str(path)] = digest
    prior = read(H256_RUN / 'launch.json'); outcome = read(H256_RUN / 'eval/acceptance.json')
    _require(prior['status'] == 'evaluated' and prior['checkpoint_id'] == outcome['checkpoint_id']
             == '970cc279-a735-46bd-b279-b2cf98cd6ee8'
             and all(int((H256_RUN / name).read_text()) == 0 for name in ('train.exit', 'run.exit')),
             'The prior complete Qwen H256 evidence changed.')
    model = Path(outcome['checkpoint_dir'])
    evidence = [H256_PROOF, H256_PROOF.parent / 'run.exit', H256_RUN / 'launch.json',
        H256_RUN / 'train.exit', H256_RUN / 'run.exit', H256_RUN / 'eval/acceptance.json',
        model / 'checkpoint_meta.json', model.parent / 'normalized_cat_runtime_args.json']
    return sources, {str(p): sha256(p) for p in evidence}, {
        'new_gpu_short': False, 'proof_path': str(H256_PROOF),
        'unchanged_h256_core_sources': len(sources),
        'rotation_source_boundary': 'Only hadamard_utils differs from the old proof; the authorized snapshot factor extension and unchanged legal matrices are covered by inherited snapshot CPU/native rotation evidence. This candidate retains block32.',
        'prior_qwen_mean_percent': outcome['mean_percent'],
        'prior_qwen_boundary': 'Qwen all-seven-category H256/R0 BF16 with O/down D16, block64 scored below its H128 reference. Different model/category scope/block and RNG trajectory; this is not evidence of a benefit for Llama down-only H256.'}


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template, audit, control, _, evidence = historical_arm(CONTROL, 32)
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(template['model_type'] == 'llama' and template['dimension'] == 16
             and meta['checkpoint_id'] == template['base_checkpoint_id'] == EXPECTED_IDS['llama'],
             'Fresh assembly base must be original 3577, not the widened historical decoder.')
    budget = estimate(load_v6_meta(str(CONTROL / 'final_model')), audit['bit_budget'])
    argv, normalized = recipe(template, output)
    proof_sources, proof_evidence, reuse = prior_evidence()
    sources = {**template['source_sha256'], **proof_sources}
    for path in [ROOT / 'experiments/llama_down_dimension/compare.py',
                 *(ROOT / 'experiments/llama_down_width').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = {**template['input_sha256'], **evidence, **proof_evidence}
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'llama', 'categories': ['down_proj'], 'codebook_bits': {'down_proj': 24},
        'dimension': 16, 'hidden_dim': 256, 'block_size': 32, 'architecture': template['architecture'],
        'base_dir': str(base), 'base_checkpoint_id': meta['checkpoint_id'],
        'control_result_paths': template['control_result_paths'], 'historical_control_dir': str(CONTROL),
        'historical_control_checkpoint_id': control['checkpoint_id'],
        'historical_control_result_paths': [str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        'historical_control_audit_path': str(CONTROL / 'audit.json'),
        'steps_per_stage': 10000, 'full_evaluation': True, 'expected_user_linear_bits': EXPECTED_BITS,
        'budget_estimate': budget, 'training_argv': [sys.executable, 'tools/cat_train.py', *argv],
        'normalized_expected': normalized, 'source_sha256': sources, 'input_sha256': evidence,
        'protected_stats': template['protected_stats'], 'snapshot_import_paths': template['snapshot_import_paths'],
        'environment': template['environment'], 'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_down_width.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': reuse,
        'scope': 'Fresh original-HF all32 down D16/K24/S2/H256R0 symmetric BF16/block32/input256 INT8, MSE10Kx2/B8192/seed31; retain other192 and backbone from3577. No KD/LoRA.',
        'comparison_boundary': 'Primary versus completed fresh D16/H128/block32; original3577 secondary. Symmetric widens both encoder and decoder. Fixed updates/batch/LR are not equal FLOPs; changed parameter shapes change initialization RNG consumption. This existing-configuration quality comparison does not isolate deployed decoder capacity or imply equal training trajectories.',
        'resource_estimate': {'seconds_limit': 2700,
            'historical_h128_full_chain_seconds': template['finished_unix'] - template['started_unix'],
            'historical_h128_stage_seconds': {name: item['finished_unix'] - item['started_unix']
                                             for name, item in template['stages'].items()},
            'uncertainty': 'H256 Llama full-category runtime/peak is not measured; reuse real Qwen H256/B8192/36-module support evidence and Llama H128 full chain. No automatic queue or retry.'},
        'protocol': template['protocol'], 'cuda_initialized': False}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(evidence),
           'estimated_bpw': budget['estimated_bpw'], 'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output_dir)
