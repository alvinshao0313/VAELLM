"""Prepare the existing asymmetric encoder256/decoder128 option against K34True."""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch

from experiments.llama_gate_small_rate_pair.common import ROOT, ORIGINAL, RESULT, BASE_ID, DENOMINATOR, LOGICAL_BITS, PACKED_BITS, check, read, _require
from experiments.llama_gate_decoder_width.prepare import prior_evidence
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import file_stat, sha256, write_json
from litebsq.autoencoder import MultiLayerVAE
from litebsq.fused_multistage_decoder import packed_symmetric_decoder_supports_fuse
from litebsq.vae_args import apply_autoencoder_arch_defaults
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import _build_decoder_from_spec, _rebuild_converted_modules, _collect_vae_linear_specs, load_v6_meta

CONTROL = RESULT / 'llama_gate_k34_transpose/true'
CONTROL_ID = '98a80541-fd2b-4ce4-b88d-0809f82d4ef3'


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    changes = {'base_ch': 256, 'decoder_type': 'asymmetric', 'decoder_base_ch': 128, 'decoder_num_res_blocks': 0}
    for key, value in {'--output_dir': str(output / 'donor'),
                       **{'--' + k: 'default=' + str(v) for k, v in changes.items()}}.items():
        _require(argv.count(key) <= 1, 'Ambiguous native argument: ' + key)
        if key in argv:
            argv[argv.index(key) + 1] = value
        else:
            argv.extend([key, value])
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    for key, value in changes.items():
        for section in ('canonical_config', 'vae_args'):
            expected[section][key]['default'] = value
        expected['resolved_category_runtime']['gate_proj'][key] = value
    expected['canonical_config']['explicit_cli_flags'] = sorted(set(expected['canonical_config']['explicit_cli_flags'])
        | {'--decoder_base_ch', '--decoder_num_res_blocks'})
    _require(normalized == expected, 'Native recipe differs beyond output and encoder-only asymmetric selection.')
    cfg = resolved['gate_proj']
    _require(cat.compression_categories == 'gate_proj' and cat.target_layers == 'all'
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,gate_proj,down_proj'
             and cat.batch_size == 8192 and cat.linear_group_size == 32 and cat.seed == 31
             and cat.after_category_mode == 'none' and not cat.resume_from_checkpoint
             and cfg.steps == 10000 and cfg.codebook_bits == 34 and cfg.codebook_dim == 32
             and cfg.residual_stages == 2 and cfg.base_ch == 256 and cfg.num_res_blocks == 0
             and cfg.decoder_type == 'asymmetric' and cfg.decoder_base_ch == 128 and cfg.decoder_num_res_blocks == 0
             and cfg.channel_protect_count == 64 and vae.weight_rotation == 'two_sided'
             and vae.weight_rotation_block_size == 32 and vae.lr == .003 and training.bf16 and not training.fp16,
             'Wrong fresh K34True encoder-only recipe.')
    return argv, normalized, vae, cfg


def construction(meta, budget, vae_args, cfg):
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    old = specs[0]['stage_decoders'][0]
    _require(len(specs) == 32 and all(s['transpose'] and s['weight_rotation']['block_size'] == 32
             and s['compressed_in_features'] == 4032 and s['compressed_out_features'] == 14336
             and s['stage_decoders'] == [old, old] and s['protected_input_indices']['shape'] == [64] for s in specs)
             and old == {'in_dim': 34, 'out_dim': 32, 'hidden_dim': 128, 'num_res_blocks': 0,
                 'norm_type': 'layer', 'activation_type': 'swish', 'decoder_type': 'symmetric',
                 'use_checkpoint': True, 'param_dtype': 'bfloat16'}, 'Wrong control native geometry.')
    new = {**old, 'decoder_type': 'asymmetric'}
    with torch.random.fork_rng(devices=[]):
        original_decoder = _build_decoder_from_spec(old); decoder = _build_decoder_from_spec(new)
        shapes = lambda module: {k: [list(v.shape), str(v.dtype)] for k, v in module.state_dict().items()}
        _require(shapes(original_decoder) == shapes(decoder) and len(shapes(decoder)) == 6
                 and sum(p.numel() for p in decoder.parameters()) == 8864
                 and packed_symmetric_decoder_supports_fuse(original_decoder)
                 and not packed_symmetric_decoder_supports_fuse(decoder), 'Deployed decoder payload shape/count differs.')
        args = SimpleNamespace(**{**vars(vae_args), **vars(cfg), 'parallel_layers': 32})
        apply_autoencoder_arch_defaults(args); model = MultiLayerVAE(args)
        arch = model.model.arch_spec
        _require(arch.encoder_hidden_dim == 256 and arch.decoder_hidden_dim == 128
                 and arch.encoder_num_res_blocks == arch.decoder_num_res_blocks == 0
                 and model.model.decoder.num_models == 32
                 and sum(p.numel() for p in model.model.decoder.parameters()) == 32 * 8864,
                 'Actual grouped native VAE did not separate encoder/decoder widths.')
        result = {'arch_spec': asdict(arch), 'decoder_parameters_per_stage': 8864, 'decoder_tensor_shapes': shapes(decoder),
            'grouped_encoder_parameters': sum(p.numel() for p in model.model.encoder.parameters()),
            'grouped_decoder_parameters': sum(p.numel() for p in model.model.decoder.parameters()),
            'decoder_delta_bits': 0, 'encoder_is_not_deployed': True}
        # Exact native v6 reconstruction also checks packed K34 padding geometry and alias storage.
        holder = torch.nn.Module(); holder.add_module('gate_proj', torch.nn.Linear(4096, 14336, bias=False))
        expected = deepcopy(specs[0]); expected['name'] = 'gate_proj'
        for d in expected['decoders'] + expected['stage_decoders']:
            d['decoder_type'] = 'asymmetric'
        _rebuild_converted_modules(holder, [expected]); state = holder.state_dict()
        physical = sum({v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in state.values()}.values()) * 8
        _require(_collect_vae_linear_specs(holder)[0] == expected
                 and physical == budget['per_target'][specs[0]['name']]['storage_bits'], 'Native deployed storage changed.')
        result.update(physical_bits_per_gate=physical, serialized_tensors_per_gate=len(state), expected_spec=expected,
            scope='Actual CPU native VAE and v6 geometry only; no learned weights, GPU execution or quality claim.')
    return result


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Require one fresh CPU-only snapshot preparation.')
    template = read(CONTROL / 'launch.json'); audit = read(CONTROL / 'audit.json')
    summary = read(CONTROL / 'full8_summary.json'); review = read(CONTROL / 'independent_review.json')
    current, _ = score([CONTROL / (s + '.json') for s in ('mmlu', 'seven')], template['architecture'], CONTROL / 'final_model')
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and all((CONTROL / (s + '.exit')).read_text().strip() == '0'
                     for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'))
             and audit['status'] == 'verified' and review['status'] == 'independent_cpu_verified'
             and current['checkpoint_id'] == audit['checkpoint_id'] == template['checkpoint_id'] == CONTROL_ID
             and audit['base_checkpoint_id'] == BASE_ID and current == review['actual_full8']
             and current['mean_percent'] == 61.81960860492682 and current['integer_counts'] == summary['integer_counts']
             and summary['audit_sha256'] == sha256(CONTROL / 'audit.json')
             and template['full8_summary_sha256'] == sha256(CONTROL / 'full8_summary.json'), 'Wrong completed primary control.')
    budget = audit['bit_budget']; meta = load_v6_meta(str(CONTROL / 'final_model'))
    _require(meta['checkpoint_id'] == CONTROL_ID and budget['user_linear_bits'] == LOGICAL_BITS
             and budget['packed_linear_bits'] == PACKED_BITS and budget['code_padding_bits'] == 693633024
             and budget['original_linear_weights'] == DENOMINATOR and budget['budget_passed'], 'Wrong actual control budget.')
    argv, normalized, vae, cfg = recipe(template, output); native = construction(meta, budget, vae, cfg)
    proof_sources, proof_evidence, reuse = prior_evidence()
    width = RESULT / 'llama_gate_decoder_width'; width_audit = read(width / 'audit.json')
    _require(width_audit['status'] == 'verified' and width_audit['actual_decoder_spec']['decoder_type'] == 'asymmetric'
             and width_audit['actual_decoder_spec']['num_res_blocks'] == 0
             and read(width / 'launch.json')['status'] == 'completed'
             and all((width / (s + '.exit')).read_text().strip() == '0'
                     for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')), 'Require measured Llama asymmetric R0 full path.')
    sources = {**template['source_sha256'], **proof_sources}
    for p in [*sorted(Path(__file__).parent.glob('*.py')), ROOT / 'experiments/llama_gate_decoder_width/prepare.py',
              ROOT / 'experiments/llama_gate_transpose/audit.py']:
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
        _require(sha256(ORIGINAL / p.relative_to(ROOT)) == sha256(p), 'Public/snapshot source differs.')
    evidence = [*[CONTROL / f for f in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json', 'independent_review.json')],
        CONTROL / 'final_model/checkpoint_meta.json', CONTROL / 'final_model/assembly_report.json',
        *[width / f for f in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json', 'independent_review.json')]]
    inputs = {**template['input_sha256'], **proof_evidence, **{str(p): sha256(p) for p in evidence}}
    base = Path(template['base_dir']); base_meta = load_v6_meta(str(base))
    _require(base_meta['checkpoint_id'] == BASE_ID, 'Wrong retained e767 assembly base.')
    # Historical 3577/K34False payloads are not inputs to this new comparison.
    stats = [s for s in template['protected_stats'] if '/result/' not in s['path']]
    stats += [file_stat(base / base_meta['state_dict_file']), file_stat(CONTROL / 'final_model' / meta['state_dict_file'])]
    record = {key: deepcopy(template[key]) for key in ('cwd', 'architecture', 'base_dir', 'base_checkpoint_id',
        'control_result_paths', 'block_size', 'steps_per_stage', 'full_evaluation', 'environment', 'protocol', 'transpose', 'codebook_bits')}
    record.update(schema='llama_gate_encoder_width_v1', status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        python_executable=sys.executable, output_dir=str(output), model_type='llama', categories=['gate_proj'],
        encoder_hidden_dim=256, decoder_hidden_dim=128, historical_control_dir=str(CONTROL), historical_control_checkpoint_id=CONTROL_ID,
        training_argv=[sys.executable, 'tools/cat_train.py', *argv], normalized_expected=normalized,
        source_sha256=sources, input_sha256=inputs, protected_stats=stats, native_cpu_construction=native,
        expected_budget_components=budget['components'], expected_user_linear_bits=LOGICAL_BITS,
        expected_packed_linear_bits=PACKED_BITS, expected_padding_bits=693633024,
        timeout_seconds=4500, termination_grace_seconds=20,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_gate_encoder_width.run', '--launch-record', str(output / 'launch.json')],
        validation_reuse={**reuse, 'llama_asymmetric_r0_full8': str(width / 'audit.json'),
            'k34_native_proof': str(RESULT / 'llama_gate_small_rate_pair/validate_k34_s2/verification.json'),
            'boundary': 'Reuse existing grouped H256 CAT, asymmetric saved v6/group8 and K34 native paths. New CPU construction verifies actual H256 encoder/H128 decoder with unchanged packed payload. No short weights consumed or new GPU short.',
            'future_recovery_boundary': 'The existing task-specific subset_state.validate_recipe only accepts H128 symmetric/H256 asymmetric; future recovery needs its structure compatibility expanded, not part of this CAT/full8 run.'},
        comparison_boundary='Primary fresh encoder256 minus98a8 encoder128 with the same e767 other192/backbone. Only encoder capacity, explicit asymmetric metadata and output change. Encoder initialization RNG consumption/compute differ; not equal FLOPs or identical optimizer trajectories. Symmetric K34 is eligible for existing group8 wholefused (no K%8 restriction); asymmetric uses existing packed sequential prewarm. Inference branch may add ordinary floating-point differences. e767 secondary; old decoderH256 and symmetricH256 quality negatives are not this matched comparison. No LongBench or additive-gain claim.',
        experience={'source': str(ORIGINAL / 'docs/experience/lessons/compression_choices.md'),
            'prior': 'Widening both encoder/decoder or only deployed decoder did not improve previous Llama controls.',
            'new_condition': 'No matched encoder-only H256/H128 result among the41 actual Llama normalized records reviewed by root; test complete downstream quality while retaining deployed capacity.'},
        resource_estimate={'estimated_seconds': [1800, 3000], 'seconds_limit': 4500,
            'matched_control_chain_seconds': template['finished_unix'] - template['started_unix'],
            'single_arm_disk_provision_bytes': 34000000000, 'filesystem_free_at_preparation': shutil.disk_usage(RESULT).free,
            'boundary': 'Encoder hidden activations/compute increase; existing H256 grouped CAT supports this resource scale. Native asymmetric prewarm can be slower. Timing is an estimate, one GPU and no retry.'}, cuda_initialized=False)
    _require(not torch.cuda.is_initialized() and shutil.disk_usage(RESULT).free > 34000000000, 'Unexpected CUDA or insufficient disk.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print({'prepared_sha256': sha256(output / 'launch.prepared.json'), 'source': len(sources), 'input': len(inputs), 'stat': len(stats),
           'logical_bpw': LOGICAL_BITS / DENOMINATOR, 'packed_bpw': PACKED_BITS / DENOMINATOR, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
