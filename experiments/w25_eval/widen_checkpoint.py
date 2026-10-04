"""Create a fresh v6 base using the existing H128-to-H256 identity initializer.

Only selected packed decoder leaves change. This CPU migration verifies storage
and fresh reload, not finite-precision forward parity or downstream accuracy.
Keep CUDA visible for the existing FA2-aware Qwen CPU loader when actually run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require, _save_and_verify
from experiments.qwen3_w25.expand_decoder import validate_packed_r0_decoder
from experiments.qwen3_w25.widen_decoder import widen_packed_decoder_h128_to_h256_
from experiments.w25_eval.acceptance import validate_checkpoint
from experiments.w25_eval.model_inventory import model_inventory, sha256
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import _atomic_write_json, refresh_vae_linear_runtime_after_state_load
from train_utils.config.targets import parse_target_layers, parse_target_modules, resolve_target_layers
from train_utils.utils import extract_layer_idx
from train_utils.v6_model_loader import load_v6_model_checkpoint


REPO = Path(__file__).resolve().parents[2]
SUFFIXES = ('linear_in.conv.weight', 'linear_in.conv.bias', 'norm_out.weight',
            'norm_out.bias', 'linear_out.conv.weight', 'linear_out.conv.bias')
CODE_FILES = (
    'experiments/w25_eval/widen_checkpoint.py', 'experiments/w25_eval/model_inventory.py',
    'experiments/w25_eval/acceptance.py', 'experiments/qwen3_w25/assemble_categories.py',
    'experiments/qwen3_w25/widen_decoder.py', 'experiments/qwen3_w25/expand_decoder.py',
    'train_utils/checkpoint_v6.py', 'train_utils/checkpoint_bpw.py', 'train_utils/v6_model_loader.py',
    'train_utils/config/targets.py', 'train_utils/utils.py', 'rotation/model_utils.py',
    'litebsq/autoencoder.py', 'litebsq/parallel_layers.py', 'litebsq/vae_linear.py',
)


def selected_projection_names(architecture, target_layers, target_modules):
    layers = resolve_target_layers(parse_target_layers(target_layers), num_layers=architecture['num_hidden_layers'])
    requested = parse_target_modules(target_modules)
    available = {name.rsplit('.', 1)[-1] for name in architecture['projection_shapes']}
    modules = available if requested == 'all' else set(requested)
    _require(modules <= available, f'Unknown projection suffixes: {sorted(modules - available)}.')
    names = tuple(name for name in architecture['projection_shapes']
                  if extract_layer_idx(name) in layers and name.rsplit('.', 1)[-1] in modules)
    _require(len(names) == len(layers) * len(modules), 'Selection must cover every requested layer/module pair.')
    return names


def widened_specs(meta, architecture, names):
    expected = deepcopy(meta['converted_modules'])
    shapes = architecture['projection_shapes']
    _require({s['name'] for s in expected} == set(shapes) and len(expected) == len(shapes)
             and set(meta['compressed_targets']) == set(shapes), 'Source must contain the complete actual architecture.')
    _require(bool(names) and len(set(names)) == len(names) and set(names) <= set(shapes), 'Invalid selected projection inventory.')
    added_parameters = 0
    for spec in expected:
        name = spec['name']
        _require([spec['out_features'], spec['in_features']] == shapes[name], f'Wrong architecture shape: {name}.')
        if name not in names:
            continue
        _require(spec['parallel_parts'] == 1 and spec['residual_stages'] == 2
                 and spec['parallel_stage_decode'], f'Expected two packed single-part stages: {name}.')
        _require(len(spec['stage_decoders']) == 2 and len(spec['decoders']) == 1,
                 f'Invalid packed decoder metadata: {name}.')
        for field in ('decoders', 'stage_decoders'):
            for decoder in spec[field]:
                _require(decoder['hidden_dim'] == 128 and decoder['num_res_blocks'] == 0
                         and decoder['decoder_type'] == 'symmetric' and decoder['norm_type'] == 'layer'
                         and decoder['activation_type'] == 'swish' and decoder['param_dtype'] == 'bfloat16',
                         f'Selected source must use the existing BF16 H128/R0 LayerNorm/SiLU recipe: {name}.')
                if field == 'stage_decoders':
                    added_parameters += 128 * (decoder['in_dim'] + decoder['out_dim'] + 3)
                decoder.update(hidden_dim=256, decoder_type='asymmetric')
    return expected, added_parameters


def expected_widened_state(original, meta, names):
    """Construct the exact storage reference from original per-stage tensors."""
    expected = dict(original)
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    keys = set()
    for name in names:
        prefix = name + '._parallel_stage_decoder.'
        selected = {prefix + suffix for suffix in SUFFIXES}
        _require({key for key in original if key.startswith(prefix)} == selected,
                 f'Expected exactly six original decoder leaves: {name}.')
        _require(all(original[key].device.type == 'cpu' and original[key].dtype == torch.bfloat16
                     and bool(torch.isfinite(original[key]).all()) for key in selected),
                 f'Expected finite CPU BF16 source decoder leaves: {name}.')
        stages = specs[name]['stage_decoders']
        _require(len(stages) == 2 and stages[0]['in_dim'] == stages[1]['in_dim']
                 and stages[0]['out_dim'] == stages[1]['out_dim'], f'Packed stage dimensions differ: {name}.')
        inputs, outputs = stages[0]['in_dim'], stages[0]['out_dim']
        weight = original[prefix + 'linear_in.conv.weight'].reshape(2, 128, inputs, 1)
        expected[prefix + 'linear_in.conv.weight'] = torch.cat((weight, weight), dim=1).reshape(512, inputs, 1)
        bias = original[prefix + 'linear_in.conv.bias'].reshape(2, 128)
        expected[prefix + 'linear_in.conv.bias'] = torch.cat((bias, bias), dim=1).reshape(512)
        for suffix in ('norm_out.weight', 'norm_out.bias'):
            value = original[prefix + suffix]
            _require(value.shape[-1] == 128, f'Invalid original LayerNorm width: {name}.')
            expected[prefix + suffix] = torch.cat((value, value), dim=-1)
        weight = original[prefix + 'linear_out.conv.weight'].reshape(2, outputs, 128, 1)
        expected[prefix + 'linear_out.conv.weight'] = torch.cat((weight, torch.zeros_like(weight)), dim=2).reshape(2 * outputs, 256, 1)
        # The final output bias is deliberately preserved, including its bits.
        keys.update(selected)
    return expected, keys


def _compare_budgets(before, after, added_bits):
    _require(before['budget_passed'] and after['budget_passed'], 'Source or widened base exceeds the rank-zero 2.5-bpw budget.')
    for key in ('original_linear_weights', 'outside_linear_bits', 'code_padding_bits'):
        _require(before[key] == after[key], f'Widening changed non-decoder accounting: {key}.')
    _require({k: v for k, v in before['components'].items() if k != 'decoder'}
             == {k: v for k, v in after['components'].items() if k != 'decoder'},
             'Codes, protection, rotation or another non-decoder storage component changed.')
    for key in ('user_linear_bits', 'packed_linear_bits'):
        _require(after[key] - before[key] == added_bits, f'Decoder-only bit increment disagrees: {key}.')
    _require((after['full_model_tensor_bytes'] - before['full_model_tensor_bytes']) * 8 == added_bits
             and after['components']['decoder']['storage_bits'] - before['components']['decoder']['storage_bits'] == added_bits,
             'Physical decoder/full-model storage increment disagrees.')


def widen_checkpoint(source_dir, output_dir, report_path, *, target_layers, target_modules):
    source, output, report_path = (Path(p).resolve() for p in (source_dir, output_dir, report_path))
    _require(not output.exists() and not report_path.exists(), 'Use new checkpoint/report paths.')
    _require(not output.is_relative_to(source) and not report_path.is_relative_to(source)
             and not source.is_relative_to(output), 'Output/report must not overlap the protected source.')
    _require(not report_path.is_relative_to(output), 'Keep the migration report outside the checkpoint directory.')
    started = time.monotonic()
    raw_meta = json.loads((source / 'checkpoint_meta.json').read_text())
    architecture = model_inventory(raw_meta['base_model_path'])
    _, meta, source_budget = validate_checkpoint(source, architecture)
    _require(meta.get('lora_config') is None and not meta.get('post_norm_head_linear'),
             'This migration requires the existing rank-zero model without an auxiliary head branch.')
    names = selected_projection_names(architecture, target_layers, target_modules)
    specs, added_parameters = widened_specs(meta, architecture, names)
    code_pins = {name: sha256(REPO / name) for name in CODE_FILES}
    source_pins = {name: sha256(source / name) for name in ('checkpoint_meta.json', 'pytorch_model.bin', 'config.json')}
    report = {'status': 'running', 'source_checkpoint': str(source), 'source_checkpoint_id': meta['checkpoint_id'],
              'checkpoint_dir': str(output), 'selected_targets': list(names), 'architecture': architecture,
              'source_sha256': source_pins, 'code_sha256': code_pins,
              'worker_pid': os.getpid(), 'worker_start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
              'started_at': datetime.now().astimezone().isoformat(), 'numerical_forward_checked': False,
              'purpose': 'fresh independent inference/round base; no training, optimizer or exact-resume state'}
    try:
        original = _load_state(source)
        expected, keys = expected_widened_state(original, meta, names)
        actual_added = sum(expected[key].numel() - original[key].numel() for key in keys)
        _require(actual_added == added_parameters, 'Actual source decoder dimensions disagree with metadata.')
        widened_meta = dict(meta, converted_modules=specs)
        budget = audit_state_dict(widened_meta, expected, max_bpw=2.5, max_lora_rank=0)
        _compare_budgets(source_budget, budget, added_parameters * 16)
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True, expected_kind='final_model')
        _require(loaded_meta['checkpoint_id'] == meta['checkpoint_id'], 'Loaded source identity changed.')
        report['source_loaded_tensors_compared'] = _assert_state_equal(model.state_dict(), original, 'Complete original source before widening')
        model.eval().requires_grad_(False)
        for name in names:
            validate_packed_r0_decoder(model.get_submodule(name))
        transformations = {name: widen_packed_decoder_h128_to_h256_(model.get_submodule(name)) for name in names}
        for name in names:
            refresh_vae_linear_runtime_after_state_load(model.get_submodule(name))
        report['initialized_tensors_compared'] = _assert_state_equal(model.state_dict(), expected, 'Exact per-stage duplication and zero-column initialization')
        frozen = {key: value for key, value in original.items() if key not in keys}
        report['frozen_tensors_compared'] = _assert_state_equal({key: model.state_dict()[key] for key in frozen}, frozen, 'Every nonselected decoder/code/backbone tensor')
        provenance = {'base_checkpoint_dir': str(source), 'base_checkpoint_id': meta['checkpoint_id'],
                      'donor_checkpoint_id': meta['checkpoint_id'], 'donor_kind': 'same_source_identity_initializer_no_donor_weights',
                      'method': 'per-stage input/LayerNorm duplication and output [B,0]; unchanged output bias',
                      'selected_targets': list(names), 'selected_decoder_keys': sorted(keys),
                      'initialization': transformations, 'source_sha256': source_pins, 'code_sha256': code_pins,
                      'added_decoder_parameters': added_parameters, 'added_decoder_storage_bits': added_parameters * 16,
                      'purpose': report['purpose']}
        tokenizer = AutoTokenizer.from_pretrained(architecture['base_model_path'], use_fast=True, local_files_only=True)
        result = _save_and_verify(model, output, widened_meta, expected, provenance, tokenizer)
        _compare_budgets(source_budget, result['bit_budget'], added_parameters * 16)
        _require(result['serialized_tensors_compared'] == result['reloaded_tensors_compared'] == len(original),
                 'Incomplete actual whole-state saved/fresh verification.')
        for name, digest in code_pins.items():
            _require(sha256(REPO / name) == digest, f'Migration code changed: {name}.')
        for name, digest in source_pins.items():
            _require(sha256(source / name) == digest, f'Protected source changed: {name}.')
        report.update(status='verified', checkpoint_id=result['checkpoint_id'], source_tensor_count=len(original),
                      selected_decoder_tensor_count=len(keys), frozen_code_banks=source_budget['components']['code']['unique_storages'],
                      serialized_tensors_compared=result['serialized_tensors_compared'], reloaded_tensors_compared=result['reloaded_tensors_compared'],
                      fresh_strict_reload_state_exact=True, converted_specs_only_selected_width_type_changed=True,
                      nonselected_tensor_bytes_unchanged=True, source_inputs_unchanged=True,
                      added_decoder_parameters=added_parameters, added_decoder_storage_bits=added_parameters * 16,
                      source_bit_budget=source_budget, bit_budget=result['bit_budget'], initialization=transformations,
                      cuda_initialized=torch.cuda.is_initialized())
        return report
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, completed_at=datetime.now().astimezone().isoformat())
        _atomic_write_json(report, str(report_path))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source_dir', 'output_dir', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('target_layers', 'target_modules'):
        parser.add_argument('--' + name, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    result = widen_checkpoint(args.source_dir, args.output_dir, args.report,
                              target_layers=args.target_layers, target_modules=args.target_modules)
    print(json.dumps({key: result[key] for key in ('status', 'checkpoint_id', 'selected_targets',
                                                 'added_decoder_storage_bits', 'reloaded_tensors_compared')}), flush=True)


if __name__ == '__main__':
    main()
