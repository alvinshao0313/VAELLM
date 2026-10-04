"""Assemble Qwen3/Llama v6 categories; create an inference-only checkpoint.

Original architecture/tokenizer metadata determines the inventory. A partial
fresh-CAT donor may retain dense unselected projections; those never enter the
complete compressed output. CUDA visibility is preserved for the native loader.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import (
    _assert_state_equal, _expected_state, _load_state, _replace_modules,
    _require, _save_and_verify,
)
from experiments.w25_eval.model_inventory import CATEGORIES, model_inventory, sha256
from train_utils.checkpoint_bpw import _code_specs, audit_state_dict
from train_utils.checkpoint_v6 import load_v6_meta, resolve_v6_checkpoint_dir, validate_v6_meta
from train_utils.v6_model_loader import load_v6_model_checkpoint


def assembly_tokenizer(original):
    tokenizer = AutoTokenizer.from_pretrained(original, use_fast=True, local_files_only=True)
    _require(tokenizer.is_fast, 'Assembly requires the original fast tokenizer.')
    original_pad = tokenizer.pad_token_id
    # Same rule as e2e_common.data.build_tokenizer, without online resolution.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _require(type(tokenizer.pad_token_id) is int and tokenizer.pad_token_id >= 0,
             'Original tokenizer supplies neither a pad token nor a usable EOS token.')
    return tokenizer, original_pad


def validate_runtime_configs(base_config, donor_config, architecture, tokenizer, original_pad):
    configs = [dict(base_config), dict(donor_config)]
    runtime = {}
    for label, config in zip(('base', 'donor'), configs):
        _require(config.get('model_type') == architecture['model_type']
                 and config.get('num_hidden_layers') == architecture['num_hidden_layers']
                 and config.get('architectures') == [architecture['model_class']],
                 f'{label}: saved configuration disagrees with the actual architecture.')
        cache, pad = config.get('use_cache'), config.get('pad_token_id')
        _require(type(cache) is bool, f'{label}: use_cache must be bool.')
        _require(pad is None or (type(pad) is int and pad == tokenizer.pad_token_id),
                 f'{label}: pad_token_id must be None or the original tokenizer resolved pad ID.')
        runtime[label] = {'use_cache': cache, 'pad_token_id': pad}
        for field in ('_name_or_path', 'transformers_version', 'use_cache', 'pad_token_id'):
            config.pop(field, None)
    _require(configs[0] == configs[1],
             'Source configurations differ outside the permitted runtime fields.')
    runtime.update({
        'differences': {key: {label: runtime[label][key] for label in ('base', 'donor')}
                        for key in ('use_cache', 'pad_token_id')
                        if runtime['base'][key] != runtime['donor'][key]},
        'original_tokenizer_pad_token_id': original_pad,
        'resolved_tokenizer_pad_token_id': tokenizer.pad_token_id,
        'tokenizer_eos_token_id': tokenizer.eos_token_id,
        'pad_resolution': 'original pad; if absent, original tokenizer EOS, as build_tokenizer',
        'load_and_save_config_source': 'shared original HF model via canonical v6 loader',
    })
    return runtime


def validate_source_metadata(meta, architecture, *, complete):
    meta = validate_v6_meta(meta)
    shapes = architecture['projection_shapes']
    _require(meta['checkpoint_kind'] in ('final_model', 'category_boundary'),
             'Source must be a stable final_model or category_boundary.')
    _require(Path(meta['base_model_path']).resolve() == Path(architecture['base_model_path']),
             'Source references a different original architecture/model directory.')
    if complete:
        _require(meta['checkpoint_kind'] == 'final_model'
                 and set(meta['compressed_targets']) == set(shapes)
                 and not meta['pending_dense_targets'] and not meta['skip_targets'],
                 'Base must contain every actual projection compressed, without dense/skip targets.')
    for field in ('compressed_targets', 'pending_dense_targets', 'skip_targets'):
        _require(len(meta[field]) == len(set(meta[field])) and set(meta[field]) <= set(shapes),
                 f'Source {field} contains duplicate or unknown projections.')
    _require(meta.get('lora_config') is None
             and (meta.get('extra_meta') or {}).get('residual_lora') is None,
             'LoRA is forbidden.')
    _require(not meta.get('shared_protected_residual_decoders'),
             'Shared residual decoder registries cannot be transplanted.')
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    _require(set(specs) == set(meta['compressed_targets']),
             'Converted specifications must match the compressed inventory.')
    for name, spec in specs.items():
        _require(name in shapes and [spec['out_features'], spec['in_features']] == shapes[name],
                 f'Projection shape differs from the actual architecture: {name}.')
        _require(not spec.get('has_original_weight') and not spec.get('always_use_original')
                 and not spec.get('protect_original_weight'),
                 f'Original dense projection payload is forbidden: {name}.')
        _require(spec.get('low_rank_a') is None and spec.get('low_rank_b') is None,
                 f'LoRA is forbidden: {name}.')
        _require(not spec.get('protected_residual_shared_decoder_refs'),
                 f'Shared residual decoder references are forbidden: {name}.')
        dims = spec.get('stage_codebook_dims') or [spec['codebook_dim']]
        _require(all(type(dim) is int and dim > 0 for dim in dims),
                 f'Invalid codebook dimensions: {name}.')
        counts = []
        for axis, other in (('input', 'out_features'), ('output', 'in_features')):
            prefix = f'protected_{axis}_'
            index, values, scales = (spec.get(prefix + field) for field in ('indices', 'qvalues', 'scales'))
            _require(spec.get(prefix + 'weight') is None, f'Dense protected payload is forbidden: {name}.')
            count = 0 if index is None else index['shape'][0]
            _require(type(count) is int and 0 <= count < spec['in_features' if axis == 'input' else 'out_features']
                     and all(count % dim == 0 for dim in dims),
                     f'Protection count must be codebook-aligned and leave a nonempty matrix: {name}.')
            if count:
                _require(index == {'shape': [count], 'dtype': 'int64'}
                         and values == {'shape': [count, spec[other]], 'dtype': 'uint8'}
                         and scales == {'shape': [count], 'dtype': 'bfloat16'}
                         and spec.get('protected_channel_quant_format') in ('int8', 'fp8_e4m3', 'fp8_e5m2'),
                         f'Invalid 8-bit protected channel layout: {name}.')
            else:
                _require(index is None and values is None and scales is None,
                         f'Empty protection must not retain tensor payloads: {name}.')
            counts.append(count)
        _require(not all(counts), f'Simultaneous input/output protection is unsupported: {name}.')
        _require(spec['compressed_in_features'] == spec['in_features'] - counts[0]
                 and spec['compressed_out_features'] == spec['out_features'] - counts[1],
                 f'Protection and compressed geometry disagree: {name}.')
        rotation = spec.get('weight_rotation')
        if rotation and rotation['block_size']:
            _require(all(spec[field] % rotation['block_size'] == 0
                         for field in ('compressed_in_features', 'compressed_out_features')),
                     f'Rotation block does not divide the protected geometry: {name}.')
        list(_code_specs(spec))  # Existing packed schema and complete matrix coverage check.
    return meta, specs


def validate_sources(base_dir, donor_dir, categories):
    raw = [load_v6_meta(str(path)) for path in (base_dir, donor_dir)]
    architecture = model_inventory(raw[0]['base_model_path'])
    base_meta, base_specs = validate_source_metadata(raw[0], architecture, complete=True)
    donor_meta, donor_specs = validate_source_metadata(raw[1], architecture, complete=False)
    _require(categories and len(categories) == len(set(categories))
             and set(categories) <= set(CATEGORIES), 'Select unique explicit projection categories.')
    names = sorted(name for name in architecture['projection_shapes']
                   if name.rsplit('.', 1)[1] in categories)
    _require(set(names) <= set(donor_specs), 'Donor must compress every selected category in every actual layer.')
    tokenizer, original_pad = assembly_tokenizer(architecture['base_model_path'])
    configs = [json.loads((path / 'config.json').read_text()) for path in (base_dir, donor_dir)]
    runtime = validate_runtime_configs(*configs, architecture, tokenizer, original_pad)
    differences = {name: sorted(key for key in set(base_specs[name]) | set(donor_specs[name])
                                if base_specs[name].get(key) != donor_specs[name].get(key)) for name in names}
    return base_meta, donor_meta, names, architecture, tokenizer, runtime, differences


def assemble(base_dir, donor_dir, categories, output_dir):
    requested_output = Path(output_dir)
    _require(not requested_output.exists() and not requested_output.is_symlink(),
             f'Output already exists: {requested_output}.')
    base_dir, donor_dir = [Path(resolve_v6_checkpoint_dir(str(path))) for path in (base_dir, donor_dir)]
    output_dir = requested_output.resolve()
    _require(not any(output_dir.is_relative_to(path) for path in (base_dir, donor_dir)),
             'Output must be outside both source checkpoints.')
    categories = [part.strip() for part in categories.split(',')]
    base_meta, donor_meta, names, architecture, tokenizer, runtime, differences = validate_sources(
        base_dir, donor_dir, categories)
    base_state, donor_state = _load_state(base_dir), _load_state(donor_dir)
    expected = _expected_state(base_state, donor_state, names, architecture['projection_shapes'])
    donor_specs = {spec['name']: spec for spec in donor_meta['converted_modules']}
    assembled_meta = dict(base_meta)
    assembled_meta['converted_modules'] = [donor_specs[spec['name']] if spec['name'] in names else spec
                                           for spec in base_meta['converted_modules']]
    budget = audit_state_dict(assembled_meta, expected, max_bpw=2.5, max_lora_rank=0)
    _require(budget['budget_passed'], f'Assembled budget failed: {budget["contract_violations"]}')
    _require(not budget['unclassified_target_keys'], 'Unclassified projection state cannot be transplanted.')
    _require(not budget['components'].get('dense_weights'), 'Dense projections cannot enter the output.')
    provenance = {
        'base_checkpoint_dir': str(base_dir), 'base_checkpoint_id': base_meta['checkpoint_id'],
        'donor_checkpoint_dir': str(donor_dir), 'donor_checkpoint_id': donor_meta['checkpoint_id'],
        'base_meta_sha256': sha256(base_dir / 'checkpoint_meta.json'),
        'donor_meta_sha256': sha256(donor_dir / 'checkpoint_meta.json'),
        'base_config_sha256': sha256(base_dir / 'config.json'),
        'donor_config_sha256': sha256(donor_dir / 'config.json'),
        'original_model_path': architecture['base_model_path'], 'architecture': architecture,
        'categories': categories, 'replaced_targets': names,
        'source_config_runtime_fields': runtime, 'selected_spec_differences': differences,
        'purpose': 'inference_only; no CAT exact-resume contract',
    }
    print(f'Verified {architecture["model_type"]} inventory; replacing {len(names)}/'
          f'{architecture["projection_count"]}, expected {budget["user_linear_bpw"]:.9f} bpw.', flush=True)
    base, _, _ = load_v6_model_checkpoint(str(base_dir), map_location='cpu', strict=True, expected_kind='final_model')
    donor, _, _ = load_v6_model_checkpoint(str(donor_dir), map_location='cpu', strict=True)
    _replace_modules(base, donor, names)
    _assert_state_equal(base.state_dict(), expected, 'In-memory assembly')
    del donor
    gc.collect()
    return _save_and_verify(base, output_dir, assembled_meta, expected, provenance, tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Complete compressed Qwen3/Llama v6 final_model.')
    parser.add_argument('--donor', type=Path, required=True, help='Stable v6 with every selected category compressed.')
    parser.add_argument('--categories', required=True, help='Explicit comma-separated categories, e.g. gate_proj,down_proj.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = assemble(args.base, args.donor, args.categories, args.output)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'Assembly rejected: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'status': report['status'], 'checkpoint_id': report['checkpoint_id'],
                      'user_linear_bpw': report['bit_budget']['user_linear_bpw'],
                      'report': str(args.output.resolve() / 'assembly_report.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
