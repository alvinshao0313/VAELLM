"""Read selected deployed INT8 payloads and compare isolated protected codecs on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from experiments.protected_int4_probe.codec import (decode_output_scale, encode_output_scale,
                                                    require, rotate)
from experiments.protected_int4_probe.mechanism import checks
from litebsq.protected_channel_quant import decode_protected_channel_weight

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'qwen3': (ROOT / 'result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model',
              'f927ffda-dd5b-4d00-b63a-5e839a182d70', (0, 18, 35), 3423, 252),
    'llama': (ROOT / 'result/qwen3_8b_mmlu65_20261004/llama_fineweb1600_down_d16_combo/final_model',
              'e767c232-56fd-4b03-a4ae-dd0781b6e047', (0, 16, 31), 2979, 224),
}
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
            'storage_bits': value.untyped_storage().nbytes() * 8}


def stat_record(path):
    value = path.stat()
    return {'path': str(path), 'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns,
            'device': value.st_dev, 'inode': value.st_ino}


def errors(candidate, reference):
    error = candidate.double() - reference.double()
    sse, energy = float(error.square().sum()), float(reference.double().square().sum())
    require(energy > 0, 'Actual protected reference must have positive energy.')
    return {'sse': sse, 'reference_energy': energy, 'mse': sse / error.numel(),
            'nmse': sse / energy, 'relative_l2': (sse / energy) ** .5,
            'max_abs': float(error.abs().max()), 'changed_elements': int((error != 0).sum())}


def measure(output):
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Run from public root with CUDA hidden.')
    output = Path(output).resolve(); require(not output.exists(), 'Use a fresh isolated output directory.')
    torch.set_num_threads(4)
    started = time.time(); mechanism = checks()
    source_paths = [*sorted(Path(__file__).parent.glob('*.py')),
                    ROOT / 'litebsq/protected_channel_quant.py', ROOT / 'litebsq/vae_linear.py']
    source_hashes = {str(p): digest(p) for p in source_paths}
    reports, saved_payloads, model_stats, meta_hashes = {}, {}, {}, {}
    for label, (directory, uuid, layers, state_count, projection_count) in SOURCES.items():
        meta_path = directory / 'checkpoint_meta.json'; meta = json.loads(meta_path.read_text())
        require(meta['checkpoint_id'] == uuid and len(meta['converted_modules']) == projection_count,
                'Unexpected fixed deployment identity: ' + label)
        model_path = directory / meta['state_dict_file']
        require(model_path.resolve().parent == directory, 'State path escaped checkpoint directory.')
        model_stats[label] = stat_record(model_path); meta_hashes[str(meta_path)] = digest(meta_path)
        # mmap avoids materializing the full state; only the selected protected tensors are copied.
        state = torch.load(model_path, map_location='cpu', weights_only=True, mmap=True)
        require(len(state) == state_count, 'Unexpected persistent tensor inventory: ' + label)
        specs = {spec['name']: spec for spec in meta['converted_modules']}
        report, saved = {}, {}
        for layer in layers:
            for category in CATEGORIES:
                section = 'self_attn' if category in CATEGORIES[:4] else 'mlp'
                name = f'model.layers.{layer}.{section}.{category}'; spec = specs[name]
                count = 256 if category == 'down_proj' else 64
                require(spec['protected_channel_quant_format'] == 'int8'
                        and spec['protected_output_indices'] is None
                        and spec['protected_input_indices'] == {'shape': [count], 'dtype': 'int64'}
                        and count % spec['codebook_dim'] == 0, 'Fixed input protection differs: ' + name)
                indices = state[name + '.protected_input_indices'].clone()
                qvalues = state[name + '.protected_input_qvalues'].clone()
                scales = state[name + '.protected_input_scales'].clone()
                out_features = spec['out_features']
                require(indices.dtype == torch.int64 and indices.shape == (count,)
                        and bool((indices[1:] > indices[:-1]).all())
                        and 0 <= int(indices.min()) <= int(indices.max()) < spec['in_features']
                        and qvalues.dtype == torch.uint8 and qvalues.shape == (count, out_features)
                        and scales.dtype == torch.bfloat16 and scales.shape == (count,), 'Payload geometry differs.')
                reference = decode_protected_channel_weight(qvalues, scales, quant_format='int8',
                    dtype=torch.bfloat16, device=torch.device('cpu')).t().contiguous()
                oracle = (qvalues.view(torch.int8).float() * scales.float().unsqueeze(1)).to(torch.bfloat16).t().contiguous()
                require(torch.equal(reference, oracle), 'Native INT8 input-scale exact control failed.')
                native = {'values': qvalues, 'scales': scales, 'indices': indices}
                native_components = {key: tensor_record(value) for key, value in native.items()}
                native_bits = sum(v['storage_bits'] for v in native_components.values())
                arms = {'native_input_scale_int8': {'storage': {'components': native_components,
                    'total_bits': native_bits, 'logical_value_bits': reference.numel() * 8,
                    'padding_bits': 0, 'scale_axis': 'protected_input', 'scale_count': count,
                    'zero_point_bits': 0, 'sign_bits': 0}, 'versus_native_bf16': errors(reference, reference),
                    'exact_control': True, 'bits_saved_including_indices_scales_signs': 0}}
                saved[name] = {'shape_out_by_protected_input': list(reference.shape), 'native': native}
                for arm, bits, rotation in (('output_scale_int8', 8, False), ('output_scale_int4', 4, False),
                                            ('rht_output_scale_int4', 4, True)):
                    payload = encode_output_scale(reference, bits=bits, use_rotation=rotation, seed=31)
                    before_bf16 = decode_output_scale(payload, dtype=torch.float32)
                    deployed = decode_output_scale(payload, dtype=torch.bfloat16)
                    require(torch.equal(deployed, before_bf16.to(torch.bfloat16)), 'Final BF16 cast order differs.')
                    storage = payload.storage()
                    storage['components']['indices'] = native_components['indices']
                    storage['total_bits'] += native_components['indices']['storage_bits']
                    require(storage['padding_bits'] == 0, 'Real64/256 input protection must have no padding.')
                    item = {'storage': storage, 'payload': {k: tensor_record(v) for k, v in payload.tensors().items()},
                        'before_final_bf16_vs_native_bf16': errors(before_bf16, reference),
                        'versus_native_bf16': errors(deployed, reference),
                        'bits_saved_including_indices_scales_signs': native_bits - storage['total_bits'],
                        'saved_fraction_of_native_protection': 1 - storage['total_bits'] / native_bits}
                    if rotation:
                        forward = rotate(reference.float(), payload.signs)
                        inverse = rotate(forward, payload.signs, inverse=True)
                        item['unquantized_rotation_fp32'] = errors(inverse, reference)
                        item['unquantized_rotation_after_bf16'] = errors(inverse.to(torch.bfloat16), reference)
                        item['rotation_energy_relative_error'] = float(abs(forward.double().square().sum()
                            / reference.double().square().sum() - 1))
                    arms[arm] = item
                    saved[name][arm] = {'shape': payload.shape, 'bits': bits, **payload.tensors()}
                report[name] = {'layer': layer, 'category': category, 'out_features': out_features,
                    'protected_input_count': count, 'reference_elements': reference.numel(), 'arms': arms}
        require(len(report) == len(layers) * len(CATEGORIES) == 21, 'Require the declared complete representative selection.')
        require(stat_record(model_path) == model_stats[label], 'Source checkpoint stat changed during read.')
        del state
        aggregates = {}
        for arm in next(iter(report.values()))['arms']:
            values = [x['arms'][arm] for x in report.values()]
            sse = sum(x['versus_native_bf16']['sse'] for x in values)
            energy = sum(x['versus_native_bf16']['reference_energy'] for x in values)
            aggregates[arm] = {'selected_matrix_count': 21, 'sse': sse, 'reference_energy': energy,
                'pooled_nmse': sse / energy, 'total_bits': sum(x['storage']['total_bits'] for x in values),
                'bits_saved': sum(x['bits_saved_including_indices_scales_signs'] for x in values),
                'boundary': 'Actual tensors for these21 protected subsets only, not full-model mean bits or quality.'}
        reports[label] = {'checkpoint_dir': str(directory), 'checkpoint_id': uuid, 'layers': list(layers),
                          'projections': report, 'aggregates': aggregates}
        saved_payloads[label] = saved
    for path, value in source_hashes.items():
        require(digest(path) == value, 'Probe source changed during measurement.')
    for path, value in meta_hashes.items():
        require(digest(path) == value, 'Checkpoint metadata changed during measurement.')
    require(not torch.cuda.is_initialized(), 'CPU probe initialized CUDA.')
    output.mkdir(parents=True)
    torch.save(saved_payloads, output / 'selected_payloads.pt')
    result = {'status': 'cpu_protected_codec_comparison_complete', 'mechanism': mechanism, 'models': reports,
        'source_sha256': source_hashes, 'checkpoint_meta_sha256': meta_hashes, 'checkpoint_weight_stats': model_stats,
        'whole_model_hash_computed': False, 'model_weights_modified': False, 'model_exported': False,
        'payload_artifact': {'path': str(output / 'selected_payloads.pt'),
            'sha256': digest(output / 'selected_payloads.pt'), 'bytes': (output / 'selected_payloads.pt').stat().st_size,
            'scope': 'Selected original indices/INT8 payload plus all candidate packed values/BF16 scales/int8 signs.'},
        'protocol': {'reference': 'Actual currently deployed native INT8 dequantization with BF16 materialization; not original HF unquantized weights.',
            'selection': 'Every projection category at prespecified first/middle/last layers; no score-based selection.',
            'scale_rule': 'One BF16 amax/7 oramax/127 scale per original output row; zero row scale1; round-even and signed clipping.',
            'rotation': 'Only the protected input subspace: Z=(W D)H; inverse=(Zhat H)D; normalized Sylvester H; explicit int8 signs, seed31 separately for each projection.',
            'storage': 'Actual tensor storage for values, BF16 scales, original int64 indices and explicit int8 signs. Container bytes separate; no hidden zero points or persisted Hadamard matrix.',
            'boundary': 'No main VQ, protection selection, model activation, checkpoint schema, inference loader or downstream task changes.'},
        'experience_boundary': 'Prior local rotation/reconstruction gains did not reliably predict complete tasks; these errors are diagnostics only.',
        'elapsed_seconds': time.time() - started, 'cuda_initialized': False,
        'downstream_quality': 'not_evaluated', 'full_model_budget': 'not_evaluated', 'formal_integration': 'not_performed'}
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
        'report_sha256': digest(output / 'report.json'),
        'aggregates': {k: v['aggregates'] for k, v in reports.items()}}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    measure(parser.parse_args().output_dir)
