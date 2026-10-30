"""Read one real native projection and measure SPACE geometry on CPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/home/star/VAELLM')
HERE = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01'
sys.path.insert(0, str(SNAPSHOT))
sys.dont_write_bytecode = True

import torch
from safetensors import safe_open
from train_utils.checkpoint_v6 import _rebuild_converted_modules, refresh_vae_linear_runtime_after_state_load


def read(path):
    return json.loads(Path(path).read_text())


def energy(value):
    return float(value.double().square().sum())


@torch.no_grad()
def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    parent = Path('/data/star/VAELLM/cat_aligned_recovery_20261028/G')
    audits = [read(parent / stage / 'native_audit.json') for stage in
              ('01_qk_fineweb2000', '02_online_opd32')]
    assert all(a['status'] == 'verified' for a in audits)
    assert audits[1]['source_checkpoint_id'] == audits[0]['checkpoint_id']
    dirs = [Path(a['checkpoint_dir']) for a in audits]
    metas = [read(d / 'checkpoint_meta.json') for d in dirs]
    # A single complete small matrix at median depth is a cost probe, not a layer rule.
    layers = sorted({int(s['name'].split('.')[2]) for s in metas[0]['converted_modules']})
    name = f'model.layers.{layers[len(layers) // 2]}.self_attn.k_proj'
    specs = [next(s for s in m['converted_modules'] if s['name'] == name) for m in metas]
    assert specs[0] == specs[1]
    for a, d, m in zip(audits, dirs, metas):
        assert m['checkpoint_id'] == a['checkpoint_id']
        stat = (d / m['state_dict_file']).stat()
        saved = a['checkpoint_weight_stat']
        assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (saved['size'], saved['mtime_ns'], saved['inode'])
    mapped = [torch.load(d / m['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
              for d, m in zip(dirs, metas)]
    states = [{k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + '.')}
              for state in mapped]
    assert states[0].keys() == states[1].keys()
    frozen = [k for k in states[0] if not k.startswith('_parallel_stage_decoder.')]
    assert all(torch.equal(states[0][k], states[1][k]) for k in frozen)
    model = torch.nn.Module()
    model.projection = torch.nn.Linear(specs[0]['in_features'], specs[0]['out_features'], bias=False)
    _rebuild_converted_modules(model, [dict(specs[0], name='projection')])
    module = model.projection.eval().requires_grad_(False)
    weights, timings = [], []
    for state in states:
        module.load_state_dict(state, strict=True)
        refresh_vae_linear_runtime_after_state_load(model)
        t = time.perf_counter()
        weight = module._decode_weight(dtype=torch.bfloat16)
        assert weight.dtype == torch.bfloat16 and torch.isfinite(weight).all()
        weights.append(weight.float())
        timings.append(time.perf_counter() - t)
    base, post = weights
    delta = post - base
    t = time.perf_counter()
    _, singular, vh = torch.linalg.svd(base, full_matrices=False)
    squared = singular.square()
    cumulative = squared.cumsum(0)
    k = int(torch.searchsorted(cumulative, .5 * cumulative[-1])) + 1
    directions = vh[:k].T
    removed = (delta @ directions) @ directions.T
    kept = delta - removed
    calibrated = post - removed
    ratio = (energy(kept) / energy(delta)) ** .5
    shrink = base + ratio * delta
    svd_seconds = time.perf_counter() - t
    protected = states[0]['protected_input_indices'].long()
    assert torch.count_nonzero(delta[:, protected]) == 0
    deployed = calibrated.bfloat16().float()
    shrunk_deployed = shrink.bfloat16().float()
    hf = Path(metas[0]['base_model_path'])
    index = read(hf / 'model.safetensors.index.json')
    key = name + '.weight'
    with safe_open(hf / index['weight_map'][key], framework='pt', device='cpu') as f:
        reference = f.get_tensor(key).float()
    output = {
        'status': 'complete_cpu_geometry_only', 'source_ids': [m['checkpoint_id'] for m in metas],
        'source_paths': [str(d) for d in dirs], 'source_weight_stats': [a['checkpoint_weight_stat'] for a in audits],
        'projection': name, 'selection_rule': 'median depth k_proj; full smallest attention matrix; diagnostic only',
        'matrix_shape': list(base.shape), 'source_state_tensors_touched_per_checkpoint': len(states[0]),
        'source_tensor_bytes_per_checkpoint': sum(v.numel() * v.element_size() for v in states[0].values()),
        'frozen_tensors_equal': len(frozen), 'decoder_parameters': sum(v.numel() for n, v in states[0].items() if n.startswith('_parallel_stage_decoder.')),
        'decode_backend': 'native CPU BF16 reference; not CUDA packed-kernel parity',
        'decode_seconds': timings, 'svd_projection_seconds': svd_seconds,
        'rho': .5, 'alpha': 1., 'rank': k, 'captured_weight_energy': float(cumulative[k-1] / cumulative[-1]),
        'delta_energy': energy(delta), 'relative_delta_norm': (energy(delta) / energy(base)) ** .5,
        'removed_delta_energy_fraction': energy(removed) / energy(delta), 'retained_norm_shrink_beta': ratio,
        'orthogonality_relative_residual': float(torch.linalg.vector_norm(kept @ directions) / torch.linalg.vector_norm(delta)),
        'protected_input_columns': len(protected), 'original_delta_on_protection_energy': energy(delta[:, protected]),
        'unreachable_protected_energy_fraction_of_calibration_move': energy(removed[:, protected]) / energy(removed),
        'unreachable_protected_energy_fraction_of_target_delta': energy(removed[:, protected]) / energy(kept),
        'bf16_calibration_quantization_energy_over_move': energy(deployed - calibrated) / energy(removed),
        'bf16_actual_move_energy_over_ideal_move': energy(deployed - post) / energy(removed),
        'bf16_protected_changed_elements': int(torch.count_nonzero((deployed - post)[:, protected])),
        'bf16_retained_norm_calibration': energy(deployed - base) ** .5,
        'bf16_retained_norm_shrink': energy(shrunk_deployed - base) ** .5,
        'hf_weight_relative_squared_error': {label: energy(weight - reference) / energy(reference) for label, weight in
                                              [('G01', base), ('G02', post), ('SPACE_FP32', calibrated), ('SPACE_BF16', deployed), ('shrink_BF16', shrunk_deployed)]},
        'no_capability_or_native_refit_claim': True, 'cuda_initialized': torch.cuda.is_initialized(),
        'torch_version': torch.__version__, 'python': sys.executable, 'cpu_threads': torch.get_num_threads(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_source_sha256': {str(p.relative_to(SNAPSHOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [SNAPSHOT / 'train_utils/checkpoint_v6.py', SNAPSHOT / 'litebsq/vae_linear.py']},
        'elapsed_seconds': time.perf_counter() - started,
    }
    assert not output['cuda_initialized']
    (HERE / 'cpu_diagnostic.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
