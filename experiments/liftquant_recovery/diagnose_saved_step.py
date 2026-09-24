"""Inspect the already-saved one-step decoder update; no optimizer or new training.

Use identical saved calibration IDs and the same teacher/native packed arithmetic.
Interpolate one FIXED saved direction, not different optimizer runs or test scores.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from experiments.liftquant_recovery.recovery_runtime import (
    block_output, clear_caches, first_inputs, prime_packed_cache,
    teacher_outputs, tree_to,
)
from rotation.model_utils import get_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(0)  # Reproduce the historical smoke, not a new seed-42 run.
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(
        6 * 2**30 / torch.cuda.get_device_properties(device).total_memory, device)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    source_meta = json.loads((args.checkpoint / 'checkpoint_meta.json').read_text())
    if source_meta['checkpoint_id'] != manifest['source_checkpoint_id']:
        raise ValueError('The saved step must originate from this exact checkpoint.')
    ids = torch.load(args.run / 'calibration_ids.pt', weights_only=True, map_location='cpu')
    if tuple(ids.shape) != (4, 64):
        raise ValueError('Only the historical 4-by-64 smoke is supported by this diagnostic.')
    saved_metrics = json.loads((args.run / 'block_metrics.json').read_text())['9']
    if saved_metrics['ntrain'] != 2 or saved_metrics['optimizer_steps'] != 1:
        raise ValueError('Expected exactly one historical update on two training sequences.')
    restored, meta, load = load_v6_model_checkpoint(str(args.checkpoint), map_location='cpu', strict=True)
    if load.missing_keys or load.unexpected_keys:
        raise ValueError('Strict source loading failed.')
    restored.eval().requires_grad_(False)
    block = restored.model.layers[9]
    prefix = 'model.layers.9.'
    bmeta = json.loads((args.run / 'recovered_model/checkpoint_meta.json').read_text())
    bstate = torch.load(args.run / 'recovered_model' / bmeta['state_dict_file'],
                        map_location='cpu', mmap=True, weights_only=True)
    initial = {name: p.detach().clone() for name, p in block.named_parameters()
               if '._parallel_stage_decoder.' in '.' + name}
    if len(initial) != 42:
        raise ValueError(f'Expected 42 decoder tensors, found {len(initial)}.')
    delta = {name: bstate[prefix + name].float() - value.float() for name, value in initial.items()}
    for name, buffer in block.named_buffers():
        if name.endswith('vq_weight') and not torch.equal(buffer, bstate[prefix + name]):
            raise ValueError('Packed codes differ: interpolation would not isolate the decoder step.')
    del bstate
    teacher = get_model(meta['base_model_path']).eval().requires_grad_(False)
    hidden, kwargs = first_inputs(teacher, ids, 2, device)
    for index in range(10):
        target = teacher_outputs(teacher.model.layers[index], hidden, kwargs, 2, device)
        if index != 9:
            hidden = target
    del teacher
    gc.collect()
    kwargs = tree_to(kwargs, device)
    block.to(device).eval()
    params = dict(block.named_parameters())
    report = dict(status='RUNNING', source_checkpoint_id=meta['checkpoint_id'],
                  scope='fixed historical decoder direction; no training or checkpoint writing',
                  physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__,
                  block=9, training_sequences=2, heldout_sequences=2, seqlen=64,
                  original_before_mse=saved_metrics['before_mse'],
                  original_after_mse=saved_metrics['after_mse'],
                  decoder_delta_l2=sum(d.double().square().sum().item() for d in delta.values())**0.5,
                  points=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = None
    for alpha in (0.0, 1.0 / 16, 1.0 / 4, 1.0):
        for name, value in initial.items():
            params[name].copy_((value.float() + alpha * delta[name]).to(device))
        clear_caches(block)
        prime_packed_cache(block)
        prediction = torch.cat([block_output(block, hidden[start:start+2].to(device), kwargs).cpu()
                                for start in (0, 2)])
        errors = (prediction.float() - target.float()).square().mean(dim=(1, 2))
        if baseline is None:
            baseline = prediction.clone()
        row = dict(alpha=alpha, train_mse=errors[:2].mean().item(),
                   heldout_mse=errors[2:].mean().item(), all_mse=errors.mean().item(),
                   block_output_delta_relative_to_teacher=(prediction.float()-baseline.float()).norm().item()
                   / target.float().norm().item())
        report['points'].append(row)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        del prediction, errors
    for actual, expected in ((report['points'][0]['all_mse'], saved_metrics['before_mse']),
                             (report['points'][-1]['all_mse'], saved_metrics['after_mse'])):
        if abs(actual - expected) > 1e-6 * max(1., abs(expected)):
            raise ValueError(f'Historical endpoint not reproduced: {actual} vs {expected}')
    report.update(status='COMPLETE_HISTORICAL_ENDPOINTS_REPRODUCED',
                  max_memory_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  max_memory_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  caution='Forward interpolation supports a step-size diagnosis, not full gradient correctness or an optimal LR.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    clear_caches(block)
    block.cpu()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
