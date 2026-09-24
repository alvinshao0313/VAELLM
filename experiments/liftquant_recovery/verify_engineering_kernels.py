"""Permanent equivalence checks for recovery engineering, using real payloads.

Call from the approved GPU runner; this module does not select a GPU, alter
precision policy, write checkpoints, or launch a workload when imported.
"""
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch

from experiments.liftquant_recovery import all_bits
from experiments.liftquant_recovery.proxy_coordinates import hard_bits
from experiments.liftquant_recovery.verify_proxy_coordinates import check_decoder
from experiments.liftquant_recovery.verify_recovery import check_ste
from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool
from litebsq.vae_linear import VAELinear


def load_reference(reference_dir):
    path = Path(reference_dir) / 'all_bits.py'
    spec = importlib.util.spec_from_file_location('_recovery_engineering_reference', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_gradient_flags(state, spec, old):
    name = spec['name']
    prefix = name + '._parallel_stage_decoder.linear_in.linear.'
    w = state[prefix + 'weight'].unsqueeze(0).cuda().float()
    b = state[prefix + 'bias'].unsqueeze(0).cuda().float()
    packed = state[name + '.vq_weight'][:16385].contiguous().cuda()
    shape, scale = (len(packed), 1, w.shape[-1]), 0.005
    bits = unpack_uint8_tensor_to_bool(packed, logical_shape=shape)
    base = (bits.float() - 0.5) * scale
    base[0, 0, :7] = torch.tensor([-1.5, -1., -.5, 0., .5, 1., 1.5], device=w.device) * scale
    packed = pack_bool_tensor_to_uint8(hard_bits(base, scale), logical_shape=shape)
    cotangent = torch.randn(len(packed), 1, w.shape[1], device=w.device, dtype=torch.bfloat16)
    flags = dict(joint=(True, True, True), decoder_only=(True, True, False),
                 code_only=(False, False, True), bias_only=(False, True, False),
                 weight_only=(True, False, False), weight_and_code=(True, False, True),
                 bias_and_code=(False, True, True))
    report = {}
    for label, enabled in flags.items():
        runs = []
        for implementation in (old, all_bits):
            values = [v.detach().clone().requires_grad_(need)
                      for v, need in zip((w, b, base), enabled)]
            out = implementation._AllBitsLinear.apply(packed, *values, torch.bfloat16, scale)
            out.backward(cotangent)
            for value, need in zip(values, enabled):
                if (value.grad is not None) != need:
                    raise AssertionError(f'{label}: incorrect gradient ownership')
            runs.append([out.detach().cpu()] +
                        [value.grad.detach().cpu() for value, need in zip(values, enabled) if need])
            del out, values
        for actual, reference in zip(runs[1], runs[0]):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        report[label] = dict(status='PASS', forward_and_required_gradients_exact=True)
    return dict(source=name, rows=len(packed), cases=report)


def run_checks(reference_dir, checkpoint):
    """Check BF16 decoder math and exact old/new gradient ownership branches."""
    old = load_reference(reference_dir)
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint / 'checkpoint_meta.json').read_text())
    state = torch.load(checkpoint / meta['state_dict_file'], mmap=True,
                       map_location='cpu', weights_only=True)
    specs = [s for s in meta['converted_modules']
             if any(s['name'].startswith(f'model.layers.{i}.') for i in (9, 10))]
    if len(specs) != 14:
        raise ValueError('Expected the fourteen established block-9/10 decoder payloads.')
    qspec = next(s for s in specs if s['name'] == 'model.layers.9.self_attn.q_proj')
    return dict(status='PASS', checkpoint_id=meta['checkpoint_id'],
                linear_ste=check_ste('cuda'),
                gradient_flags=_check_gradient_flags(state, qspec, old),
                linears=[check_decoder(state, spec) for spec in specs])


def _interleaved_timing(old_call, new_call, reset):
    calls = dict(old=old_call, new=new_call)
    times = dict(old=[], new=[])
    for _ in range(2):
        for call in calls.values():
            reset()
            call()
    for repetition in range(5):
        for label in (('old', 'new') if repetition % 2 == 0 else ('new', 'old')):
            reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            calls[label]()
            torch.cuda.synchronize()
            times[label].append(time.perf_counter() - start)
    return dict(seconds=times, median_seconds={k: statistics.median(v) for k, v in times.items()},
                protocol='two warmups each, five interleaved synchronized wall-time repetitions',
                interpretation='component measurement under shared-GPU conditions, not an end-to-end speedup')


@torch.no_grad()
def run_projection_checks(module, old_all_bits):
    """Exercise a real q_proj across chunk boundaries and restore its native state.

    Artificial coordinates below are boundary-test inputs, not evidence of
    training improvements. The caller owns GPU allocation and module lifetime.
    """
    if type(module) is not VAELinear or getattr(module, '_recovery_runtime', None) is not None:
        raise ValueError('Pass an unattached native VAELinear from the real checkpoint.')
    storage = module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)
    if storage.device.type != 'cuda' or len(storage) <= 65536:
        raise ValueError('Expected a CUDA q_proj with more than one 65536-row chunk.')
    original = {key: value.detach().clone() for key, value in module.state_dict().items()}
    trainable = {name: p.requires_grad for name, p in module.named_parameters()}
    flags = {key: getattr(module, key) for key in
             ('training', 'trainable_decode', 'cache_decoded_weight', 'parallel_stage_decode')}
    decoder_checkpoint = all_bits.decoder_for(module).use_checkpoint
    try:
        scores, _ = all_bits.attach(module)
        runtime = module._recovery_runtime
        scores[::17].mul_(-1)
        edges = torch.tensor([-1.5, -1., -.5, 0., .5, 1., 1.5], device=scores.device) * runtime.proxy_scale
        scores[0, 0, :7] = edges
        scores[65536, 0, :7] = edges.flip(0)
        baseline_packed = runtime.initial_packed
        def reset():
            storage.copy_(baseline_packed)
            grouped = getattr(module, '_parallel_stage_grouped_vq_packed', None)
            if grouped is not None:
                grouped.copy_(baseline_packed)
        old_stats = old_all_bits.proxy_statistics(module)
        new_stats = all_bits.proxy_statistics(module)
        if old_stats != new_stats:
            raise AssertionError(f'Statistics differ: {old_stats} versus {new_stats}')
        device_stats = all_bits.proxy_statistics(module, device=True)
        for key, reference in old_stats.items():
            actual = device_stats[key].item()
            if key in ('saturated_fraction', 'proxy_delta_over_initial_margin'):
                if abs(actual - reference) > 1e-14 * max(1., abs(reference)):
                    raise AssertionError(f'Device ratio differs: {key}')
            elif actual != reference:
                raise AssertionError(f'Device statistic differs: {key}')
        reset()
        old_counts = old_all_bits.project(module)
        expected_packed = storage.clone()
        reset()
        new_counts = all_bits.project(module)
        if new_counts != old_counts:
            raise AssertionError('Projection counters differ.')
        torch.testing.assert_close(storage, expected_packed, rtol=0, atol=0)
        reset()
        device_counts = all_bits.project(module, device=True)
        if {k: int(v.item()) for k, v in device_counts.items()} != old_counts:
            raise AssertionError('Device projection counters differ.')
        torch.testing.assert_close(storage, expected_packed, rtol=0, atol=0)
        reset()
        if all_bits.project(module, collect_stats=False) != {}:
            raise AssertionError('No-stat projection returned counters.')
        torch.testing.assert_close(storage, expected_packed, rtol=0, atol=0)
        report = dict(status='PASS', rows=len(storage), counts=old_counts, statistics=new_stats,
                      packed_exact_all_modes=True, boundary_inputs='explicit threshold/saturation values',
                      timing=dict(
                          projection=_interleaved_timing(lambda: old_all_bits.project(module),
                                                        lambda: all_bits.project(module), reset),
                          statistics=_interleaved_timing(lambda: old_all_bits.proxy_statistics(module),
                                                        lambda: all_bits.proxy_statistics(module), reset)))
    finally:
        if getattr(module, '_recovery_runtime', None) is not None:
            all_bits.detach(module)
        module.load_state_dict(original, strict=True)
        # The grouped packed cache is nonpersistent: state_dict restoration does
        # not overwrite the artificial codes projected into it above.
        primary = module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)
        grouped = getattr(module, '_parallel_stage_grouped_vq_packed', None)
        if grouped is not None:
            grouped.copy_(primary)
            torch.testing.assert_close(grouped, primary, rtol=0, atol=0)
        all_bits.decoder_for(module).use_checkpoint = decoder_checkpoint
        for name, p in module.named_parameters():
            p.requires_grad_(trainable[name])
        for key, value in flags.items():
            setattr(module, key, value)
        module.clear_decoded_weight_cache()
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    if type(module) is not VAELinear or getattr(module, '_recovery_runtime', None) is not None:
        raise AssertionError('Recovery adapter or soft proxies survived restoration.')
    if getattr(module, '_cached_weight', None) is not None:
        raise AssertionError('Restoration left a stale decoded-weight cache.')
    report['native_state_restored_exact'] = True
    report['soft_proxies_removed_and_cache_invalidated'] = True
    return report
