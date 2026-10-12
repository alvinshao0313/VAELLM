"""The existing fixed-code conditional gate fit with explicit per-layer inputs."""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_student_context_probe.core import Objective as OriginalObjective, records
from experiments.joint_reconstruction_probe.core import EPS, LR, error_stats, require, schema, stage_norm, storage_bits
from experiments.rotation_factor_probe.metrics import aggregate, squared_error


TARGETS = tuple(f'model.layers.{index}.mlp.gate_proj' for index in (0, 9, 18, 27))
STEPS = 500


def ready(layer):
    require(not layer.transpose and layer.residual_stages == 2 and layer.parallel_parts == 1
            and layer._parallel_stage_layout == [(0, 0), (1, 0)]
            and layer.in_features == 4096 and layer.out_features == 12288
            and layer.compressed_in_features == 4032 and layer.compressed_out_features == 12288
            and layer.stage_codebook_dims == [32, 32] and layer.protected_input_indices.numel() == 64
            and layer.protected_channel_quant_format == 'int8'
            and layer.weight_rotation is not None and layer.weight_rotation.block_size == 32
            and getattr(layer, '_sparse_bit_binding', None) is None
            and layer.bias is None and layer.original_weight is None
            and layer.low_rank_a is None and layer.low_rank_b is None,
            'Require the real f927 gate geometry, its own block32 rotation and INT8 protection.')
    for stage in range(2):
        bank = layer.get_stage_part_vq_storage(stage, 0)
        require(bank.dtype == torch.uint8 and tuple(bank.shape) == (12288 * 4032 // 32, 1, 4),
                'Keep the native two K32 packed code banks.')
    require(len(layer.state_dict()) == 13 and len(list(layer._parallel_stage_decoder.parameters())) == 6,
            'The current gate inventory is thirteen tensors including six decoder parameters.')
    layer._decoder_compute_dtype = torch.bfloat16
    layer.clear_decoded_weight_cache()
    layer._build_parallel_stage_decode_plan()


def check_target(archive, cache, target_name):
    require(target_name in TARGETS and cache['target'] == target_name
            and target_name in archive['targets']
            and target_name.replace('gate_proj', 'down_proj') in archive['targets'],
            'Use the explicit matching layer archive and frozen student context.')


class Objective(OriginalObjective):
    """Reuse the exact original prediction/loss; bind this layer's HF weights."""

    def __init__(self, archive, cache, target_name, device):
        nn.Module.__init__(self)
        check_target(archive, cache, target_name)
        self.lengths = cache['document_lengths']['train']
        x = cache['inputs']['train'].to(device)
        require(len(self.lengths) == 32 and sum(self.lengths) == len(x) and x.dtype == torch.bfloat16,
                'Fit exactly the original32 documents, using their valid student-prefix tokens.')
        self.register_buffer('inputs', x)
        self.register_buffer('student_up', cache['student_up_weight'].to(device))
        self.register_buffer('student_down', cache['student_down_weight'].to(device))
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            gate = archive['targets'][target_name]['reference_weight'].to(device)
            up = archive['up_proj_weight'].to(device)
            down = archive['targets'][target_name.replace('gate_proj', 'down_proj')]['reference_weight'].to(device)
            require(all(value.dtype == torch.bfloat16 and not value.requires_grad
                        for value in (x, gate, up, down, self.student_up, self.student_down)),
                    'Real per-layer HF targets and fixed student context must remain BF16 and frozen.')
            self.register_buffer('up_output', F.linear(x, self.student_up))
            self.register_buffer('reference', F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down))
        self.normalizer = float(self.reference.double().square().sum()) / self.reference.numel()
        require(self.normalizer > 0 and math.isfinite(self.normalizer) and not list(self.parameters()),
                'Conditional HF target and frozen context must have finite nonzero energy.')


@torch.no_grad()
def measure(layer, archive, cache, target_name, device):
    """The unchanged per-document endpoint metric at an explicitly named gate."""
    check_target(archive, cache, target_name)
    ready(layer)
    weight = layer._decode_weight(dtype=torch.bfloat16)
    gate = archive['targets'][target_name]['reference_weight'].to(device)
    up = archive['up_proj_weight'].to(device)
    down = archive['targets'][target_name.replace('gate_proj', 'down_proj')]['reference_weight'].to(device)
    student_up, student_down = (cache[k].to(device) for k in ('student_up_weight', 'student_down_weight'))
    result = {'original_coordinate_weight': error_stats(weight, gate)}
    with torch.autocast(device_type=device.type, enabled=False):
        for split, values in cache['inputs'].items():
            documents, offset = [], 0
            for index, length in enumerate(cache['document_lengths'][split]):
                x = values[offset:offset + length].to(device)
                reference = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
                candidate = F.linear(F.silu(F.linear(x, weight)) * F.linear(x, student_up), student_down)
                error, signal = squared_error(candidate, reference)
                require(signal > 0 and math.isfinite(error), 'Invalid conditional endpoint energy.')
                documents.append({'index': index, 'tokens': length, 'squared_error': error,
                                  'squared_reference': signal, 'nmse': error / signal})
                offset += length
            require(offset == len(values) and len(documents) == 32, 'Incomplete endpoint population.')
            result[split] = aggregate(documents)
    return result


def train(layer, archive, cache, target_name, *, deadline, observer):
    check_target(archive, cache, target_name)
    require(target_name != 'model.layers.18.mlp.gate_proj', 'Reuse the existing L18 endpoint; do not train it again.')
    ready(layer)
    device = next(layer.parameters()).device
    require(device.type == 'cuda' and not layer.training and all(not p.requires_grad for p in layer.parameters()),
            'Start from the assigned CUDA device and a frozen fresh f927 gate.')
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    original, before_schema, before_bits = records(layer), schema(layer), storage_bits(layer)
    banks = [layer.get_stage_part_vq_storage(i, 0).clone() for i in range(2)]
    initial = {name: p.detach().clone() for name, p in layer._parallel_stage_decoder.named_parameters()}
    require(len(initial) == 6 and all(value.dtype == torch.bfloat16 for value in initial.values()),
            'All six decoder tensors must start from real f927 BF16 values.')
    objective = Objective(archive, cache, target_name, device).eval()
    fixed = records(objective)
    for p in layer._parallel_stage_decoder.parameters():
        p.data = p.data.float()
    layer.enable_trainable_decode(parallel_stage_decode=True)
    layer.training = True
    layer._parallel_stage_decoder.eval()
    ready(layer)
    params = dict(layer._parallel_stage_decoder.named_parameters())
    active = {'_parallel_stage_decoder.' + key for key in params}
    require({name for name, p in layer.named_parameters() if p.requires_grad} == active
            and all(p.dtype == torch.float32 for p in params.values()), 'Only the six FP32 decoder tensors may train.')
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        weight = layer._decode_weight(dtype=torch.bfloat16)
        prediction = objective.prediction(weight)
        direct = F.linear(F.silu(layer(objective.inputs)) * F.linear(objective.inputs, objective.student_up), objective.student_down)
        require(torch.equal(prediction, direct), 'Restored weight/context differs from same-shape native VAELinear forward.')
        loss, raw = objective(weight)
        before = {'nmse_loss': float(loss), 'raw_mse': float(raw)}
        per_document = measure(layer, archive, {**cache, 'inputs': {'train': cache['inputs']['train']},
            'document_lengths': {'train': cache['document_lengths']['train']}}, target_name, device)
    del weight, prediction, direct, loss, raw
    optimizer = torch.optim.AdamW(params.values(), lr=LR, betas=(.9, .95), eps=EPS, weight_decay=0)
    trajectory, maximum = [], [0., 0.]
    for step in range(1, STEPS + 1):
        require(time.time() < deadline, 'The assigned fitting deadline expired; no retry.')
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr'] = LR * (1 - (step - 1) / STEPS)
        with torch.autocast('cuda', enabled=False):
            weight = layer._decode_weight(dtype=torch.bfloat16)
            loss, raw = objective(weight)
        require(bool(torch.isfinite(loss)), 'Nonfinite conditional output loss.')
        loss.backward()
        require(all(p.grad is not None and p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all())
                    for p in params.values()) and all(p.grad is None for name, p in layer.named_parameters() if name not in active),
                'Missing/invalid FP32 decoder gradient or a gradient on frozen state.')
        norms = stage_norm([p.grad for p in params.values()])
        require(all(value > 0 for value in norms), 'Both stages need real nonzero output gradients.')
        maximum = [max(a, b) for a, b in zip(maximum, norms, strict=True)]
        optimizer.step()
        torch.cuda.synchronize(device)
        row = {'step': step, 'nmse_loss': float(loss), 'raw_mse': float(raw),
               'lr': optimizer.param_groups[0]['lr'], 'stage_gradient_l2': norms, 'seconds': time.perf_counter() - tick}
        trajectory.append(row)
        if step <= 2 or step % 50 == 0:
            observer(row)
        del weight, loss, raw
    require({str(state[key].dtype) for state in optimizer.state.values() for key in ('exp_avg', 'exp_avg_sq')}
            == {'torch.float32'}, 'Adam moments must remain FP32.')
    with torch.no_grad():
        live = layer._decode_weight(dtype=torch.bfloat16)
        ready(layer)
        require(torch.equal(live, layer._decode_weight(dtype=torch.bfloat16)), 'Packed plan rebuild changed live decode.')
    del optimizer, live
    delta_fp32 = stage_norm([p.detach() - initial[key].float() for key, p in params.items()])
    for key, p in params.items():
        p.grad = None
        p.data = p.data.to(initial[key].dtype)
    layer.eval().disable_trainable_decode()
    layer.requires_grad_(False)
    ready(layer)
    delta_bf16 = stage_norm([p.detach().float() - initial[key].float() for key, p in params.items()])
    with torch.no_grad():
        deployed = layer._decode_weight(dtype=torch.bfloat16)
        loss, raw = objective(deployed)
    final = records(layer)
    require(all(final[key] == value for key, value in original.items() if key not in active)
            and all(popcounts(value, layer.get_stage_part_vq_storage(i, 0)) == 0 for i, value in enumerate(banks))
            and schema(layer) == before_schema and storage_bits(layer) == before_bits and records(objective) == fixed,
            'Frozen codes/context/protection/rotation or persistent storage changed.')
    require(all(value > 0 for value in delta_fp32 + delta_bf16), 'Both stage updates must survive BF16 deployment.')
    return {'status': 'completed', 'target_name': target_name, 'steps': STEPS, 'schedule_steps': STEPS,
        'before': before, 'after_deployment': {'nmse_loss': float(loss), 'raw_mse': float(raw)},
        'step0_native_full_forward_exact': True, 'step0_per_document': per_document,
        'step0_whole_minus_document_nmse': before['nmse_loss'] - per_document['train']['nmse'],
        'trajectory': trajectory, 'stage_gradient_l2_max': maximum,
        'stage_delta_l2_fp32': delta_fp32, 'stage_delta_l2_bf16': delta_bf16,
        'initial_payload': original, 'final_payload': final, 'selected_decoder_tensors': 6,
        'frozen_state_exact': True, 'frozen_objective_buffers_exact': True, 'changed_code_bits': 0,
        'persistent_bits': before_bits, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'elapsed_seconds': time.perf_counter() - started,
        'settings': {'lr': LR, 'betas': [.9, .95], 'eps': EPS, 'weight_decay': 0, 'horizon': STEPS,
            'warmup': 0, 'normalizer': objective.normalizer, 'train_tokens': len(objective.inputs),
            'target': 'HF_MLP(frozen f927 student input), not independent full-teacher-forward output',
            'student_context': 'Actual fixed f927 compressed up/down; full native gate decode includes rotation and INT8',
            'population': 'All original train32 valid tokens on every update; no heldout/replay in gradients',
            'precision': 'FP32 decoder and Adam; native BF16 decode/stage sum/MLP; FP32 MSE',
            'boundary': 'One named gate fitted independently; local gains across layers do not establish additive downstream gains.'}}
