"""Fixed-code gate/down scope ablation on the same frozen real-student contexts."""
from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_student_context_probe.core import (
    Objective as GateObjective, measure as gate_measure, records as module_records,
)
from experiments.joint_reconstruction_probe.core import EPS, LR, error_stats, require, schema, stage_norm, storage_bits
from experiments.rotation_factor_probe.capture import tensor_record


NAMES = ('gate_proj', 'down_proj')
PREFIX = 'model.layers.18.mlp.'
STEPS = 500
BANKS = ('vq_weight', 'vq_weight_s1')


def records(layers):
    return {name + '.' + key: tensor_record(value) for name in NAMES for key, value in layers[name].state_dict().items()}


def ready(layers):
    require(set(layers) == set(NAMES), 'Use exactly the real f927 L18 gate/down pair.')
    for name in NAMES:
        layer = layers[name]
        down = name == 'down_proj'
        in_features, out_features, protected, k = (12288, 4096, 256, 48) if down else (4096, 12288, 64, 32)
        require(layer.transpose == down and layer.residual_stages == 2 and layer.parallel_parts == 1
                and layer._parallel_stage_layout == [(0, 0), (1, 0)]
                and layer.in_features == in_features and layer.out_features == out_features
                and layer.compressed_in_features == in_features - protected
                and layer.compressed_out_features == out_features and layer.stage_codebook_dims == [32, 32]
                and layer.protected_input_indices.numel() == protected
                and layer.protected_channel_quant_format == 'int8'
                and layer.weight_rotation is not None and layer.weight_rotation.block_size == 32
                and getattr(layer, '_sparse_bit_binding', None) is None
                and layer.bias is None and layer.original_weight is None
                and layer.low_rank_a is None and layer.low_rank_b is None,
                'Keep native source geometry, rotation and input protection for ' + name)
        for stage in range(2):
            require(layer.get_stage_part_vq_storage(stage, 0).dtype == torch.uint8
                    and tuple(layer.get_stage_part_vq_storage(stage, 0).shape)
                    == ((in_features - protected) * out_features // 32, 1, k // 8),
                    'Unexpected native packed K32/K48 bank: ' + name)
        layer._decoder_compute_dtype = torch.bfloat16
        layer.clear_decoded_weight_cache()
        layer._build_parallel_stage_decode_plan()


class Objective(GateObjective):
    """Keep the previous target/population/normalizer; only the mutable scope changes."""

    def prediction(self, gate_weight, down_weight):
        require(gate_weight.dtype == down_weight.dtype == torch.bfloat16
                and tuple(gate_weight.shape) == (12288, 4096) and tuple(down_weight.shape) == (4096, 12288),
                'Use both native full original-coordinate matrices without another transpose or inverse.')
        return F.linear(F.silu(F.linear(self.inputs, gate_weight)) * self.up_output, down_weight)

    def forward(self, gate_weight, down_weight):
        raw = F.mse_loss(self.prediction(gate_weight, down_weight).float(), self.reference.float())
        return raw / self.normalizer, raw


@torch.no_grad()
def measure(layers, archive, cache, device):
    ready(layers)
    down = layers['down_proj']._decode_weight(dtype=torch.bfloat16)
    result = gate_measure(layers['gate_proj'], archive, {**cache, 'student_down_weight': down}, device)
    target = archive['targets'][PREFIX + 'down_proj']['reference_weight'].to(device)
    result['down_original_coordinate_weight'] = error_stats(down, target)
    return result


def train(layers, archive, cache, *, mode, steps, deadline, observer):
    require(mode in ('down_only', 'gate_down') and steps in (2, STEPS), 'Use the approved scope and two/500 updates.')
    ready(layers)
    selected = ('down_proj',) if mode == 'down_only' else NAMES
    device = next(layers['gate_proj'].parameters()).device
    require(device.type == 'cuda' and all(next(layer.parameters()).device == device for layer in layers.values()),
            'Place both real projections on the assigned CUDA device.')
    require(all(not p.requires_grad and p.dtype == torch.bfloat16 and p.grad is None
                for layer in layers.values() for p in layer.parameters()), 'Every arm must begin from frozen native BF16 f927 values.')
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    original = records(layers)
    initial = {name: {key: value.detach().clone() for key, value in layers[name]._parallel_stage_decoder.named_parameters()}
               for name in selected}
    original_schema = {name: schema(layers[name]) for name in NAMES}
    original_bits = {name: storage_bits(layers[name]) for name in NAMES}
    banks = {name: [layers[name].get_stage_part_vq_storage(stage, 0).clone() for stage in range(2)] for name in NAMES}
    objective = Objective(archive, cache, device).eval()
    objective_state = module_records(objective)
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        native_gate = layers['gate_proj']._decode_weight(dtype=torch.bfloat16)
        native_down = layers['down_proj']._decode_weight(dtype=torch.bfloat16)
        initial_loss, initial_raw = objective(native_gate, native_down)
        before_native = {'nmse_loss': float(initial_loss), 'raw_mse': float(initial_raw)}
        # The fixed-gate arm can reuse this exact deployed weight; the joint arm never does.
        frozen_gate = native_gate.detach() if mode == 'down_only' else None
    del native_gate, native_down, initial_loss, initial_raw
    for name in selected:
        layer = layers[name]
        for p in layer._parallel_stage_decoder.parameters():
            p.data = p.data.float()
        layer.enable_trainable_decode(parallel_stage_decode=True)
        layer.training = True
        layer._parallel_stage_decoder.eval()
    ready(layers)
    params = {name: dict(layers[name]._parallel_stage_decoder.named_parameters()) for name in selected}
    active = {name + '._parallel_stage_decoder.' + key for name in selected for key in params[name]}
    actual_active = {name + '.' + key for name in NAMES for key, p in layers[name].named_parameters() if p.requires_grad}
    require(active == actual_active and len(active) == 6 * len(selected)
            and all(p.dtype == torch.float32 for values in params.values() for p in values.values()),
            'Only the selected six/twelve FP32 decoder tensors may train.')

    def weights():
        gate = frozen_gate if frozen_gate is not None else layers['gate_proj']._decode_weight(dtype=torch.bfloat16)
        return gate, layers['down_proj']._decode_weight(dtype=torch.bfloat16)

    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        gate, down = weights()
        prediction = objective.prediction(gate, down)
        direct = layers['down_proj'](F.silu(layers['gate_proj'](objective.inputs))
                                     * F.linear(objective.inputs, objective.student_up))
        require(torch.equal(prediction, direct), 'Whole-batch objective differs from native gate/down forward.')
        loss, raw = objective(gate, down)
        before_training = {'nmse_loss': float(loss), 'raw_mse': float(raw)}
        per_document = measure(layers, archive, {**cache,
            'inputs': {'train': cache['inputs']['train']},
            'document_lengths': {'train': cache['document_lengths']['train']}}, device)
    del gate, down, prediction, direct, loss, raw
    optimizer = torch.optim.AdamW([p for values in params.values() for p in values.values()],
                                 lr=LR, betas=(.9, .95), eps=EPS, weight_decay=0)
    maximum, trajectory = {name: [0., 0.] for name in selected}, []
    for step in range(1, steps + 1):
        require(time.time() < deadline, 'The assigned fitting deadline expired; no retry.')
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr'] = LR * (1 - (step - 1) / STEPS)
        with torch.autocast('cuda', enabled=False):
            gate, down = weights()
            loss, raw = objective(gate, down)
        require(bool(torch.isfinite(loss)), 'Nonfinite conditional output loss.')
        loss.backward()
        require(all(p.grad is not None and p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all())
                    for values in params.values() for p in values.values()), 'Missing/invalid selected decoder gradient.')
        require(all(p.grad is None for name in NAMES for key, p in layers[name].named_parameters()
                    if name + '.' + key not in active), 'A frozen tensor received a gradient.')
        norms = {name: stage_norm([p.grad for p in values.values()]) for name, values in params.items()}
        require(all(value > 0 for norm in norms.values() for value in norm), 'Every selected decoder stage needs a real gradient.')
        for name in selected:
            maximum[name] = [max(a, b) for a, b in zip(maximum[name], norms[name], strict=True)]
        optimizer.step()
        torch.cuda.synchronize(device)
        row = {'step': step, 'nmse_loss': float(loss), 'raw_mse': float(raw),
            'lr': optimizer.param_groups[0]['lr'], 'stage_gradient_l2': norms, 'seconds': time.perf_counter() - tick}
        trajectory.append(row)
        if step <= 2 or step % 50 == 0:
            observer(row)
        del gate, down, loss, raw
    require({str(state[key].dtype) for state in optimizer.state.values() for key in ('exp_avg', 'exp_avg_sq')}
            == {'torch.float32'}, 'Adam moments must remain FP32.')
    del optimizer
    fp32_delta = {name: stage_norm([p.detach() - initial[name][key].float() for key, p in params[name].items()]) for name in selected}
    for name in selected:
        for key, p in params[name].items():
            p.grad = None
            p.data = p.data.to(initial[name][key].dtype)
        layers[name].eval().disable_trainable_decode()
        layers[name].requires_grad_(False)
    ready(layers)
    bf16_delta = {name: stage_norm([p.detach().float() - initial[name][key].float() for key, p in params[name].items()]) for name in selected}
    require(all(value > 0 for delta in (fp32_delta, bf16_delta) for norm in delta.values() for value in norm),
            'Each selected stage update must survive BF16 deployment.')
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        loss, raw = objective(layers['gate_proj']._decode_weight(dtype=torch.bfloat16),
                              layers['down_proj']._decode_weight(dtype=torch.bfloat16))
        after = {'nmse_loss': float(loss), 'raw_mse': float(raw)}
    final = records(layers)
    require(all(final[key] == value for key, value in original.items() if key not in active)
            and all(popcounts(old, layers[name].get_stage_part_vq_storage(stage, 0)) == 0
                    for name in NAMES for stage, old in enumerate(banks[name]))
            and {name: schema(layers[name]) for name in NAMES} == original_schema
            and {name: storage_bits(layers[name]) for name in NAMES} == original_bits
            and module_records(objective) == objective_state, 'Frozen codes, protection, rotation, context or storage changed.')
    return {'status': 'completed', 'mode': mode, 'steps': steps, 'schedule_steps': STEPS,
        'before_native': before_native, 'before': before_training, 'after_deployment': after,
        'step0_native_full_forward_exact': True, 'step0_per_document': per_document,
        'step0_whole_minus_document_nmse': before_training['nmse_loss'] - per_document['train']['nmse'],
        'trajectory': trajectory, 'stage_gradient_l2_max': maximum,
        'stage_delta_l2_fp32': fp32_delta, 'stage_delta_l2_bf16': bf16_delta,
        'initial_payload': original, 'final_payload': final, 'selected_decoder_tensors': len(active),
        'frozen_state_exact': True, 'frozen_objective_buffers_exact': True, 'changed_code_bits': 0,
        'persistent_bits': original_bits, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'elapsed_seconds': time.perf_counter() - started,
        'settings': {'lr': LR, 'betas': [.9, .95], 'eps': EPS, 'weight_decay': 0, 'horizon': STEPS, 'warmup': 0,
            'normalizer': objective.normalizer, 'train_tokens': len(objective.inputs),
            'target': 'HF_MLP(frozen f927 student input), same target as the completed gate-only control',
            'population': 'All original train32 valid tokens on every update; heldout/replay only at endpoints',
            'precision': 'FP32 decoder and Adam; native BF16 decode/stage sum/MLP; FP32 MSE',
            'scope': 'Only selected gate/down decoders; original hard codes, up, norm and all other state frozen',
            'joint_gate_output': 'Recomputed from current gate parameters at every optimizer update',
            'boundary': 'Single L18 isolated scope ablation; local gains alone do not authorize all-layer rollout.'}}
