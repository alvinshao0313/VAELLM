"""Two detached CPU hidden streams, advanced one real Qwen3 block at a time."""
from __future__ import annotations

from dataclasses import dataclass
import weakref

from .data import EntryPool, InputEntry

import torch
from transformers import Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_vector_rvq.common import digest
from litebsq.vae_linear import VAELinear


PROJECTIONS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
               'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


@dataclass(frozen=True)
class BlockBatch:
    """Read-only views into the current CPU stream; never retain across layers."""
    hidden_states: torch.Tensor
    target_hidden_states: torch.Tensor
    forward_kwargs: dict
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ReconstructionGroup:
    group_index: int
    batches: tuple[BlockBatch, ...]
    valid_tokens: int
    denominator: int


class _InputsCaptured(Exception):
    pass


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_to(item, device) for item in value)
    return value


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_copy(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'Unsupported captured Qwen3 argument: {type(value).__name__}.')


def _matching_kwargs(left, right, path='kwargs'):
    if type(left) is not type(right):
        raise ValueError(f'Teacher/student {path} types differ.')
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right.detach().cpu()):
            raise ValueError(f'Teacher/student {path} tensors differ.')
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f'Teacher/student {path} keys differ.')
        for key in left:
            _matching_kwargs(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError(f'Teacher/student {path} lengths differ.')
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _matching_kwargs(a, b, f'{path}[{index}]')
    elif left != right:
        raise ValueError(f'Teacher/student {path} values differ.')


def _require_frozen_eval(module):
    if any(child.training for child in module.modules()) or any(p.requires_grad for p in module.parameters()):
        raise ValueError('Rolling capture/advance requires frozen eval-mode models and blocks.')


def _validate_entry(entry: InputEntry):
    ids, mask = entry.inputs.get('input_ids'), entry.inputs.get('attention_mask')
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 2 or not ids.numel()
            or not isinstance(mask, torch.Tensor) or mask.shape != ids.shape
            or any(value.device.type != 'cpu' or value.requires_grad or torch.is_inference(value)
                   for value in (*entry.inputs.values(), entry.valid_mask))
            or entry.valid_mask.dtype != torch.bool or not torch.equal(entry.valid_mask, mask.bool())
            or not bool(((mask == 0) | (mask == 1)).all()) or not bool(entry.valid_mask.any(dim=1).all())):
        raise ValueError(f'{entry.entry_id}: require original detached CPU IDs, binary mask and valid tokens.')
    if set(entry.inputs) not in ({'input_ids', 'attention_mask'}, {'input_ids', 'attention_mask', 'position_ids'}):
        raise ValueError(f'{entry.entry_id}: unsupported extra model inputs.')
    if 'position_ids' in entry.inputs:
        positions = entry.inputs['position_ids']
        if positions.dtype != torch.int64 or positions.shape != ids.shape:
            raise ValueError(f'{entry.entry_id}: position_ids must preserve the complete native batch shape.')
    if entry.domain == 'mmlu' and (ids.shape[0] != 32 or 'position_ids' not in entry.inputs):
        raise ValueError('M entries require the actual native M32 batch and explicit position_ids.')
    if entry.domain != 'mmlu' and ids.shape[0] != 1:
        raise ValueError('QA entries require the original complete B1 teacher-forcing input.')


@torch.no_grad()
def _capture_inputs(model, pool, device, expected_kwargs=None):
    embedding = model.get_input_embeddings()
    if embedding.weight.dtype != torch.bfloat16:
        raise ValueError('Capture requires native BF16 embeddings.')
    original_device = embedding.weight.device
    rotary = model.model.rotary_emb
    if (rotary.inv_freq.dtype != torch.float32 or rotary.original_inv_freq.dtype != torch.float32
            or rotary.inv_freq is not rotary.original_inv_freq):
        raise ValueError('Keep the original FP32 root RoPE buffer and original_inv_freq alias.')
    rope_objects = (rotary.inv_freq, rotary.original_inv_freq)
    hidden_states, shared_kwargs = [], [] if expected_kwargs is None else expected_kwargs
    current = None

    def capture(_module, args, kwargs):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError('Expected one positional hidden_states tensor at the real first Qwen3 block.')
        entry = pool.entries[current]
        hidden = args[0]
        if (hidden.shape != (*entry.valid_mask.shape, model.config.hidden_size)
                or hidden.dtype != torch.bfloat16 or not bool(torch.isfinite(hidden).all())):
            raise ValueError(f'{entry.entry_id}: expected the complete finite native BF16 batch.')
        if kwargs.get('past_key_value') is not None or kwargs.get('use_cache', False) or kwargs.get('output_attentions', False):
            raise ValueError('Capture requires no KV cache and no attention outputs.')
        rope = kwargs.get('position_embeddings')
        if (not isinstance(rope, (tuple, list)) or len(rope) != 2
                or any(not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 for value in rope)
                or not isinstance(kwargs.get('cache_position'), torch.Tensor)):
            raise ValueError('Capture requires the actual root Qwen3 RoPE tensors and cache_position.')
        if 'position_ids' in entry.inputs:
            _matching_kwargs(entry.inputs['position_ids'], kwargs.get('position_ids'), f'{entry.entry_id}.position_ids')
        hidden_states.append(_cpu_copy(hidden))
        if expected_kwargs is None:
            shared_kwargs.append(_cpu_copy(kwargs))
        else:
            _matching_kwargs(expected_kwargs[current], kwargs, f'{entry.entry_id}.kwargs')
        raise _InputsCaptured()

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        # Qwen3 root RoPE transfers only its FP32 expanded inv_freq internally.
        # The first-block pre-hook stops before any decoder or lm_head executes.
        embedding.to(device=device)
        for current, entry in enumerate(pool.entries):
            try:
                with torch.inference_mode(False), torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                    model(**tree_to(entry.inputs, device), use_cache=False,
                          output_attentions=False, output_hidden_states=False)
            except _InputsCaptured:
                if len(hidden_states) != current + 1:
                    raise RuntimeError('The real first-block pre-hook did not capture exactly one entry.')
            else:
                raise RuntimeError('The model did not execute its real first Qwen3 block.')
    finally:
        handle.remove()
        embedding.to(device=original_device)
        if rotary.inv_freq is not rope_objects[0] or rotary.original_inv_freq is not rope_objects[1]:
            raise RuntimeError('Capture changed the original root RoPE buffer or alias.')
    return hidden_states, shared_kwargs


def _require_deployed_block(block):
    _require_frozen_eval(block)
    device = next(block.parameters()).device
    if any(value.device != device for value in (*block.parameters(), *block.buffers())):
        raise ValueError('Place the complete current block on one execution device before rolling.')
    if any(p.dtype != torch.bfloat16 or p.grad is not None for p in block.parameters()):
        raise ValueError('Commit actual frozen BF16 block parameters and clear training gradients before rolling.')
    for module in block.modules():
        if isinstance(module, VAELinear):
            if (module.trainable_decode or not module.cache_decoded_weight
                    or getattr(module, '_decoder_compute_dtype', None) != torch.bfloat16
                    or 'forward' in module.__dict__ or getattr(module, '_sparse_bit_binding', None) is not None):
                raise ValueError('Restore native BF16 projection policy before rolling; caller owns group8 cache preparation.')
            cache = module._cached_weight
        elif isinstance(module, CompactRVQLinear):
            if 'forward' in module.__dict__:
                raise ValueError('Restore the real compact projection forward before rolling.')
            cache = module._decoded_cache
            versions = tuple(value._version for value in (*module.parameters(), *module.buffers()))
            if module._cache_versions != versions:
                raise ValueError('Caller must prepare current compact BF16 deployment caches before rolling.')
        else:
            continue
        if (cache is None or cache.dtype != torch.bfloat16 or cache.device != device
                or cache.requires_grad or cache.grad_fn is not None or torch.is_inference(cache)):
            raise ValueError('Caller must prepare ordinary no_grad BF16 deployment caches before rolling.')
    return device


def tensor_record(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest({'value': value}))


def _cpu_storage(value):
    return dict(tensor_identity=id(value), data_ptr=value.data_ptr(),
                storage_data_ptr=value.untyped_storage().data_ptr(),
                storage_bytes=value.untyped_storage().nbytes())


@torch.no_grad()
def _advance_in_place(block, hidden_states, forward_kwargs, pool, targets=None):
    device = _require_deployed_block(block)
    rows = []
    # Keep CPU storage identities and replace each value before moving on. There
    # is never a third pool of targets or a list of all GPU block outputs.
    with torch.inference_mode(False), torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for index, (hidden, kwargs, entry) in enumerate(zip(hidden_states, forward_kwargs, pool.entries, strict=True)):
            before = _cpu_storage(hidden)
            row = dict(entry_index=index, entry_id=entry.entry_id, domain=entry.domain,
                input=tensor_record(hidden), valid_tokens=entry.valid_tokens,
                padded_tokens=entry.padded_tokens, CPU_storage_before=before)
            result = block(hidden.detach().to(device), **tree_to(kwargs, device))
            if not isinstance(result, tuple) or len(result) != 1 or not isinstance(result[0], torch.Tensor):
                raise ValueError('Expected native Qwen3 complete block output without attention or KV outputs.')
            value = result[0]
            if value.shape != hidden.shape or value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise ValueError('Expected finite BF16 complete block output with the original entry shape.')
            if targets is not None:
                target_cpu = targets[index]
                if (target_cpu.shape != hidden.shape or target_cpu.dtype != torch.bfloat16
                        or target_cpu.device.type != 'cpu' or target_cpu.requires_grad):
                    raise ValueError('Compare only to the independently advanced detached CPU BF16 teacher target.')
                target = target_cpu.detach().to(device)
                # Match fit.entry_objective's original valid-token mask and
                # FP32 reduction exactly; do not normalize individual entries.
                mask = entry.valid_mask.to(device)
                difference = (value.float() - target.float()).masked_fill(~mask.unsqueeze(-1), 0.)
                sse = difference.square().sum(dtype=torch.float32)
                energy = target.float().masked_fill(~mask.unsqueeze(-1), 0.).square().sum(dtype=torch.float32)
                if not bool(torch.isfinite(sse)) or not bool(torch.isfinite(energy)):
                    raise FloatingPointError('Nonfinite FP32 valid-token deployed block statistics.')
                row.update(squared_error=float(sse), teacher_squared_energy=float(energy),
                           teacher_target=tensor_record(target_cpu))
                del target, mask, difference, sse, energy
            hidden.copy_(value.detach())
            del result, value
            after = _cpu_storage(hidden)
            if (after != before or hidden.device.type != 'cpu' or hidden.requires_grad
                    or hidden.grad_fn is not None or torch.is_inference(hidden)):
                raise RuntimeError('Rolling must preserve ordinary detached CPU hidden storage and data pointers.')
            row.update(output=tensor_record(hidden), CPU_storage_after=after, CPU_storage_preserved=True)
            rows.append(row)
    return dict(entries=rows, valid_tokens=sum(row['valid_tokens'] for row in rows),
        padded_tokens=sum(row['padded_tokens'] for row in rows), CPU_hidden_storage_preserved=True,
        ordinary_detached_CPU_hidden=True, each_unique_entry_advanced_once=True)


def _group_statistics(pool, rows, hidden_size):
    groups = []
    for group in pool.groups:
        entries = [rows[index] for index in group.entry_indices]
        sse = sum(row['squared_error'] for row in entries)
        energy = sum(row['teacher_squared_energy'] for row in entries)
        denominator = group.denominator(hidden_size)
        groups.append(dict(group_index=group.group_index, valid_tokens=group.valid_tokens,
            denominator=denominator, MSE=sse/denominator, teacher_mean_square=energy/denominator,
            normalized_squared_error=sse/energy if energy else None, entries=entries))
    return groups


class RollingStreams:
    """Keep T_i/S_i on CPU; group() lends read-only references to T_i+1/S_i.

    The caller moves one block at a time and owns deployment cache preparation,
    including category group8 peers. This component neither clears nor primes
    any projection cache. Failed partial advances invalidate this object.
    """

    @classmethod
    def capture(cls, teacher, student, entries: EntryPool, *, device='cuda:0'):
        if (not isinstance(teacher, Qwen3ForCausalLM) or not isinstance(student, Qwen3ForCausalLM)
                or teacher is student):
            raise TypeError('Capture requires separate real Qwen3ForCausalLM teacher/student instances.')
        for model in (teacher, student):
            _require_frozen_eval(model)
        fields = ('num_hidden_layers', 'hidden_size', 'intermediate_size', 'num_attention_heads',
                  'num_key_value_heads', 'head_dim', 'max_position_embeddings', 'rope_theta',
                  'rope_scaling', 'use_sliding_window', 'sliding_window', '_attn_implementation')
        if any(getattr(teacher.config, key) != getattr(student.config, key) for key in fields):
            raise ValueError('Teacher/student geometry, positional and attention configuration must match.')
        if not teacher.model.layers or len(teacher.model.layers) != len(student.model.layers):
            raise ValueError('Teacher/student require the same nonzero real block count.')
        if any(type(block.get_submodule(name)) is not torch.nn.Linear
               for block in teacher.model.layers for name in PROJECTIONS):
            raise ValueError('Teacher must be the original full dense Qwen3 model.')
        if not isinstance(entries, EntryPool) or not entries.entries or not entries.groups:
            raise TypeError('Capture requires the fixed nonempty H96 EntryPool.')
        for entry in entries.entries:
            _validate_entry(entry)
        for group in entries.groups:
            if (len(group.entry_indices) != 9 or len(set(group.entry_indices)) != 9
                    or any(not 0 <= index < len(entries.entries) for index in group.entry_indices)
                    or group.valid_tokens != sum(entries.entries[index].valid_tokens for index in group.entry_indices)):
                raise ValueError('Each original group must reference one M32 and eight QA entries with its real token count.')
        device = torch.device(device)
        result = cls()
        with torch.inference_mode(False), torch.no_grad():
            result._teacher_hidden, result._kwargs = _capture_inputs(teacher, entries, device)
            result._student_hidden, shared = _capture_inputs(student, entries, device, result._kwargs)
        if shared is not result._kwargs:
            raise RuntimeError('Teacher/student kwargs must share one exact CPU copy.')
        result.entries = entries
        result.hidden_size = teacher.config.hidden_size
        result._teacher_blocks = tuple(weakref.ref(block) for block in teacher.model.layers)
        result._student_blocks = tuple(weakref.ref(block) for block in student.model.layers)
        result._layer_index, result._prepared, result._failed = 0, False, False
        result.initialization_passes = dict(teacher=len(entries.entries), student=len(entries.entries))
        return result

    @property
    def layer_index(self):
        return self._layer_index

    @property
    def complete(self):
        return self.layer_index == len(self._teacher_blocks)

    def _check_block(self, index, block, references):
        if self._failed:
            raise RuntimeError('A partially advanced stream failed; recapture fresh streams before continuing.')
        if self.complete:
            raise RuntimeError('All sequential blocks have already completed.')
        if (type(index) is not int or index != self.layer_index or not isinstance(block, Qwen3DecoderLayer)
                or references[index]() is not block):
            raise ValueError(f'Expected the original model block {self.layer_index}; no skip or replacement is supported.')

    def prepare_layer(self, index, teacher_block):
        self._check_block(index, teacher_block, self._teacher_blocks)
        if self._prepared:
            raise RuntimeError('Commit the prepared student block before preparing another layer.')
        try:
            evidence = _advance_in_place(teacher_block, self._teacher_hidden, self._kwargs, self.entries)
        except BaseException:
            self._failed = True
            raise
        self._prepared = True
        return dict(schema='BR_independent_teacher_block_advance_v1', layer=index,
            source='Independent original BF16 teacher prefix; never the student stream.', **evidence)

    def group(self, index) -> ReconstructionGroup:
        if self._failed or not self._prepared:
            raise RuntimeError('Prepare the current independent teacher layer before requesting training groups.')
        group = self.entries.group(index)
        batches = tuple(BlockBatch(self._student_hidden[i], self._teacher_hidden[i],
            self._kwargs[i], self.entries.entries[i].valid_mask) for i in group.entry_indices)
        return ReconstructionGroup(index, batches, group.valid_tokens, group.denominator(self.hidden_size))

    def commit_and_advance(self, index, student_block):
        self._check_block(index, student_block, self._student_blocks)
        if not self._prepared:
            raise RuntimeError('Prepare independent teacher targets before committing the student block.')
        try:
            evidence = _advance_in_place(student_block, self._student_hidden, self._kwargs,
                                         self.entries, self._teacher_hidden)
            after_deployed = _group_statistics(self.entries, evidence['entries'], self.hidden_size)
        except BaseException:
            self._failed = True
            raise
        self._layer_index += 1
        self._prepared = False
        return dict(schema='BR_committed_student_block_advance_v1', layer=index,
            target_source='Current independent teacher stream T_i+1; BF16 committed student output S_i+1.',
            after_deployed=after_deployed, **evidence)
