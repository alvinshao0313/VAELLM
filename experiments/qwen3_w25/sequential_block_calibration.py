"""Rolling teacher/student streams for ordered, detached Qwen3 block recovery."""

from __future__ import annotations

import weakref

import torch
from transformers import Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from experiments.qwen3_w25.block_calibration import _cpu_copy
from experiments.qwen3_w25.local_block import BlockBatch, _tree_to
from litebsq.vae_linear import VAELinear


class _InputsCaptured(Exception):
    """Stop the real model immediately before its first decoder block."""


def _require_frozen_eval(module):
    if any(child.training for child in module.modules()) or any(p.requires_grad for p in module.parameters()):
        raise ValueError("Sequential calibration requires eval-mode, frozen models/blocks.")


def _matching_kwargs(left, right, path="kwargs"):
    if type(left) is not type(right):
        raise ValueError(f"Teacher/student {path} types differ.")
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"Teacher/student {path} tensors differ.")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Teacher/student {path} keys differ.")
        for key in left:
            _matching_kwargs(left[key], right[key], path + "." + key)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError(f"Teacher/student {path} lengths differ.")
        for index, (a, b) in enumerate(zip(left, right)):
            _matching_kwargs(a, b, f"{path}[{index}]")
    elif left != right:
        raise ValueError(f"Teacher/student {path} values differ.")


@torch.no_grad()
def _capture_inputs(model, ids, mask):
    embedding = model.get_input_embeddings().weight
    if embedding.dtype != torch.bfloat16:
        raise ValueError("Sequential calibration requires native BF16 embeddings.")
    device = embedding.device
    hidden_states, forward_kwargs = [], []

    def capture(_module, args, kwargs):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("Expected one positional hidden_states tensor at the first Qwen3 block.")
        hidden = args[0]
        if (hidden.shape != (1, ids.shape[1], model.config.hidden_size)
                or hidden.dtype != torch.bfloat16 or not bool(torch.isfinite(hidden).all())):
            raise ValueError("Expected finite BF16 inputs [1, collated_tokens, hidden].")
        if kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False) or kwargs.get("output_attentions", False):
            raise ValueError("Sequential capture requires no KV cache and no attention outputs.")
        rope = kwargs.get("position_embeddings")
        if not isinstance(rope, (tuple, list)) or len(rope) != 2 or any(not isinstance(t, torch.Tensor) for t in rope):
            raise ValueError("The model did not supply its real Qwen3 RoPE tensors.")
        if not isinstance(kwargs.get("cache_position"), torch.Tensor):
            raise ValueError("The model did not supply its real cache_position tensor.")
        hidden_states.append(_cpu_copy(hidden))
        forward_kwargs.append(_cpu_copy(kwargs))
        raise _InputsCaptured()

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for index in range(len(ids)):
            try:
                with torch.autocast(device_type=device.type, enabled=False):
                    model(input_ids=ids[index:index + 1].to(device),
                          attention_mask=mask[index:index + 1].to(device),
                          use_cache=False, output_attentions=False, output_hidden_states=False)
            except _InputsCaptured:
                if len(hidden_states) != index + 1:
                    raise RuntimeError("First-block capture did not produce exactly one document.")
            else:
                raise RuntimeError("The model did not execute its first Qwen3 block.")
    finally:
        handle.remove()
    return hidden_states, forward_kwargs


@torch.no_grad()
def _run_native_block(block, hidden_states, forward_kwargs):
    _require_frozen_eval(block)
    device = next(block.parameters()).device
    if any(t.device != device for t in block.state_dict().values()):
        raise ValueError("Place the current block on one execution device before calibration.")
    projections = [child for child in block.modules() if isinstance(child, VAELinear)]
    for projection in projections:
        if (projection.trainable_decode or getattr(projection, "_sparse_bit_binding", None) is not None
                or "forward" in projection.__dict__
                or getattr(projection, "_decoder_compute_dtype", None) not in (None, torch.bfloat16)):
            raise ValueError("Restore the student projections' native BF16 inference policy before advancing.")
    cache_flags = [projection.cache_decoded_weight for projection in projections]
    outputs = []
    try:
        # Reuse native decoded weights across documents, discarding stale weights
        # from before the caller's decoder update. No differentiable decode here.
        for projection in projections:
            projection.clear_decoded_weight_cache()
            projection.cache_decoded_weight = True
        for hidden, kwargs in zip(hidden_states, forward_kwargs):
            with torch.autocast(device_type=device.type, enabled=False):
                result = block(hidden.detach().to(device), **_tree_to(kwargs, device))
            if not isinstance(result, tuple) or len(result) != 1 or not isinstance(result[0], torch.Tensor):
                raise ValueError("Expected the native Qwen3 block output tuple without KV cache/attentions.")
            value = result[0]
            if value.shape != hidden.shape or value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise ValueError("Expected finite native BF16 block output matching the cached input.")
            outputs.append(_cpu_copy(value))
    finally:
        for projection, flag in zip(projections, cache_flags):
            projection.cache_decoded_weight = flag
            projection.clear_decoded_weight_cache()
    return outputs


class SequentialBlockCalibration:
    """Keep CPU hT/hS caches and advance through every block exactly once.

    ``prepare(teacher.model.layers[i])`` returns detached student-input /
    teacher-target BlockBatch objects for the current layer. After external
    recovery restores the student's native BF16 inference policy and freezes
    its parameters, ``advance(student.model.layers[i])`` propagates its actual
    updated output. Teacher targets always follow the original teacher stream;
    neither stream is periodically reset. The last target is before final norm.

    Initialization executes only embeddings, mask/RoPE construction and the
    first block's pre-hook, separately for both models and every document.
    Qwen3 constructs these kwargs once and shares them across its layers. The
    caller owns model placement; this object keeps only weak layer references
    and detached CPU tensors, and never moves a model or performs optimization.
    """

    def __init__(self, teacher, student, batch: dict):
        if not isinstance(teacher, Qwen3ForCausalLM) or not isinstance(student, Qwen3ForCausalLM):
            raise TypeError("Sequential calibration requires native Qwen3ForCausalLM models.")
        if teacher is student:
            raise ValueError("Teacher and student must be separate model instances.")
        for model in (teacher, student):
            _require_frozen_eval(model)
        if any(isinstance(child, VAELinear) for child in teacher.modules()):
            raise ValueError("Teacher must be the original dense Qwen3 model.")
        config_fields = ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
                         "num_key_value_heads", "head_dim", "max_position_embeddings", "rope_theta",
                         "rope_scaling", "use_sliding_window", "sliding_window", "_attn_implementation")
        if any(getattr(teacher.config, key) != getattr(student.config, key) for key in config_fields):
            raise ValueError("Teacher/student Qwen3 geometry and position/attention configurations must match.")
        if not teacher.model.layers or len(teacher.model.layers) != len(student.model.layers):
            raise ValueError("Teacher/student must have the same nonzero number of blocks.")
        if not isinstance(batch, dict) or any(key not in batch for key in ("input_ids", "attention_mask")):
            raise ValueError("A collated batch with input_ids and attention_mask is required.")
        ids, mask = batch["input_ids"], batch["attention_mask"]
        if (not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor)
                or ids.ndim != 2 or ids.shape != mask.shape or not ids.numel()
                or ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Expected nonempty integer input_ids and matching [documents, tokens] attention_mask.")
        if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.bool().any(dim=1).all()):
            raise ValueError("Each document needs a binary attention_mask with at least one valid token.")
        self._teacher_hidden, self._teacher_kwargs = _capture_inputs(teacher, ids, mask)
        self._student_hidden, self._student_kwargs = _capture_inputs(student, ids, mask)
        for index, (left, right) in enumerate(zip(self._teacher_kwargs, self._student_kwargs)):
            _matching_kwargs(left, right, f"document[{index}].kwargs")
        self._valid_masks = [_cpu_copy(row.unsqueeze(0).bool()) for row in mask]
        self._teacher_blocks = tuple(weakref.ref(block) for block in teacher.model.layers)
        self._student_blocks = tuple(weakref.ref(block) for block in student.model.layers)
        self._layer_index = 0
        self._prepared = False
        self.initialization_passes = {"teacher": len(ids), "student": len(ids)}

    @property
    def layer_index(self):
        return self._layer_index

    @property
    def complete(self):
        return self.layer_index == len(self._teacher_blocks)

    def _check_block(self, block, references):
        if self.complete:
            raise RuntimeError("All sequential blocks have already completed.")
        if not isinstance(block, Qwen3DecoderLayer) or references[self.layer_index]() is not block:
            raise ValueError(f"Expected the original model's block {self.layer_index}; skipping/replacing blocks is unsupported.")

    def prepare(self, teacher_block) -> list[BlockBatch]:
        """Build this layer's fixed student inputs and independent teacher targets."""
        self._check_block(teacher_block, self._teacher_blocks)
        if self._prepared:
            raise RuntimeError("Advance the prepared student block before preparing the next layer.")
        targets = _run_native_block(teacher_block, self._teacher_hidden, self._teacher_kwargs)
        batches = [BlockBatch(hidden_states=_cpu_copy(hidden), target_hidden_states=_cpu_copy(target),
                              forward_kwargs=_cpu_copy(kwargs), valid_mask=_cpu_copy(mask))
                   for hidden, target, kwargs, mask in
                   zip(self._student_hidden, targets, self._student_kwargs, self._valid_masks)]
        self._teacher_hidden = targets
        self._prepared = True
        return batches

    def advance(self, student_block) -> None:
        """Commit the updated native student output, detached from this layer."""
        self._check_block(student_block, self._student_blocks)
        if not self._prepared:
            raise RuntimeError("Prepare the teacher targets before advancing the student block.")
        outputs = _run_native_block(student_block, self._student_hidden, self._student_kwargs)
        self._student_hidden = outputs
        self._layer_index += 1
        self._prepared = False
