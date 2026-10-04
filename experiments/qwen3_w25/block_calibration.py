"""Capture one native Qwen3 block per document without executing later layers."""

from __future__ import annotations

import torch
from transformers import Qwen3ForCausalLM

from experiments.qwen3_w25.local_block import BlockBatch


class _BlockCaptured(Exception):
    """The target block completed; skip the remaining teacher forward."""


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_copy(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported captured block argument: {type(value).__name__}.")


@torch.no_grad()
def capture_block_batches(teacher, block_index: int, batch: dict) -> list[BlockBatch]:
    """Capture BF16 teacher inputs, actual kwargs and outputs on CPU.

    Each document retains the collated sequence length and padding. The block's
    causal mask and RoPE values come from that document's real teacher forward;
    valid_mask only determines which tokens contribute to reconstruction loss.
    """
    if not isinstance(teacher, Qwen3ForCausalLM):
        raise TypeError("Block calibration requires a native Qwen3ForCausalLM teacher.")
    if any(module.training for module in teacher.modules()) or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Block calibration requires an eval-mode, frozen teacher.")
    if type(block_index) is not int or not 0 <= block_index < len(teacher.model.layers):
        raise ValueError("block_index must select an existing teacher decoder layer.")
    if not isinstance(batch, dict) or any(key not in batch for key in ("input_ids", "attention_mask")):
        raise ValueError("A collated batch with input_ids and attention_mask is required.")
    ids, mask = batch["input_ids"], batch["attention_mask"]
    if (not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor)
            or ids.ndim != 2 or ids.shape != mask.shape or not ids.numel()
            or ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Expected nonempty integer input_ids and matching [documents, tokens] attention_mask.")
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.bool().any(dim=1).all()):
        raise ValueError("Each document needs a binary attention_mask with at least one valid token.")
    embedding = teacher.get_input_embeddings().weight
    if embedding.dtype != torch.bfloat16:
        raise ValueError("Block calibration requires the native BF16 teacher path.")
    device = embedding.device
    block = teacher.model.layers[block_index]
    captured = []
    current = {}

    def pre_hook(_module, args, kwargs):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("Expected one positional hidden_states tensor at the Qwen3 block.")
        hidden = args[0]
        if (hidden.shape != (1, ids.shape[1], teacher.config.hidden_size)
                or hidden.dtype != torch.bfloat16 or not bool(torch.isfinite(hidden).all())):
            raise ValueError("Expected finite BF16 teacher block inputs [1, collated_tokens, hidden].")
        if kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False) or kwargs.get("output_attentions", False):
            raise ValueError("Teacher block capture requires no KV cache and output_attentions=False.")
        rope = kwargs.get("position_embeddings")
        if not isinstance(rope, (tuple, list)) or len(rope) != 2 or any(not isinstance(t, torch.Tensor) for t in rope):
            raise ValueError("The teacher did not supply the actual Qwen3 RoPE tensors.")
        if not isinstance(kwargs.get("cache_position"), torch.Tensor):
            raise ValueError("The teacher did not supply its actual cache_position tensor.")
        current["hidden_states"] = _cpu_copy(hidden)
        current["forward_kwargs"] = _cpu_copy(kwargs)

    def post_hook(_module, _args, output):
        if not isinstance(output, tuple) or len(output) != 1 or not isinstance(output[0], torch.Tensor):
            raise ValueError("Expected the native Qwen3 block output tuple without attentions or KV cache.")
        target = output[0]
        if (target.shape != current["hidden_states"].shape or target.dtype != torch.bfloat16
                or not bool(torch.isfinite(target).all())):
            raise ValueError("Expected finite BF16 teacher block outputs matching the captured inputs.")
        captured.append(BlockBatch(
            hidden_states=current["hidden_states"], target_hidden_states=_cpu_copy(target),
            forward_kwargs=current["forward_kwargs"], valid_mask=current["valid_mask"],
        ))
        raise _BlockCaptured()

    pre_handle = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = block.register_forward_hook(post_hook)
    try:
        for index in range(len(ids)):
            current = {"valid_mask": _cpu_copy(mask[index:index + 1].bool())}
            try:
                teacher(input_ids=ids[index:index + 1].to(device),
                        attention_mask=mask[index:index + 1].to(device),
                        use_cache=False, output_attentions=False, output_hidden_states=False)
            except _BlockCaptured:
                if len(captured) != index + 1:
                    raise RuntimeError("Teacher block capture did not produce exactly one document.")
            else:
                raise RuntimeError(f"Teacher decoder layer {block_index} was not executed.")
    finally:
        pre_handle.remove()
        post_handle.remove()
    return captured
