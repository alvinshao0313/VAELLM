"""Group compatible captured Qwen3 documents without changing their positions."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from experiments.qwen3_w25.local_block import BlockBatch


_KWARGS = {
    "attention_mask", "position_ids", "past_key_value", "output_attentions",
    "use_cache", "cache_position", "position_embeddings",
}


def _signature(batch):
    if not isinstance(batch, BlockBatch):
        raise TypeError("Expected captured BlockBatch documents.")
    x, y, valid, kwargs = (batch.hidden_states, batch.target_hidden_states,
                           batch.valid_mask, batch.forward_kwargs)
    if (not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor)
            or x.ndim != 3 or x.shape[0] != 1 or x.shape != y.shape or not x.numel()
            or x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16):
        raise ValueError("Expected single-document BF16 hidden/target tensors with matching [1,L,H] shapes.")
    if (not isinstance(valid, torch.Tensor) or valid.shape != x.shape[:2]
            or not bool(((valid == 0) | (valid == 1)).all()) or not bool(valid.any())):
        raise ValueError("Expected a nonempty binary valid_mask for each document.")
    if not isinstance(kwargs, dict) or set(kwargs) != _KWARGS:
        raise ValueError("Only the seven actually captured Qwen3 block kwargs are supported.")
    if (kwargs["past_key_value"] is not None or kwargs["use_cache"] is not False
            or kwargs["output_attentions"] is not False):
        raise ValueError("Captured documents must disable KV cache and output_attentions.")
    length = x.shape[1]
    positions, cache, rope = (kwargs["position_ids"], kwargs["cache_position"], kwargs["position_embeddings"])
    if (not isinstance(positions, torch.Tensor) or positions.shape != (1, length)
            or positions.dtype not in (torch.int32, torch.int64)
            or not isinstance(cache, torch.Tensor) or cache.shape != (length,)
            or cache.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Expected captured integer position_ids [1,L] and cache_position [L].")
    if (not isinstance(rope, (tuple, list)) or len(rope) != 2
            or any(not isinstance(t, torch.Tensor) or t.ndim != 3 or t.shape[:2] != (1, length)
                   or t.dtype != torch.bfloat16 for t in rope) or rope[0].shape != rope[1].shape):
        raise ValueError("Expected the two captured BF16 RoPE tensors [1,L,head_dim].")
    tensors = [x, y, valid, positions, cache, *rope]
    mask = kwargs["attention_mask"]
    if mask is None:
        if not bool(valid.all()):
            raise ValueError("A missing attention mask requires an entirely valid document.")
        mask_signature = None
    elif isinstance(mask, torch.Tensor):
        if mask.ndim == 2:
            if mask.shape != (1, length) or not torch.equal(mask, valid):
                raise ValueError("The captured two-dimensional attention mask must match valid_mask.")
        elif mask.ndim == 4:
            if mask.shape != (1, 1, length, length) or mask.dtype != torch.bfloat16:
                raise ValueError("Expected the no-cache BF16 eager causal mask [1,1,L,L].")
        else:
            raise ValueError("Only None, FA2 two-dimensional, or eager four-dimensional masks are supported.")
        tensors.append(mask)
        mask_signature = (tuple(mask.shape[1:]), mask.dtype)
    else:
        raise TypeError("Invalid captured attention mask type.")
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("Combine the captured CPU documents before moving them to the execution device.")
    return (tuple(x.shape[1:]), valid.dtype, positions.dtype, cache.dtype,
            type(rope), tuple(rope[0].shape[1:]), mask_signature)


def _combine(documents):
    def cat(values):
        return torch.cat([value.detach() for value in values], dim=0)

    first = documents[0].forward_kwargs
    kwargs = {
        "attention_mask": None if first["attention_mask"] is None else cat(
            [batch.forward_kwargs["attention_mask"] for batch in documents]),
        "position_ids": cat([batch.forward_kwargs["position_ids"] for batch in documents]),
        "cache_position": first["cache_position"].detach().clone(),
        "position_embeddings": type(first["position_embeddings"])(
            cat([batch.forward_kwargs["position_embeddings"][index] for batch in documents]) for index in range(2)),
        "past_key_value": None, "output_attentions": False, "use_cache": False,
    }
    return BlockBatch(
        hidden_states=cat([batch.hidden_states for batch in documents]),
        target_hidden_states=cat([batch.target_hidden_states for batch in documents]),
        forward_kwargs=kwargs, valid_mask=cat([batch.valid_mask for batch in documents]),
    )


def batch_block_documents(batches: Sequence[BlockBatch], *, microbatch_size: int = 8) -> list[BlockBatch]:
    """Concatenate globally compatible CPU documents on the batch axis.

    Groups follow their first occurrence; each group's documents retain their
    original order. Full padded lengths, per-document RoPE and masks are retained.
    None/2D/4D masks and different non-batched cache_position values stay separate.
    A microbatch size of one retains the original document execution order.
    No positions or teacher values are recomputed. The caller still accumulates
    every document's valid-token error before its single optimizer step; grouping
    changes execution and floating-point reduction order, not that objective.

    Keep the original per-document batches for pilot selection/confirmation
    reporting; this helper changes only the execution batch of the block core.
    """
    if type(microbatch_size) is not int or microbatch_size < 1:
        raise ValueError("microbatch_size must be a positive integer.")
    if not batches:
        raise ValueError("At least one captured document is required.")
    groups = {}
    for batch in batches:
        signature = (_signature(batch), tuple(batch.forward_kwargs["cache_position"].tolist()))
        groups.setdefault(signature, []).append(batch)
    if microbatch_size == 1:
        return [_combine([batch]) for batch in batches]
    return [_combine(documents[start:start + microbatch_size])
            for documents in groups.values()
            for start in range(0, len(documents), microbatch_size)]
