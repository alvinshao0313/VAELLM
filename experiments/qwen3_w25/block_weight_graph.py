"""One decoder graph per block step, with FP32 weight-gradient accumulation."""

from contextlib import contextmanager
from types import MethodType

import torch
from torch.nn import functional as F


class BlockWeightGraph:
    """Transient decoded weights; never registered in the model or optimizer."""

    def __init__(self, decoded, leaves, requires_grad):
        self.decoded = decoded
        self.leaves = leaves
        self.requires_grad = requires_grad
        self._backward_done = False

    def backward_to_decoders(self):
        if not self.requires_grad or self._backward_done:
            raise RuntimeError("A training weight graph supports exactly one decoder backward.")
        gradients = []
        for name, leaf in self.leaves.items():
            gradient = leaf.grad
            if gradient is None or gradient.dtype != torch.float32 or not torch.isfinite(gradient).all():
                raise RuntimeError(f"{name}: missing, non-FP32 or non-finite accumulated weight gradient.")
            gradients.append(gradient)
        # FP32 accumulation happens before this VJP. Summing gradients on a
        # BF16 leaf would add rounding absent from the original per-batch path.
        with torch.autocast(device_type=gradients[0].device.type, enabled=False):
            torch.autograd.backward(list(self.decoded.values()), gradients)
        self._backward_done = True


def _leaf_forward(leaf):
    def forward(module, inputs):
        if inputs.dtype != torch.bfloat16:
            raise ValueError("Block projections must retain BF16 activations.")
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            bias = None if module.bias is None else module.bias.to(torch.bfloat16)
            return F.linear(inputs, leaf.to(torch.bfloat16), bias)
    return forward


@contextmanager
def block_weight_graph(targets, *, requires_grad):
    """Decode validated local-block targets once and temporarily project with leaves.

    The caller owns source validation, frozen parameters, FP32 decoder setup,
    loss weighting, zero_grad and optimizer.step. With requires_grad=True, call
    backward on each batch loss and then backward_to_decoders once, before any
    parameter update. With False, all batches share fixed decoded weights for
    endpoint measurement. No model state, parameter identity or dtype is changed.
    The context removes temporary forwards and graph references on every exit.
    """
    targets = list(targets)
    if not targets or len({name for name, _ in targets}) != len(targets):
        raise ValueError("Nonempty uniquely named projection targets are required.")
    for name, module in targets:
        if "forward" in module.__dict__:
            raise ValueError(f"{name}: an instance-specific projection forward is already installed.")
        if not module.trainable_decode or module.cache_decoded_weight or module._decoder_compute_dtype != torch.float32:
            raise ValueError(f"{name}: prepare trainable FP32 decode without a weight cache first.")
    decoded, leaves, installed = {}, {}, []
    graph = BlockWeightGraph(decoded, leaves, requires_grad)
    try:
        for name, module in targets:
            device = next(module._parallel_stage_decoder.parameters()).device
            with torch.set_grad_enabled(requires_grad), torch.autocast(device_type=device.type, enabled=False):
                weight = module._decode_weight(dtype=torch.float32, include_low_rank=False)
            if weight.dtype != torch.float32 or bool(weight.requires_grad) != requires_grad:
                raise RuntimeError(f"{name}: expected an FP32 decode with requires_grad={requires_grad}.")
            decoded[name] = weight
            leaves[name] = weight.detach().requires_grad_(requires_grad)
            module.forward = MethodType(_leaf_forward(leaves[name]), module)
            installed.append(module)
        yield graph
    finally:
        for module in installed:
            del module.forward
        for leaf in leaves.values():
            leaf.grad = None
        leaves.clear()
        decoded.clear()
