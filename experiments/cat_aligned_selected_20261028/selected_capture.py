"""Observe the actual unchanged F.linear weight; no extra VJP or replacement grad."""
import math
from types import MethodType

from selected_bindings import core, torch


def cpu_sums(value):
    signed = absolute = squared = 0.
    for start in range(0, value.shape[0], 128):
        part = value[start:start + 128].detach().to('cpu').double()
        signed += float(part.sum()); absolute += float(part.abs().sum())
        squared += float(part.square().sum())
    return signed, absolute, squared


class WeightCapture:
    def __init__(self, layer):
        self.layer = layer
        self.original = layer._decode_weight
        self.override = layer.__dict__.get('_decode_weight')
        self.gradient = torch.zeros((4096, 12288), dtype=torch.float32,
            device=next(layer._parallel_stage_decoder.parameters()).device)
        self.weight = None
        self.current, self.handle, self.guard = None, None, False
        self.rows = []
        layer._decode_weight = MethodType(self.decode, layer)

    def decode(self, _layer, *args, **kwargs):
        weight = self.original(*args, **kwargs)
        if self.guard:
            core.require(weight.shape == self.gradient.shape and weight.dtype == torch.bfloat16
                and weight.requires_grad and not weight.is_leaf and weight.grad_fn is not None
                and self.current['registrations'] == 0, 'Capture exactly one original live BF16 down weight per forward.')
            if self.weight is None:
                self.weight = weight.detach().to('cpu', copy=True)
            self.current['registrations'] += 1
            identity = self.current['identity']
            self.handle = weight.register_hook(lambda grad: self.hook(identity, grad))
        return weight

    def hook(self, identity, gradient):
        core.require(not self.guard and self.current is not None and identity == self.current['identity']
            and self.current['callbacks'] == 0 and gradient.shape == self.gradient.shape
            and gradient.dtype == torch.bfloat16, 'Missing/duplicate checkpoint or cross-microbatch gW callback.')
        signed, absolute, _ = cpu_sums(gradient)
        core.require(math.isfinite(signed) and math.isfinite(absolute), 'Nonfinite actual BF16 gW contribution.')
        self.current.update(callbacks=1, BF16_gradient_sum_FP64_CPU=signed,
            BF16_gradient_abs_sum_FP64_CPU=absolute)
        # add_ converts the BF16 source element to the FP32 accumulator without
        # materializing a second full FP32 gradient or changing autograd's input.
        self.gradient.add_(gradient.detach())
        return None

    def begin(self, identity):
        core.require(self.current is None and self.handle is None and not self.guard, 'Prior graph retained.')
        self.current = {'identity': identity, 'registrations': 0, 'callbacks': 0}
        self.guard = True

    def close_forward(self):
        self.guard = False
        core.require(self.current['registrations'] == 1, 'Missing original forward weight hook.')

    def end_backward(self):
        core.require(not self.guard and self.current['callbacks'] == 1, 'One backward must contribute once.')
        self.handle.remove(); self.handle = None
        self.rows.append(self.current); self.current = None

    def proof(self):
        core.require(len(self.rows) == 16 and self.current is None and self.handle is None
            and all(r['registrations'] == r['callbacks'] == 1 for r in self.rows), 'Require M1+QA8+R7 actual contributions.')
        expected = math.fsum(r['BF16_gradient_sum_FP64_CPU'] for r in self.rows)
        absolute = math.fsum(r['BF16_gradient_abs_sum_FP64_CPU'] for r in self.rows)
        actual, _, squared = cpu_sums(self.gradient)
        epsilon = torch.finfo(torch.float32).eps
        bound = 16 * epsilon / (1 - 16 * epsilon) * absolute + 1e-10
        core.require(abs(actual - expected) <= bound, 'Unscaled FP32 gW checksum exceeds sequential accumulation bound.')
        return {'contributions': 16, 'rows': self.rows, 'hook_return': None,
            'extra_whole_backward_or_VJP': False, 'checkpoint_double_registration': False,
            'accumulator_dtype': str(self.gradient.dtype), 'original_weight_dtype': str(self.weight.dtype),
            'callback_sum_FP64_CPU': expected, 'buffer_sum_FP64_CPU': actual,
            'absolute_error': abs(actual - expected), 'FP32_accumulation_bound': bound,
            'gW_l2_FP64_CPU': math.sqrt(squared), 'gW_RMS_FP64_CPU': math.sqrt(squared / self.gradient.numel()),
            'GPU_full_matrix_clone_or_double_per_microbatch': False}

    def remove(self):
        self.guard = False
        if self.handle is not None:
            self.handle.remove(); self.handle = None
        if self.override is None:
            self.layer.__dict__.pop('_decode_weight', None)
        else:
            self.layer._decode_weight = self.override
        self.current = None
