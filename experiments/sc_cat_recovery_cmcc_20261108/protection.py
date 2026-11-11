"""Joint signed-INT8 STE and positive BF16 scale, owned by each native projection."""
from types import MethodType

import torch
from torch import nn

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require


def integer_coordinate(master):
    clipped = master.clamp(-128., 127.)
    return clipped + (clipped.round() - clipped).detach()


def effective_scale(source, delta):
    value = source.float() * delta.exp()
    hard = value.bfloat16()
    require(bool(torch.isfinite(value).all() and (value > 0).all()
        and torch.isfinite(hard).all() and (hard > 0).all()), 'Protection scale must remain finite and positive.')
    return hard


class Protection:
    def __init__(self, modules):
        self.modules = dict(modules)
        self.calls = {name: 0 for name in modules}
        for name, module in self.modules.items():
            q, s, index = module.protected_input_qvalues, module.protected_input_scales, module.protected_input_indices
            require(module.protected_channel_quant_format == 'int8' and q.dtype == torch.uint8
                and q.shape == (index.numel(), module.out_features) and s.shape == index.shape
                and s.dtype == torch.bfloat16 and index.numel() > 0 and module.protected_output_indices is None,
                'Unsupported native protection topology: ' + name)
            require('_decode_protected_input_weight_rows' not in module.__dict__, 'Protection already adapted.')
            # Registered parameters participate in native clipping, mutable saves,
            # optimizer state and exact resume; finalization removes both masters.
            module.register_parameter('protected_code_coordinate', nn.Parameter(q.view(torch.int8).float()))
            module.register_parameter('protected_log_scale_delta', nn.Parameter(torch.zeros_like(s, dtype=torch.float32)))
            decoder_trainable = any(p.requires_grad for p in module._parallel_stage_decoder.parameters())
            module.trainable_decode = True
            module.cache_decoded_weight = False
            module.parallel_stage_decode = True
            module.clear_decoded_weight_cache()
            module.pack_parallel_stage_decoder_(trainable=decoder_trainable)
            module._cached_weight = None
            initial = module._decode_protected_input_weight_rows(dtype=torch.bfloat16, device=q.device)
            require(torch.equal(initial, self.rows(name, dtype=torch.bfloat16, device=q.device)), 'Initial joint QAT changes native rows.')
            def rows(instance, *, dtype, device, key=name):
                require(instance is self.modules[key], 'Protection owner changed.')
                self.calls[key] += 1
                return self.rows(key, dtype=dtype, device=device)
            module._decode_protected_input_weight_rows = MethodType(rows, module)

    def rows(self, name, *, dtype, device):
        module = self.modules[name]
        require(module.protected_code_coordinate.device == torch.device(device), 'Protection device mismatch.')
        q = integer_coordinate(module.protected_code_coordinate)
        s = effective_scale(module.protected_input_scales, module.protected_log_scale_delta)
        return (q * s.float().unsqueeze(1)).to(dtype).contiguous()

    def parameter_groups(self):
        return {
            'q': {name+'.protected_code_coordinate': m.protected_code_coordinate for name, m in self.modules.items()},
            'scale': {name+'.protected_log_scale_delta': m.protected_log_scale_delta for name, m in self.modules.items()},
        }

    @torch.no_grad()
    def finalize(self):
        changed = set()
        for name, module in self.modules.items():
            q = integer_coordinate(module.protected_code_coordinate).to(torch.int8).view(torch.uint8)
            s = effective_scale(module.protected_input_scales, module.protected_log_scale_delta)
            module.protected_input_qvalues.copy_(q)
            module.protected_input_scales.copy_(s)
            del module.protected_code_coordinate, module.protected_log_scale_delta
            del module._decode_protected_input_weight_rows
            module.disable_trainable_decode()
            changed.update((name+'.protected_input_qvalues', name+'.protected_input_scales'))
        return changed
