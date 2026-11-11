"""Use the unchanged mature Muon math with actual symmetric up ownership."""
import importlib.util

from . import bindings as b
from litebsq.parallel_layers import Normalize, ParallelLinear

spec = importlib.util.spec_from_file_location(__package__ + '._mature_optimizer', b.MATURE_OPTIMIZER)
mature = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mature)


def derive_layout(vae):
    named, layout, affine = dict(vae.named_parameters()), {}, []
    for role in ('encoder','decoder'):
        module = getattr(vae.model,role)
        b.require(module.num_models == 36 and module.num_res_blocks == 0, 'Use real symmetric up M36/R0 owners.')
        for part in ('linear_in','linear_out'):
            linear = getattr(module,part)
            expected = [36,128,32] if part == 'linear_in' else [36,32,128]
            b.require(isinstance(linear,ParallelLinear) and linear.num_models == linear.conv.groups == 36
                and linear.conv.kernel_size == (1,) and [36,linear.out_features,linear.in_features] == expected,
                'The actual symmetric up matrix owner differs: ' + role + '.' + part)
            name = f'model.{role}.{part}.conv.weight'
            b.require(tuple(named[name].shape) == (expected[0]*expected[1],expected[2],1), 'Packed up matrix shape mismatch.')
            layout[name] = dict(role=role,num_models=36,matrix_shape=expected,packed_shape=list(named[name].shape))
            bias = f'model.{role}.{part}.conv.bias'
            b.require(named[bias] is linear.conv.bias, 'Route the actual bias to original auxiliary AdamW.')
            affine.append(bias)
        norm = module.norm_out
        b.require(isinstance(norm,Normalize) and norm.norm_type == 'layer' and norm.num_models == 36, 'Up layer norm is auxiliary AdamW.')
        for leaf in ('weight','bias'):
            name = f'model.{role}.norm_out.{leaf}'
            b.require(named[name] is getattr(norm,leaf) and tuple(named[name].shape) == (36,128), 'Actual up norm leaf differs.')
            affine.append(name)
    b.require(len(named) == len({id(value) for value in named.values()}) == 12
        and set(layout) | set(affine) == set(named) and len(layout) == 4 and len(affine) == 8
        and sum(value.numel() for value in named.values()) == 619776
        and sum(named[name].numel() for name in layout) == 589824
        and sum(named[name].numel() for name in affine) == 29952
        and all(value.dtype == b.torch.bfloat16 for value in named.values()), 'Assign every actual symmetric-up BF16 leaf exactly once.')
    return named, layout, affine


mature.derive_layout = derive_layout
mature.SCHEMA = 'up_symmetric_D32_vae_muon_bf16_v1'
VAEMuon, install, matrices, zeropower = mature.VAEMuon, mature.install, mature.matrices, mature.zeropower
