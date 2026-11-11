"""Reuse mature Muon math and stage factories with the actual gate owners."""
import importlib.util

from . import bindings as b
from litebsq.parallel_layers import Normalize, ParallelLinear

spec = importlib.util.spec_from_file_location(__package__ + '._mature_optimizer', b.MATURE_OPTIMIZER)
mature = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mature)


def derive_layout(vae):
    named, layout, affine = dict(vae.named_parameters()), {}, []
    for role in ('encoder', 'decoder'):
        module = getattr(vae.model, role)
        b.require(module.num_models == 36 and module.num_res_blocks == 0, 'Use actual gate M36/R0 owners.')
        for part in ('linear_in', 'linear_out'):
            linear = getattr(module, part)
            b.require(isinstance(linear, ParallelLinear) and linear.num_models == linear.conv.groups == 36
                and linear.conv.kernel_size == (1,), 'Use genuine grouped ParallelLinear matrices.')
            name = f'model.{role}.{part}.conv.weight'
            shape = [36, linear.out_features, linear.in_features]
            expected = [36,128,64] if part == 'linear_in' else [36,64,128]
            b.require(shape == expected and tuple(named[name].shape) == (shape[0]*shape[1],shape[2],1),
                'The gate matrix owner dimensions differ: ' + name)
            layout[name] = dict(role=role, num_models=36, matrix_shape=shape, packed_shape=list(named[name].shape))
            bias = f'model.{role}.{part}.conv.bias'
            b.require(named[bias] is linear.conv.bias, 'Route the actual convolution bias to native BF16 Adam.')
            affine.append(bias)
        norm = module.norm_out
        b.require(isinstance(norm, Normalize) and norm.norm_type == 'layer' and norm.num_models == 36,
            'Keep the original gate layer norm as auxiliary Adam.')
        for leaf in ('weight','bias'):
            name = f'model.{role}.norm_out.{leaf}'
            b.require(named[name] is getattr(norm,leaf) and tuple(named[name].shape) == (36,128), 'Wrong gate norm leaf.')
            affine.append(name)
    b.require(len(named) == 12 and set(layout) | set(affine) == set(named)
        and len({id(p) for p in named.values()}) == 12
        and sum(p.numel() for p in named.values()) == 1211904
        and sum(named[name].numel() for name in layout) == 1179648
        and sum(named[name].numel() for name in affine) == 32256
        and all(p.dtype == b.torch.bfloat16 for p in named.values()), 'Every gate BF16 leaf needs exactly one owner.')
    return named, layout, affine


mature.derive_layout = derive_layout
mature.SCHEMA = 'gate_D64_vae_muon_bf16_v1'
VAEMuon, install, matrices = mature.VAEMuon, mature.install, mature.matrices
