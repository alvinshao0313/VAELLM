"""Batch encoder/decoder GEMMs while retaining each original BSQ objective."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from experiments.adaptive_vector_rvq import common as c
from litebsq.autoencoder import MultiLayerVAE, pack_decoders
from litebsq.parallel_layers import pack_normalizes, pack_parallel_linears


def fresh_models(count, dtype='bf16', device='cpu', *, args=None):
    models = []
    for _ in range(count):
        torch.manual_seed(2031)
        models.append(MultiLayerVAE(deepcopy(args) if args is not None else c.arguments(dtype)).to(device))
    return models


class IndependentGroupedVAE(nn.Module):
    def __init__(self, originals):
        super().__init__()
        c.require(len(originals) > 0, 'Group at least one independently optimized real layer.')
        models = [item.model for item in originals]
        c.require(all(item.encoder.num_res_blocks == item.decoder.num_res_blocks == 0
                      and item.decoder.norm_type == 'layer' for item in models),
                  'Use the existing R0/LayerNorm stage geometry.')
        self.count = len(models)
        self.args, self.params_dtype = deepcopy(models[0].args), models[0].params_dtype
        self.dim = int(self.args.codebook_dim)
        c.require(all(item.arch_spec == models[0].arch_spec and item.params_dtype == self.params_dtype
                      for item in models), 'Independent grouped layers must have the same actual architecture and dtype.')
        self.encoder = deepcopy(models[0].encoder)
        self.encoder.num_models = self.count
        self.encoder.linear_in = pack_parallel_linears([item.encoder.linear_in for item in models])
        self.encoder.norm_out = pack_normalizes([item.encoder.norm_out for item in models])
        self.encoder.linear_out = pack_parallel_linears([item.encoder.linear_out for item in models])
        self.decoder = pack_decoders([item.decoder for item in models])
        self.quantizers = nn.ModuleList([deepcopy(item.quantizer) for item in models])
        self.requires_grad_(True)

    def forward(self, values):
        c.require(values.shape[1:] == (self.count, self.dim), 'Use [batch, independent layer, actual D].')
        context = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
                   if values.device.type == 'cuda' and self.params_dtype == torch.bfloat16
                   else torch.autocast(values.device.type, enabled=False))
        with context:
            hidden = self.encoder(values)
            results = [quantizer(hidden[:, index:index+1])
                       for index, quantizer in enumerate(self.quantizers)]
            quantized = torch.cat([result.quantized.to(self.params_dtype) for result in results], dim=1)
            reconstructed = self.decoder(quantized)
        # Original MSE is outside autocast. Each BSQ sees only its own layer.
        losses = [F.mse_loss(reconstructed[:, index:index+1], values[:, index:index+1]) * self.args.l1_weight
                  + result.entropy_aux_loss * self.args.lfq_weight
                  for index, result in enumerate(results)]
        return reconstructed, torch.stack(losses)

    def single_parameters(self, index, *, gradients=False):
        values = {}
        for part in ('encoder', 'decoder'):
            module = getattr(self, part)
            for name in ('linear_in', 'linear_out'):
                linear = getattr(module, name)
                weight = linear.conv.weight.grad if gradients else linear.conv.weight
                bias = linear.conv.bias.grad if gradients else linear.conv.bias
                c.require(weight is not None and bias is not None, 'Missing grouped affine gradient.')
                values[f'model.{part}.{name}.linear.weight'] = weight[:, :, 0].reshape(
                    self.count, linear.out_features, linear.in_features)[index]
                values[f'model.{part}.{name}.linear.bias'] = bias.reshape(self.count, linear.out_features)[index]
            norm = module.norm_out
            for name in ('weight', 'bias'):
                parameter = getattr(norm, name)
                tensor = parameter.grad if gradients else parameter
                c.require(tensor is not None, 'Missing independent grouped normalization gradient.')
                values[f'model.{part}.norm_out.norm.{name}'] = tensor[index]
        return values

    @torch.no_grad()
    def single_model(self, index):
        model = MultiLayerVAE(deepcopy(self.args)).to(next(self.parameters()).device)
        target = dict(model.named_parameters())
        values = self.single_parameters(index)
        c.require(target.keys() == values.keys(), 'Export must cover every independent encoder/decoder leaf.')
        for name, parameter in target.items():
            parameter.copy_(values[name])
        model.model.quantizer.load_state_dict(self.quantizers[index].state_dict(), strict=True)
        return model
