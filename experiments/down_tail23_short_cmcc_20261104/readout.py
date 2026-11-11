"""Actual BF16 Linear outputs and persistence fingerprints on one captured input."""
from . import bindings as b
from experiments.adaptive_native_model_20261029.native_prefix import inventory
from torch.nn import functional as F


@b.torch.no_grad()
def outputs(weight, inputs, *, device):
    weight = weight.to(device)
    b.c.require(weight.dtype == inputs.dtype == b.torch.bfloat16 and tuple(inputs.shape) == (4096, 12288),
        'Read all sixteen real captured fixed256 BF16 inputs with the actual BF16 weight.')
    parts = [F.linear(inputs[start:start + 256].to(device), weight).cpu() for start in range(0, len(inputs), 256)]
    result = b.torch.cat(parts).contiguous()
    b.c.require(result.dtype == b.torch.bfloat16 and bool(b.torch.isfinite(result).all()), 'Nonfinite actual BF16 F.linear readout.')
    return result


@b.torch.no_grad()
def deployment(bundle, lengths, module, inputs, *, device):
    module = module.to(device).eval()
    weight = module.decode_weight()
    b.c.require(weight.dtype == b.torch.bfloat16 and bool(b.torch.isfinite(weight).all()), 'Actual compact deployment must decode finite BF16 weights.')
    actual = outputs(weight, inputs, device=device)
    fingerprint = dict(bundle_digest=b.c.digest(inventory(bundle)), length_digest=b.c.digest({'lengths': lengths}),
        state_digest=b.c.digest(module.state_dict()), BF16_weight_digest=b.c.digest({'weight': weight}),
        captured_input_digest=b.c.digest({'inputs': inputs}), BF16_output_digest=b.c.digest({'output': actual}))
    return weight, actual, fingerprint


@b.torch.no_grad()
def metrics(weight, source, actual, reference):
    difference = weight.float() - source.to(weight.device).float()
    w_sse = float(difference.double().square().sum())
    w_energy = float(source.double().square().sum())
    groups = {}
    for key, start, stop in (('train_first8', 0, 2048), ('train_last8_functional_only', 2048, 4096)):
        error = actual[start:stop].float() - reference[start:stop].float()
        sse = float(error.double().square().sum())
        energy = float(reference[start:stop].double().square().sum())
        b.c.require(energy > 0, 'The real functional reference must have positive output energy.')
        groups[key] = dict(SSE=sse, reference_energy=energy, NMSE=sse / energy,
            elements=actual[start:stop].numel(), documents=8)
    return dict(weight=dict(SSE=w_sse, reference_energy=w_energy, NMSE=w_sse / w_energy,
        elements=source.numel()), output=groups,
        compute='Actual source and compact decoded BF16 W, BF16 F.linear, fixed256-row chunks; FP64 error reduction')


def aggregate(layers):
    result = {}
    for group in ('train_first8', 'train_last8_functional_only'):
        sse = sum(layer['metrics']['output'][group]['SSE'] for layer in layers)
        energy = sum(layer['metrics']['output'][group]['reference_energy'] for layer in layers)
        result[group] = dict(SSE=sse, reference_energy=energy, NMSE=sse / energy)
    return result
