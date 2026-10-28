"""Read packed endpoints against original weights and fixed teacher MLP inputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.longbench_v1.common import file_stat, write_json
from experiments.rotation_factor_probe.capture import require
from .prepare import CATEGORIES, LAYERS
from .runtime import load_endpoint


def ratio(error, reference):
    error, reference = float(error), float(reference)
    require(math.isfinite(error) and error >= 0 and math.isfinite(reference) and reference > 0,
            'Require finite nonnegative error and positive reference energy.')
    return dict(squared_error=error, reference_squared_sum=reference, nmse=error / reference)


@torch.no_grad()
def observe(prepared_dir, arm_dir, output_file):
    prepared, arm, output = map(Path, (prepared_dir, arm_dir, output_file))
    require(not output.exists(), 'Observation output must be new.')
    manifest = json.loads((prepared / 'manifest.json').read_text())
    require(manifest['status'] == 'captured', 'Require completed original-model activation capture.')
    weights = torch.load(prepared / 'inputs.pt', weights_only=True, map_location='cpu', mmap=True)['targets']
    inputs = torch.load(prepared / 'activations.pt', weights_only=True, map_location='cpu', mmap=True)['inputs']
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Assign exactly one GPU.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started, linear_results, mlp_results, sources = time.time(), {}, {}, {}
    for layer in LAYERS:
        prefix = f'model.layers.{layer}.mlp.'
        original, candidate = {}, {}
        for category in CATEGORIES:
            name = prefix + category
            matches = list(arm.rglob(name.replace('.', '_') + '.pt'))
            require(len(matches) == 1, f'Expected exactly one completed endpoint for {name}: {matches}')
            sources[name] = file_stat(matches[0])
            endpoint = load_endpoint(matches[0], device='cuda:0')
            candidate[category] = endpoint.decode_weight().float()
            original[category] = weights[name]['reference_weight'].to('cuda:0').float()
            delta = candidate[category] - original[category]
            weight_error = ratio(delta.double().square().sum(), original[category].double().square().sum())
            input_name = prefix + ('down_proj' if category == 'down_proj' else 'gate_proj')
            error, reference = torch.zeros(2, device='cuda:0', dtype=torch.float64).unbind()
            for batch in inputs[input_name].split(256):
                x = batch.to('cuda:0').float()
                difference = F.linear(x, delta)
                target = F.linear(x, original[category])
                error += difference.double().square().sum()
                reference += target.double().square().sum()
            linear_results[name] = dict(weight=weight_error, output=ratio(error, reference))
            print(json.dumps(dict(target=name, **linear_results[name])), flush=True)
            del endpoint, delta, x, difference, target
        error, reference = torch.zeros(2, device='cuda:0', dtype=torch.float64).unbind()
        for batch in inputs[prefix + 'gate_proj'].split(256):
            x = batch.to('cuda:0').float()
            ref_hidden = F.silu(F.linear(x, original['gate_proj'])) * F.linear(x, original['up_proj'])
            pred_hidden = F.silu(F.linear(x, candidate['gate_proj'])) * F.linear(x, candidate['up_proj'])
            target = F.linear(ref_hidden, original['down_proj'])
            prediction = F.linear(pred_hidden, candidate['down_proj'])
            error += (prediction - target).double().square().sum()
            reference += target.double().square().sum()
        mlp_results[str(layer)] = ratio(error, reference)
        print(json.dumps(dict(layer=layer, mlp=mlp_results[str(layer)])), flush=True)
        del original, candidate, x, ref_hidden, pred_hidden, target, prediction
    report = dict(status='completed', arm=str(arm.resolve()), prepared=str(prepared.resolve()),
        protocol=dict(valid_tokens=manifest['valid_tokens'], documents=manifest['protocol']['documents'],
            max_length=manifest['protocol']['max_length'], split='train', fitting_uses_text=False,
            weights='Actual packed deployment decoded to BF16, then cast to FP32 for diagnostic arithmetic.',
            accumulation='FP64 sums of FP32 weight/output errors; TF32 disabled.',
            linear_context='Original BF16 teacher input for each individual projection.',
            mlp_context='Original gate input; candidate gate/up -> SiLU product -> candidate down. No teacher down substitution.',
            boundary='Local fixed-prefix diagnostics only; not native BF16 full-model inference or downstream task quality.'),
        endpoints=sources, linear=linear_results, mlp=mlp_results,
        macro=dict(weight_nmse=sum(item['weight']['nmse'] for item in linear_results.values()) / len(linear_results),
            linear_output_nmse=sum(item['output']['nmse'] for item in linear_results.values()) / len(linear_results),
            mlp_output_nmse=sum(item['nmse'] for item in mlp_results.values()) / len(mlp_results)),
        seconds=time.time() - started)
    write_json(output, report)
    print(json.dumps(dict(status=report['status'], macro=report['macro'], seconds=report['seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', required=True)
    parser.add_argument('--arm-dir', required=True)
    parser.add_argument('--output-file', required=True)
    observe(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
