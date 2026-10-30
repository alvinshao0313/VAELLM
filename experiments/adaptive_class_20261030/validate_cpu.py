"""Necessary actual D64 grouped-loss/gradient and Rice artifact CPU contracts."""
import argparse
import os
from pathlib import Path
import time

import torch

from experiments.adaptive_grouped_fit_20261029.grouped import IndependentGroupedVAE, fresh_models
from experiments.adaptive_native_model_20261029 import objective
from experiments.adaptive_vector_rvq import common as c
from . import artifact, sources as s
from .fit import arguments
from .sparse import Fixed2RiceLinear, build
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not args.output.exists(), 'Use a fresh CPU-only actual contract result.')
    torch.set_num_threads(8)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    gate, specs, manifest, live = s.raw_inputs()
    by_layer = {int(spec['name'].split('.')[2]): spec for spec in specs}
    batches, records, bundles = [], [], []
    for layer in (0, 18, 35):
        bundle, lengths, resolved = s.prefix(gate, by_layer[layer])
        target = objective.live_vectors(bundle, live[by_layer[layer]['name'] + '.weight'])
        prefixes = objective.decode_prefixes(bundle, start=0, stop=129)
        residual = target[:129] - prefixes[-1].float()
        batches.append((residual / residual.std()).float().unsqueeze(1))
        bundles.append(bundle)
        records.append(dict(layer=layer, n_vectors=resolved['n_vectors'], residual_digest=c.digest({'residual': residual})))
    config = arguments(bundles[0], 'fp32')
    originals = fresh_models(3, args=config)
    grouped = IndependentGroupedVAE(originals)
    reconstructed, losses = grouped(torch.cat(batches, dim=1))
    losses.sum().backward()
    for index, (single, batch) in enumerate(zip(originals, batches, strict=True)):
        expected, parts = single(batch)
        parts['loss'].backward()
        torch.testing.assert_close(reconstructed[:, index:index + 1], expected, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(losses[index], parts['loss'], rtol=2e-5, atol=1e-6)
        reference = dict(single.named_parameters())
        gradients = grouped.single_parameters(index, gradients=True)
        denominator = sum(float(parameter.grad.double().square().sum()) for parameter in reference.values())
        numerator = sum(float((parameter.grad.double() - gradients[name].double()).square().sum()) for name, parameter in reference.items())
        c.require((numerator / denominator) ** .5 <= 1e-4, 'Actual D64 grouped independent leaf gradients disagree with single-layer objectives.')
        exported = grouped.single_model(index)
        c.require(all(torch.equal(parameter, dict(exported.named_parameters())[name]) for name, parameter in single.named_parameters()), 'D64 grouped extraction changed an independent parameter.')
        records[index].update(reference_loss=float(parts['loss']), grouped_loss=float(losses[index]), gradient_relative_l2=(numerator / denominator) ** .5)
    bundle = bundles[0]
    lengths = torch.full((bundle['n_vectors'],), 2, dtype=torch.uint8)
    base = CompactRVQLinear(bundle, lengths)
    fields, rice = build(bundle, live[by_layer[0]['name'] + '.weight'], base.decode_weight())
    module = Fixed2RiceLinear(bundle, fields)
    size = artifact.save_module(module, args.output / 'real_L0_control.pt')
    fresh, saved = artifact.load(args.output / 'real_L0_control.pt')
    c.require('length_frame' not in saved['state'] and torch.equal(module.decode_weight(), fresh.decode_weight()),
              'Actual Rice control fresh deployment or implicit length2 changed.')
    c.write(args.output / 'report.json', dict(status='actual_D64_independent_math_Rice_fresh_verified', exit_code=0,
        elapsed_seconds=time.monotonic() - started, grouped_math=records, real_L0_control_bytes=size,
        real_L0_control_sha256=c.sha(args.output / 'real_L0_control.pt'), rice=rice,
        source_sha256=s.code_identity(), cuda_initialized=torch.cuda.is_initialized(),
        source_raw_B=manifest['source_raw']['checkpoint_id'], source_after_o=s.BASE_ID,
        scope='Actual D64/K64 three-layer FP32 loss/all-leaf gradients/extraction and complete L0 Rice artifact; no GPU or quality claim'))
    print(dict(status='actual_D64_independent_math_Rice_fresh_verified', seconds=time.monotonic() - started), flush=True)


if __name__ == '__main__':
    main()
