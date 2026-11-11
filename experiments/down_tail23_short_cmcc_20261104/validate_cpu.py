"""Real D16 source/grouped-BSQ/strict2->3/unequal-K CPU contracts, not quality."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from . import allocation, artifact, fit
from experiments.adaptive_grouped_fit_20261029.grouped import IndependentGroupedVAE, fresh_models
from experiments.adaptive_native_model_20261029 import objective
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_model_20261029.native_prefix import inventory, load_native
from experiments.adaptive_native_model_20261029.validate_native_prefix import check_fields, numerical
from torch.nn import functional as F


def event(name, **fields):
    print(dict(event=name, **fields), flush=True)


@b.torch.no_grad()
def deployment(module):
    generator = b.torch.Generator().manual_seed(2031)
    inputs = b.torch.randn((3, module.in_features), generator=generator).bfloat16()
    weight = module.decode_weight()
    output = module(inputs)
    b.c.require(b.torch.equal(output, F.linear(inputs, weight)), 'Consumer must perform the same actual BF16 F.linear.')
    return dict(weight_digest=b.c.digest({'weight': weight}), input_digest=b.c.digest({'input': inputs}),
        output_digest=b.c.digest({'output': output}), state_digest=b.c.digest(module.state_dict()))


def fresh(path, output):
    b.cpu_environment()
    bundle, lengths, module = artifact.load(path)
    record = deployment(module)
    record.update(bundle_digest=b.c.digest(inventory(bundle)), length_digest=b.c.digest({'lengths': lengths}),
        cuda_initialized=b.torch.cuda.is_initialized())
    b.c.write(output, record)


def grouped(values, bits):
    originals = fresh_models(3, args=b.arguments(bits, 'fp32'))
    model = IndependentGroupedVAE(originals)
    batches = [value['normalized'][:129].float() for value in values]
    reconstruction, losses = model(b.torch.cat(batches, dim=1))
    losses.sum().backward()
    result = []
    for index, (single, batch) in enumerate(zip(originals, batches, strict=True)):
        expected, parts = single(batch)
        parts['loss'].backward()
        b.torch.testing.assert_close(reconstruction[:, index:index + 1], expected, rtol=2e-5, atol=1e-6)
        b.torch.testing.assert_close(losses[index], parts['loss'], rtol=2e-5, atol=1e-6)
        gradients = model.single_parameters(index, gradients=True)
        reference = dict(single.named_parameters())
        denominator = sum(float(parameter.grad.double().square().sum()) for parameter in reference.values())
        numerator = sum(float((parameter.grad.double() - gradients[name].double()).square().sum())
            for name, parameter in reference.items())
        relative = (numerator / denominator) ** .5
        b.c.require(relative <= 1e-4, 'Per-Linear BSQ grouped gradients changed the independent objective.')
        exported = model.single_model(index)
        b.c.require(all(b.torch.equal(parameter, dict(exported.named_parameters())[name])
            for name, parameter in single.named_parameters()), 'Grouped extraction changed an independent parameter.')
        result.append(dict(layer=index, single_loss=float(parts['loss']), grouped_loss=float(losses[index]),
            all_leaf_gradient_relative_L2=relative, extracted_parameters_exact=True))
    return result


def coordinate_mapping(value):
    target, geometry = value['target'], value['bundle']['geometry']
    out_features, in_features = geometry['compressed_shape']
    b.c.require(out_features // b.DIM == 256 and geometry['transpose'], 'D16 transpose has256 vectors per input coordinate, never oldD32 factor128.')
    h = b.torch.arange(1, in_features + 1, dtype=b.torch.float64)
    h = (h / h.mean()).contiguous()
    start, stop = 254, 259
    actual = objective.prefix_errors(target[start:stop], [value['prefixes'][-1][start:stop]], h,
        geometry, value['info']['source_variance'], start=start)
    expected = []
    for row in range(start, stop):
        coordinates = b.torch.arange(row * b.DIM, (row + 1) * b.DIM)
        residual = (target[row] - value['prefixes'][-1][row].float()).double()
        expected.append((h[coordinates // out_features] * residual.square()).sum() / (b.DIM * value['info']['source_variance']))
    b.torch.testing.assert_close(actual[:, 0], b.torch.stack(expected), rtol=1e-12, atol=1e-15)
    return dict(vectors_per_input_coordinate=256, checked_global_rows=list(range(start, stop)),
        checked_input_coordinates=[int(row * b.DIM // out_features) for row in range(start, stop)],
        exact_coordinate_weighted_formula=True)


def route_boundaries(full):
    n = full['n_vectors']
    # Actual complete source banks, crafted score fixtures to test zero/ties/budget.
    scores = b.torch.zeros((n, 2), dtype=b.torch.float64)
    scores[:3, 0] = 1
    scores[3, 1] = 1
    all2 = b.torch.full((n,), 2, dtype=b.torch.uint8)
    empty = artifact.measure(allocation.materialize(full, all2))
    lengths, _selected, record = allocation.allocate(scores, full, b.CAP)
    b.c.require(b.torch.equal(b.torch.nonzero(lengths == 3).flatten(), b.torch.tensor([0, 1, 2])),
        'Zero/negative gains must remain2; tied positive gains preserve original indices.')
    row_bytes = (int(full['stages'][2]['decoder_spec']['in_dim']) + 7) // 8
    tied_lengths, _candidate, tied = allocation.allocate(scores, full, empty['payload_tensor_bytes'] + row_bytes)
    ids = b.torch.nonzero(tied_lengths == 3).flatten()
    b.c.require(len(ids) <= 1 and (not len(ids) or int(ids[0]) == 0), 'Tied budget fixture must prefer original vector0.')
    return dict(strictly_positive_and_tie_indices=[0, 1, 2], zero_and_negative_disabled=True,
        tight_cap_selected=ids.tolist(), tight_cap_rounds=tied['rounds'], rows_preserved=record['chosen']['stage_active_rows'][:2])


@b.torch.no_grad()
def validate_bank(value, bits, output):
    started = time.monotonic()
    single = fresh_models(1, args=b.arguments(bits))[0]
    stage = fit.export_stage(single, value)
    scores, full = fit.errors(value, stage, b.torch.ones(value['bundle']['geometry']['compressed_shape'][1], dtype=b.torch.float64))
    lengths, selected, allocation_record = allocation.allocate(scores, full, b.CAP)
    # The stage3 decoder is real, with its genuine nonzero norm/bias/mean leaves.
    disabled = allocation.materialize(full, b.torch.full((b.N,), 2, dtype=b.torch.uint8))
    base = CompactRVQLinear(value['bundle'], b.torch.full((b.N,), 2, dtype=b.torch.uint8)).eval()
    inactive = CompactRVQLinear(disabled, b.torch.full((b.N,), 2, dtype=b.torch.uint8)).eval()
    b.c.require(b.torch.equal(base.decode_weight(), inactive.decode_weight()), 'Inactive third-stage bias or fused mean leaked into an all2 weight.')
    del base, inactive
    boundary = route_boundaries(full)
    # Force a genuine final1-row active chunk using actual encoded rows; no mocked decoder.
    tail_lengths = b.torch.full((b.N,), 2, dtype=b.torch.uint8)
    tail_lengths[:8192], tail_lengths[-1] = 3, 3
    tail = allocation.materialize(full, tail_lengths)
    tail_module = CompactRVQLinear(tail, tail_lengths).eval()
    calls = []
    handle = tail_module.decoders[2].register_forward_pre_hook(lambda _module, inputs: calls.append(len(inputs[0])))
    tail_module.decode_weight()
    handle.remove()
    b.c.require(calls == [8192, 1], 'Actual active third bank must consume its real last1-row chunk only.')
    del tail_module, tail
    path = output / f'untrained_K{bits}_actual.pt'
    measured = artifact.save(selected, path)
    bundle, loaded_lengths, module = artifact.load(path)
    b.c.require(b.c.digest(inventory(bundle)) == b.c.digest(inventory(selected))
        and b.torch.equal(loaded_lengths, lengths), 'Same-process real compact fields or routes changed.')
    expected = deployment(module)
    expected.update(bundle_digest=b.c.digest(inventory(bundle)), length_digest=b.c.digest({'lengths': lengths}),
        cuda_initialized=False)
    fresh_output = output / f'fresh_K{bits}.json'
    subprocess.run([sys.executable, '-B', '-u', '-m', 'experiments.down_tail23_short_cmcc_20261104.validate_cpu',
        '--fresh', str(path), '--output', str(fresh_output)], cwd=b.REPO, check=True)
    observed = b.c.read(fresh_output)
    b.c.require(expected == observed, 'Independent fresh process changed persistent fields/routes/BF16 weight or same-input output.')
    record = dict(K=bits, trained_updates=0, real_full_bank_shape=list(stage['codes'].shape),
        actual_decoder_leaves={name: dict(shape=list(value.shape), dtype=str(value.dtype), bytes=b.c.compact(value))
            for name, value in stage['decoder_state'].items()}, allocation=allocation_record,
        actual_file=measured, artifact_sha256=b.c.sha(path), fresh=observed, fresh_process_exact=True,
        original_bundle_fields_digest=b.c.digest(inventory(value['bundle'])),
        disabled_stage_bias_and_mean_no_leak=True, actual_active_tail_chunks=calls, score_boundaries=boundary,
        elapsed_seconds=time.monotonic() - started,
        boundary='Actual full L0 banks, untrained real stage3. Uniform h is a CPU contract fixture, not calibrated quality evidence.')
    # No future run uses these untrained generated weights; retain hashes/metrics/log only.
    record['cleanup'] = dict(paths=[str(path), str(fresh_output)], removed_file_bytes=path.stat().st_size + fresh_output.stat().st_size,
        reason='Completed untrained CPU contract artifacts; no downstream use; evidence merged into report')
    del module, bundle, stage, full, selected, disabled, single
    gc.collect()
    path.unlink()
    fresh_output.unlink()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh', type=Path)
    args = parser.parse_args()
    b.cpu_environment()
    b.torch.set_num_threads(8)
    if args.fresh:
        fresh(args.fresh, args.output)
        return
    b.c.require(not args.output.exists(), 'Never overwrite prior CPU evidence.')
    args.output.mkdir(parents=True)
    started = time.time()
    record = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv], started_unix=started,
        GPU_used=False, training_updates=0, no_full_model_constructed=True, cases=[])
    b.c.write(args.output / 'report.json', record)
    try:
        raw, live, specs, source = b.sources()
        record['source_binding'] = source
        values = []
        for spec in specs:
            event('actual_D16_common_residual_start', layer=spec['name'])
            bundle, lengths, resolved, fields = b.prefix(raw, spec, source)
            module = load_native(spec, fields)
            base = CompactRVQLinear(bundle, lengths)
            exact = check_fields(module, fields, bundle, lengths, resolved)
            numerical_record, _digest = numerical(module, base, b.torch.arange(8193))
            value = fit.common_residual(bundle, live[spec['name'] + '.weight'])
            payload = sum(b.c.compact(tensor) for name, tensor in inventory(bundle).items() if name != 'length_frame')
            b.c.require(payload == b.BASE_PAYLOAD, 'Actual source two-stage payload differs from the common budget base.')
            record['cases'].append(dict(name=spec['name'], source_payload_bytes_without_added_frame=payload,
                original_fields=exact, native_vs_compact_numerical=numerical_record, common_residual=value['info'],
                layout=coordinate_mapping(value)))
            values.append(value)
            del module, base, fields
            event('actual_D16_common_residual_done', layer=spec['name'], residual=value['info'])
        record['grouped_independent_BSQ'] = {str(bits): grouped(values, bits) for bits in (8, 24)}
        # Same independent shuffle generators for both K, including genuine epoch crossing.
        sequences = []
        for _bits in (8, 24):
            generators = [b.torch.Generator().manual_seed(2031) for _ in values]
            orders, positions = [None] * 3, [0] * 3
            selected = []
            for _step in range(190):
                batch, ids = fit.batches(values, generators, orders, positions)
                b.c.require(tuple(batch.shape) == (b.BATCH, 3, b.DIM), 'Real D16 short batch shape changed.')
                selected.append(b.c.digest({str(index): value for index, value in enumerate(ids)}))
            sequences.append(selected)
        b.c.require(sequences[0] == sequences[1], 'K8 and K24 must sample the same independent per-layer indices.')
        record['same_K_shuffle_190_updates_exact'] = True
        record['banks'] = []
        for bits in (8, 24):
            event('untrained_real_full_bank_start', K=bits)
            record['banks'].append(validate_bank(values[0], bits, args.output))
            event('untrained_real_full_bank_done', K=bits)
        record.update(status='cpu_contracts_verified', exit_code=0, cuda_initialized=b.torch.cuda.is_initialized(),
            source_sha256=b.pins(), scope='Real three source Linears and full common residuals; independent BSQ bothK; L0 full actual untrained unequal-K banks/fresh consumption. No GPU, quality claim, training or36-layer path validation.')
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - started)
        b.c.write(args.output / 'report.json', record)


if __name__ == '__main__':
    main()
