"""One calibration-input-energy weighted allocation on the frozen real L0 three-stage bank."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from safetensors import safe_open

from . import codec, common as c
from .entropy_budget import iterate
from .observe import original_weight
from .third_stage import vector_errors
from .third_stage_export import inventory, load, save
from experiments.night_sequence_20261026.queue_wait import free_gpu
from rotation.weight_preprocess import _transform_last, build_weight_rotation


def decode(bundle, lengths):
    vectors = torch.zeros((bundle['n_vectors'], c.DIM), dtype=torch.bfloat16)
    for index, stage in enumerate(bundle['stages'], start=1):
        active = torch.nonzero(lengths >= index, as_tuple=False).flatten()
        codec.add_stage(vectors, stage, active, 'cuda:0')
    return vectors


def weights_and_checks(source, target, prefix, x, split, e2):
    geometry, fixed = source['geometry'], source['fixed']
    out_features, in_features = geometry['compressed_shape']
    keep = torch.ones(geometry['original_shape'][1], dtype=torch.bool)
    keep[fixed['protected_input_indices']] = False
    kept = torch.nonzero(keep, as_tuple=False).flatten()
    transform = build_weight_rotation(out_features, in_features, geometry['rotation'])
    transform.input_signs.copy_(fixed['rotation_input_signs'])
    transform.output_signs.copy_(fixed['rotation_output_signs'])
    c.require(len(kept) == in_features and out_features == 4096 and out_features % c.DIM == 0,
        'Preserve ascending unprotected columns and actual output-vector layout.')
    factor = out_features // c.DIM
    c.require(in_features * factor == len(target), 'Vector-to-input-row layout changed.')
    # This checks direction/sign order with real X and a real fixed2 error output block.
    block = transform.output_block
    c.require(block == 32, 'The specified real L0 rotation uses a32-output Hadamard block.')
    ids = torch.arange(in_features) * factor
    error = (target.index_select(0, ids).float() - prefix.index_select(0, ids).float()).t().contiguous()
    xb = x[:64].index_select(1, kept).float()
    sin, sout = transform.input_signs.float(), transform.output_signs[:block].float()
    xr = _transform_last(xb * sin, block_size=transform.input_block,
        factor=transform.input_factor, had=transform.input_had, inverse=False)
    original_error = _transform_last(error, block_size=transform.input_block,
        factor=transform.input_factor, had=transform.input_had, inverse=True) * sin
    original_error = _transform_last(original_error.t(), block_size=transform.output_block,
        factor=transform.output_factor, had=transform.output_had, inverse=True).t() * sout[:, None]
    lhs = _transform_last((xb @ original_error.t()) * sout, block_size=transform.output_block,
        factor=transform.output_factor, had=transform.output_had, inverse=False)
    rhs = xr @ error.t()
    relative = float(torch.linalg.vector_norm((lhs - rhs).double()) / torch.linalg.vector_norm(rhs.double()))
    c.require(relative <= 2e-5, 'Real FP32 bilinear rotation identity failed; do not fall back to other weights.')
    energy = torch.zeros(in_features, dtype=torch.float64, device='cuda:0')
    signs = sin.to('cuda:0')
    for start in range(0, split, 256):
        values = x[start:min(start + 256, split)].index_select(1, kept).to('cuda:0', dtype=torch.float32)
        rotated = _transform_last(values * signs, block_size=transform.input_block,
            factor=transform.input_factor, had=transform.input_had, inverse=False)
        energy += rotated.double().square().sum(0)
    h = (energy / split).cpu()
    c.require(bool(torch.isfinite(h).all()) and bool((h >= 0).all()) and float(h.mean()) > 0,
        'Actual calibration energy must be finite/nonnegative with a positive global mean.')
    normalized = h / h.mean()
    weights = normalized.repeat_interleave(factor)
    # Independent matrix-column accumulation prevents a repeated/tiled-vector layout swap.
    column_energy = torch.empty(in_features, dtype=torch.float64)
    for start in range(0, in_features, 128):
        end = min(start + 128, in_features)
        residual = target[start * factor:end * factor].double() - prefix[start * factor:end * factor].double()
        column_energy[start:end] = residual.reshape(end - start, out_features).square().sum(-1)
    left = float((normalized.repeat_interleave(factor) * e2 * c.DIM).sum())
    right = float((normalized * column_energy).sum())
    layout_relative = abs(left - right) / max(abs(right), torch.finfo(torch.float64).tiny)
    c.require(layout_relative < 1e-12, 'Actual input-row/vector-repeat weighted-error identity failed.')
    return h, normalized, weights, kept, dict(bilinear_FP32_relative_L2=relative,
        bilinear_relative_L2_tolerance=2e-5,
        tolerance_basis='FP32 signed FWHT and12032-term dot reductions; BF16 input values are promoted before the identity, without inverse-rotation BF16 rounding',
        weighted_vector_matrix_relative_error=layout_relative, vectors_per_input_row=factor,
        input_sign_before_forward_hadamard=True, original_unprotected_columns_ascending=True,
        no_epsilon_or_clipping=True, calibration_tokens=split)


def output_metrics(weight, reference, x, split):
    result = {}
    for name, begin, end in [('calibration64', 0, split), ('heldout_train64', split, len(x))]:
        error_sum, target_sum, elements = 0., 0., 0
        for start in range(begin, end, 256):
            values = x[start:min(start + 256, end)].to('cuda:0')
            actual = torch.nn.functional.linear(values, weight)
            expected = torch.nn.functional.linear(values, reference)
            c.require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), 'Nonfinite native BF16 outputs.')
            error_sum += float((actual.double() - expected.double()).square().sum())
            target_sum += float(expected.double().square().sum())
            elements += expected.numel()
        result[name] = dict(tokens=end - begin, output_elements=elements, squared_error_sum=error_sum,
            target_energy_sum=target_sum, MSE=error_sum / elements, NMSE=error_sum / target_sum)
    a, b = result['calibration64'], result['heldout_train64']
    error, energy, count = a['squared_error_sum'] + b['squared_error_sum'], a['target_energy_sum'] + b['target_energy_sum'], a['output_elements'] + b['output_elements']
    result['whole128'] = dict(tokens=len(x), output_elements=count, squared_error_sum=error,
        target_energy_sum=energy, MSE=error / count, NMSE=error / energy,
        aggregation='Sum numerator/target energy/count from both groups, never average NMSE.')
    error = (weight.double() - reference.double()).square().sum()
    result['original_BF16_weight_MSE'] = float(error / reference.numel())
    result['original_BF16_weight_NMSE'] = float(error / reference.double().square().sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root_dir.resolve(), args.output_dir.resolve()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and not output.exists(), 'Use the assigned free GPU0 and an independent output.')
    resource = free_gpu(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = c.check_prepared(root / 'prepared')
    layer = prepared['layers']['0']
    parent_path = root / 'entropy_stage3/report.json'
    compact_report_path = root / 'entropy_stage3/compact_validation/report.json'
    parent, compact_report = c.read(parent_path), c.read(compact_report_path)
    c.require(parent['status'] == compact_report['status'] == 'completed', 'Use the completed frozen three-stage source and compact candidate.')
    fixed2_path = root / 'formal_l0/layer_0/fixed2.avrvq'
    stage_path = Path(parent['stage3_artifact']['path'])
    errors_path = Path(parent['errors_artifact']['path'])
    unweighted_path = Path(compact_report['actual_file']['path'])
    for path, expected in [(stage_path, parent['stage3_artifact']['sha256']),
        (errors_path, parent['errors_artifact']['sha256']), (unweighted_path, compact_report['actual_file']['sha256'])]:
        c.require(c.sha(path) == expected, 'A source three-stage artifact changed.')
    source = codec.load(fixed2_path)
    banks = [*source['stages'], torch.load(stage_path, map_location='cpu', weights_only=True)]
    data = torch.load(layer['path'], map_location='cpu', weights_only=True, mmap=True)
    per_vector = torch.load(errors_path, map_location='cpu', weights_only=True)
    errors = torch.stack([per_vector[f'e{i}_MSE'] for i in range(1, 4)], dim=1)
    c.require(per_vector['target_digest'] == layer['target_digest'], 'Actual prefix-error source changed.')
    activation = torch.load(c.ACTIVATIONS, map_location='cpu', weights_only=True, mmap=True)
    manifest = c.read(c.ACTIVATION_MANIFEST)
    documents = manifest['documents']
    lengths = activation['document_lengths']
    c.require(len(documents) == len(lengths) == 128 and lengths == [row['valid_tokens'] for row in documents]
        and [row['row_index'] for row in documents] == list(range(700000, 700128)), 'Cached document identity/order/lengths changed.')
    x = activation['inputs'][layer['geometry']['name']]
    split = sum(lengths[:64])
    c.require(split == 15183 and sum(lengths[64:]) == 15492 and sum(lengths) == len(x) == 30675
        and x.dtype == torch.bfloat16 and tuple(x.shape) == (30675, 12288), 'Use the exact fixed front/back document halves.')
    paths = [Path(__file__), c.HERE / 'entropy_budget.py', c.HERE / 'third_stage_export.py',
        c.HERE / 'third_stage.py', c.HERE / 'length_codec_probe.py', c.HERE / 'codec.py', c.HERE / 'common.py',
        c.HERE / 'observe.py', c.ROOT / 'rotation/weight_preprocess.py', root / 'prepared/prepared.json',
        parent_path, compact_report_path, fixed2_path, stage_path, errors_path, unweighted_path, c.ACTIVATION_MANIFEST]
    frozen = c.digest({f'{i}/{name}': value for i, stage in enumerate(banks)
        for name, value in {'codes': stage['codes'], **stage['decoder_state']}.items()})
    output.mkdir(parents=True)
    report = dict(status='running', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), resource=resource,
        source_sha256={str(path.resolve()): c.sha(path) for path in paths},
        activation_input=c.stat(c.ACTIVATIONS), target_digest=layer['target_digest'], frozen_bank_digest=frozen,
        protocol=dict(decoder_training=False, actual_stages=3, decoder_reservation=4, maximum_allocation_rounds=4,
            calibration_rows=[700000, 700063], calibration_tokens=15183,
            heldout_train_rows=[700064, 700127], heldout_train_tokens=15492,
            data_source=manifest['data_source'], split='train', both_halves_previously_observed=True,
            heldout_role='Only this round excludes the latter64train documents from weighting/routing; not project-wide unseen or blind evaluation',
            score='normalized_mean_calibration_input_rotation_energy_per_input * actual_prefix_MSE / original_source_variance',
            no_clipping=True, no_quality_based_selection=True))
    c.write(output / 'report.json', report)
    try:
        prefix = codec.decode(source, 'cuda:0')
        h, normalized, vector_weight, kept, checks = weights_and_checks(source, data['vectors'], prefix, x, split, errors[:, 1])
        weight_path = output / 'calibration_input_energy.pt'
        torch.save(dict(h=h, normalized_h=normalized, kept_original_columns=kept,
            document_lengths=lengths, calibration_documents=64, calibration_tokens=split,
            target_digest=layer['target_digest']), weight_path)
        report.update(weight_chain_checks=checks, input_energy=dict(path=str(weight_path), sha256=c.sha(weight_path),
            zero_count=int((h == 0).sum()), mean=float(h.mean()), normalized_mean=float(normalized.mean()),
            normalized_min=float(normalized.min()), normalized_max=float(normalized.max())))
        c.write(output / 'report.json', report)
        print(dict(event='rotation_layout_split_checks_passed', **checks), flush=True)
        weighted_scores = errors / layer['source_variance'] * vector_weight[:, None]
        routed, allocation = iterate(weighted_scores, layer['budget'], output / 'weighted_three_stages',
            original_elements=math.prod(layer['geometry']['original_shape']), actual_stages=3, ideal_stages=[])
        frame_path = Path(allocation['chosen']['length_frame']['artifact']['path'])
        frame = frame_path.read_bytes()
        selected = [{**stage, 'codes': stage['codes'].index_select(0, torch.nonzero(routed >= i, as_tuple=False).flatten()).contiguous()}
            for i, stage in enumerate(banks, start=1)]
        bundle = dict(n_vectors=layer['vector_count'], geometry=source['geometry'], fixed=source['fixed'], stages=selected,
            length_frame=torch.frombuffer(bytearray(frame), dtype=torch.uint8).clone())
        candidate_path = output / 'weighted_three_stage.avrvqz'
        size = save(bundle, candidate_path)
        c.require(size + layer['budget']['decoder_bytes'] == allocation['chosen']['complete_accounting_bytes']
            and allocation['chosen']['feasible'], 'Weighted candidate must meet the same four-decoder conservative budget.')
        fresh, fresh_lengths = load(candidate_path)
        c.require(c.digest(inventory(fresh)) == c.digest(inventory(bundle)) and torch.equal(fresh_lengths, routed), 'Fresh selected codes/decoder/fixed/frame/routes changed.')
        weighted_prefix = decode(fresh, fresh_lengths)
        actual_errors = vector_errors(data['vectors'], weighted_prefix)
        expected_errors = errors.gather(1, (routed.long() - 1)[:, None]).flatten()
        c.require(torch.equal(actual_errors, expected_errors), 'Fresh actual weighted-prefix code ordering or numerical errors changed.')
        unweighted, unweighted_lengths = load(unweighted_path)
        unweighted_prefix = decode(unweighted, unweighted_lengths)
        key = source['geometry']['source_key']
        index = c.read(c.HF / 'model.safetensors.index.json')['weight_map']
        with safe_open(str(c.HF / index[key]), framework='pt', device='cpu') as stream:
            reference = stream.get_tensor(key).to('cuda:0')
        metrics = {}
        for name, state, values in [('fixed2', source, prefix),
            ('unweighted_dynamic3', unweighted, unweighted_prefix), ('weighted_dynamic3', fresh, weighted_prefix)]:
            weight = original_weight(state, values, 'cuda:0')
            metrics[name] = output_metrics(weight, reference, x, split)
            del weight
        for filename, expected_sha in report['source_sha256'].items():
            c.require(c.sha(filename) == expected_sha, 'A bound source/data/codec file changed: ' + filename)
        c.require(c.stat(c.ACTIVATIONS) == report['activation_input'] and frozen == c.digest({f'{i}/{name}': value
            for i, stage in enumerate(banks) for name, value in {'codes': stage['codes'], **stage['decoder_state']}.items()}),
            'Input archive or frozen three-stage bank changed.')
        report.update(status='completed', exit_code=0, source_unchanged=True, frozen_banks_unchanged=True,
            fresh_all_fields_and_routes_exact=True, fresh_actual_prefix_errors_exact=True,
            allocation=allocation, actual_candidate=dict(path=str(candidate_path), bytes=size, sha256=c.sha(candidate_path),
                bits_per_original_linear_weight=8 * size / 50331648, fourth_decoder_reservation_only_bytes=21312,
                conservative_complete_bytes=size + 21312, actual_stages=3),
            actual_unweighted_prefix_NMSE=float(actual_errors.mean() / layer['source_variance']),
            weighted_proxy_score=float((actual_errors / layer['source_variance'] * vector_weight).mean()),
            metrics=metrics,
            boundary='Diagonal unprotected-input covariance proxy ignores input cross-terms, protected-error cross-terms and BF16 inverse-rotation rounding. Actual native original-coordinate outputs above determine this local comparison. No downstream claim, grid, learned gate or decoder update.')
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'report.json', report)
    print(dict(status=report['status'], actual_candidate=report['actual_candidate'], metrics=report['metrics']), flush=True)


if __name__ == '__main__':
    main()
