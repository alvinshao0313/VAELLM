"""Compare native abs005 and two frozen dynamic3 routes on actual L18/L35, GPU2."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from safetensors import safe_open

from . import sources
from experiments.adaptive_layer_transfer_20261028 import helpers as h
from experiments.adaptive_sparse_pair_20261028 import allocate, readout, sparse_file
from experiments.adaptive_vector_rvq import codec, common as c
from experiments.adaptive_vector_rvq.observe import original_weight
from experiments.adaptive_vector_rvq.third_stage import stage_digest, vector_errors
from experiments.adaptive_vector_rvq.third_stage_export import inventory, load, save
from litebsq.sparse_residual import decode_blocked_quantized_sparse_residual
from train_utils.cat_train_residual_protection import build_sparse_residual_payload

DEVICE = 'cuda:0'


def patched_weight(source, payload, vectors):
    weight = original_weight(source, vectors, DEVICE)
    fields = {name: payload['sparse_residual_' + name] for name in sparse_file.FIELDS}
    rows, cols, values = decode_blocked_quantized_sparse_residual(**fields,
        out_features=4096, in_features=12288, block_rows=256, block_cols=256,
        index_bits=8, value_bits=8, value_dtype=torch.bfloat16, device=DEVICE)
    protected = source['fixed']['protected_input_indices'].to(DEVICE)
    c.require(values.dtype == torch.bfloat16 and values.numel() == 251659
        and not bool(torch.isin(cols, protected).any()), 'Native abs005 must exclude all protected columns.')
    weight.index_put_((rows, cols), values, accumulate=True)
    return weight


@torch.no_grad()
def run(output, layer):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '2' and not output.exists(),
        'Use only assigned physical GPU2 and a new isolated output directory.')
    resource = sources.free_gpu2()
    c.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one GPU must be visible.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared, row, source, data, banks, paths = sources.bind(layer)
    manifest = c.read(c.ACTIVATION_MANIFEST)
    c.require(manifest['data_source']['split'] == 'train'
        and c.sha(c.ACTIVATIONS) == manifest['activations_archive']['sha256'], 'Use the unchanged train archive.')
    target, variance = data['vectors'], data['source_variance']
    frozen = [stage_digest(s) for s in banks]
    output.mkdir(parents=True)
    report = dict(status='running', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), resource=resource,
        layer=layer, used_cuda=True, cpu_threads=4, target_digest=row['target_digest'],
        source_sha256={str(p.resolve()): c.sha(p) for p in sorted(paths)},
        input_stats=prepared['input_stats'], endpoints={}, allocations={},
        protocol=dict(train_new_decoder=False, actual_dynamic_stages=3,
            unmaterialized_decoder_reservation_bytes=0, maximum_allocation_rounds=4,
            selection='Last complete-file-feasible allocation round, never quality based.',
            sparse_recipe=dict(rank_metric='sparse_residual_abs', top_p=.005, min_abs=0.,
                codec='blocked_quantized', index_bits=8, value_bits=8, block_shape=[256, 256]),
            data_source=manifest['data_source'], calibration_tokens=15183, heldout_train_tokens=15492,
            calibration_rows=[700000, 700063], heldout_train_rows=[700064, 700127],
            both_halves_previously_observed=True, readout_dtype='torch.bfloat16', batch_tokens=256,
            teacher_calls_per_batch=1, readout_device='physical GPU2',
            output_scope='Local original-HF-context down-linear outputs; no downstream/model-wide score.'),
        compact_format=prepared['compact_format'],
        budget_boundary='One measured common cap, actual bytes and unused bytes for each full file; no exact-size-equality claim.',
        compute_boundary='Frozen dynamic3 bank has extra prior training; the two allocators share the exact same bank.')
    c.write(output / 'report.json', report)

    def emit(event):
        event = {**event, 'layer': layer}
        if event['event'] == 'first_formal_BF16_CPU_batch_complete':
            event['event'] = 'first_formal_BF16_GPU_batch_complete'
            report['first_batch_timing'] = event
        print(event, flush=True)
        c.write(output / 'report.json', report)

    try:
        full = torch.arange(row['vector_count'])
        prefix = torch.zeros((row['vector_count'], c.DIM), dtype=torch.bfloat16)
        errors_list, dense_prefixes = [], []
        for stage in banks:
            codec.add_stage(prefix, stage, full, DEVICE)
            errors_list.append(vector_errors(target, prefix))
            dense_prefixes.append(prefix.clone())
        errors, fixed2_vectors = torch.stack(errors_list, dim=1), dense_prefixes[1]
        archive = torch.load(c.ACTIVATIONS, map_location='cpu', weights_only=True, mmap=True)
        x, split = h.activation_inputs(archive, manifest, row['geometry']['name'])
        _, normalized, vector_weight, _, checks = h.input_energy(
            source, target, fixed2_vectors, x, split, errors[:, 1], DEVICE)
        stored_h = torch.load(row['energy']['path'], map_location='cpu', weights_only=True)['normalized_h']
        h_relative = float(((normalized - stored_h).abs() / stored_h).max())
        c.require(h_relative <= 2e-5, 'Actual GPU input-energy weights differ materially from preparation.')
        index = c.read(c.HF / 'model.safetensors.index.json')['weight_map']
        key = row['geometry']['source_key']
        with safe_open(str(c.HF / index[key]), framework='pt', device='cpu') as stream:
            reference = stream.get_tensor(key).clone().to(DEVICE)
        c.require(reference.dtype == torch.bfloat16 and tuple(reference.shape) == (4096, 12288),
            'Use the complete true BF16 matrix for this exact layer.')
        base_weight = original_weight(source, fixed2_vectors, DEVICE)
        protected = source['fixed']['protected_input_indices']
        sparse_target, base_cpu = reference.cpu(), base_weight.cpu()
        sparse_target[:, protected] = base_cpu[:, protected]
        payload, nnz, storage = build_sparse_residual_payload(linear_name=row['geometry']['name'],
            target_weight=sparse_target, reconstructed_weight=base_cpu, activation_weight=None,
            rank_metric='sparse_residual_abs', top_p=.005, min_abs=0., codec='blocked_quantized',
            index_bits=8, value_bits=8, block_shape=(256, 256))
        c.require(payload is not None and nnz == 251659, 'Keep the native full-original-matrix abs005 density.')
        blocks = payload['sparse_residual_active_block_ids'].numel()
        sparse_bytes = sum(c.compact(payload['sparse_residual_' + name]) for name in sparse_file.FIELDS)
        baseline_bytes = codec.ledger(source)['total_bytes']
        cap = baseline_bytes + sparse_bytes
        c.require(sparse_bytes == storage['codec_bytes'] == 3 * nnz + 14 * blocks + 4
            and baseline_bytes == Path(row['fixed2_path']).stat().st_size == 19607424,
            'Bill complete actual fixed2 plus actual native sparse payload.')
        sparse_path = output / 'sparse_abs005.avrqsp'
        sparse_ledger = sparse_file.save(source, payload, sparse_path)
        fresh_source, fresh_payload = sparse_file.load(sparse_path)
        c.require(c.digest(sparse_file.inventory(fresh_source, fresh_payload))
            == c.digest(sparse_file.inventory(source, payload)) and sources.same_metadata(fresh_source, source)
            and all(fresh_payload['sparse_residual_' + name] == payload['sparse_residual_' + name]
                for name in sparse_file.SPEC), 'Fresh sparse fields or metadata changed.')
        fresh_vectors = codec.decode(fresh_source, DEVICE)
        c.require(torch.equal(fresh_vectors, fixed2_vectors), 'Fresh sparse file changed the full fixed2 decode.')
        sparse_weight = patched_weight(fresh_source, fresh_payload, fresh_vectors)
        c.require(torch.equal(sparse_weight, patched_weight(source, payload, fixed2_vectors))
            and torch.equal(sparse_weight[:, protected], base_weight[:, protected]),
            'Fresh native BF16 patch/protected columns changed.')
        report.update(common_cap_bytes=cap, baseline_complete_bytes=baseline_bytes,
            baseline_fixed_digest=c.digest(source['fixed']), frozen_bank_digests=frozen,
            actual_HF_reference_digest=c.digest({'weight': reference}), weight_chain_checks=checks,
            GPU_prefix_NMSE=[float(v.mean() / variance) for v in errors_list],
            sparse=dict(nnz=nnz, active_blocks=blocks, native_codec_bytes=sparse_bytes,
                ledger=sparse_ledger, old_state_preserved_exact=True, protected_columns_exact=True,
                fresh_fields_metadata_and_BF16_weight_exact=True),
            activation_weights=dict(normalized_mean=float(normalized.mean()),
                normalized_min=float(normalized.min()), normalized_max=float(normalized.max()),
                CPU_GPU_max_relative_difference=h_relative))
        weights, paths_by_name = {'sparse_abs005': sparse_weight}, {'sparse_abs005': sparse_path}
        emit(dict(event='native_sparse_and_common_cap_ready', cap_bytes=cap, nnz=nnz, active_blocks=blocks))
        for name, scores in [('unweighted_dynamic3', errors / variance),
            ('weighted_dynamic3', errors / variance * vector_weight[:, None])]:
            lengths, allocation = allocate.iterate(scores, row['budget'], cap, output / name)
            active = [torch.nonzero(lengths >= i, as_tuple=False).flatten() for i in range(1, 4)]
            selected = [{**stage, 'codes': stage['codes'].index_select(0, ids).contiguous()}
                for stage, ids in zip(banks, active, strict=True)]
            frame = Path(allocation['chosen']['length_frame']['artifact']['path']).read_bytes()
            bundle = dict(n_vectors=row['vector_count'], geometry=source['geometry'], fixed=source['fixed'],
                stages=selected, length_frame=torch.frombuffer(bytearray(frame), dtype=torch.uint8).clone())
            path = output / (name + '.avrvqz')
            size = save(bundle, path)
            c.require(size == allocation['chosen']['actual_complete_bytes'] <= cap, 'Actual dynamic file exceeds the complete-file cap.')
            fresh, loaded_lengths = load(path)
            c.require(c.digest(inventory(fresh)) == c.digest(inventory(bundle))
                and sources.same_metadata(fresh, bundle) and torch.equal(loaded_lengths, lengths),
                'Fresh dynamic fields, normalization or routing changed.')
            vectors = h.decode_compact(fresh, loaded_lengths, DEVICE)
            c.require(torch.equal(vectors, h.decode_compact(bundle, lengths, DEVICE)),
                'Same-path fresh dynamic BF16 decoding changed.')
            expected_vectors = dense_prefixes[0].clone()
            for i in (2, 3):
                ids = torch.nonzero(lengths == i, as_tuple=False).flatten()
                expected_vectors.index_copy_(0, ids, dense_prefixes[i - 1].index_select(0, ids))
            different = (vectors != expected_vectors).any(1)
            diagnostic = dict(exact=not bool(different.any()), differing_vectors=int(different.sum()),
                active_rows=[len(ids) for ids in active], ragged_rows=[len(ids) % c.CHUNK for ids in active],
                maximum_absolute_difference=float((vectors.float() - expected_vectors.float()).abs().max()))
            report.setdefault('dense_selected_prefix_checks', {})[name] = diagnostic
            c.write(output / 'report.json', report)
            c.require(diagnostic['exact'], 'Dense versus selected-prefix mismatch; inspect batch geometry before proceeding.')
            actual_errors = vector_errors(target, vectors)
            c.require(torch.equal(actual_errors, errors.gather(1, (lengths.long() - 1)[:, None]).flatten()),
                'Fresh selected actual errors changed.')
            weight = original_weight(fresh, vectors, DEVICE)
            c.require(torch.equal(weight[:, protected], base_weight[:, protected]), 'Dynamic protected columns changed.')
            weights[name], paths_by_name[name] = weight, path
            report['allocations'][name] = allocation
            report['endpoints'][name] = dict(compressed_coordinate_NMSE=float(actual_errors.mean() / variance),
                weighted_proxy_score=float((actual_errors / variance * vector_weight).mean()),
                fresh_all_fields_routes_and_same_path_BF16_vectors_exact=True,
                dense_selected_prefix_errors_exact=True, protected_columns_exact=True)
            emit(dict(event='dynamic_complete_file_ready', endpoint=name, bytes=size, unused_bytes=cap - size))
        reference_energy = float(reference.double().square().sum())
        for name, weight in weights.items():
            path = paths_by_name[name]
            squared = float((weight.double() - reference.double()).square().sum())
            report['endpoints'].setdefault(name, {}).update(original_BF16_weight_MSE=squared / reference.numel(),
                original_BF16_weight_NMSE=squared / reference_energy,
                file=dict(path=str(path), bytes=path.stat().st_size, sha256=c.sha(path), cap_bytes=cap,
                    unused_bytes=cap - path.stat().st_size,
                    bits_per_original_linear_weight=8 * path.stat().st_size / reference.numel()))
        x_gpu = x.to(DEVICE)
        c.require(reference.is_cuda and x_gpu.is_cuda and all(w.is_cuda and w.dtype == torch.bfloat16
            for w in weights.values()), 'Every teacher and endpoint F.linear must run on the same GPU in BF16.')
        outputs = readout.matched_outputs(weights, reference, x_gpu, split, emit)
        for name, metrics in outputs.items():
            report['endpoints'][name]['outputs'] = metrics
        c.require([stage_digest(s) for s in banks] == frozen, 'Allocation/readout modified a frozen bank.')
        for filename, expected in report['source_sha256'].items():
            c.require(c.sha(filename) == expected, 'Bound source/input changed: ' + filename)
        for item in report['input_stats']:
            c.require(c.stat(item['path']) == item, 'Bound input identity changed: ' + item['path'])
        report.update(status='completed', exit_code=0, source_unchanged=True, frozen_banks_unchanged=True,
            same_cap_and_full_file_accounting_passed=True,
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            conclusion_boundary='One true layer, three fresh local BF16 endpoints and one measured cap; no model-wide or downstream claim.')
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'report.json', report)
    print(dict(status=report['status'], layer=layer, cap_bytes=cap, endpoints=report['endpoints'],
        elapsed_seconds=report['elapsed_seconds']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layer', type=int, choices=h.LAYERS, required=True)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.layer)


if __name__ == '__main__':
    main()
