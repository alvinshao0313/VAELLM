"""Fresh L0 fixed2+abs005 versus two dynamic3 routes under one measured storage cap."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from safetensors import safe_open

from . import allocate, readout, sparse_file
from experiments.adaptive_layer_transfer_20261028.helpers import activation_inputs, decode_compact, input_energy
from experiments.adaptive_vector_rvq import codec, common as c
from experiments.adaptive_vector_rvq.observe import original_weight
from experiments.adaptive_vector_rvq.third_stage import stage_digest, vector_errors
from experiments.adaptive_vector_rvq.third_stage_export import inventory, load, save
from litebsq.sparse_residual import decode_blocked_quantized_sparse_residual
from train_utils.cat_train_residual_protection import build_sparse_residual_payload

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/star/VAELLM/adaptive_vector_rvq_20261027/repair_20261027')


def same_metadata(left, right):
    return (all(left[key] == right[key] for key in ('n_vectors', 'geometry'))
        and all(a['decoder_spec'] == b['decoder_spec'] and a['normalization'] == b['normalization']
            for a, b in zip(left['stages'], right['stages'], strict=True)))


def patched_weight(source, payload, vectors):
    weight = original_weight(source, vectors, 'cpu')
    fields = {name: payload['sparse_residual_' + name] for name in sparse_file.FIELDS}
    rows, cols, values = decode_blocked_quantized_sparse_residual(**fields,
        out_features=4096, in_features=12288, block_rows=256, block_cols=256,
        index_bits=8, value_bits=8, value_dtype=torch.bfloat16, device='cpu')
    c.require(values.dtype == torch.bfloat16 and values.numel() == 251659
        and not bool(torch.isin(cols, source['fixed']['protected_input_indices']).any()),
        'Native abs005 BF16 payload must preserve the fixed protected columns.')
    weight.index_put_((rows, cols), values, accumulate=True)
    return weight


@torch.no_grad()
def run(output):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
        and not output.exists(), 'Hide CUDA and use a new isolated CPU result directory.')
    torch.set_num_threads(4)
    prepared = c.check_prepared(ROOT / 'prepared')
    row = prepared['layers']['0']
    fixed2_path = ROOT / 'formal_l0/layer_0/fixed2.avrvq'
    control_path = fixed2_path.with_name('report.json')
    parent_path = ROOT / 'entropy_stage3/report.json'
    parent = c.read(parent_path)
    stage_path = Path(parent['stage3_artifact']['path'])
    c.require(parent['status'] == 'completed' and parent['exit_code'] == 0
        and parent['original_stage1_stage2_frozen'] and parent['fresh_stage3_state_and_prefix_exact']
        and parent['target_digest'] == row['target_digest']
        and c.sha(stage_path) == parent['stage3_artifact']['sha256']
        and c.sha(fixed2_path) == parent['source_sha256'][str(fixed2_path)]
        and c.sha(ROOT / 'prepared/prepared.json') == parent['source_sha256'][str(ROOT / 'prepared/prepared.json')],
        'Use the actual completed fresh-HF L0 fixed2 and unchanged trained dense stage3 bank.')
    manifest = c.read(c.ACTIVATION_MANIFEST)
    c.require(c.sha(c.ACTIVATIONS) == manifest['activations_archive']['sha256']
        and manifest['data_source']['split'] == 'train', 'Use the original fixed FineWeb train archive.')
    paths = {Path(path) for path in prepared['source_sha256']}
    paths.update(c.HERE.glob('*.py'))
    paths.update(HERE.glob('*.py'))
    paths.update(HERE.glob('*.sh'))
    paths.update((ROOT / 'prepared/prepared.json', Path(row['path']), fixed2_path, control_path,
        parent_path, stage_path, c.ACTIVATIONS, c.ACTIVATION_MANIFEST,
        c.ROOT / 'experiments/adaptive_layer_transfer_20261028/helpers.py',
        c.ROOT / 'litebsq/sparse_residual.py', c.ROOT / 'litebsq/vae_linear.py',
        c.ROOT / 'train_utils/cat_train_residual_protection.py'))
    output.mkdir(parents=True)
    report = dict(status='running', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), used_cuda=False, cpu_threads=4,
        source_sha256={str(path.resolve()): c.sha(path) for path in sorted(paths)},
        input_stats=[c.stat(row['path']), c.stat(c.ACTIVATIONS), row['HF_shard'], c.stat(fixed2_path)],
        target_digest=row['target_digest'], endpoints={}, allocations={},
        protocol=dict(layer=0, train_new_decoder=False, actual_dynamic_stages=3,
            unmaterialized_decoder_reservation_bytes=0, maximum_allocation_rounds=4,
            selection='Last complete-file-feasible allocation round, never quality based.',
            sparse_recipe=dict(rank_metric='sparse_residual_abs', top_p=.005, min_abs=0.,
                codec='blocked_quantized', index_bits=8, value_bits=8, block_shape=[256, 256]),
            data_source=manifest['data_source'], calibration_tokens=15183, heldout_train_tokens=15492,
            calibration_rows=[700000, 700063], heldout_train_rows=[700064, 700127],
            both_halves_previously_observed=True, readout_dtype='torch.bfloat16', batch_tokens=256,
            output_scope='Actual local down-linear outputs in original HF context, no end-model/downstream accuracy.'),
        budget_boundary='One common measured cap, with each actual file and unused bytes reported. Cap equality is not exact file-size equality.',
        compute_boundary='Dynamic3 reuses an extra pretrained stage3; it is not compute-matched with fixed2+abs005. Its two allocators share one exact bank.')
    c.write(output / 'report.json', report)

    def emit(event):
        print(event, flush=True)
        if event['event'] == 'first_formal_BF16_CPU_batch_complete':
            report['first_batch_timing'] = event
        c.write(output / 'report.json', report)

    try:
        data = torch.load(row['path'], map_location='cpu', weights_only=True, mmap=True)
        source = codec.load(fixed2_path)
        stage3 = torch.load(stage_path, map_location='cpu', weights_only=True)
        target, variance = data['vectors'], data['source_variance']
        control = c.read(control_path)
        c.require(source['mode'] == 'fixed2' and source['geometry'] == data['geometry'] == row['geometry']
            and c.digest(data['fixed']) == c.digest(source['fixed']) == row['fixed_digest']
            and c.digest({'vectors': target}) == row['target_digest']
            and variance == row['source_variance']
            and control['freshdecode']['fixed2']['exact'] and control['allfinite']
            and [stage_digest(stage) for stage in source['stages']] == parent['frozen_stages'],
            'The common true target, fixed2 state, protected payload or original variance changed.')
        banks = [*source['stages'], stage3]
        frozen = [stage_digest(stage) for stage in banks]
        c.require(all(tuple(stage['codes'].shape) == (row['vector_count'], 6) for stage in banks),
            'Reuse all real dense code rows; do not train or synthesize a stage.')
        prefix = torch.zeros((row['vector_count'], c.DIM), dtype=torch.bfloat16)
        full, errors_list, fixed2_vectors = torch.arange(row['vector_count']), [], None
        for index, stage in enumerate(banks, start=1):
            codec.add_stage(prefix, stage, full, 'cpu')
            errors_list.append(vector_errors(target, prefix))
            if index == 2:
                fixed2_vectors = prefix.clone()
        errors = torch.stack(errors_list, dim=1)
        report['CPU_prefix_NMSE'] = [float(value.mean() / variance) for value in errors_list]
        archive = torch.load(c.ACTIVATIONS, map_location='cpu', weights_only=True, mmap=True)
        x, split = activation_inputs(archive, manifest, row['geometry']['name'])
        _, normalized, vector_weight, _, checks = input_energy(
            source, target, fixed2_vectors, x, split, errors[:, 1], 'cpu')
        index = c.read(c.HF / 'model.safetensors.index.json')['weight_map']
        key = row['geometry']['source_key']
        with safe_open(str(c.HF / index[key]), framework='pt', device='cpu') as stream:
            reference = stream.get_tensor(key).clone()
        c.require(reference.dtype == torch.bfloat16 and tuple(reference.shape) == (4096, 12288),
            'Use the original real HF BF16 down matrix as the single reference.')
        base_weight = original_weight(source, fixed2_vectors, 'cpu')
        protected = source['fixed']['protected_input_indices']
        sparse_target = reference.clone()
        sparse_target[:, protected] = base_weight[:, protected]
        payload, nnz, storage = build_sparse_residual_payload(linear_name=row['geometry']['name'],
            target_weight=sparse_target, reconstructed_weight=base_weight, activation_weight=None,
            rank_metric='sparse_residual_abs', top_p=.005, min_abs=0., codec='blocked_quantized',
            index_bits=8, value_bits=8, block_shape=(256, 256))
        c.require(payload is not None and nnz == 251659, 'Preserve the full-original-matrix abs005 selector density.')
        blocks = payload['sparse_residual_active_block_ids'].numel()
        sparse_bytes = sum(c.compact(payload['sparse_residual_' + name]) for name in sparse_file.FIELDS)
        baseline_bytes = codec.ledger(source)['total_bytes']
        cap = baseline_bytes + sparse_bytes
        c.require(sparse_bytes == storage['codec_bytes'] == 3 * nnz + 14 * blocks + 4
            and baseline_bytes == fixed2_path.stat().st_size == row['budget']['control_bytes'] == 19607424,
            'The common cap must come from complete actual fixed2 plus actual native sparse tensors.')
        sparse_path = output / 'sparse_abs005.avrqsp'
        sparse_ledger = sparse_file.save(source, payload, sparse_path)
        fresh_source, fresh_payload = sparse_file.load(sparse_path)
        c.require(c.digest(sparse_file.inventory(fresh_source, fresh_payload))
            == c.digest(sparse_file.inventory(source, payload)) and same_metadata(fresh_source, source)
            and all(fresh_payload['sparse_residual_' + name] == payload['sparse_residual_' + name]
                for name in sparse_file.SPEC), 'Fresh complete sparse fields or metadata changed.')
        fresh_vectors = codec.decode(fresh_source, 'cpu')
        c.require(torch.equal(fresh_vectors, fixed2_vectors), 'Fresh sparse file changed its full fixed2 baseline.')
        sparse_weight = patched_weight(fresh_source, fresh_payload, fresh_vectors)
        c.require(torch.equal(sparse_weight, patched_weight(source, payload, fixed2_vectors))
            and torch.equal(sparse_weight[:, protected], base_weight[:, protected]),
            'Fresh native BF16 sparse application or protected columns changed.')
        report.update(common_cap_bytes=cap, baseline_complete_bytes=baseline_bytes,
            baseline_fixed_digest=c.digest(source['fixed']), frozen_bank_digests=frozen,
            actual_HF_reference_digest=c.digest({'weight': reference}), weight_chain_checks=checks,
            sparse=dict(nnz=nnz, active_blocks=blocks, native_codec_bytes=sparse_bytes,
                ledger=sparse_ledger, old_state_preserved_exact=True, protected_columns_exact=True,
                fresh_fields_metadata_and_BF16_weight_exact=True),
            activation_weights=dict(normalized_mean=float(normalized.mean()),
                normalized_min=float(normalized.min()), normalized_max=float(normalized.max())))
        weights = {'sparse_abs005': sparse_weight}
        paths_by_name = {'sparse_abs005': sparse_path}
        for name, scores in [('unweighted_dynamic3', errors / variance),
            ('weighted_dynamic3', errors / variance * vector_weight[:, None])]:
            lengths, allocation = allocate.iterate(scores, row['budget'], cap, output / name)
            selected = [{**stage, 'codes': stage['codes'].index_select(0,
                torch.nonzero(lengths >= i, as_tuple=False).flatten()).contiguous()}
                for i, stage in enumerate(banks, start=1)]
            frame = Path(allocation['chosen']['length_frame']['artifact']['path']).read_bytes()
            bundle = dict(n_vectors=row['vector_count'], geometry=source['geometry'], fixed=source['fixed'],
                stages=selected, length_frame=torch.frombuffer(bytearray(frame), dtype=torch.uint8).clone())
            path = output / (name + '.avrvqz')
            size = save(bundle, path)
            c.require(size == allocation['chosen']['actual_complete_bytes'] <= cap,
                'Actual three-decoder dynamic file must match the complete common-cap ledger.')
            fresh, loaded_lengths = load(path)
            c.require(c.digest(inventory(fresh)) == c.digest(inventory(bundle))
                and same_metadata(fresh, bundle) and torch.equal(loaded_lengths, lengths),
                'Fresh dynamic fields, metadata or routing changed.')
            vectors = decode_compact(fresh, loaded_lengths, 'cpu')
            actual_errors = vector_errors(target, vectors)
            c.require(torch.equal(actual_errors, errors.gather(1, (lengths.long() - 1)[:, None]).flatten()),
                'Fresh CPU dynamic decoding changed selected actual prefix errors.')
            weight = original_weight(fresh, vectors, 'cpu')
            c.require(torch.equal(weight[:, protected], base_weight[:, protected]),
                'Dynamic allocation changed common INT8 protected BF16 columns.')
            weights[name], paths_by_name[name] = weight, path
            report['allocations'][name] = allocation
            report['endpoints'][name] = dict(compressed_coordinate_NMSE=float(actual_errors.mean() / variance),
                weighted_proxy_score=float((actual_errors / variance * vector_weight).mean()),
                fresh_all_fields_routes_and_prefix_errors_exact=True, protected_columns_exact=True)
        reference_energy = float(reference.double().square().sum())
        for name, weight in weights.items():
            path = paths_by_name[name]
            squared = float((weight.double() - reference.double()).square().sum())
            report['endpoints'].setdefault(name, {}).update(original_BF16_weight_MSE=squared / reference.numel(),
                original_BF16_weight_NMSE=squared / reference_energy,
                file=dict(path=str(path), bytes=path.stat().st_size, sha256=c.sha(path),
                    cap_bytes=cap, unused_bytes=cap - path.stat().st_size,
                    bits_per_original_linear_weight=8 * path.stat().st_size / reference.numel(),
                    delta_bits_per_weight_vs_sparse=8 * (path.stat().st_size - cap) / reference.numel()))
        emit(dict(event='complete_files_and_weight_metrics_ready', cap_bytes=cap,
            endpoints=report['endpoints']))
        outputs = readout.matched_outputs(weights, reference, x, split, emit)
        for name, metrics in outputs.items():
            report['endpoints'][name]['outputs'] = metrics
        c.require([stage_digest(stage) for stage in banks] == frozen,
            'Routing or observations modified the original frozen bank.')
        for filename, expected in report['source_sha256'].items():
            c.require(c.sha(filename) == expected, 'Bound source/input changed: ' + filename)
        for item in report['input_stats']:
            c.require(c.stat(item['path']) == item, 'Bound input identity changed: ' + item['path'])
        c.require(not torch.cuda.is_initialized(), 'The CPU comparison initialized CUDA.')
        report.update(status='completed', exit_code=0, source_unchanged=True, frozen_banks_unchanged=True,
            GPU_initialized=False, same_cap_and_full_file_accounting_passed=True,
            conclusion_boundary='Three fresh L0 local BF16 CPU endpoints under one measured cap. No comparison to recovered bestabs end-model, no downstream accuracy or model-wide advantage claimed.')
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'report.json', report)
    print(dict(status=report['status'], cap_bytes=cap, endpoints=report['endpoints'],
        elapsed_seconds=report['elapsed_seconds']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == '__main__':
    main()
