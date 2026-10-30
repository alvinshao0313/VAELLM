"""Fit one new dense stage3, then compare two allocators on the same frozen bank."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from safetensors import safe_open

from . import helpers as h
from experiments.adaptive_vector_rvq import codec, common as c
from experiments.adaptive_vector_rvq.entropy_budget import iterate
from experiments.adaptive_vector_rvq.observe import original_weight
from experiments.adaptive_vector_rvq.third_stage import stage_digest, vector_errors
from experiments.adaptive_vector_rvq.third_stage_export import inventory, load, save
from experiments.adaptive_vector_rvq.training import fit_stage
from experiments.night_sequence_20261026.queue_wait import free_gpu


def run(prepared_directory, output, layer, steps, contract_directory):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    c.require(visible.isdigit() and not output.exists() and steps in (2, 6000)
        and layer in h.LAYERS and (steps != 2 or layer == 18),
        'Assign one free physical GPU, a new directory, and L18 steps2 or L18/L35 steps6000.')
    resource = free_gpu(int(visible))
    c.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
        'Exactly one visible CUDA device is required for this full-geometry path.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = h.check_prepared(prepared_directory)
    prepared_path = prepared_directory / 'prepared.json'
    contract_path = None
    if steps == 6000:
        c.require(contract_directory is not None, 'Complete the new full-geometry L18 steps2 path before6000.')
        contract_path = contract_directory / 'report.json'
        contract = c.read(contract_path)
        c.require(contract['status'] == 'completed' and contract['exit_code'] == 0
            and contract['steps'] == 2 and contract['layer'] == 18
            and contract['prepared_sha256'] == c.sha(prepared_path)
            and contract['source_unchanged'] and contract['frozen_banks_unchanged']
            and contract['fresh_all_fields_and_routes_exact'] and contract['fresh_actual_prefix_errors_exact'],
            'Use the successful same-preparation full-geometry2-update contract, never its trained state.')
    row = prepared['layers'][str(layer)]
    source = codec.load(Path(row['fixed2_path']))
    data = torch.load(row['path'], map_location='cpu', weights_only=True, mmap=True)
    target, variance = data['vectors'], data['source_variance']
    c.require(source['geometry'] == data['geometry'] == row['geometry']
        and source['n_vectors'] == row['vector_count']
        and c.digest(source['fixed']) == row['fixed_digest']
        and variance == row['source_variance']
        and [stage_digest(stage) for stage in source['stages']] == row['frozen_stage_digests'],
        'The prepared real source, geometry, variance or frozen stages changed.')
    archive = torch.load(prepared['activations']['path'], map_location='cpu', weights_only=True, mmap=True)
    manifest = c.read(prepared['activations']['manifest'])
    x, split = h.activation_inputs(archive, manifest, row['geometry']['name'])
    frozen = [stage_digest(stage) for stage in source['stages']]
    source_hashes = {**prepared['source_sha256'], str(prepared_path): c.sha(prepared_path)}
    source_hashes[str(Path(free_gpu.__code__.co_filename).resolve())] = c.sha(free_gpu.__code__.co_filename)
    if contract_path is not None:
        source_hashes[str(contract_path)] = c.sha(contract_path)
    output.mkdir(parents=True)
    report = dict(status='running', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), resource=resource,
        layer=layer, steps=steps, prepared_sha256=c.sha(prepared_path), source_sha256=source_hashes,
        target_digest=row['target_digest'], frozen_stage1_stage2=frozen,
        protocol={**prepared['protocol'], 'steps': steps, 'observation_steps': [max(1, steps // 2), steps],
            'final_node': steps, 'short_contract_only': steps == 2,
            'native_arguments': vars(c.arguments()), 'starts_fresh_stage3_seed2031': True,
            'gpu_contract_report': str(contract_path) if contract_path else None},
        compact_format=prepared['compact_format'], boundary=prepared['activations']['boundary'],
        compute_boundary='Extra stage3 is not compute-matched against fixed2; only the two allocators share exactly the same trained bank.',
        allocations={}, endpoints={})
    c.write(output / 'report.json', report)
    try:
        full = torch.arange(len(target))
        prefix = torch.zeros((len(target), c.DIM), dtype=torch.bfloat16)
        prefix_errors = []
        for stage in source['stages']:
            codec.add_stage(prefix, stage, full, 'cuda:0')
            prefix_errors.append(vector_errors(target, prefix))
        report['recomputed_fixed2_compressed_NMSE'] = float(prefix_errors[-1].mean() / variance)
        print(dict(event='new_dense_stage3_start', layer=layer, steps=steps, seed=2031,
            vectors=len(target), batch=c.BATCH), flush=True)
        stage3, following, training = fit_stage(target, prefix, full, steps=steps, seed=2031,
            device='cuda:0', output=output / 'stage3_training/stage3', source_variance=variance)
        stage_path = output / 'stage3.pt'
        torch.save(stage3, stage_path)
        fresh_stage3 = torch.load(stage_path, map_location='cpu', weights_only=True)
        reconstructed = prefix.clone()
        codec.add_stage(reconstructed, fresh_stage3, full, 'cuda:0')
        c.require(stage_digest(fresh_stage3) == stage_digest(stage3)
            and fresh_stage3['decoder_spec'] == stage3['decoder_spec']
            and fresh_stage3['normalization'] == stage3['normalization']
            and torch.equal(reconstructed, following), 'Fresh actual dense-stage3 fields or BF16 prefix changed.')
        prefix_errors.append(vector_errors(target, reconstructed))
        errors = torch.stack(prefix_errors, dim=1)
        errors_path = output / 'per_vector_errors.pt'
        torch.save(dict(e1_MSE=prefix_errors[0], e2_MSE=prefix_errors[1], e3_MSE=prefix_errors[2],
            source_variance=variance, target_digest=row['target_digest']), errors_path)
        energy, normalized, vector_weight, kept, checks = h.input_energy(
            source, target, prefix, x, split, errors[:, 1], 'cuda:0')
        cpu_energy = torch.load(row['energy']['path'], map_location='cpu', weights_only=True)
        relative_h = float(((normalized - cpu_energy['normalized_h']).abs()
            / cpu_energy['normalized_h'].clamp_min(torch.finfo(torch.float64).tiny)).max())
        c.require(torch.allclose(normalized, cpu_energy['normalized_h'], rtol=2e-5, atol=0),
            'CPU preparation and assigned-GPU FP32 input-energy transforms disagree.')
        energy_path = output / 'calibration_input_energy.pt'
        torch.save(dict(h=energy, normalized_h=normalized, kept_original_columns=kept,
            document_lengths=archive['document_lengths'], calibration_documents=64,
            calibration_tokens=split, target_digest=row['target_digest']), energy_path)
        banks = [*source['stages'], fresh_stage3]
        frozen_bank = [stage_digest(stage) for stage in banks]
        index = c.read(c.HF / 'model.safetensors.index.json')['weight_map']
        key = row['geometry']['source_key']
        with safe_open(str(c.HF / index[key]), framework='pt', device='cpu') as stream:
            reference = stream.get_tensor(key).to('cuda:0')
        control_weight = original_weight(source, prefix, 'cuda:0')
        report['endpoints']['fixed2'] = dict(compressed_coordinate_MSE=float(errors[:, 1].mean()),
            compressed_coordinate_NMSE=float(errors[:, 1].mean() / variance),
            metrics=h.output_metrics(control_weight, reference, x, split, 'cuda:0'),
            actual_file=dict(path=row['fixed2_path'], bytes=row['fixed2_ledger']['total_bytes'],
                sha256=row['fixed2_sha256']))
        del control_weight, reconstructed, following
        for name, scores in [('unweighted_dynamic3', errors / variance),
            ('weighted_dynamic3', errors / variance * vector_weight[:, None])]:
            allocation_output = output / name
            lengths, allocation = iterate(scores, row['budget'], allocation_output,
                original_elements=50331648, actual_stages=3, ideal_stages=[])
            frame_path = Path(allocation['chosen']['length_frame']['artifact']['path'])
            selected = [{**stage, 'codes': stage['codes'].index_select(0,
                torch.nonzero(lengths >= i, as_tuple=False).flatten()).contiguous()}
                for i, stage in enumerate(banks, start=1)]
            bundle = dict(n_vectors=row['vector_count'], geometry=source['geometry'],
                fixed=source['fixed'], stages=selected,
                length_frame=torch.frombuffer(bytearray(frame_path.read_bytes()), dtype=torch.uint8).clone())
            path = allocation_output / (name + '.avrvqz')
            size = save(bundle, path)
            c.require(size + row['budget']['decoder_bytes'] == allocation['chosen']['complete_accounting_bytes']
                and allocation['chosen']['feasible'] and size <= row['budget']['control_bytes'],
                'Actual3decoder file plus the unmaterialized4th reserve must equal the complete feasible ledger.')
            fresh, loaded_lengths = load(path)
            c.require(c.digest(inventory(fresh)) == c.digest(inventory(bundle))
                and torch.equal(loaded_lengths, lengths) and fresh['geometry'] == bundle['geometry']
                and fresh['n_vectors'] == bundle['n_vectors']
                and all(a['decoder_spec'] == b['decoder_spec'] and a['normalization'] == b['normalization']
                    for a, b in zip(fresh['stages'], bundle['stages'], strict=True)),
                'Fresh actual tensor fields, stage metadata, layer geometry or routing changed.')
            values = h.decode_compact(fresh, loaded_lengths, 'cuda:0')
            actual_errors = vector_errors(target, values)
            expected_errors = errors.gather(1, (lengths.long() - 1)[:, None]).flatten()
            c.require(torch.equal(actual_errors, expected_errors),
                'Fresh compact code order or deployed BF16 prefix/error contract changed.')
            weight = original_weight(fresh, values, 'cuda:0')
            report['endpoints'][name] = dict(compressed_coordinate_MSE=float(actual_errors.mean()),
                compressed_coordinate_NMSE=float(actual_errors.mean() / variance),
                weighted_proxy_score=float((actual_errors / variance * vector_weight).mean()),
                metrics=h.output_metrics(weight, reference, x, split, 'cuda:0'),
                actual_file=dict(path=str(path), bytes=size, sha256=c.sha(path),
                    bits_per_original_linear_weight=8 * size / 50331648,
                    actual_decoder_stages=3, fourth_decoder_reservation_only_bytes=row['budget']['decoder_bytes'],
                    conservative_complete_bytes=size + row['budget']['decoder_bytes']))
            report['allocations'][name] = allocation
            c.write(output / 'report.json', report)
            print(dict(event='matched_endpoint_complete', layer=layer, endpoint=name,
                actual_file_bytes=size, metrics=report['endpoints'][name]['metrics']), flush=True)
            del fresh, bundle, values, weight, actual_errors, expected_errors
        c.require([stage_digest(stage) for stage in source['stages']] == frozen
            and [stage_digest(stage) for stage in banks] == frozen_bank,
            'Training, allocation or observations modified a frozen bank.')
        for filename, expected in report['source_sha256'].items():
            c.require(c.sha(filename) == expected, 'Bound source/input changed: ' + filename)
        for item in prepared['input_stats']:
            c.require(c.stat(item['path']) == item, 'Prepared input identity changed: ' + item['path'])
        report.update(status='completed', exit_code=0, training=training, source_unchanged=True,
            frozen_bank_digests=frozen_bank, frozen_banks_unchanged=True,
            fresh_stage3_state_and_prefix_exact=True, fresh_all_fields_and_routes_exact=True,
            fresh_actual_prefix_errors_exact=True, weight_chain_checks=checks,
            CPU_GPU_normalized_h_max_relative_difference=relative_h,
            CPU_GPU_h_relative_tolerance=2e-5, no_clipping_or_epsilon_in_h=True,
            stage3_artifact=dict(path=str(stage_path), bytes=stage_path.stat().st_size, sha256=c.sha(stage_path)),
            errors_artifact=dict(path=str(errors_path), sha256=c.sha(errors_path)),
            input_energy_artifact=dict(path=str(energy_path), sha256=c.sha(energy_path)),
            stage4_training_started=False,
            conclusion_boundary='Actual local down-linear readouts only; diagonal input proxy omits covariance/protected-error cross terms and BF16 inverse rounding. No full-model or downstream claim.')
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'report.json', report)
    print(dict(status=report['status'], layer=layer, steps=steps,
        elapsed_seconds=report['elapsed_seconds']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layer', type=int, choices=h.LAYERS, required=True)
    parser.add_argument('--steps', type=int, choices=(2, 6000), required=True)
    parser.add_argument('--gpu-contract-dir', type=Path)
    args = parser.parse_args()
    run(args.prepared_dir.resolve(), args.output_dir.resolve(), args.layer, args.steps,
        args.gpu_contract_dir.resolve() if args.gpu_contract_dir else None)


if __name__ == '__main__':
    main()
