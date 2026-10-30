"""Bind existing true L18/L35 inputs and check input-energy geometry on CPU."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from . import helpers as h
from experiments.adaptive_vector_rvq import codec, common as c
from experiments.adaptive_vector_rvq.third_stage import stage_digest, vector_errors


def prepare(source_directory, output):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
        and not output.exists(), 'Hide CUDA and use a fresh independent preparation directory.')
    torch.set_num_threads(4)
    original = c.check_prepared(source_directory)
    manifest = c.read(c.ACTIVATION_MANIFEST)
    expected_data = dict(repo='HuggingFaceFW/fineweb-edu',
        revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9', config='sample-10BT', split='train',
        license='odc-by', license_version='Open Data Commons Attribution License v1.0')
    c.require(all(manifest['data_source'][key] == value for key, value in expected_data.items()),
        'Use the existing exact FineWeb-Edu train source/version/license.')
    c.require(c.sha(c.ACTIVATIONS) == manifest['activations_archive']['sha256'],
        'The original complete activation archive SHA changed.')
    archive = torch.load(c.ACTIVATIONS, map_location='cpu', weights_only=True, mmap=True)
    paths = {Path(path) for path in original['source_sha256']}
    paths.update(c.HERE.glob('*.py'))
    paths.update(h.HERE.glob('*.py'))
    paths.update(h.HERE.glob('*.sh'))
    paths.update((source_directory / 'prepared.json', c.ACTIVATION_MANIFEST, c.ACTIVATIONS))
    paths.add(c.ROOT / 'experiments/night_sequence_20261026/queue_wait.py')
    stats = [c.stat(c.ACTIVATIONS)]
    for layer in h.LAYERS:
        row = original['layers'][str(layer)]
        fixed2_path = source_directory.parent / f'formal_l18_l35/layer_{layer}/fixed2.avrvq'
        paths.update((Path(row['path']), fixed2_path, fixed2_path.with_name('report.json')))
        stats.extend((c.stat(row['path']), row['HF_shard'], c.stat(fixed2_path)))
    output.mkdir(parents=True)
    report = dict(status='running', schema='adaptive_layer_transfer_20261028_inputs_v1',
        pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), used_cuda=False,
        cpu_threads=4, source_prepared=str(source_directory / 'prepared.json'), layers={},
        source_sha256={str(path.resolve()): c.sha(path) for path in sorted(paths)},
        input_stats=list({item['path']: item for item in stats}.values()),
        protocol=dict(layers=list(h.LAYERS), seed=2031, steps=6000, batch=8192, lr=.003,
            scheduler='linear', warmup_steps=0, dim=32, bits=48, hidden=128, residual_blocks=0,
            actual_stages=3, reserved_decoder_stages=4, maximum_allocation_rounds=4,
            shared_fixed_stage1_stage2=True, fresh_dense_stage3=True, stage4_trained=False,
            both_allocators_share_one_frozen_bank=True, no_quality_based_selection=True),
        activations=dict(path=str(c.ACTIVATIONS), manifest=str(c.ACTIVATION_MANIFEST),
            data_source=manifest['data_source'], document_lengths=archive['document_lengths'],
            calibration_rows=[700000, 700063], calibration_tokens=15183,
            heldout_train_rows=[700064, 700127], heldout_train_tokens=15492,
            both_halves_previously_observed=True,
            boundary='Only this round excludes latter64train documents from weighting/routing; no project-wide blind or downstream evaluation.'),
        compact_format=dict(reused_module='experiments.adaptive_vector_rvq.third_stage_export',
            historical_schema_tag='adaptive_vector_rvq_zlib_l0_three_stage_v1',
            actual_layer_identity='Always the exact bound geometry.name/source_key; historical l0 schema text is a format name.'),
        gpu_contract_status='not_run; require full-geometry steps2 on a genuinely available GPU before steps6000')
    c.write(output / 'prepared.json', report)
    try:
        for layer in h.LAYERS:
            row = original['layers'][str(layer)]
            matrix_path = Path(row['path'])
            fixed2_path = source_directory.parent / f'formal_l18_l35/layer_{layer}/fixed2.avrvq'
            control_path = fixed2_path.with_name('report.json')
            control = c.read(control_path)
            data = torch.load(matrix_path, map_location='cpu', weights_only=True, mmap=True)
            target = data['vectors']
            source = codec.load(fixed2_path)
            ledger = codec.ledger(source)
            c.require(tuple(target.shape) == (1540096, 32) and target.dtype == torch.float32
                and bool(torch.isfinite(target).all())
                and c.digest({'vectors': target}) == row['target_digest']
                and c.digest(data['fixed']) == c.digest(source['fixed']) == row['fixed_digest']
                and source['mode'] == 'fixed2' and source['n_vectors'] == row['vector_count']
                and source['geometry'] == row['geometry'] == data['geometry']
                and row['geometry']['name'] == f'model.layers.{layer}.mlp.down_proj'
                and row['geometry']['transpose'] and row['geometry']['rotation']['block_size'] == 32
                and source['fixed']['protected_input_indices'].numel() == 256,
                'Bind the exact fresh-HF target, complete geometry and frozen control payload.')
            variance = float(target.double().var(unbiased=False))
            c.require(variance == row['source_variance'] == data['source_variance']
                and control['allfinite'] and control['freeze']['passed']
                and control['freshdecode']['fixed2']['exact']
                and ledger == control['outcomes']['fixed2']['ledger']
                and ledger['total_bytes'] == fixed2_path.stat().st_size == row['budget']['control_bytes'] == 19607424
                and ledger['decoder_bytes'] == 2 * row['budget']['decoder_bytes'] == 42624,
                'The original fixed2 control, complete ledger or source variance changed.')
            fresh = codec.load(fixed2_path)
            c.require(c.digest(codec.tensor_inventory(fresh)) == c.digest(codec.tensor_inventory(source))
                and fresh['geometry'] == source['geometry'] and fresh['n_vectors'] == source['n_vectors']
                and all(a['decoder_spec'] == b['decoder_spec'] and a['normalization'] == b['normalization']
                    for a, b in zip(fresh['stages'], source['stages'], strict=True)),
                'Independent CPU fixed2 load changed persistent state or metadata.')
            frozen = [stage_digest(stage) for stage in source['stages']]
            prefix = torch.zeros((len(target), c.DIM), dtype=torch.bfloat16)
            full = torch.arange(len(target))
            prefix_nmse = []
            for stage in source['stages']:
                codec.add_stage(prefix, stage, full, 'cpu')
                errors = vector_errors(target, prefix)
                prefix_nmse.append(float(errors.mean() / variance))
            x, split = h.activation_inputs(archive, manifest, row['geometry']['name'])
            energy, normalized, weights, kept, checks = h.input_energy(source, target, prefix, x, split, errors, 'cpu')
            energy_path = output / f'layer_{layer}_input_energy.pt'
            torch.save(dict(h=energy, normalized_h=normalized, kept_original_columns=kept,
                calibration_documents=64, calibration_tokens=split, document_lengths=archive['document_lengths'],
                target_digest=row['target_digest']), energy_path)
            c.require([stage_digest(stage) for stage in source['stages']] == frozen,
                'CPU observations modified a frozen source stage.')
            old_nmse = control['outcomes']['fixed2']['normalized_whole_MSE']
            report['layers'][str(layer)] = dict(**row, fixed2_path=str(fixed2_path),
                control_report=str(control_path), fixed2_sha256=c.sha(fixed2_path),
                fixed2_ledger=ledger, frozen_stage_digests=frozen, fixed2_control=control['outcomes']['fixed2'],
                energy=dict(path=str(energy_path), sha256=c.sha(energy_path), bytes=energy_path.stat().st_size,
                    zero_count=int((energy == 0).sum()), mean=float(energy.mean()),
                    normalized_mean=float(normalized.mean()), normalized_min=float(normalized.min()),
                    normalized_max=float(normalized.max())), weight_chain_checks=checks,
                cpu_checks=dict(target_digest_exact=True, source_variance_exact=True,
                    fresh_persistent_fields_exact=True, source_stages_unchanged=True,
                    actual_prefix_NMSE=prefix_nmse, old_GPU_fixed2_NMSE=old_nmse,
                    relative_CPU_vs_old_GPU_NMSE_difference=prefix_nmse[-1] / old_nmse - 1,
                    boundary='CPU BF16 kernel readout is a preparation check; formal GPU endpoints are recomputed together on their assigned device.'))
            report['source_sha256'][str(energy_path)] = c.sha(energy_path)
            c.write(output / 'prepared.json', report)
            print(dict(event='cpu_layer_prepared', layer=layer, cpu_prefix_NMSE=prefix_nmse,
                control_actual_bytes=ledger['total_bytes'], weight_chain_checks=checks), flush=True)
            del data, target, source, fresh, prefix, errors, energy, normalized, weights
            gc.collect()
        for filename, expected in report['source_sha256'].items():
            c.require(c.sha(filename) == expected, 'Bound source/input changed during CPU preparation: ' + filename)
        for item in report['input_stats']:
            c.require(c.stat(item['path']) == item, 'Bound input identity changed during CPU preparation: ' + item['path'])
        c.require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
        report.update(status='completed', exit_code=0, source_unchanged=True,
            GPU_initialized=False, full_geometry_CPU_checks_passed=True)
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'prepared.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-prepared-dir', type=Path,
        default=c.OUTPUT / 'repair_20261027/prepared')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.source_prepared_dir.resolve(), args.output_dir.resolve())
    print(dict(status=report['status'], layers=list(report['layers']),
        elapsed_seconds=report['elapsed_seconds'], gpu_contract_status=report['gpu_contract_status']), flush=True)


if __name__ == '__main__':
    main()
