"""One dense L0 third-stage fit on frozen actual fixed2 prefixes, then bounded zlib routing."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from . import codec, common as c
from .entropy_budget import iterate
from .observe import compare
from .training import fit_stage
from experiments.night_sequence_20261026.queue_wait import free_gpu


def stage_digest(stage):
    return c.digest({'codes': stage['codes'], **stage['decoder_state']})


def vector_errors(target, prefix):
    errors = torch.empty(len(target), dtype=torch.float64)
    for start in range(0, len(target), 65536):
        residual = target[start:start + 65536].double() - prefix[start:start + 65536].double()
        errors[start:start + len(residual)] = residual.square().mean(-1)
    c.require(bool(torch.isfinite(errors).all()), 'Actual deployed-prefix errors must be finite.')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root_dir.resolve(), args.output_dir.resolve()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available()
        and torch.cuda.device_count() == 1 and not output.exists(), 'Use free physical GPU0 and a new output.')
    resource = free_gpu(0)
    torch.set_num_threads(4)
    prepared = c.check_prepared(root / 'prepared')
    layer = prepared['layers']['0']
    fixed2_path = root / 'formal_l0/layer_0/fixed2.avrvq'
    control_report_path = fixed2_path.with_name('report.json')
    diagnostic_path = root / 'threshold_diagnostic/report.json'
    diagnostic = c.read(diagnostic_path)
    errors_path = Path(diagnostic['per_vector']['path'])
    ideal_path = root / 'entropy_budget/report.json'
    ideal = c.read(ideal_path)
    c.require(ideal['status'] == 'completed' and ideal['independent_reference_matches']
        and ideal['ideal_point_better_than_fixed2'], 'Complete the bounded ideal CPU evidence first.')
    c.require(diagnostic['status'] == 'completed' and c.sha(errors_path) == diagnostic['per_vector']['sha256']
        and diagnostic['target_digest'] == layer['target_digest']
        and c.sha(fixed2_path) == diagnostic['source_sha256'][str(fixed2_path)], 'Use the same real frozen L0 source.')
    source = codec.load(fixed2_path)
    data = torch.load(layer['path'], map_location='cpu', weights_only=True, mmap=True)
    stored_errors = torch.load(errors_path, map_location='cpu', weights_only=True)
    control = c.read(control_report_path)
    c.require(source['mode'] == 'fixed2' and source['geometry'] == layer['geometry'] == data['geometry']
        and c.digest(source['fixed']) == layer['fixed_digest']
        and codec.ledger(source)['total_bytes'] == 19607424
        and control['freshdecode']['fixed2']['exact'], 'Actual source geometry/storage/freshdecode changed.')
    frozen = [stage_digest(stage) for stage in source['stages']]
    paths = [Path(__file__), c.HERE / 'entropy_budget.py', c.HERE / 'length_codec_probe.py',
        c.HERE / 'common.py', c.HERE / 'codec.py', c.HERE / 'training.py', c.HERE / 'observe.py',
        root / 'prepared/prepared.json', fixed2_path, control_report_path, errors_path, diagnostic_path, ideal_path]
    output.mkdir(parents=True)
    record = dict(status='running', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), resource=resource,
        source_sha256={str(path.resolve()): c.sha(path) for path in paths},
        target_digest=layer['target_digest'], frozen_stages=frozen,
        protocol=dict(steps=6000, seed=2031, batch_size=8192, lr=.003, scheduler='linear', warmup_steps=0,
            observation_steps=[3000, 6000], final_node=6000, dense_all_vectors=True,
            native_arguments=vars(c.arguments()), decoder_reservation=4,
            actual_trained_stages=3, stage4_trained=False),
        boundary='Extra dense-stage3 fitting is not compute-matched to fixed2. Readouts are L0 weight/local FineWeb-train outputs, not downstream accuracy. Stage4 zero-error allocation is mathematical only.')
    c.write(output / 'report.json', record)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        target, variance = data['vectors'], data['source_variance']
        n, full = len(target), torch.arange(len(target))
        prefix = torch.zeros_like(target, dtype=torch.bfloat16)
        actual_errors = []
        for index, stage in enumerate(source['stages'], start=1):
            codec.add_stage(prefix, stage, full, 'cuda:0')
            error = vector_errors(target, prefix)
            c.require(torch.equal(error, stored_errors[f'e{index}_MSE']), 'Actual frozen BF16 source prefix changed.')
            actual_errors.append(error)
        record['actual_source_prefixes_exact'] = True
        c.write(output / 'report.json', record)
        print(dict(event='dense_stage3_start', steps=6000, seed=2031, vectors=n, batch=8192), flush=True)
        stage3, following, training = fit_stage(target, prefix, full, steps=6000, seed=2031,
            device='cuda:0', output=output / 'stage3_training/stage3', source_variance=variance)
        c.require([stage_digest(stage) for stage in source['stages']] == frozen,
            'Dense-stage3 fitting modified the frozen source stages.')
        stage_path = output / 'stage3.pt'
        torch.save(stage3, stage_path)
        fresh = torch.load(stage_path, map_location='cpu', weights_only=True)
        reconstructed = prefix.clone()
        codec.add_stage(reconstructed, fresh, full, 'cuda:0')
        c.require(stage_digest(fresh) == stage_digest(stage3) and torch.equal(reconstructed, following),
            'Fresh saved dense stage3 state, code order or actual BF16 prefix changed.')
        e3 = vector_errors(target, reconstructed)
        actual_errors.append(e3)
        errors = torch.stack(actual_errors, dim=1)
        per_vector_path = output / 'per_vector_errors.pt'
        torch.save(dict(e1_MSE=actual_errors[0], e2_MSE=actual_errors[1], e3_MSE=e3,
            source_variance=variance, target_digest=layer['target_digest']), per_vector_path)
        original_elements = math.prod(layer['geometry']['original_shape'])
        lengths, real = iterate(errors / variance, layer['budget'], output / 'actual_three_stages',
            original_elements=original_elements, actual_stages=3, ideal_stages=[])
        _, ideal4 = iterate(torch.cat((errors / variance, torch.zeros((n, 1), dtype=torch.float64)), dim=1),
            layer['budget'], output / 'ideal_fourth_stage', original_elements=original_elements,
            actual_stages=3, ideal_stages=[4])
        banks = [*source['stages'], fresh]
        candidate = torch.zeros_like(target, dtype=torch.bfloat16)
        selected_stages = []
        for index, stage in enumerate(banks, start=1):
            active = torch.nonzero(lengths >= index, as_tuple=False).flatten()
            selected = {**stage, 'codes': stage['codes'].index_select(0, active).contiguous()}
            codec.add_stage(candidate, selected, active, 'cuda:0')
            selected_stages.append(selected)
        candidate_errors = vector_errors(target, candidate)
        expected_errors = errors.gather(1, (lengths.long() - 1)[:, None]).flatten()
        c.require(torch.equal(candidate_errors, expected_errors),
            'Actual selected-code BF16 decoding differs from the routed prefix errors.')
        metrics = dict(compressed_coordinate_MSE=float(candidate_errors.mean()),
            compressed_coordinate_NMSE=float(candidate_errors.mean() / variance),
            **compare(source, candidate, prepared['activations'], 'cuda:0'))
        reference = control['outcomes']['fixed2']
        improved = metrics['compressed_coordinate_NMSE'] < reference['normalized_whole_MSE']
        local_improved = metrics['local_output_NMSE'] < reference['local_output_NMSE']
        code_bytes = sum(c.compact(stage['codes']) for stage in selected_stages)
        decoder_bytes = sum(c.compact(value) for stage in selected_stages for value in stage['decoder_state'].values())
        fixed_bytes = sum(c.compact(value) for value in source['fixed'].values())
        frame_bytes = real['chosen']['length_frame']['compressed_total_actual_bytes']
        complete_without_reservation = c.HEADER_BYTES + fixed_bytes + decoder_bytes + code_bytes + frame_bytes
        c.require(decoder_bytes == 3 * layer['budget']['decoder_bytes']
            and complete_without_reservation + layer['budget']['decoder_bytes']
                == real['chosen']['complete_accounting_bytes'], 'Actual three-stage state and fourth-decoder reservation accounting differ.')
        for filename, expected in record['source_sha256'].items():
            c.require(c.sha(filename) == expected, 'Bound source/input changed: ' + filename)
        c.require([stage_digest(stage) for stage in source['stages']] == frozen,
            'Routed readout modified the frozen source banks.')
        record.update(status='completed', exit_code=0, training=training,
            fresh_stage3_state_and_prefix_exact=True, original_stage1_stage2_frozen=True,
            actual_routed_prefix_errors_exact=True, source_unchanged=True,
            stage3_artifact=dict(path=str(stage_path), sha256=c.sha(stage_path), bytes=stage_path.stat().st_size),
            errors_artifact=dict(path=str(per_vector_path), sha256=c.sha(per_vector_path)),
            actual_three_stage_allocation=real, ideal_fourth_stage_allocation=ideal4,
            actual_candidate_metrics=metrics, fixed2_reference=reference,
            compressed_coordinate_improved=improved, local_output_improved=local_improved,
            requires_compact_file_validation=improved or local_improved,
            complete_accounting=dict(code_bytes=code_bytes, actual_decoder_bytes=decoder_bytes,
                fixed_bytes=fixed_bytes, length_frame_bytes=frame_bytes, model_header_bytes=c.HEADER_BYTES,
                unmaterialized_fourth_decoder_reservation_bytes=layer['budget']['decoder_bytes'],
                actual_state_accounting_bytes=complete_without_reservation,
                conservative_budget_accounting_bytes=real['chosen']['complete_accounting_bytes'],
                complete_model_file_written=False),
            stage4_training_started=False)
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        c.write(output / 'report.json', record)
    print({key: record[key] for key in ('status', 'actual_candidate_metrics', 'compressed_coordinate_improved',
        'local_output_improved', 'requires_compact_file_validation')}, flush=True)


if __name__ == '__main__':
    main()
