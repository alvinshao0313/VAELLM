"""Run the approved three-layer K8/K24 pair once, then independent GPU fresh."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from . import allocation, artifact, capture, fit, readout
from experiments.adaptive_native_model_20261029 import objective
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_model_20261029.native_prefix import inventory

CPU_REPORT = b.REPO / 'result/qwen3_8b_mmlu65_20261004/down_tail23_short_cmcc_20261104/cpu01/report.json'


def remaining(deadline):
    value = deadline - time.time()
    b.c.require(value > 0, 'The common60min pair deadline has expired; stop without retries or extensions.')
    return value


def check_pins(pins):
    for name, digest in pins.items():
        b.c.require(b.c.sha(name) == digest, 'Bound source changed: ' + name)


def event(name, **fields):
    print(dict(event=name, **fields), flush=True)


def fresh(output, deadline):
    remaining(deadline)
    manifest = b.c.read(output / 'fresh_request.json')
    check_pins(manifest['source_sha256'])
    allowed = {str((output / 'capture_inputs.pt').resolve()),
        *(str(Path(row['path']).resolve()) for row in manifest['artifacts'])}
    original_load = b.torch.load
    reads = []

    def only_expected(path, *args, **kwargs):
        resolved = str(Path(path).resolve())
        b.c.require(resolved in allowed, 'Independent fresh consumer attempted an external parent weight read.')
        reads.append(resolved)
        return original_load(path, *args, **kwargs)

    b.torch.load = only_expected
    inputs = b.torch.load(output / 'capture_inputs.pt', map_location='cpu', weights_only=True, mmap=True)
    rows = []
    for row in manifest['artifacts']:
        remaining(deadline)
        bundle, lengths, module = artifact.load(row['path'])
        weight, actual, fingerprint = readout.deployment(bundle, lengths, module, inputs[row['name']], device='cuda:0')
        b.c.require(fingerprint == row['expected'], 'Independent GPU fresh changed fields/routes/codes/BF16 W or same-input F.linear outputs.')
        rows.append(dict(tag=row['tag'], name=row['name'], all_expected_fingerprints_exact=True,
            actual_file_sha256=b.c.sha(row['path']), actual_active_rows=[int((lengths >= index + 1).sum()) for index in range(len(bundle['stages']))]))
        del module, weight, actual, bundle
        gc.collect()
        b.torch.cuda.empty_cache()
    check_pins(manifest['source_sha256'])
    b.c.write(output / 'fresh_report.json', dict(status='independent_GPU_fresh_verified', exit_code=0,
        pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        command=[sys.executable, *sys.argv], artifacts=rows, actual_weight_reads=reads,
        no_external_parent_weight_read=True, deadline_unix=deadline,
        peak_allocated_bytes=b.torch.cuda.max_memory_allocated(), peak_reserved_bytes=b.torch.cuda.max_memory_reserved()))
    event('independent_GPU_fresh_verified', artifacts=len(rows))


def execute(output, deadline):
    b.c.require(output.is_dir() and not (output / 'report.json').exists(), 'Use only the newly created once-only sequence output.')
    remaining(deadline)
    previous = b.c.read(CPU_REPORT)
    b.c.require(previous['status'] == 'cpu_contracts_verified' and previous['exit_code'] == 0
        and previous['review_patch_validation']['exit_code'] == 0, 'Reuse the completed core CPU review without rerunning it.')
    check_pins(previous['source_sha256'])
    started = time.time()
    report = dict(status='running', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        command=[sys.executable, *sys.argv], started_unix=started, deadline_unix=deadline,
        source_sha256=b.pins(), reused_CPU_report=str(CPU_REPORT), reused_CPU_report_sha256=b.c.sha(CPU_REPORT),
        GPU='physical2/logical0', arms={}, control=[], boundary='Three complete Linears only; no36-layer/full8/recovery500/nine path.')
    path = output / 'report.json'
    b.c.write(path, report)
    try:
        raw, live, specs, source = b.sources()
        report['source_binding'] = source
        bundles = [b.prefix(raw, spec, source)[0] for spec in specs]
        frozen = [b.c.digest({name: value for name, value in inventory(bundle).items() if name != 'length_frame'}) for bundle in bundles]
        event('real_after_up_capture_start')
        inputs, energies, captured = capture.capture(bundles, device='cuda:0')
        gc.collect()
        b.torch.cuda.empty_cache()
        b.torch.save(inputs, output / 'capture_inputs.pt')
        b.torch.save(energies, output / 'route_h.pt')
        report['capture'] = dict(**captured, capture_input_file_sha256=b.c.sha(output / 'capture_inputs.pt'),
            h_file_sha256=b.c.sha(output / 'route_h.pt'), h_digest=b.c.digest(energies))
        b.c.write(path, report)
        event('real_after_up_capture_done', **report['capture'])
        values, references, artifacts = [], {}, []
        for spec, bundle in zip(specs, bundles, strict=True):
            remaining(deadline)
            name = spec['name']
            source_weight = live[name + '.weight']
            value = fit.common_residual(bundle, source_weight, device='cuda:0')
            value['name'] = name
            values.append(value)
            references[name] = readout.outputs(source_weight, inputs[name], device='cuda:0')
            lengths = b.torch.full((b.N,), 2, dtype=b.torch.uint8)
            control_path = output / (name + '.fixed2.pt')
            measured = artifact.save(bundle, control_path)
            module = CompactRVQLinear(bundle, lengths, 'cuda:0').eval()
            weight, actual, fingerprint = readout.deployment(bundle, lengths, module, inputs[name], device='cuda:0')
            errors = objective.prefix_errors(value['target'], value['prefixes'], energies[name], bundle['geometry'], value['info']['source_variance'])[:, 1]
            report['control'].append(dict(name=name, common_residual=value['info'], artifact=measured,
                weighted_residual_NMSE=float(errors.mean()), metrics=readout.metrics(weight, source_weight, actual, references[name])))
            artifacts.append(dict(tag='fixed2', name=name, path=str(control_path), expected=fingerprint))
            del module, weight, actual, errors
            gc.collect()
            b.torch.cuda.empty_cache()
        report['control_aggregate'] = readout.aggregate(report['control'])
        b.c.write(path, report)
        for bits in (8, 24):
            remaining(deadline)
            event('real_fresh_arm_start', K=bits, actual_updates=2000, schedule_steps=10000)
            arm_started = time.time()
            stages, training = fit.fit_group(values, bits, device='cuda:0')
            arm = dict(training=training, layers=[])
            report['arms'][str(bits)] = arm
            for index, (value, stage) in enumerate(zip(values, stages, strict=True)):
                remaining(deadline)
                name = value['name']
                errors, full = fit.errors(value, stage, energies[name], device='cuda:0')
                lengths, selected, routing = allocation.allocate(errors, full, b.CAP)
                actual_frozen = b.c.digest({name: tensor for name, tensor in inventory(selected).items()
                    if name != 'length_frame' and not name.startswith('stage2/')})
                b.c.require(actual_frozen == frozen[index], 'Training/routing changed any frozen code, decoder, protection or rotation sign.')
                for old, current in zip(value['bundle']['stages'], selected['stages'][:2], strict=True):
                    b.c.require(old['decoder_spec'] == current['decoder_spec'] and old['normalization'] == current['normalization'],
                        'Frozen stage metadata changed or normalization was fused again.')
                model_path = output / (name + f'.K{bits}.pt')
                measured = artifact.save(selected, model_path)
                b.c.require(measured['payload_tensor_bytes'] <= b.CAP and bool(((lengths == 2) | (lengths == 3)).all()),
                    'Actual full payload exceeds the independent layer cap or permits a forbidden stage1 exit.')
                module = CompactRVQLinear(selected, lengths, 'cuda:0').eval()
                weight, actual, fingerprint = readout.deployment(selected, lengths, module, inputs[name], device='cuda:0')
                chosen_error = errors.gather(1, (lengths.long() - 2).unsqueeze(1)).mean()
                row = dict(name=name, artifact=measured, artifact_sha256=b.c.sha(model_path), allocation=routing,
                    coverage=float((lengths == 3).double().mean()), weighted_residual_NMSE=float(chosen_error),
                    dense_third_weighted_residual_NMSE=float(errors[:, 1].mean()),
                    frozen_prefix_codes_decoders_protection_signs_exact=True,
                    metrics=readout.metrics(weight, live[name + '.weight'], actual, references[name]))
                arm['layers'].append(row)
                artifacts.append(dict(tag=f'K{bits}', name=name, path=str(model_path), expected=fingerprint))
                del module, weight, actual, errors, full, selected
                gc.collect()
                b.torch.cuda.empty_cache()
            arm['aggregate'] = readout.aggregate(arm['layers'])
            arm['elapsed_seconds_including_export_route_readout'] = time.time() - arm_started
            b.c.write(path, report)
            event('real_arm_completed', K=bits, aggregate=arm['aggregate'])
            del stages
        # One genuinely independent process consumes only the nine compressed artifacts and captured inputs.
        request = dict(source_sha256=b.pins(), artifacts=artifacts, deadline_unix=deadline)
        b.c.write(output / 'fresh_request.json', request)
        gc.collect()
        b.torch.cuda.empty_cache()
        command = [sys.executable, '-B', '-u', '-m', 'experiments.down_tail23_short_cmcc_20261104.run',
            '--output', str(output), '--deadline-unix', str(deadline), '--fresh']
        with (output / 'fresh.log').open('x') as log:
            process = subprocess.Popen(command, cwd=b.REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            report['fresh_process'] = dict(pid=process.pid, command=command, log=str(output / 'fresh.log'), status='running')
            b.c.write(path, report)
            code = process.wait(timeout=remaining(deadline))
        report['fresh_process'].update(exit_code=code, status='completed' if code == 0 else 'failed')
        b.c.require(code == 0, 'Independent fresh failed; stop without retry or dependent followup.')
        report['fresh'] = b.c.read(output / 'fresh_report.json')
        baseline = report['control_aggregate']['train_last8_functional_only']['SSE']
        for arm in report['arms'].values():
            actual = arm['aggregate']['train_last8_functional_only']['SSE']
            arm['last8_output_SSE_relative_improvement_over_same_compact2'] = 1 - actual / baseline
        check_pins(request['source_sha256'])
        remaining(deadline)
        report.update(status='completed_short_pair', exit_code=0,
            both_last8_functional_errors_not_improved=all(arm['aggregate']['train_last8_functional_only']['SSE'] >= baseline
                for arm in report['arms'].values()),
            no_automatic_followup=True, source_sha256=request['source_sha256'])
    except BaseException:
        report.update(status='failed_stop_no_followup', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - started,
            peak_allocated_bytes=b.torch.cuda.max_memory_allocated(), peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
        b.c.write(path, report)
    event('completed_short_pair', arms={key: value['aggregate'] for key, value in report['arms'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    b.c.require(Path(sys.executable).resolve() == b.original.PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == '2',
        'Use the approved caller bitvae and physicalGPU2 only.')
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    if args.fresh:
        fresh(args.output.resolve(), args.deadline_unix)
    else:
        execute(args.output.resolve(), args.deadline_unix)
