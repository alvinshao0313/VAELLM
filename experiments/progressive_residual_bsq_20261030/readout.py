"""Wait for the existing Muon, validate saved S4 readout, then run BF16-origin stage1."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from bindings import CATEGORIES, HF, OUTPUT_ROOT, PYTHON, boundary, process_event, s, source_pins
from run import bind_existing_control, execute as formal, prepare as prepare_formal, stage, wait_existing_control
from validate_cpu import pins_digest


def manual_resource_release(retired_plan):
    from experiments.cmcc_runtime_20261030.sequence import proc_start
    previous = s.read(retired_plan)
    failed = s.read(retired_plan.parent / 'run.json')
    s.require(failed['status'] == 'failed' and failed['exit_code'] == 1
        and failed['S4_formal_fit_started'] is False and not failed['stages']
        and (retired_plan.parent / 'continuation.exit').read_text().strip() == '1',
        'This manual release applies only to the actual unstarted, failed S4 continuation.')
    old_control = Path(previous['after_control']['run_report'])
    control_plan = s.read(old_control.parent / 'plan.json')
    pipeline = Path(control_plan['target'])
    reports = [retired_plan.parent / 'run.json', old_control, pipeline / 'run.json',
        Path(control_plan['control_plan']).parent / 'run.json']
    pipeline_run = s.read(pipeline / 'run.json')
    absent_evaluations = []
    for item in pipeline_run['stages'].values():
        argv = item.get('argv', [])
        if item.get('name') == 'full9' and '--output' in argv:
            evaluation = Path(argv[argv.index('--output') + 1])
            if evaluation.exists():
                reports.append(evaluation / 'evaluation.json')
                reports.extend(sorted((evaluation / 'longbench/groups').glob('*/runs/*.json')))
            else:
                s.require(item['status'] == 'exited' and item['exit_code'] != 0,
                    'A successful old evaluation must have its actual output report.')
                absent_evaluations.append(dict(path=str(evaluation), stage_exit_code=item['exit_code'],
                    stage_log=dict(path=item['log'], sha256=s.sha(item['log']))))
    observed = {}

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get('pid'), int):
                pid = value['pid']
                expected = value.get('start_ticks', value.get('start_tick'))
                current = proc_start(pid)
                s.require(current is None or expected is not None and current != expected,
                    f'Old resource predecessor process {pid} is still alive; do not start S4.')
                observed[str(pid)] = dict(expected_start_ticks=expected, current_start_ticks=current, retired=True)
            for entry in value.values():
                visit(entry)
        elif isinstance(value, list):
            for entry in value:
                visit(entry)

    terminal = {}
    for report in reports:
        data = s.read(report)
        visit(data)
        terminal[str(report)] = dict(sha256=s.sha(report), status=data.get('status'), exit_code=data.get('exit_code'))
    visit(previous['after_control'])
    needles = (str(pipeline), str(Path(control_plan['control_plan']).parent))
    other_model_readers = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            argv = [os.fsdecode(value) for value in (entry / 'cmdline').read_bytes().split(b'\0') if value]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if (argv and Path(argv[0]).name.startswith('python') and argv[1:2] != ['-c']
                and any(needle in ' '.join(argv) for needle in needles)):
            other_model_readers.append(dict(pid=int(entry.name), argv=argv))
    from longbench import canonical
    resources = canonical().check_resources(['6', '7'])
    return dict(status='manually_verified_retired_resource_predecessor', checked_unix=time.time(),
        old_plan=dict(path=str(retired_plan), sha256=s.sha(retired_plan)), terminal_reports=terminal,
        retired_processes=observed, other_current_model_readers=other_model_readers, resources=resources,
        reader_scope='Same-checkpoint readers with different process identities are other workspace tasks, not predecessor dependencies.',
        failed_evaluation_outputs_absent=absent_evaluations,
        predecessor_exit_success_required=False,
        scope='Explicit user-woken independent S4 run. Old C0/Muon has no input or selection dependency; no automatic failure continuation.')


def prepare(short, output, retired_resource_plan=None):
    s.require(Path(sys.executable).resolve() == PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU bitvae only.')
    output = output.resolve()
    s.require(output.is_relative_to(OUTPUT_ROOT) and not output.exists(), 'Use one new isolated fixed continuation directory.')
    request = s.read(short / 'request.json')
    fit = s.read(short / 'fit.json')
    s.require((short / 'run.exit').read_text().strip() == '1' and len(fit['boundaries']) == 7
        and [row['category'] for row in fit['groups']] == list(CATEGORIES)
        and all(row['status'] == 'fitted_and_full_encoded' for row in fit['groups'])
        and 'OutOfMemoryError' in fit['error'], 'Bind only the actual all-S4 saved endpoint and original OOM.')
    checkpoint = boundary(fit['boundaries'][-1]['path'])
    s.require(checkpoint['checkpoint_id'] == fit['boundaries'][-1]['checkpoint_id']
        and s.hf_identity(HF) == request['hf_identity'], 'The actual saved S4 or original BF16 changed.')
    cpu_path = short / 'formal_cpu_validation.json'
    cpu = s.read(cpu_path)
    pins = source_pins()
    s.require(cpu['status'] == 'verified_S4_CPU_fixes_pending_GPU_prewarm'
        and cpu['exit_code'] == 0 and cpu['full_native_S4_state_exact'] and cpu['next_real_M36_VAE_initialization_exact'],
        'Current CPU state/RNG/dependency/partition evidence is missing.')
    release = None
    if retired_resource_plan is not None:
        previous = s.read(retired_resource_plan)
        old_pins = previous['source_sha256']
        s.require(cpu['current_source_pins_digest'] == pins_digest(old_pins)
            and previous['CPU_validation']['sha256'] == s.sha(cpu_path)
            and previous['checkpoint'] == checkpoint, 'The original CPU proof/source/model binding changed.')
        changed = {name: dict(previous_sha256=old_pins.get(name), current_sha256=pins.get(name))
            for name in old_pins.keys() | pins.keys() if old_pins.get(name) != pins.get(name)}
        s.require(set(changed) == {str(Path(__file__).resolve())},
            'Only the manual readout control branch may change; reuse requires identical numerical sources.')
        release = manual_resource_release(retired_resource_plan)
        parent = {}
    else:
        s.require(cpu['current_source_pins_digest'] == pins_digest(pins), 'The bound CPU implementation changed.')
        parent = bind_existing_control()
    output.mkdir(parents=True)
    if release is not None:
        original_fit_pins = s.read(short / 'run.json')['source_sha256']
        inherited = ['full_native_S4_state_exact', 'observation_CPU_RNG_preserved',
            'next_real_M36_VAE_initialization_exact', 'runtime_scratch_CPU_release_preserves_persistent_state_exact']
        s.require(all(field in cpu and cpu[field] is True for field in inherited),
            'Each inherited exact property must exist in the original CPU proof.')
        cpu_reuse = {**cpu, 'current_source_pins_digest': pins_digest(pins),
            'source_delta': {name: dict(GPU_verified_sha256=original_fit_pins.get(name), current_sha256=pins.get(name))
                for name in original_fit_pins.keys() | pins.keys() if original_fit_pins.get(name) != pins.get(name)},
            'CPU_execution_scope': 'Manual resource-release control branch only; no full-model CPU test repeated.',
            'reused_CPU_evidence': dict(path=str(cpu_path), sha256=s.sha(cpu_path),
                fields=inherited),
            'control_source_delta': changed, 'manual_resource_release': release,
            'formal_after_control': {}, 'resource_success_is_not_algorithm_dependency': True}
        cpu_path = output / 'control_cpu_validation.json'
        s.write(cpu_path, cpu_reuse)
        s.write(output / 'resource_release.json', release)
    plan = dict(schema='fixed_S4_readout_then_BF16_initialization_v1', short_directory=str(short), output=str(output),
        source_sha256=pins, checkpoint=checkpoint, hf_identity=request['hf_identity'], after_control=parent,
        manual_resource_release=None if release is None else dict(path=str(output / 'resource_release.json'),
            sha256=s.sha(output / 'resource_release.json')),
        CPU_validation=dict(path=str(cpu_path), sha256=s.sha(cpu_path)), original_fit_report=str(short / 'run.json'),
        original_fit_report_sha256=s.sha(short / 'run.json'), original_fit_source_sha256=s.read(short / 'run.json')['source_sha256'],
        formal_output=str(output / 'formal'), readout_output=str(output / 'remaining_validation'),
        original_fitter_live_logits_available=False, automatic_retry=False,
        scope='Only fixed S4 no-recovery initialization, seven full MMLU nodes and final complete nine. '
            'No retraining of the saved short, enhancement, gates or other research.')
    s.write(output / 'plan.json', plan)
    print(json.dumps(dict(status='prepared_fixed_S4_readout_then_formal', plan=str(output / 'plan.json'),
        dependency=parent, checkpoint_id=checkpoint['checkpoint_id'])), flush=True)


def run(plan_path):
    plan = s.read(plan_path)
    output = Path(plan['output'])
    s.require(Path(sys.executable).resolve() == PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU bitvae only.')
    s.require(not (output / 'run.json').exists(), 'This continuation is one-shot; no retry or resume.')
    s.check_pins(plan['source_sha256'])
    record = dict(status='starting_independent_S4_readout_validation' if plan.get('manual_resource_release')
        else 'waiting_existing_C0_Muon32_before_S4_validation', **process_event(os.getpid()),
        started_unix=time.time(), command=[sys.executable, *sys.argv], plan_sha256=s.sha(plan_path),
        S4_formal_fit_started=False, stages=[])
    path = output / 'run.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    try:
        if plan['after_control']:
            wait_existing_control(plan['after_control'], output)
        else:
            release = plan['manual_resource_release']
            s.require(s.sha(release['path']) == release['sha256']
                and s.read(release['path'])['status'] == 'manually_verified_retired_resource_predecessor',
                'Require the explicit manual resource-release evidence for this independent run.')
        s.check_pins(plan['source_sha256'])
        from longbench import canonical
        record['initial_resource_check'] = canonical().check_resources(['6', '7'])
        record['status'] = 'validating_saved_S4_prewarm_and_readout'
        s.write(path, record)
        short = Path(plan['short_directory'])
        original = s.read(short / 'request.json')
        checkpoint = boundary(plan['checkpoint']['path'])
        s.require(checkpoint == plan['checkpoint'] and s.sha(plan['original_fit_report']) == plan['original_fit_report_sha256'],
            'The original saved short evidence changed while waiting.')
        remaining = Path(plan['readout_output'])
        remaining.mkdir()
        request = {**original, 'output_dir': str(remaining), 'input_dir': str(short),
            'source_sha256': plan['source_sha256'], 'consumer_reference_logits': str(remaining / 'fresh_reference_logits.pt')}
        request_path = remaining / 'request.json'
        s.write(request_path, request)
        record['stages'].append(stage(request_path, 'reference', '--component', 'reference',
            '--checkpoint', checkpoint['path'], '--output', str(remaining / 'reference.json')))
        s.write(path, record)
        (remaining / 'nodes/down_proj').mkdir(parents=True)
        record['stages'].append(stage(request_path, 'mmlu', '--component', 'mmlu', '--checkpoint', checkpoint['path'],
            '--output', str(remaining / 'nodes/down_proj/mmlu.json'), '--short',
            '--functional-output', str(remaining / 'nodes/down_proj/functional.json')))
        s.write(path, record)
        record['stages'].append(stage(request_path, 'longbench', '--component', 'longbench',
            '--checkpoint', checkpoint['path'], '--output', str(remaining / 'longbench'), '--short'))
        reference = s.read(remaining / 'reference.json')
        mmlu = s.read(remaining / 'nodes/down_proj/mmlu.json')
        s.require(reference['status'] == 'verified_original_group8_and_streaming_S4_full_prewarm'
            and reference['checkpoint_id'] == mmlu['checkpoint_id'] == checkpoint['checkpoint_id']
            and reference['load_audit']['warmup']['warmed'] == 252
            and reference['load_audit']['warmup']['per_group_scratch_release']['compared_weights'] == 8
            and mmlu['load_audit']['same_input_two_fresh_BF16_logits_exact']
            and mmlu['evals']['lm_eval']['limit'] == 1, 'Actual saved-S4 GPU path evidence is incomplete.')
        proof = dict(status='complete_S4_native_path_validation_with_reused_fit', exit_code=0,
            checkpoint=checkpoint, source_sha256=plan['original_fit_source_sha256'], source_sha256_scope='Original GPU fitter snapshot.',
            readout_source_sha256=plan['source_sha256'], CPU_validation=plan['CPU_validation'],
            original_fit_report=plan['original_fit_report'], original_fit_report_sha256=plan['original_fit_report_sha256'],
            original_fitter_live_logits_available=False, two_independent_fresh_BF16_logits_exact=True,
            actual_saved_fitter_stages=28, actual_updates_per_stage=2, schedule_steps_per_stage=10000,
            readout_reference=dict(path=str(remaining / 'reference.json'), sha256=s.sha(remaining / 'reference.json')),
            mmlu=dict(path=str(remaining / 'nodes/down_proj/mmlu.json'), sha256=s.sha(remaining / 'nodes/down_proj/mmlu.json'), limit=1),
            LongBench='Original three deterministic diagnostic requests, unmodified generation; no task score claimed.',
            boundary='Reused genuine saved S4 fitting evidence plus separately verified current CPU and GPU readout paths. '
                'The lost original fitter-live logits comparison remains unavailable; short quality is not formal quality.')
        proof_path = remaining / 'validation.json'
        s.write(proof_path, proof)
        record['status'] = 'saved_S4_readout_validated_preparing_formal_BF16'
        record['validation_report'] = str(proof_path)
        s.write(path, record)
        prepare_formal(Path(plan['formal_output']), False, proof_path, dependency=plan['after_control'])
        record['status'] = 'executing_authorized_formal_S4_from_BF16'
        record.pop('S4_formal_fit_started', None)
        record['formal_request'] = str(Path(plan['formal_output']) / 'request.json')
        s.write(path, record)
        formal(Path(plan['formal_output']) / 'request.json')
        record.update(status='completed_formal_S4_first_stage', exit_code=0)
    except BaseException as error:
        record.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.time() - record['started_unix']
        s.write(path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--short-directory', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--retired-resource-plan', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.short_directory.resolve(strict=True), args.prepare,
            None if args.retired_resource_plan is None else args.retired_resource_plan.resolve(strict=True))
    else:
        run(args.plan.resolve(strict=True))
