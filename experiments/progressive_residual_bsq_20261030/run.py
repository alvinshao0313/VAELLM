"""Fixed BF16 -> seven cumulative S4 nodes; no recovery or research decisions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from bindings import BASELINE, CATEGORIES, HF, HERE, OUTPUT_ROOT, PYTHON, ROOT, boundary, environment, process_event, s, source_pins
import recipe


def bind_existing_control():
    from experiments.fresh_muon_cmcc_20261030 import after_control
    controller = after_control.process_identity(420498)
    shell = after_control.process_identity(420496)
    report_path = after_control.DIRECTORY / 'run.json'
    report = s.read(report_path)
    exit_path = after_control.DIRECTORY / 'continuation.exit'
    expected = [str(PYTHON), '-B', '-u', '-m', 'experiments.fresh_muon_cmcc_20261030.after_control', '--run']
    s.require((controller['pid'], controller['start_ticks']) == (420498, 11732332)
        and (shell['pid'], shell['start_ticks']) == (420496, 11732331)
        and controller['parent_pid'] == shell['pid'] and controller['argv'] == expected
        and shell['argv'][:2] == ['bash', '-c'] and len(shell['argv']) == 3
        and ' '.join(expected) in shell['argv'][2] and str(exit_path) in shell['argv'][2]
        and (report['pid'], report['start_ticks']) == (420498, 11732332)
        and report['plan_sha256'] == s.sha(after_control.DIRECTORY / 'plan.json'),
        'Bind only the actual existing C0->Muon controller and its real exit-publication shell.')
    return dict(controller=controller, shell=shell, run_report=str(report_path), exit_file=str(exit_path),
        expected_status='completed_existing_C0_Muon32_pipeline', plan_sha256=report['plan_sha256'])


def wait_existing_control(parent, output):
    from experiments.cmcc_runtime_20261030.sequence import wait_identity
    path = output / 'waiting_dependency.json'
    record = dict(status='waiting_existing_C0_Muon32_controller', **process_event(os.getpid()),
        dependency=parent, started_unix=time.time(), S4_fit_started=False)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    try:
        observed = [wait_identity(parent[name]['pid'], parent[name]['start_ticks']) for name in ('controller', 'shell')]
        actual = s.read(parent['run_report'])
        identity = parent['controller']
        s.require((actual['pid'], actual['start_ticks']) == (identity['pid'], identity['start_ticks'])
            and actual['status'] == parent['expected_status'] and actual['exit_code'] == 0
            and actual['plan_sha256'] == parent['plan_sha256']
            and Path(parent['exit_file']).read_text().strip() == '0',
            'The actual C0->Muon controller/shell did not publish a successful terminal state.')
        record.update(status='existing_C0_Muon32_dependency_completed', observed=observed)
    except BaseException as error:
        record.update(status='failed_existing_dependency', error=repr(error))
        raise
    finally:
        s.write(path, record)


def validation_binding(proof_path, pins):
    import hashlib
    verified = s.read(proof_path)
    s.require(verified['status'] in ('complete_s4_initialization_short', 'complete_S4_native_path_validation_with_reused_fit') and verified['exit_code'] == 0,
        'Require complete genuine S4 GPU short evidence, preserving its original hashes.')
    result = dict(path=str(proof_path.resolve()), sha256=s.sha(proof_path), GPU_source_sha256=verified['source_sha256'])
    if verified['source_sha256'] != pins or verified['status'] == 'complete_S4_native_path_validation_with_reused_fit':
        cpu_path = Path(verified['CPU_validation']['path']) if verified.get('CPU_validation') else proof_path.parent / 'formal_cpu_validation.json'
        cpu = s.read(cpu_path)
        digest = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
        delta = {path: dict(GPU_verified_sha256=verified['source_sha256'].get(path), current_sha256=pins.get(path))
            for path in sorted(verified['source_sha256'].keys() | pins.keys())
            if verified['source_sha256'].get(path) != pins.get(path)}
        s.require(cpu['status'] == 'verified_S4_CPU_fixes_pending_GPU_prewarm'
            and cpu['exit_code'] == 0 and cpu['original_GPU_report_sha256'] == verified.get('original_fit_report_sha256', s.sha(proof_path))
            and cpu['current_source_pins_digest'] == digest and cpu['source_delta'] == delta
            and cpu['full_native_S4_state_exact'] and cpu['observation_CPU_RNG_preserved']
            and cpu['next_real_M36_VAE_initialization_exact'] and cpu['cuda_initialized'] is False,
            'Changed CPU branches lack current, scope-specific real evidence.')
        if verified['status'] == 'complete_S4_native_path_validation_with_reused_fit':
            s.require(verified['readout_source_sha256'] == pins and verified['original_fitter_live_logits_available'] is False,
                'Readout evidence must bind its actual current source and preserve the lost fitter-live boundary.')
        result['CPU_followup'] = dict(path=str(cpu_path), sha256=s.sha(cpu_path), source_delta=delta,
            boundary=cpu['reuse_boundary'])
    return result



def prepare(output, short, proof, dependency=None):
    s.require(Path(sys.executable).resolve() == PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
        'Prepare on CPU in the caller-activated bitvae environment.')
    output = output.resolve()
    s.require(output.is_relative_to(OUTPUT_ROOT) and not output.exists(), 'Use a new isolated artifact directory.')
    resolved = recipe.describe(output / 'native')
    for category, cfg in resolved['category_configs'].items():
        dim = 64 if category == 'gate_proj' else 32
        s.require(cfg['codebook_dim'] == cfg['codebook_bits'] == dim and cfg['residual_stages'] == 4
            and cfg['steps'] == 10000 and cfg['base_ch'] == 128 and cfg['num_res_blocks'] == 0,
            'Resolved native S4 geometry differs from the confirmed recipe.')
    pins = source_pins()
    validation = None
    if not short:
        s.require(proof is not None, 'Bind completed real S4 short-path evidence for formal execution.')
        validation = validation_binding(proof, pins)
    migration = s.read(BASELINE / 'migration_binding.json')
    s.require(migration['status'] == 'verified', 'Use verified same-model BF16 evaluation evidence.')
    for item in migration['bindings'].values():
        s.require(s.sha(item['path']) == item['sha256'], 'Verified BF16 baseline changed.')
    for shard in ('mmlu', 'seven'):
        reference = s.read(BASELINE / (shard + '.json'))
        from experiments.w25_eval.acceptance import leaf_metrics
        metrics, _, batch = leaf_metrics(reference['results'], reference['n-samples'], reference['n-shot'], reference['configs'])
        s.require(shard == 'mmlu' and set(metrics) == {'mmlu'} and batch == 64 or
            shard == 'seven' and len(metrics) == 7 and batch == 32, 'Wrong actual BF16 shard schema.')
    output.mkdir(parents=True)
    import torch
    from transformers import AutoTokenizer
    from train_utils.lora_data import build_calibration_input_ids
    tokenizer = AutoTokenizer.from_pretrained(str(HF), use_fast=False, trust_remote_code=True, local_files_only=True)
    ids = build_calibration_input_ids('alpaca=1', tokenizer=tokenizer, nsamples=4, seqlen=256, seed=2031)
    from fit import tensor_digest
    input_sha = tensor_digest({str(index): value for index, value in enumerate(ids)})
    torch.save(dict(input_ids=ids, input_ids_sha256=input_sha), output / 'functional_inputs.pt')
    request = dict(schema='progressive_bf16_s4_request_v1', output_dir=str(output), fit_gpu=6, eval_gpu=7,
        validation_only=short, formal_parent_eligible=not short, source_sha256=pins,
        hf_identity=s.hf_identity(HF), recipe=resolved, base_model_path=str(HF),
        baseline_binding=dict(path=str(BASELINE / 'migration_binding.json'), sha256=s.sha(BASELINE / 'migration_binding.json')),
        calibration=dict(repository='vicgalle/alpaca-gpt4', split='train', license='cc-by-nc-4.0',
            samples=512, length=256, seed=31, use='Original BF16 protection calibration only; no distillation.'),
        functional_inputs=dict(repository='vicgalle/alpaca-gpt4', split='train', license='cc-by-nc-4.0',
            samples=4, length=256, seed=2031, input_ids_sha256=input_sha,
            tensors_sha256=s.sha(output / 'functional_inputs.pt'), use='Fixed train functional probe; excluded from optimization.'),
        actual_updates_per_stage=2 if short else 10000, schedule_steps_per_stage=10000,
        fixed_pipeline='GPU6 native q/k/v/o/gate/up/down S4; GPU7 sequential cumulative full MMLU; final seven+LB21 reuse MMLU.',
        validation_report=validation)
    data_cache = Path('/home/cmcc/.cache/huggingface/hub/datasets--vicgalle--alpaca-gpt4/snapshots/f7e3ded725cb81e8e564e32feb12860f376f2b51')
    data_files = (data_cache / 'README.md', data_cache / 'data/train-00000-of-00001-6ef3991c06080e14.parquet')
    request['data_source'] = dict(repository='vicgalle/alpaca-gpt4', revision=data_cache.name, split='train',
        files={str(path): dict(sha256=s.sha(path), bytes=path.stat().st_size) for path in data_files})
    if not short:
        request['after_control'] = dependency if dependency is not None else bind_existing_control()
    s.write(output / 'request.json', request)
    print(json.dumps(dict(status='prepared', request=str(output / 'request.json'), validation_only=short,
        source_files=len(pins), hf= str(HF), functional_input_sha256=input_sha)), flush=True)


def stage(request_path, name, *arguments, gpu=None):
    request = s.read(request_path)
    output = Path(request['output_dir'])
    command = [str(PYTHON), '-B', '-u', str(HERE / 'evaluation.py'), '--request', str(request_path), *arguments]
    with (output / (name + '.log')).open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment(request['eval_gpu'] if gpu is None else gpu), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
        event = dict(name=name, command=command, **process_event(process.pid), started_unix=time.time())
        code = process.wait()
    event.update(exit_code=code, elapsed_seconds=time.time() - event['started_unix'])
    s.write(output / (name + '.stage.json'), event)
    s.require(code == 0, name + ' failed; inspect its original log. No automatic retry.')
    return event


def evaluate_stream(request_path, event_fd):
    request = s.read(request_path)
    output = Path(request['output_dir'])
    report = dict(status='running', **process_event(os.getpid()), nodes=[], stages=[], validation_only=request['validation_only'])
    path = output / 'evaluation.json'
    s.write(path, report)
    try:
        report['stages'].append(stage(request_path, 'teacher', '--component', 'teacher'))
        with os.fdopen(event_fd, 'r', encoding='utf-8') as stream:
            for index, line in enumerate(stream):
                event = json.loads(line)
                s.require(index < 7 and event['category'] == CATEGORIES[index]
                    and boundary(event['path'])['checkpoint_id'] == event['checkpoint_id'], 'Wrong cumulative node order or identity.')
                report['nodes'].append(event)
                s.write(path, report)
                if request['validation_only'] and index == 0:
                    report['stages'].append(stage(request_path, 'q_partial_probe', '--component', 'probe',
                        '--checkpoint', event['path'], '--output', str(output / 'q_partial_probe.json')))
                    s.write(path, report)
                if request['validation_only'] and index < 6:
                    continue
                node_dir = output / 'nodes' / event['category']
                node_dir.mkdir(parents=True)
                arguments = ['--component', 'mmlu', '--checkpoint', event['path'], '--output', str(node_dir / 'mmlu.json'),
                    '--functional-output', str(node_dir / 'functional.json')]
                if request['validation_only']:
                    arguments += ['--short']
                report['stages'].append(stage(request_path, event['category'] + '_mmlu', *arguments))
                s.write(path, report)
        s.require(len(report['nodes']) == 7, 'Fit stream ended before all seven genuine saves.')
        if request['validation_only']:
            final = report['nodes'][-1]
            report['stages'].append(stage(request_path, 'short_longbench', '--component', 'longbench',
                '--checkpoint', final['path'], '--output', str(output / 'short_longbench'), '--short'))
        if not request['validation_only']:
            from longbench import execute as terminal_evaluation
            terminal_evaluation(request_path)
        report.update(status='complete_s4_initialization_short' if request['validation_only'] else 'complete_s4_initialization_evaluation', exit_code=0)
    except BaseException as error:
        report.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        s.write(path, report)


def summarize(output, request, report_path=None):
    from experiments.w25_eval.model_inventory import model_inventory
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
    from experiments.compare_longbench_v1 import comparison
    from train_utils.checkpoint_bpw import audit_checkpoint
    last = s.read(output / 'evaluation.json')['nodes'][-1]
    directory = Path(last['path'])
    meta = s.read(directory / 'checkpoint_meta.json')
    inventory = model_inventory(HF)
    node = output / 'nodes/down_proj'
    paths = [node / 'mmlu.json', node / 'seven.json']
    low = collect_results(paths, inventory, checkpoint=(directory, meta))
    baseline = collect_results([BASELINE / 'mmlu.json', BASELINE / 'seven.json'], inventory)
    s.require(low['versions'] == baseline['versions'], 'Full8 versions differ from same-model BF16.')
    lb = comparison(BASELINE / 'longbench/scores.json', node / 'longbench/merged/scores.json', str(__import__('bindings').MANIFEST))
    budget = audit_checkpoint(directory, max_bpw=16, max_lora_rank=0)
    s.require(not budget['contract_violations'] and budget['budget_passed'], 'Actual S4 storage contract failed.')
    import math
    scores = {name: value * 100 for name, value in low['task_metrics'].items()}
    dense = {name: value * 100 for name, value in baseline['task_metrics'].items()}
    scores['longbench'] = lb['task_macro_mean']['compressed']
    dense['longbench'] = lb['task_macro_mean']['dense']
    s.write(report_path if report_path is not None else output / 'result.json',
        dict(status='complete_four_stage_research_parent', checkpoint=last, checkpoint_id=meta['checkpoint_id'],
        unprotected_main_bpw=4, scores_percent=scores, BF16_percent=dense,
        deltas_pp={name: scores[name] - dense[name] for name in scores},
        nine_mean_percent=math.fsum(scores.values()) / 9, BF16_nine_mean_percent=math.fsum(dense.values()) / 9,
        bit_budget={**budget, 'complete_directory_bytes': sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())},
        raw_samples=_sample_artifacts(paths),
        final_lowbit_acceptance_evaluated=False, boundary='Fixed S4 initialization only; no recovery, gates or later compression.'))


def execute(request_path):
    request = s.read(request_path)
    output = Path(request['output_dir'])
    s.require(Path(sys.executable).resolve() == PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
        'CPU driver requires caller-activated bitvae and no visible GPU.')
    s.require(not (output / 'run.json').exists(), 'Never repeat an executed request.')
    s.check_pins(request['source_sha256'])
    s.require(s.hf_identity(HF) == request['hf_identity'], 'Original HF changed after preparation.')
    if request.get('after_control'):
        wait_existing_control(request['after_control'], output)
        s.check_pins(request['source_sha256'])
    command = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader']
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader'], text=True)
    apps = subprocess.check_output(command, text=True)
    for row in inventory.splitlines():
        index, uuid, used = [value.strip() for value in row.split(',')]
        if int(index) in (6, 7):
            s.require(int(used.split()[0]) < 1024 and uuid not in apps, 'Assigned GPU is occupied; no waiting, retry or interference.')
    read_fd, write_fd = os.pipe()
    report = dict(schema='progressive_bf16_s4_run_v1', status='running', **process_event(os.getpid()),
        started_unix=time.time(), validation_only=request['validation_only'], source_sha256=request['source_sha256'],
        command=[sys.executable, *sys.argv],
        request_sha256=s.sha(request_path), resources=dict(inventory=inventory, compute_apps=apps), processes={})
    s.write(output / 'run.json', report)
    children = []
    try:
        for role, script, arguments, fd, gpu in (
            ('evaluation', 'run.py', ['--evaluate-stream', '--event-fd', str(read_fd)], read_fd, ''),
            ('fit', 'fit.py', ['--event-fd', str(write_fd)], write_fd, request['fit_gpu'])):
            argv = [str(PYTHON), '-B', '-u', str(HERE / script), '--request', str(request_path), *arguments]
            with (output / (role + '.log')).open('x', encoding='utf-8') as log:
                process = subprocess.Popen(argv, cwd=ROOT, env=environment(gpu), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
            report['processes'][role] = dict(command=argv, **process_event(process.pid), gpu=gpu)
            children.append((role, process))
            s.write(output / 'run.json', report)
        os.close(read_fd); os.close(write_fd)
        for role, process in reversed(children):
            code = process.wait()
            report['processes'][role]['exit_code'] = code
            s.write(output / 'run.json', report)
        s.require(all(event['exit_code'] == 0 for event in report['processes'].values()), 'A fixed pipeline component failed.')
        report.update(status='complete_s4_initialization_short' if request['validation_only'] else 'complete_s4_initialization', exit_code=0)
    except BaseException as error:
        report.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        s.write(output / 'run.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--validation-report', type=Path)
    parser.add_argument('--evaluate-stream', action='store_true')
    parser.add_argument('--event-fd', type=int)
    parser.add_argument('--summarize-output', type=Path)
    args = parser.parse_args()
    if args.summarize_output:
        request = s.read(args.request)
        s.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU summary only.')
        summarize(Path(request['output_dir']), request, args.summarize_output)
    elif args.prepare:
        prepare(args.prepare, args.validation_only, args.validation_report)
    elif args.evaluate_stream:
        evaluate_stream(args.request.resolve(strict=True), args.event_fd)
    else:
        execute(args.request.resolve(strict=True))
