"""One bound execution: existing KL proposal, existing decoder500, native full8."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.gate_kl_calibration_probe.run import teacher_chunks, teacher_populations
from experiments.gate_final_kl_accept_probe.run import validate_teacher_cache
from experiments.gate_local_delta_probe.run import load_assets
from experiments.gate_student_context_probe.core import measure, ready, records, train
from experiments.gate_tail_objective_probe.core import teacher_pass
from experiments.gate_kl_coverage16_probe.common import ARM, BANKS, SOURCE_ID, check, read, replay_codes
from experiments.gate_kl_coverage16_probe.core import propose
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def bound(record):
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd'])
            and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and not record['queued'] and not record['automatic_retry'], 'Require one root-bound execution, no queue/retry.')
    require(all(os.environ.get(key) == value for key, value in record['environment'].items())
            and not any(key in os.environ for key in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
            'The prepared numerical/offline environment differs.')
    require(record['preparation_sha256'] == sha256(Path(record['output_dir']) / 'launch.prepared.json'),
            'Binding must identify the immutable CPU preparation.')
    check(record)


def compute(path):
    p = read(path); bound(p)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Use only the bound GPU.')
    output = Path(p['output_dir']) / 'run'
    require(not output.exists(), 'Never overwrite an existing computation.')
    output.mkdir(); started = time.time(); deadline = started + p['compute_budget_seconds']
    report = {'status': 'running', 'source_checkpoint_id': SOURCE_ID, 'preparation_sha256': p['preparation_sha256'],
        'started_unix': started, 'pid': os.getpid(), 'protocol': p['protocol']}
    write_json(output / 'report.json', report)
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(137)
        batches = torch.load(p['batch_path'], map_location='cpu', weights_only=True)
        require({name: {key: tensor_record(value) for key, value in batch.items()} for name, batch in batches.items()}
                == p['token_batches'], 'Actual fixed native batches changed.')
        archive = torch.load(p['paths']['archive'], map_location='cpu', weights_only=True, mmap=True)
        teacher_by_chunk, hf_mlp, teacher_report = teacher_pass(p['data'], teacher_chunks(batches), archive,
            device=torch.device('cuda:0'), deadline=deadline, observer=lambda *args: None)
        del hf_mlp, archive
        old_teacher = torch.load(p['old_teacher_path'], map_location='cpu', weights_only=True, mmap=True)
        validate_teacher_cache(old_teacher, batches, read(p['old_teacher_report']))
        teacher = teacher_populations(teacher_by_chunk, old_teacher, batches)
        old_choices = torch.load(p['reference_choices_path'], map_location='cpu', weights_only=True)
        with (output / 'trajectory.jsonl').open('x') as stream:
            def observer(kind, row):
                stream.write(json.dumps({'kind': kind, **row}) + '\n'); stream.flush()
                if kind != 'gradient' or row['index'] % 32 == 0:
                    print(json.dumps({'kind': kind, **{key: row[key] for key in ('index', 'vector_start', 'seconds') if key in row}}), flush=True)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                choices, proposal = propose(p['data'], batches, teacher, old_choices, deadline=deadline, observer=observer)
            del teacher, old_teacher, teacher_by_chunk
            gc.collect(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
            require(torch.cuda.memory_allocated() == 0, 'Release the complete proposal student before isolated fitting.')
            choice_path = output / 'choices.pt'; torch.save(choices, choice_path)
            require(torch.equal(torch.load(choice_path, map_location='cpu', weights_only=True), choices), 'Saved choices differ.')
            set_seed(31)
            layer, archive, cache, _, _ = load_assets(p['paths'])
            require(records(layer) == p['original_state'] == proposal['initial_state'], 'Fit must start with original f927 decoder.')
            chosen, xor = replay_codes(dict(layer.state_dict()), choices)
            require({key: tensor_record(value) for key, value in chosen.items()} == proposal['applied_state']
                    and xor == proposal['stage_xor'], 'Fit initialization differs from actual proposal state.')
            layer.load_state_dict(chosen, strict=True); layer.to('cuda:0'); ready(layer)
            training = train(layer, archive, cache, steps=500, deadline=deadline,
                             observer=lambda row: observer('fit', row))
            require(training['settings'] == p['expected_training_settings']
                    and training['persistent_bits'] == p['expected_persistent_bits']
                    and training['initial_payload'] == proposal['applied_state']
                    and training['frozen_state_exact'] and training['frozen_objective_buffers_exact'],
                    'Original500 schedule, context or frozen codes changed.')
            endpoint = measure(layer, archive, cache, torch.device('cuda:0'))
            state_path = output / 'gate_state.pt'
            torch.save({key: value.detach().cpu() for key, value in layer.state_dict().items()}, state_path)
            saved = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
            require({key: tensor_record(value) for key, value in saved.items()} == training['final_payload']
                    and all(torch.equal(saved[key], chosen[key]) for key in BANKS), 'Saved fit endpoint or fixed banks changed.')
        report.update(status='completed', steps=500, teacher=teacher_report, proposal=proposal, training=training,
            endpoint_path=str(state_path), endpoint_sha256=sha256(state_path), endpoint_state=training['final_payload'],
            choices_path=str(choice_path), choices_sha256=sha256(choice_path), stage_xor=xor,
            frozen_codes_exact=True, local_observations=endpoint, elapsed_seconds=time.time() - started,
            full8_and_longbench='not_yet_evaluated')
        check(p); write_json(output / 'report.json', report)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        write_json(output / 'report.json', report); raise


def run(path):
    p = read(path); bound(p); output = Path(p['output_dir'])
    require(not any((output / name).exists() for name in ('run', 'final_model', 'run.json', 'run.exit')),
            'Use a fresh experiment output without retry.')
    from experiments.qwen3_w25.step_export_callback import run_logged_stage
    result = {'status': 'running', 'bound_sha256': sha256(path), 'started_unix': time.time(),
        'pid': os.getpid(), 'stages': {}}
    code = 1
    try:
        for name, module, extra in [('compute', 'experiments.gate_kl_coverage16_probe.run', ['--compute']),
            ('evaluation', 'experiments.gate_kl_coverage16_probe.evaluation', ['--phase', 'run'])]:
            command = [sys.executable, '-m', module, '--preparation', str(path), *extra]
            result['stages'][name] = {'argv': command, 'started_unix': time.time()}
            write_json(output / 'run.json', result)
            status = run_logged_stage(command, name=name, output_dir=output, env=os.environ.copy(), allowed_exit_codes=(0,))
            result['stages'][name].update(exit_code=status, finished_unix=time.time())
            require(status == 0, 'A stage failed; stop without fallback or automatic retry.')
        check(p); result['status'] = 'completed'; code = 0
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        result.update(exit_code=code, finished_unix=time.time())
        write_json(output / 'run.json', result); (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True, type=Path)
    parser.add_argument('--compute', action='store_true')
    args = parser.parse_args()
    if args.compute:
        compute(args.preparation.resolve())
    else:
        sys.exit(run(args.preparation.resolve()))
