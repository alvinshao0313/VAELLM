"""Fresh reruns of the three zero-update path failures; original lanes stay immutable."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'experiments/night_sequence_20261026'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(FROZEN))
import recovery as original
import recovery_core as core
from recovery_eval import evaluate, run_stage

OLD_OUTPUT, OLD_DATA = original.OUTPUT, original.DATA
OUTPUT = OLD_OUTPUT.parent / 'repaired_recovery'
DATA = OLD_DATA.parent / 'repaired_recovery'
PREPARED = OUTPUT / 'prepared.json'
LANES = {0: ('C', 'ce10'), 1: ('C', 'ce50'), 2: ('G', 'ce10')}


def process_module():
    spec = importlib.util.spec_from_file_location('_repair_queue_wait', FROZEN / 'queue_wait.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


processes = process_module()
read, write, require, sha = core.read, core.write, core.require, core.sha


def bind_output_paths():
    # The frozen trainer calls its own training_argv/global OUTPUT in this process.
    # Its child is this same wrapper, which applies the identical path substitution.
    original.OUTPUT = core.OUTPUT = OUTPUT
    original.DATA = core.DATA = DATA


def check_prepared():
    record = read(PREPARED)
    require(record['status'] == 'cpu_verified' and record['schema'] == 'three_zero_update_path_repairs_v1',
            'Use the actual CPU-verified fixed three-lane repair.')
    core.check_prepared()
    require(sha(HERE / 'repair.py') == record['repair_source_sha256'], 'Repair source changed after preparation.')
    for path, digest in record['original13_source_sha256'].items():
        require(sha(path) == digest, 'An original frozen source changed: ' + path)
    require(read(OLD_OUTPUT / 'data_path_repair.json')['status'] == 'verified', 'Five real snapshot loaders must pass.')
    return record


def prepare():
    from compressed_e2e_fintuning.args import parse_args
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only repair preparation.')
    require(not PREPARED.exists() and not DATA.exists(), 'Repair output/data must be fresh.')
    core.check_prepared()
    data_proof = read(OLD_OUTPUT / 'data_path_repair.json')
    require(data_proof['status'] == 'verified' and data_proof['all_five_actual_preset_loaders_passed']
            and len(data_proof['presets']) == 5, 'Require the completed actual five-preset data check.')
    queue_manifest = read(OLD_OUTPUT.parent / 'queue/manifest.json')
    pins = queue_manifest['source_sha256']
    require(len(pins) == 13 and all(sha(path) == digest for path, digest in pins.items()),
            'The original thirteen sources must remain unchanged.')
    before = {gpu: original.training_argv(*values) for gpu, values in LANES.items()}
    failures = {}
    for gpu, (profile, recipe) in LANES.items():
        report = read(OLD_OUTPUT / profile / recipe / 'pipeline.json')
        require(report['status'] == 'failed' and 'train exited1' in report['error'], 'Require the actual original failed donor.')
        log = (OLD_OUTPUT / profile / recipe / 'train.log').read_text(encoding='utf-8')
        require('Local dataset file not found:' in log and 'snapshot_20261005_01/data/edgerazor_qwen3/' in log,
                'Only the confirmed shared input-path failures are rerun.')
        failures[str(gpu)] = dict(profile=profile, recipe=recipe, pipeline=str(OLD_OUTPUT / profile / recipe / 'pipeline.json'),
                                  pipeline_sha256=sha(OLD_OUTPUT / profile / recipe / 'pipeline.json'),
                                  training_log_sha256=sha(OLD_OUTPUT / profile / recipe / 'train.log'),
                                  optimizer_updates=0, recovery_kind='fresh rerun; no optimizer state to resume')
    bind_output_paths()
    commands = {}
    for gpu, values in LANES.items():
        after = original.training_argv(*values)
        changed = [index for index, (a, b) in enumerate(zip(before[gpu], after, strict=True)) if a != b]
        require(changed == [before[gpu].index('--run_root_dir') + 1], 'Only the independent output root may change.')
        cfg, _hf, args = parse_args(after)
        require(cfg.train_mode == 'decoder' and args.max_steps == 2000 and args.save_total_limit == 5
                and args.save_steps == 400 and args.per_device_train_batch_size == 4
                and args.gradient_accumulation_steps == 1 and args.seed == args.data_seed == 0
                and cfg.data.model_max_length == 1024 and cfg.opt.decoder_lr == 1e-5
                and tuple(cfg.opt.distill_fp32_components) == ('decoder',), 'Original fixed training contract changed.')
        commands[str(gpu)] = dict(profile=values[0], recipe=values[1], training_argv=after,
                                  original_output=before[gpu][changed[0]], repaired_output=after[changed[0]],
                                  only_output_changed=True)
    original_g = read(OLD_OUTPUT.parent / 'queue/lane_3/queue.json')
    require(original_g['status'] == 'running_donor' and original_g['gpu'] == 3,
            'The original Gce50 must remain the running peer; do not duplicate it.')
    require(processes.process_start(original_g['pid']) == original_g['start_ticks'], 'Original Gce50 queue identity changed.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    link = OUTPUT / 'G/ce50'
    link.parent.mkdir(parents=True, exist_ok=True)
    require(not os.path.lexists(link), 'Never replace a repair donor location.')
    link.symlink_to(OLD_OUTPUT / 'G/ce50', target_is_directory=True)
    record = dict(schema='three_zero_update_path_repairs_v1', status='cpu_verified', prepared_unix=time.time(),
        repair_source_sha256=sha(HERE / 'repair.py'), original_prepared_sha256=sha(core.PREPARED),
        original13_source_sha256=pins, data_path_proof=str(OLD_OUTPUT / 'data_path_repair.json'),
        data_path_proof_sha256=sha(OLD_OUTPUT / 'data_path_repair.json'), failures=failures,
        actual_training_commands=commands, source_and_training_math_unchanged=True,
        original_Gce50_peer=dict(pid=original_g['pid'], start_ticks=original_g['start_ticks'],
                                queue_report=str(OLD_OUTPUT.parent / 'queue/lane_3/queue.json')),
        Gce50_read_only_donor_reference=dict(path=str(link), target=str(OLD_OUTPUT / 'G/ce50'),
                                            purpose='Read actual original donor only after its complete_donor terminal.'),
        CUDA_initialized=False, GPU_training_executed=False)
    write(PREPARED, record)
    return record


def donor(gpu):
    prepared = check_prepared()
    profile, recipe = LANES[gpu]
    bind_output_paths()
    output = OUTPUT / profile / recipe
    require(not output.exists(), 'Never overwrite any repair attempt.')
    output.mkdir(parents=True)
    record = dict(status='running', profile=profile, recipe=recipe, gpu=gpu,
        source_checkpoint_id=core.SOURCE_IDS[profile], repair_prepared_sha256=sha(PREPARED),
        original_zero_update_failure=prepared['failures'][str(gpu)], started_unix=time.time(), stages={})
    path = output / 'pipeline.json'
    write(path, record)
    try:
        core.check_gpu(gpu)
        argv = [sys.executable, '-B', '-u', str(HERE / 'repair.py'), 'train', '--gpu', str(gpu)]
        record['stages']['train'] = dict(command=argv, started_unix=time.time())
        write(path, record)
        run_stage(argv, name='train', output=output, gpu=str(gpu), seconds=50400)
        record['stages']['train'].update(exit_code=0, finished_unix=time.time())
        trained = read(output / 'training_result.json')
        final = Path(trained['native_result']['saved_model_dir'])
        step = Path(trained['saved_training_checkpoints'][-1]['path'])
        core.audit_final(core.SOURCES[profile], final, step, output / 'native_audit.json')
        record.update(final_model=str(final), checkpoint_id=read(final / 'checkpoint_meta.json')['checkpoint_id'])
        record['stages']['native_audit'] = dict(exit_code=0, evidence=str(output / 'native_audit.json'))
        write(path, record)
        evaluated = evaluate(final, output / 'evaluation_full8', gpu, evidence=output / 'native_audit.json')
        record['stages']['full8'] = dict(exit_code=0, evidence=str(output / 'evaluation_full8/evaluation.json'))
        record.update(status='complete_donor', full8_mean_percent=evaluated['full8_mean_percent'])
    except BaseException as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        write(path, record)
    return record


def lane(gpu, peer_pid=None, peer_start=None):
    prepared = check_prepared()
    directory = OUTPUT / 'queue' / ('lane_' + str(gpu))
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(status='running_donor', gpu=gpu, pid=os.getpid(), start_ticks=processes.process_start(os.getpid()),
                  started_unix=time.time(), automatic_retry=False, optimizer_resume=False)
    path = directory / 'queue.json'
    write(path, record)
    try:
        result = donor(gpu)
        record['donor'] = dict(checkpoint_id=result['checkpoint_id'], pipeline=str(OUTPUT / LANES[gpu][0] / LANES[gpu][1] / 'pipeline.json'))
        if gpu in (0, 2):
            if gpu == 0:
                require(peer_pid is not None and peer_start is not None, 'C leader must bind the new Cce50 peer.')
                peer_report = OUTPUT / 'C/ce50/pipeline.json'
            else:
                peer = prepared['original_Gce50_peer']
                peer_pid, peer_start = peer['pid'], peer['start_ticks']
                peer_report = OLD_OUTPUT / 'G/ce50/pipeline.json'
            record.update(status='waiting_fixed_peer', peer_pid=peer_pid, peer_start_ticks=peer_start,
                          peer_pipeline=str(peer_report))
            write(path, record)
            processes.wait_for_process(peer_pid, peer_start)
            actual = read(peer_report)
            require(actual['status'] == 'complete_donor' and actual['profile'] == LANES[gpu][0],
                    'The actual same-source donor must complete; queue failure alone is not a donor result.')
            record.update(status='assembly_full9', peer_pipeline_sha256=sha(peer_report))
            write(path, record)
            core.check_gpu(gpu)
            argv = ['timeout', '--signal=TERM', '--kill-after=60s', '28800s', sys.executable,
                    '-B', '-u', str(HERE / 'repair.py'), 'assemble', '--gpu', str(gpu)]
            with (directory / 'assembly_full9.log').open('x', encoding='utf-8') as log:
                child = subprocess.Popen(argv, cwd=ROOT, env=core.environment(''), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                record['assembly_process'] = dict(pid=child.pid, start_ticks=processes.process_start(child.pid), command=argv)
                write(path, record)
                code = child.wait()
            record['assembly_exit_code'] = code
            require(code == 0 and read(OUTPUT / LANES[gpu][0] / 'assembly/pipeline.json')['status'] == 'complete_assembly_full9',
                    'Fixed assembly/full9 failed; no retry or alternative endpoint.')
        record.update(status='complete', exit_code=0)
    except BaseException as error:
        record.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        write(path, record)


def launch():
    check_prepared()
    path = OUTPUT / 'launch.json'
    require(not path.exists(), 'Launch the fixed fresh repairs once.')
    record = dict(status='launching', started_unix=time.time(), pid=os.getpid(),
                  start_ticks=processes.process_start(os.getpid()), workers={}, prepared_sha256=sha(PREPARED),
                  original_lanes_unchanged=True, automatic_retry=False)
    write(path, record)
    launched = {}
    try:
        for gpu in (1, 0, 2):
            argv = [sys.executable, '-B', '-u', str(HERE / 'repair.py'), 'lane', '--gpu', str(gpu)]
            if gpu == 0:
                peer = record['workers']['1']
                argv.extend(['--peer-pid', str(peer['pid']), '--peer-start', str(peer['start_ticks'])])
            with (OUTPUT / ('lane_' + str(gpu) + '.log')).open('x', encoding='utf-8') as log:
                child = subprocess.Popen(argv, cwd=ROOT, env=core.environment(''), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            launched[gpu] = child
            record['workers'][str(gpu)] = dict(pid=child.pid, start_ticks=processes.process_start(child.pid),
                                              command=argv, independent_session=True)
            write(path, record)
        record['status'] = 'three_fresh_repairs_running'
        write(path, record)
    except BaseException as error:
        record.update(status='launch_failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for gpu, child in launched.items():
            record['workers'][str(gpu)]['exit_code'] = child.wait()
            write(path, record)
        if len(launched) == 3:
            record['status'] = 'complete' if all(row['exit_code'] == 0 for row in record['workers'].values()) else 'failed'
        record['finished_unix'] = time.time()
        write(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare'); sub.add_parser('launch')
    for name in ('train', 'assemble', 'lane'):
        p = sub.add_parser(name)
        p.add_argument('--gpu', type=int, choices=(0,1,2), required=True)
        if name == 'lane':
            p.add_argument('--peer-pid', type=int); p.add_argument('--peer-start', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'launch':
        launch()
    else:
        check_prepared(); bind_output_paths()
        if args.action == 'train':
            profile, recipe = LANES[args.gpu]
            original.train(profile, recipe, args.gpu, core.PREPARED)
        elif args.action == 'assemble':
            require(args.gpu in (0,2), 'Only the fixed C/G leader assembles.')
            original.run_assembly(LANES[args.gpu][0], args.gpu, core.PREPARED)
        else:
            lane(args.gpu, args.peer_pid, args.peer_start)


if __name__ == '__main__':
    main()
