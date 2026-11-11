"""One fixed arm: fresh original-C1 train32, independent audit, complete nine."""
import argparse
import fcntl
import os
import sys
import time
import traceback

from . import bindings as b


def run(arm, gpu):
    executor = b.original.executor
    executor.cpu_environment()
    b.configure()
    b.require((arm, gpu) in (('full_teacher', 1), ('pure_gold', 2), ('conditional_lr_half', 4),
        ('conditional_lr_double', 5)), 'Keep the reviewed independent four-arm GPU allocation.')
    prepared_path = b.PREPARED
    prepared = b.read_prepared(prepared_path)
    probe = b.read(b.OUTPUT / 'probe.json')
    b.require(probe['status'] == 'verified_fixed_first_train_M32_real_backward', 'The new target backward probe must pass first.')
    directory = b.EVIDENCE / arm
    b.require(not directory.exists(), 'Never overwrite or retry an arm controller.')
    directory.mkdir(parents=True)
    module = 'experiments.c1_m_target_muon_cmcc_20261102.'
    common = ['--prepared', str(prepared_path), '--gpu', str(gpu), '--arm', arm]

    def request(name, suffix, arguments, visible, devices):
        return dict(name=name, argv=[str(b.PYTHON), '-B', '-u', '-m', module + suffix, *arguments],
            environment=executor.environment(visible), gpus=devices,
            log=str(directory / (name + '.log')), exit_file=str(directory / (name + '.exit')))

    stages = [request('formal32', 'run', [*common, '--timeout-seconds', '14400'], str(gpu), [str(gpu)]),
        request('fresh', 'fresh', common, str(gpu), [str(gpu)]),
        request('full9', 'evaluate', ['--support', str(b.SUPPORT), '--checkpoint', str(b.RUN_OUTPUT / arm / 'final_model'),
            '--pipeline-report', str(b.RUN_OUTPUT / arm / 'report.json'), '--output', str(directory / 'evaluation'),
            '--gpus', str(gpu)], '', [str(gpu)])]
    plan = dict(schema='C1_M_target_fixed_arm_v1', arm=arm, gpu=gpu, prepared=b.binding(prepared_path),
        target_probe=b.binding(b.OUTPUT / 'probe.json'), contract=b.CONTRACT,
        stages=stages, source_sha256={path: item['sha256'] for path, item in b.code_bindings().items()},
        automatic_retry=False, automatic_resume=False)
    b.core.write(directory / 'plan.json', plan)
    record = dict(status='starting', arm=arm, pid=os.getpid(), start_ticks=executor.proc_start(os.getpid()),
        argv=[sys.executable, *sys.argv], started_unix=time.time(), plan=b.binding(directory / 'plan.json'), stages={})
    with (directory / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def changed(item):
            record['stages'][item['name']] = item
            record['status'] = item['name'] + '_' + item['status']
            b.core.write(directory / 'run.json', record)
            print(dict(event=record['status'], pid=item.get('pid'), arm=arm), flush=True)

        code = 1
        try:
            for stage in stages:
                executor.check_pins(plan['source_sha256'])
                record[stage['name'] + '_resource_check'] = executor.check_resources(stage['gpus'])
                executor.launch_child(stage, changed)
            audit = b.read(b.RUN_OUTPUT / arm / 'fresh_audit.json')
            quality = executor.validate_evaluation(directory / 'evaluation', audit['checkpoint_id'])
            record.update(status='completed_fixed_arm_and_full9', exit_code=0, quality=quality)
            code = 0
        except BaseException:
            record.update(status='failed', exit_code=1, error=traceback.format_exc())
            raise
        finally:
            record['finished_unix'] = time.time()
            b.core.write(directory / 'run.json', record)
            (directory / 'run.exit').write_text(str(code) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=b.ARMS)
    parser.add_argument('--gpu', required=True, type=int, choices=(1, 2, 4, 5))
    arguments = parser.parse_args()
    run(arguments.arm, arguments.gpu)
