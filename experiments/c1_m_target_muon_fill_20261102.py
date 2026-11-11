"""Use the verified GPU3 gap for one fixed full_teacher train/fresh/full9 chain."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.c1_m_target_muon_cmcc_20261102 import bindings as b

HERE = Path(__file__).resolve()
ARM = 'full_teacher'
GPU = '3'
DIRECTORY = b.EVIDENCE / ARM


def requests():
    executor = b.original.executor
    prefix = 'experiments.c1_m_target_muon_cmcc_20261102.'
    common = ['--prepared', str(b.PREPARED), '--gpu', GPU, '--arm', ARM]

    def request(name, suffix, arguments, visible):
        return dict(name=name, argv=[str(b.PYTHON), '-B', '-u', '-m', prefix + suffix, *arguments],
            environment=executor.environment(visible), gpus=[GPU],
            log=str(DIRECTORY / (name + '.log')), exit_file=str(DIRECTORY / (name + '.exit')))

    return [request('formal32', 'run', [*common, '--timeout-seconds', '14400'], GPU),
        request('fresh', 'fresh', common, GPU),
        request('full9', 'evaluate', ['--support', str(b.SUPPORT), '--checkpoint', str(b.RUN_OUTPUT / ARM / 'final_model'),
            '--pipeline-report', str(b.RUN_OUTPUT / ARM / 'report.json'), '--output', str(DIRECTORY / 'evaluation'),
            '--gpus', GPU], '')]


def validate():
    from experiments.c1_m_target_muon_cmcc_20261102 import run, fresh, evaluate
    executor = b.original.executor
    executor.cpu_environment()
    b.torch.set_num_threads(4)
    b.configure()
    b.require(run.b is fresh.b is evaluate.r is b and run.run.__module__.endswith('.run')
        and fresh.fresh.__module__.endswith('.fresh') and evaluate.source.__module__.endswith('.evaluate'),
        'Keep the frozen real training/fresh/source dispatch; this file changes only the physical GPU allocation.')
    b.require(all(row['gpus'] == [GPU] for row in requests())
        and requests()[0]['argv'][-2:] == ['--timeout-seconds', '14400']
        and b.LR_MULTIPLIERS[ARM] == 1., 'The fixed GPU3 target recipe changed.')
    proof = b.EVIDENCE / 'gpu3_controller_cpu.json'
    b.require(not proof.exists(), 'Never overwrite the allocation-controller check.')
    b.core.write(proof, dict(status='verified_frozen_GPU3_CLI_and_source_dispatch', actual_GPU=GPU, arm=ARM,
        requests=requests(), controller=b.binding(HERE), frozen_package=b.code_bindings(),
        CUDA_initialized=b.torch.cuda.is_initialized(), boundary='Resource-only controller; no model load or repeated GPU short-chain.'))
    print(dict(status='verified_frozen_GPU3_CLI_and_source_dispatch'), flush=True)


def run():
    executor = b.original.executor
    executor.cpu_environment()
    b.torch.set_num_threads(4)
    b.configure()
    check_path = b.EVIDENCE / 'gpu3_controller_cpu.json'
    check = b.read(check_path)
    b.require(check['status'] == 'verified_frozen_GPU3_CLI_and_source_dispatch'
        and check['controller'] == b.binding(HERE) and check['requests'] == requests()
        and check['frozen_package'] == b.code_bindings() and not check['CUDA_initialized'],
        'Use the verified resource-only controller and unchanged frozen target package.')
    prepared = b.read_prepared(b.PREPARED)
    probe = b.read(b.OUTPUT / 'probe.json')
    b.require(probe['status'] == 'verified_fixed_first_train_M32_real_backward'
        and not DIRECTORY.exists() and not (b.RUN_OUTPUT / ARM).exists(),
        'Require the real target probe and an unused full_teacher run; never double-launch the original sequence.')
    DIRECTORY.mkdir(parents=True)
    pins = {path: item['sha256'] for path, item in b.code_bindings().items()}
    pins[str(HERE)] = b.binding(HERE)['sha256']
    plan = dict(schema='C1_full_teacher_GPU3_resource_fill_v1', arm=ARM, GPU=GPU,
        prepared=b.binding(b.PREPARED), target_probe=b.binding(b.OUTPUT / 'probe.json'),
        source_binding=prepared['source_binding'], controller=b.binding(HERE), source_sha256=pins,
        contract=b.CONTRACT, stages=requests(), automatic_retry=False, automatic_resume=False,
        boundary='GPU3 gap only. Same frozen full_teacher recipe; GPU1/2 remain stage1. No scheduler or resource wait.')
    b.core.write(DIRECTORY / 'plan.json', plan)
    record = dict(status='starting', arm=ARM, GPU=GPU, pid=os.getpid(), start_ticks=executor.proc_start(os.getpid()),
        argv=[sys.executable, *sys.argv], started_unix=time.time(), plan=b.binding(DIRECTORY / 'plan.json'), stages={})
    with (DIRECTORY / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def changed(item):
            record['stages'][item['name']] = item
            record['status'] = item['name'] + '_' + item['status']
            b.core.write(DIRECTORY / 'run.json', record)
            print(dict(event=record['status'], pid=item.get('pid'), arm=ARM, GPU=GPU), flush=True)

        code = 1
        try:
            for stage in plan['stages']:
                executor.check_pins(pins)
                record[stage['name'] + '_resource_check'] = executor.check_resources(stage['gpus'])
                executor.launch_child(stage, changed)
            audit = b.read(b.RUN_OUTPUT / ARM / 'fresh_audit.json')
            quality = executor.validate_evaluation(DIRECTORY / 'evaluation', audit['checkpoint_id'])
            record.update(status='completed_fixed_full_teacher_GPU3_and_full9', exit_code=0, quality=quality)
            code = 0
        except BaseException:
            record.update(status='failed', exit_code=1, error=traceback.format_exc())
            raise
        finally:
            record['finished_unix'] = time.time()
            b.core.write(DIRECTORY / 'run.json', record)
            (DIRECTORY / 'run.exit').write_text(str(code) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    options = parser.parse_args()
    validate() if options.validate else run()
