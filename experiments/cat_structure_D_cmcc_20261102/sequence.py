"""One fixed original-D short/fresh -> fresh formal -> own GPU0 full9 chain."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bindings as b


def child(name, argv, gpu, record):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED='31',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        OPENBLAS_NUM_THREADS='8', NUMEXPR_NUM_THREADS='8', PYTHONPATH=str(b.SNAPSHOT))
    if gpu != '':
        resources = b.free_gpu(int(gpu))
    else:
        resources = None
    log = b.RESULT / (name + '.log')
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(argv, cwd=b.REPO, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT)
        item = dict(argv=argv, gpu=gpu, pid=process.pid,
            start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            log=str(log), resources=resources, started_unix=time.time(), status='running')
        record['stages'][name] = item
        b.write(b.RESULT / 'run.json', record)
        status = process.wait()
    item.update(status='exited', exit_code=status, elapsed_seconds=time.time()-item['started_unix'])
    (b.RESULT / (name+'.exit')).write_text(str(status)+'\n')
    if name.endswith('_train'):
        (b.OUTPUT / name.split('_')[0] / 'run.exit').write_text(str(status)+'\n')
    b.write(b.RESULT / 'run.json', record)
    b.require(status == 0, 'Actual child failed; stop dependent stages: ' + name)


def run(support):
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'The fixed controller remains on CPU bitvae.')
    b.RESULT.mkdir(parents=True, exist_ok=True)
    with (b.RESULT / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        b.require(not (b.RESULT / 'run.json').exists(), 'Do not start the same fixed chain twice.')
        prepared = b.read(b.RESULT / 'prepared.json')
        b.check_pins(prepared['source_sha256'])
        record = dict(status='running', pid=os.getpid(), actual_argv=[sys.executable,*sys.argv],
            source_binding=prepared['source_binding'], training_contract=b.CONTRACT,
            prepared_sha256=b.sha(b.RESULT / 'prepared.json'), stages={})
        try:
            for phase in ('short', 'formal'):
                command = [str(b.PYTHON),'-B','-u','-m',
                    'experiments.cat_structure_D_cmcc_20261102.run','--phase',phase]
                if phase == 'formal':
                    command = ['timeout','--signal=TERM','--kill-after=20s','43200s',*command]
                child(phase+'_train', command, 0, record)
                report = b.read(b.OUTPUT / phase / 'report.json')
                child(phase+'_fresh', [str(b.PYTHON),'-B','-u','-m',
                    'experiments.cat_structure_D_cmcc_20261102.prepare','--verify',report['final_model'],
                    '--expected',str(b.OUTPUT / phase / 'expected_state.json'),
                    '--output',str(b.OUTPUT / phase / 'strict_reload.json')], '', record)
                if phase == 'short':
                    fresh = b.read(b.OUTPUT / phase / 'strict_reload.json')
                    b.require(report['status'] == 'completed_D_native_shape_short' and len(report['actual_groups']) == 1
                        and report['actual_groups'][0]['status'] == 'fitted_and_encoded'
                        and len(report['actual_groups'][0]['new_shape_actual_updates']) == 2
                        and all(len(stage) == 2 for stage in report['actual_groups'][0]['new_shape_actual_updates'])
                        and len(report['recoveries']) == 1 and report['recoveries'][0]['actual_steps'] == 2
                        and report['recoveries'][0]['actual_schedule_max_steps'] == 500
                        and report['short_recovery']['actual_steps'] == 2,
                        'The actual D256 shape/encoding/recovery short did not complete.')
                    b.write(b.RESULT / 'short_validation.json', dict(status='verified_new_D256_native_shape',
                        prepared_sha256=record['prepared_sha256'], steps_per_stage=2, stages=2,
                        short_report=str(b.OUTPUT / phase / 'report.json'), short_report_sha256=b.sha(b.OUTPUT / phase / 'report.json'),
                        fresh_full_state_exact=fresh['expected_state_equal'], fresh_report=fresh,
                        recovery_actual_steps=2, recovery_schedule_max_steps=500,
                        training_recovery_evidence=report['short_recovery'],
                        formal_uses_short_weights=False))
            from .source import publish
            publish(b.OUTPUT / 'formal/report.json')
            report = b.read(b.OUTPUT / 'formal/report.json')
            record['evaluation_resource_check'] = b.free_gpu(0)
            child('evaluate', [str(b.PYTHON),'-B','-u','-m','experiments.cat_structure_D_cmcc_20261102.evaluate',
                '--support',str(support),'--checkpoint',report['final_model'],'--pipeline-report',
                str(b.OUTPUT / 'formal/report.json'),'--output',str(b.RESULT / 'evaluation'),'--gpus','0'], '', record)
            record['status'] = 'completed_D_native_and_full9'
        except BaseException as error:
            record.update(status='failed', error=repr(error))
            raise
        finally:
            b.write(b.RESULT / 'run.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', type=Path, required=True)
    run(parser.parse_args().support.resolve(strict=True))


if __name__ == '__main__':
    main()
