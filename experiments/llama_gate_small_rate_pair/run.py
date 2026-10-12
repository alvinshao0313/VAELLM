"""Run one explicitly bound K34 path validation, never an automatic formal queue."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.llama_gate_small_rate_pair.common import ROOT, check, read, _require
from experiments.longbench_v1.common import sha256, write_json
from train_utils.checkpoint_v6 import load_v6_meta


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1

    def stage(name, argv):
        log = output / (name + '.log'); _require(not log.exists(), 'Do not overwrite a stage.')
        record['status'] = name
        with log.open('w') as handle:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            record['stages'][name] = {'argv': argv, 'pid': process.pid, 'started_unix': time.time()}
            write_json(path, record); value = process.wait()
        (output / (name + '.exit')).write_text(str(value) + '\n')
        record['stages'][name].update(exit_code=value, finished_unix=time.time()); write_json(path, record)
        _require(value == 0, f'{name} exited {value}; inspect {log}')

    try:
        _require(record['schema'] == 'llama_gate_k34_native_short_v1' and record['status'] == 'bound_ready'
                 and record['steps_per_stage'] == 2 and record['codebook_bits'] == 34 and not record['full_evaluation']
                 and not record['queued'] and not record['automatic_retry'] and Path.cwd().resolve() == ROOT
                 and type(record['gpu']) is int and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and Path(record['output_dir']).resolve() == output
                 and all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Require one explicitly bound original short validation.')
        check(record); record.update(worker_pid=os.getpid(), worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat')
            .read_text().rsplit(')', 1)[1].split()[19]), started_unix=time.time(), stages={}); write_json(path, record)
        stage('train', record['training_argv'])
        finals = list((output / 'donor').glob('*/final_model')); _require(len(finals) == 1, 'Expected one native donor.')
        native = finals[0].resolve(); dm = load_v6_meta(str(native))
        record.update(donor_checkpoint_dir=str(native), donor_checkpoint_id=dm['checkpoint_id']); write_json(path, record)
        stage('assembly', [sys.executable, '-m', 'experiments.w25_eval.assemble_categories', '--base', record['base_dir'],
            '--donor', str(native), '--categories', 'gate_proj', '--output', str(output / 'final_model')])
        stage('audit', [sys.executable, '-m', 'experiments.llama_gate_small_rate_pair.audit', '--launch-record', str(path)])
        proof = read(output / 'verification.json')
        _require(proof['status'] == 'verified_native_short', 'Missing actual native path verification.')
        check(record); record.update(status='completed', verification_sha256=sha256(output / 'verification.json'))
        code = 0
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()); raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    run(parser.parse_args().launch_record)
