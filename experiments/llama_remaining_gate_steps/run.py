"""Run one explicitly bound native after-gate3000 continuation and full8."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.llama_order_control.prepare import check, read, require
from experiments.llama_remaining_gate_steps.finish import verify_native
from experiments.longbench_v1.common import sha256, write_json


def run(record_path):
    record_path = Path(record_path).resolve(); record = read(record_path); output = Path(record['output_dir']); code = 1

    def stage(name, argv, *, cpu=False):
        check(record); record['status'] = name; environment = os.environ.copy()
        if cpu:
            environment['CUDA_VISIBLE_DEVICES'] = ''
        with (output / (name + '.log')).open('x') as log:
            child = subprocess.Popen(argv, cwd=record['cwd'], env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            record['stages'][name] = {'argv': argv, 'pid': child.pid, 'started_unix': time.time()}
            write_json(record_path, record); value = child.wait()
        (output / (name + '.exit')).write_text(str(value) + '\n')
        record['stages'][name].update(exit_code=value, finished_unix=time.time()); write_json(record_path, record)
        require(value == 0, name + ' failed; no retry or later stage.')

    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and Path.cwd().resolve() == Path(record['cwd']) and not record['queued'] and not record['automatic_retry'],
                'Use the explicitly assigned whole GPU, public source and bitvae without queue or retry.')
        require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Keep the original native environment and numerical policy.')
        require(record['prepared_sha256'] == sha256(output / 'launch.prepared.json')
                and not (output / 'run.exit').exists() and not list(output.glob('home_*')),
                'Use fresh output bound to the actual CPU preparation.')
        occupied = subprocess.check_output(['nvidia-smi', '--id=' + str(record['gpu']), '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        require(not occupied, 'Assigned GPU is occupied: ' + occupied)
        check(record)
        record.update(worker_pid=os.getpid(), worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()[19]),
            started_unix=time.time(), stages={}); write_json(record_path, record)
        stage('train', record['training_argv'])
        finals = list(output.glob('home_*/final_model'))
        require(len(finals) == 1, 'Require one real native endpoint.')
        final = finals[0].resolve(); meta, _ = verify_native(record, final)
        record.update(final_checkpoint_dir=str(final), checkpoint_id=meta['checkpoint_id']); write_json(record_path, record)
        for name, tasks, batch in [('mmlu', 'mmlu', 64), ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', 32)]:
            stage(name, [sys.executable, 'tools/cat_eval.py', '--checkpoint_dir', str(final), '--strict', '--eval_lm_eval',
                '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0', '--lm_batch_size', str(batch),
                '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8', '--eval_log_dir', str(output / ('logs_' + name)),
                '--output_json', str(output / (name + '.json'))])
        stage('summary', [sys.executable, '-m', 'experiments.llama_remaining_gate_steps.finish', '--launch-record', str(record_path)], cpu=True)
        summary = read(output / 'full8_summary.json')
        record.update(status='completed', mean_percent=summary['mean_percent'],
            mmlu_correct=summary['integer_counts']['mmlu']['correct'], full8_summary_sha256=sha256(output / 'full8_summary.json'))
        code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}'); traceback.print_exc()
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(record_path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().launch_record))
