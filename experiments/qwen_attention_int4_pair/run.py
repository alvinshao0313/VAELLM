"""Run one explicitly bound native training arm, then its isolated quality simulation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.qwen_attention_int4_pair.common import ROOT, read, check, _require
from experiments.longbench_v1.common import write_json, sha256


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1
    _require(record['status'] == 'bound_ready' and type(record['gpu']) is int
             and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']) and Path.cwd() == ROOT
             and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
             and all(os.environ.get(k) == v for k,v in record['environment'].items()), 'Require an explicit snapshot/GPU binding.')
    check(record)
    def stage(name, argv):
        log = output / (name + '.log'); _require(not log.exists(), 'No resume or overwrite.')
        with log.open('w') as handle:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            record['stages'][name] = {'argv': argv, 'pid': process.pid, 'started_unix': time.time()}
            record['status'] = name; write_json(path, record); value = process.wait()
        (output / (name + '.exit')).write_text(str(value)+'\n')
        record['stages'][name].update(exit_code=value, finished_unix=time.time()); write_json(path, record)
        _require(value == 0, name + ' failed; no automatic retry.')
    try:
        record.update(stages={}, worker_pid=os.getpid(), started_unix=time.time(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]))
        stage('train', record['training_argv'])
        finals = list((output/'donor').glob('*/final_model'))
        _require(len(finals) == 1, 'Expected one native final donor.')
        record['donor_checkpoint_dir'] = str(finals[0].resolve())
        record['donor_checkpoint_id'] = read(finals[0]/'checkpoint_meta.json')['checkpoint_id']; write_json(path,record)
        stage('simulation', [sys.executable, '-u', '-m', 'experiments.qwen_attention_int4_pair.simulate', '--launch-record', str(path)])
        result = read(output/'report.json'); _require(result['status'] == 'diagnostic_complete', 'Incomplete full-eight simulation.')
        check(record); record.update(status='completed', report_sha256=sha256(output/'report.json')); code=0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path,record)
        (output/'dispatcher.exit').write_text(str(code)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch-record',type=Path,required=True)
    run(parser.parse_args().launch_record)
