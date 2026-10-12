"""Run the three explicitly bound arms once, sequentially, without waiting/retry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.longbench_v1.common import sha256, write_json
from experiments.mmlu_attribution.protocol import require
from .protocol import check_preparation


def start_ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def dispatch(preparation):
    preparation = Path(preparation).resolve()
    root = preparation.parent
    require(not (root / 'launch.json').exists(), 'This diagnostic has already started.')
    binding = json.loads((root / 'binding.json').read_text())
    require(binding['status'] == 'bound' and binding['preparation_sha256'] == sha256(preparation)
            and type(binding['gpu']) is int and 0 <= binding['gpu'] < 8, 'Missing explicit reviewed GPU binding.')
    require(all(os.environ.get(key) == value for key, value in binding['environment'].items())
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(binding['gpu']), 'Actual bound environment differs.')
    apps = subprocess.check_output(['nvidia-smi', '--id=' + str(binding['gpu']),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    require(not apps, 'Assigned GPU is occupied; no waiting or reassignment.')
    prepared, *_ = check_preparation(preparation)
    record = {'status': 'running', 'pid': os.getpid(), 'start_ticks': start_ticks(os.getpid()),
        'started_unix': time.time(), 'binding_sha256': sha256(root / 'binding.json'),
        'preparation_sha256': sha256(preparation), 'gpu': binding['gpu'], 'stages': [],
        'automatic_retry': False, 'automatic_wallclock_stop': False}
    write_json(root / 'launch.json', record)
    try:
        commands = [(item['arm'], item['argv']) for item in prepared['arm_commands']]
        commands.append(('compare', prepared['comparison_argv']))
        for name, argv in commands:
            stage = {'name': name, 'argv': argv, 'started_unix': time.time(), 'status': 'starting'}
            record['stages'].append(stage)
            write_json(root / 'launch.json', record)
            with (root / (name + '.log')).open('xb') as log:
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                stage.update(pid=child.pid, start_ticks=start_ticks(child.pid), status='running')
                write_json(root / 'launch.json', record)
                code = child.wait()
            (root / (name + '.exit')).write_text(str(code) + '\n')
            stage.update(exit_code=code, finished_unix=time.time(), status='complete' if code == 0 else 'failed')
            write_json(root / 'launch.json', record)
            require(code == 0, 'Diagnostic stage failed without retry: ' + name)
        record.update(status='diagnostic_complete', exit_code=0, comparison_sha256=sha256(root / 'comparison.json'))
    except BaseException as error:
        record.update(status='failed', exit_code=1, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        write_json(root / 'launch.json', record)
        (root / 'run.exit').write_text(str(record['exit_code']) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    dispatch(parser.parse_args().preparation)
