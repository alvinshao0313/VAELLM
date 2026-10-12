"""Bound once by root: sequential native migration, two128-step arms and audit."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.llama_gate_r1_probe.common import DOMAINS, check, read, require, sha256, write_json


def run(path):
    path = Path(path).resolve(); record = read(path); root = Path(record['output_dir'])
    code = 1
    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and not record['queued'] and not record['automatic_retry']
                and all(os.environ.get(k) == v for k, v in record['environment'].items()), 'Root must bind one free GPU and exact S environment.')
        require(not any((root / name).exists() for name in ('migration.json', 'r0', 'r1', 'report.json')), 'Never retry/overwrite a prior run.')
        check(record)
        record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        commands = [('migration', 'migrate', []), ('initial_observations', 'evaluate', ['--phase', 'initial']),
                    ('r0_training', 'train', ['--arm', 'r0']), ('r1_training', 'train', ['--arm', 'r1']),
                    ('final_observations', 'evaluate', ['--phase', 'final'])]
        record['stages'] = {}
        for name, module, extra in commands:
            argv = [sys.executable, '-u', '-m', 'experiments.llama_gate_r1_probe.' + module, '--launch-record', str(path), *extra]
            started = time.time()
            with (root / (name + '.log')).open('w') as log:
                result = subprocess.run(argv, cwd=record['cwd'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            (root / (name + '.exit')).write_text(str(result.returncode) + '\n')
            record['stages'][name] = {'argv': argv, 'exit_code': result.returncode, 'elapsed_seconds': time.time() - started}
            write_json(path, record)
            require(result.returncode == 0, name + ' failed; no retry or remaining stages.')
        first, final = read(root / 'initial_observations.json'), read(root / 'final_observations.json')
        train = {arm: read(root / arm / 'training.json') for arm in ('r0', 'r1')}
        require(train['r0']['observations']['batches'] == train['r1']['observations']['batches'], 'The actual two training prefixes differ.')
        changes = {arm: {d: {key: final['arms'][arm]['domains'][d][key] - first['arms'][arm]['domains'][d][key]
                             for key in ('kl_mean', 'nll_mean')} for d in DOMAINS} for arm in ('r0', 'r1')}
        report = {'status': 'completed_native_short_probe', 'migration_sha256': sha256(root / 'migration.json'),
            'initial_sha256': sha256(root / 'initial_observations.json'), 'final_sha256': sha256(root / 'final_observations.json'),
            'training_sha256': {a: sha256(root / a / 'training.json') for a in train},
            'matched_training_batches': 128, 'within_arm_final_minus_initial': changes,
            'paired_final': final['paired'], 'source_checkpoint_id': record['source_checkpoint_id'],
            'boundary': record['scientific_boundary'], 'elapsed_seconds': time.time() - record['started_unix']}
        check(record); write_json(root / 'report.json', report)
        record.update(status='completed', report_sha256=sha256(root / 'report.json')); code = 0
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (root / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', required=True, type=Path)
    sys.exit(run(parser.parse_args().launch_record))
