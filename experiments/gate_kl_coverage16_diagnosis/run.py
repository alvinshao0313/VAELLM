"""Run the fixed KL diagnosis and one code-only native export/full8, without retries."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.gate_kl_coverage16_diagnosis.common import read, check, bound
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.rotation_factor_probe.capture import require


def run(path):
    p = read(path); bound(p); output = Path(p['output_dir'])
    require(not any((output / name).exists() for name in ('run.json', 'run.exit', 'kl', 'final_model', 'evaluation.json')),
            'Use one fresh diagnosis directory without queue or retry.')
    record = {'status': 'running', 'bound_sha256': sha256(path), 'pid': os.getpid(),
              'started_unix': time.time(), 'stages': {}}; code = 1
    try:
        for name, module, extra in [('kl', 'experiments.gate_kl_coverage16_diagnosis.kl', []),
                                    ('evaluation', 'experiments.gate_kl_coverage16_diagnosis.export', ['--phase', 'run'])]:
            argv = [sys.executable, '-m', module, '--preparation', str(path), *extra]
            record['stages'][name] = {'argv': argv, 'started_unix': time.time()}; write_json(output / 'run.json', record)
            exit_code = run_logged_stage(argv, name=name, output_dir=output, env=os.environ.copy(), allowed_exit_codes=(0,))
            record['stages'][name].update(exit_code=exit_code, finished_unix=time.time())
            require(exit_code == 0, 'Stage failed; no fallback, retry, truncation or automatic queue.')
        check(p); record['status'] = 'completed'; code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(output / 'run.json', record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--preparation', type=Path, required=True)
    sys.exit(run(parser.parse_args().preparation.resolve()))
