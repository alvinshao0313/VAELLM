"""One explicitly bound isolated arm; no GPU queue or automatic continuation."""
import argparse
import gc
import importlib.metadata
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import torch

from experiments.qwen_attention_decoded_target_pair.common import ROOT, CATEGORIES, read, check, _require
from experiments.qwen_attention_decoded_target_pair.core import train
from experiments.qwen_attention_decoded_target_pair.endpoint import save_and_reconstruct
from experiments.qwen_attention_decoded_target_pair.evaluate import evaluate
from experiments.longbench_v1.common import write_json, sha256
from train_utils.utils import get_logger


def logged_updates(output):
    log = (output / 'train.log').read_text(); result = {}
    for category in CATEGORIES:
        for stage in (1, 2):
            tag = f'{category}.L0-35/stage{stage}'
            rows = re.findall(r'\[' + re.escape(tag) + r'\] step=(\d+)/(\d+) loss=(\S+) recon=(\S+) commit=(\S+)', log)
            _require([int(row[0]) for row in rows] == list(range(100, 10001, 100))
                     and all(int(row[1]) == 10000 and all(math.isfinite(float(v)) for v in row[2:]) for row in rows),
                     'Incomplete/nonfinite actual native schedule: ' + tag)
            result[tag] = [{'step': int(row[0]), 'loss': float(row[2]), 'recon': float(row[3]),
                            'commit': float(row[4])} for row in rows]
    return result


def run(path):
    path = Path(path).resolve(); record = read(path); output = Path(record['output_dir']); code = 1
    _require(path.parent == output and record['status'] == 'bound_ready' and type(record['gpu']) is int
             and Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
             and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
             and all(os.environ.get(k) == v for k, v in record['environment'].items())
             and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
             'Require explicit snapshot/GPU binding with the unchanged numerical policy.')
    _require(not any((output / name).exists() for name in ('train.log', 'report.json', 'attention_state.pt')),
             'No overwrite or resume of an executed arm.')
    check(record)
    try:
        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Require one explicitly visible GPU.')
        for name, version in record['package_versions'].items():
            _require(importlib.metadata.version(name) == version, 'Package changed: ' + name)
        record.update(status='training', started_unix=time.time(), worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        os.environ['LOG_FILE'] = str(output / 'train.log'); get_logger('linear_by_category')
        model, base_state, training = train(record, torch.device('cuda:0'))
        training['native_logged_updates'] = logged_updates(output)
        write_json(output / 'training.json', training); (output / 'train.exit').write_text('0\n')
        record['status'] = 'endpoint'; write_json(path, record)
        fresh, meta, endpoint = save_and_reconstruct(model, base_state, record, training, torch.device('cuda:0'))
        write_json(output / 'endpoint_audit.json', endpoint); (output / 'endpoint.exit').write_text('0\n')
        del model, base_state; gc.collect(); torch.cuda.empty_cache()
        record['status'] = 'simulation'; write_json(path, record)
        report = evaluate(fresh, meta, record, endpoint)
        _require(report['status'] == 'diagnostic_complete', 'Incomplete full eight-task simulation.')
        (output / 'simulation.exit').write_text('0\n')
        check(record)
        record.update(status='completed', endpoint_id=endpoint['endpoint_id'],
            evidence_sha256={name: sha256(output / name) for name in ('training.json', 'endpoint_audit.json', 'report.json')})
        code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch-record', type=Path, required=True)
    run(parser.parse_args().launch_record)
