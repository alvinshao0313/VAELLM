"""Execute only a parent-bound, single-GPU isolated native column-feedback probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.llama_column_feedback_probe.core import measure, teacher_pass
from experiments.llama_column_feedback_probe.prepare import check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def run(path):
    record = read(path); output = Path(record['output_dir']); started = time.time()
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int and not record['queued']
            and not record['automatic_retry'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and torch.cuda.device_count() == 1 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and Path.cwd().resolve() == Path(record['cwd']).resolve(), 'Root must bind one available GPU and bitvae before execution.')
    require(all(os.environ.get(key) == value for key, value in record['environment'].items()), 'Prepared runtime environment differs.')
    require(not (output / 'report.json').exists() and not (output / 'endpoints.pt').exists(), 'No overwrite or automatic retry.')
    check(record)
    batches = torch.load(output / 'token_batches.pt', map_location='cpu', weights_only=True)
    require({domain: {key: tensor_record(value) for key, value in batch.items()} for domain, batch in batches.items()}
            == record['data']['token_batches'], 'Runtime token batches differ from CPU preparation.')
    configure_deterministic_mode(True); set_seed(record['data']['seed']); torch.set_num_threads(4)
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
    deadline = started + record['resource_budget_seconds']
    report = {'status': 'running', 'started_unix': started, 'worker_pid': os.getpid(),
              'launch_sha256': sha256(path), 'protocol': record['protocol'], 'data': record['data']}
    write_json(output / 'report.json', report)
    def observe(value):
        print(json.dumps({'elapsed_seconds': time.time() - started, **value}, ensure_ascii=False), flush=True)
    try:
        teacher, reference, teacher_report = teacher_pass(record, batches, device=torch.device('cuda:0'),
                                                        deadline=deadline, observer=observe)
        result = measure(record, batches, teacher, reference, device=torch.device('cuda:0'), deadline=deadline, observer=observe)
        check(record)
        report.update(result, teacher=teacher_report, elapsed_seconds=time.time() - started,
                      endpoint_sha256=sha256(output / 'endpoints.pt'), finished_unix=time.time())
        write_json(output / 'report.json', report)
        (output / 'run.exit').write_text('0\n')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                      finished_unix=time.time(), elapsed_seconds=time.time() - started)
        write_json(output / 'report.json', report)
        (output / 'run.exit').write_text('1\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    run(parser.parse_args().launch)
