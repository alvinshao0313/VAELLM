"""Run only the first 128 native CAT updates with observational PyTorch hooks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from experiments.full_rht_gate_pair.common import check
from experiments.longbench_v1.common import sha256, write_json
from experiments.llama_bsq_observation.observe import ObservationComplete, Observer, require
from tools.cat_train import main as native_cat_main


def run(record_path):
    record_path = Path(record_path).resolve()
    record = json.loads(record_path.read_text())
    require(record['status'] == 'bound' and record['gpu'] is not None
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']), 'Root must explicitly bind the GPU.')
    check(record)
    output = Path(record['output_dir'])
    require(not (output / 'observations.jsonl').exists() and not (output / 'report.json').exists(), 'Fresh diagnostics only.')
    started = time.time()
    with Observer(output) as observer:
        try:
            native_cat_main(record['training_argv'][2:])
        except ObservationComplete:
            require(observer.updates == observer.calls == 128 and len(observer.rows) == 4,
                    'Incomplete native update prefix.')
        else:
            raise RuntimeError('Native CAT returned without the explicit observation stop.')
    require(not list((output / 'native').rglob('pytorch_model.bin')), 'Diagnostic must not export a candidate.')
    write_json(output / 'report.json', {'status': 'observed_native_prefix', 'native_optimizer_updates': 128,
        'stage': 1, 'configured_stages': 2, 'configured_steps_per_stage': 10000,
        'observed_steps': observer.selected, 'elapsed_seconds': time.time() - started,
        'training_argv': record['training_argv'], 'launch_record_sha256': sha256(record_path),
        'observations_sha256': sha256(output / 'observations.jsonl'),
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'scope': 'Current BSQ loss decomposition during the first 128 real native gateFalse stage-1 updates; no candidate, quality, convergence or later-stage claim.',
        'stop_boundary': 'After native optimizer update 128; its unused next-step scheduler advance is not executed.',
        'hard_statistics_boundary': 'Per-current-batch hard bit marginals; not fixed-sample flip counts or proof of code collapse.'})
    print(json.dumps({'status': 'observed_native_prefix', 'updates': observer.updates}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', required=True)
    run(parser.parse_args().launch_record)
