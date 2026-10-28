"""Two fixed successful KL2000 endpoints, each with original full8 and24h full21."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/recompression_d_recovery_20261027'))
import core as d
from evaluate import evaluate

IDS = dict(C='a8bb3da6-0f63-44d0-8f38-40768fe289dc', G='956eb2cf-f96b-4e75-8e87-d587221617c2')
GPUS = dict(C=2, G=6)
OUTPUT = d.TOPIC / 'qwen_night_endpoint_evaluations_20261027'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('C', 'G'), required=True)
    args = parser.parse_args()
    d.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only endpoint evaluation driver.')
    d.check()
    d.native.check_prepared()
    source = d.native.OUTPUT / args.profile / '06_decoder_kl'
    pipeline_path = source / 'pipeline.json'
    pipeline = d.read(pipeline_path)
    audit_path = source / 'native_audit.json'
    audit = d.read(audit_path)
    checkpoint = Path(pipeline['final_model'])
    d.require(pipeline['status'] == 'complete_native'
        and pipeline['checkpoint_id'] == audit['checkpoint_id'] == IDS[args.profile]
        and audit['status'] == 'verified'
        and pipeline['stages']['train']['exit_code'] == pipeline['stages']['audit']['exit_code'] == 0,
        'Only the fixed real successful native KL2000/audit endpoints are eligible.')
    d.native.check_gpu(GPUS[args.profile])
    output = OUTPUT / (args.profile + '06')
    d.require(not output.exists(), 'One independent full9 per endpoint, without retry or resume.')
    output.mkdir(parents=True)
    path = output / 'lane.json'
    record = dict(status='running_full9', profile=args.profile, checkpoint_id=IDS[args.profile],
        checkpoint_dir=str(checkpoint), gpu=GPUS[args.profile], pid=os.getpid(),
        start_ticks=d.processes.process_start(os.getpid()), started_unix=time.time(),
        source_sha256={str(item): d.sha(item) for item in
            (Path(__file__), pipeline_path, audit_path, checkpoint / 'checkpoint_meta.json',
             d.HERE / 'evaluate.py', d.HERE / 'core.py', d.PREPARED, d.native.PREPARED)},
        full8_samples=24742, full21_samples=4750, longbench_timeout_seconds=86400,
        automatic_retry=False, resource_polling=False)
    d.write(path, record)
    try:
        result = evaluate(checkpoint, output / 'full9', GPUS[args.profile], audit_path)
        record.update(status='completed', exit_code=0, evaluation=result)
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        d.write(path, record)


if __name__ == '__main__':
    main()
