"""Thin CPU-check/GPU0 entry for the assigned mixed recovery handoff short chain."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def idle_gpu0():
    rows = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=index,uuid,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if rows[0].strip() != '0' or int(rows[2]) > 256:
        raise RuntimeError('Assigned GPU0 is occupied; do not wait, reserve, retry or stop another process.')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    blockers = [line for line in active.splitlines() if line.split(',')[0].strip() == rows[1].strip()]
    if blockers:
        raise RuntimeError('Assigned GPU0 has active compute processes: ' + repr(blockers))
    return dict(physical_gpu=0, uuid=rows[1].strip(), used_MiB=int(rows[2]), checked_unix=time.time(), blockers=blockers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            parser.error('Hide CUDA for the import/entry contract check; the passed full CPU preparation is reused.')
        resources = None
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            parser.error('Expose only the assigned physical GPU0.')
        # Check before importing torch/Trainer/native libraries, which may
        # initialize this process's own CUDA context during import.
        resources = None if args.fresh else idle_gpu0()
    from experiments.adaptive_native_model_20261029 import handoff_run
    if args.check_only:
        print(json.dumps(handoff_run.check_entry(args.prepared.resolve())), flush=True)
    elif args.fresh:
        handoff_run.fresh(args.prepared.resolve(), args.output.resolve())
    else:
        handoff_run.execute(args.prepared.resolve(), args.output.resolve(), resources)


if __name__ == '__main__':
    main()
