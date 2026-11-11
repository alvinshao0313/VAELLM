"""Compare the actual125/250 checkpoint contents of the two fixed-k runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def binding(path):
    with path.open('rb') as stream:
        return dict(path=str(path.resolve()), sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def compare(left, right):
    result = dict(equal=True, tensor_count=0, tensor_elements=0, mismatch_paths=[])

    def visit(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            result['tensor_count'] += 1
            result['tensor_elements'] += a.numel()
            equal = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            equal = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        elif type(a) is not type(b):
            equal = False
        elif isinstance(a, dict):
            equal = a.keys() == b.keys()
            for key in a.keys() & b.keys():
                visit(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, (list, tuple)):
            equal = len(a) == len(b)
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + '/' + str(index))
        else:
            equal = a == b
        if not equal:
            result['equal'] = False
            result['mismatch_paths'].append(path)

    visit(left, right, '')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.root).resolve()
    paths = [root / ('prefix_' + str(index)) for index in (0, 1)]
    results = [json.loads((path / 'prefix.json').read_text()) for path in paths]
    if any(result['status'] != 'completed' or result['exit_code'] != 0 for result in results):
        raise RuntimeError('Both actual250-step prefixes must complete before verification.')
    report = dict(implementation=binding(Path(__file__)),
        runs=[binding(path / 'prefix.json') for path in paths],
        initial_equal=results[0]['initial'] == results[1]['initial'],
        final_equal=results[0]['final'] == results[1]['final'],
        actual_input_equal=results[0]['actual_run']['actual_data'] == results[1]['actual_run']['actual_data'],
        exact_original_input_prefix=all(result['input_prefix_verified'] for result in results),
        actual_GPU_hours=sum(result['elapsed_seconds'] for result in results) / 3600,
        checkpoints=[])
    for step in (125, 250):
        row = dict(step=step, files={})
        directories = [path / 'train/trainer_state' / f'checkpoint-{step}' for path in paths]
        for name in ('training_model_state.pt', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            files = [directory / name for directory in directories]
            left, right = [torch.load(path, map_location='cpu', weights_only=False) for path in files]
            row['files'][name] = dict(**compare(left, right), sources=[binding(path) for path in files])
            del left, right
        histories = [json.loads((directory / 'trainer_state.json').read_text())['log_history'] for directory in directories]
        keys = {'step', 'loss', 'grad_norm', 'learning_rate', 'epoch'}
        histories = [[{key: value for key, value in line.items() if key in keys}
            for line in history if 'step' in line] for history in histories]
        row['training_log_equal'] = histories[0] == histories[1]
        report['checkpoints'].append(row)
    report['passed'] = all(report[key] for key in ('initial_equal', 'final_equal', 'actual_input_equal', 'exact_original_input_prefix'))
    report['passed'] &= all(row['training_log_equal'] and all(value['equal'] for value in row['files'].values())
        for row in report['checkpoints'])
    (root / 'prefix_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
