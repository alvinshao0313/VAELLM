"""Verify the real H96 entry organization without model loading or CUDA."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys
import time
import traceback

from . import data, rolling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    b, torch = data.legacy, data.torch
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not args.output.exists(),
        'Use caller bitvae, hidden CUDA and a new CPU evidence directory.')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    opened = set()

    def audit(event, arguments):
        if event == 'open' and arguments and isinstance(arguments[0], str):
            path = Path(arguments[0])
            if path.suffix in ('.pt', '.json') and 'qwen3_8b_mmlu65_20261004' in str(path):
                opened.add(str(path.resolve()))

    sys.addaudithook(audit)
    record = dict(status='running', command=[sys.executable, *sys.argv])
    try:
        pool, evidence = data.load()
        short = pool.select_groups(data.SHORT_GROUPS)
        for entry in pool.entries:
            rolling._validate_entry(entry)
        counts = Counter(index for group in pool.groups for index in group.entry_indices)
        b.require(all(counts[index] == (1 if entry.domain == 'mmlu' else 2)
            for index, entry in enumerate(pool.entries)), 'Do not duplicate QA state for repeated group references.')
        for group in pool.groups:
            entries = [pool.entries[index] for index in group.entry_indices]
            b.require(entries[0].domain == 'mmlu' and all(entry.domain != 'mmlu' for entry in entries[1:])
                and len(entries) == 9 and group.denominator(4096) == 4096 * sum(entry.valid_tokens for entry in entries),
                'Use the sum of all actual valid tokens, without domain or per-entry means.')
        b.require(all(any(entry is full for full in pool.entries) for entry in short.entries)
            and len({id(entry) for entry in pool.entries}) == 160,
            'Short selection must reuse the original complete entries, without tensor copies.')
        b.require(not any('teacher_target' in path or 'targets.pt' in path for path in opened),
            'Block reconstruction must not load unused cached vocabulary teacher targets.')
        expected_datasets = {value['path'] for value in evidence['bindings'].values() if value['path'].endswith('.pt')}
        b.require({path for path in opened if path.endswith('.pt')} == expected_datasets,
            'The CPU data reader opened an unbound dataset or a cached teacher payload.')
        # Exercise detached transfer on actual integer inputs under an enclosing
        # inference context; model capture separately disables inference tensors.
        with torch.inference_mode():
            with torch.inference_mode(False), torch.no_grad():
                copied = rolling._cpu_copy(short.entries[0].inputs)
        b.require(all(not torch.is_inference(value) and value.grad_fn is None
            and not value.requires_grad and value.device.type == 'cpu' for value in copied.values()),
            'Cross-layer CPU storage must be ordinary detached tensors.')
        b.require(not torch.cuda.is_initialized(), 'CPU entry validation initialized CUDA.')
        record.update(status='verified_real_H96_entries', exit_code=0, evidence=evidence,
            actual_opened_data_files=sorted(opened), QA_state_stored_once=True,
            short_entries_are_original_objects=True, token_denominators_verified=True,
            normal_detached_CPU_copy_verified=True, CUDA_initialized=False,
            source_bindings={str(path): b.binding(path) for path in
                (Path(data.__file__), Path(rolling.__file__), Path(__file__))},
            boundary='Real input organization only; real Qwen3 GPU pre-hook, block math and adjacent-layer rolling remain unverified.')
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        b.core.write(args.output / 'validation.json', record)
    print(dict(status=record['status'], entries=len(pool.entries), short_entries=len(short.entries),
        unique_valid_tokens=evidence['full']['unique_total_valid_tokens'], output=str(args.output)), flush=True)


if __name__ == '__main__':
    main()
