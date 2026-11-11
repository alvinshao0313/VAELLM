"""Verify real TF input/target identities and selected full-model peak groups on CPU."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from . import data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    b = data.legacy
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not args.output.exists(),
        'Use caller bitvae with hidden CUDA and a new evidence directory.')
    b.torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    record = dict(status='running', command=[sys.executable, *sys.argv], started_unix=time.time())
    try:
        _dataset, _targets, evidence = data.load()
        b.require(not b.torch.cuda.is_initialized(), 'CPU data verification initialized CUDA.')
        record.update(status='verified_static_TF_data', exit_code=0, evidence=evidence,
            CUDA_initialized=False, source_bindings={str(path): b.binding(path)
                for path in (Path(data.__file__), Path(__file__), Path(data.runtime.__file__))})
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        b.core.write(args.output/'validation.json', record)
    print(dict(status=record['status'], output=str(args.output),
        short_groups=evidence['short_exposure']['group_indices'],
        full_input_tokens=evidence['full_exposure']['total_input_tokens'],
        full_answer_positions=evidence['full_exposure']['total_answer_positions']), flush=True)


if __name__ == '__main__':
    main()
