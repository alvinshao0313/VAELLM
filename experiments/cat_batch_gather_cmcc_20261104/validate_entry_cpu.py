"""Verify actual runner routing and rejection; never launch a model worker."""
import argparse
import dis
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from . import search
from .fit import _validate_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Use a new CPU routing output.')
    started = time.perf_counter()
    before = dict(search.native.execute_candidates.__globals__)
    entry = search._private_execute()
    imports = [item.argval for item in dis.get_instructions(entry) if item.opname == 'IMPORT_NAME']
    assert imports.count('experiments.cat_batch_gather_cmcc_20261104.fit') == 1
    assert imports.count('fit_adapter') == 1
    assert entry.__globals__['validate_request'] is search.validate_request
    assert entry.__globals__['__package__'] == search.native.__package__
    assert all(search.native.execute_candidates.__globals__[key] is value for key, value in before.items())
    old = Path('artifacts/cat_up_rate_profiles_cmcc_20261104/batch_20261104_1538/k32/formal_request.json')
    historical = json.loads(old.read_text())
    try:
        _validate_identity(historical['run_identity'])
    except ValueError:
        pass
    else:
        raise AssertionError('An old fit identity was accepted by the changed runner.')
    command = [sys.executable, '-B', '-m', 'experiments.cat_batch_gather_cmcc_20261104.search', '--help']
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert '--request' in completed.stdout and '--worker' in completed.stdout
    assert not torch.cuda.is_initialized()
    report = dict(status='CPU_batched_runner_routing_verified', exit_code=0,
        elapsed_seconds=time.perf_counter() - started, CUDA_initialized=False,
        actual_fit_import_replaced_once=True, original_bank_installer_and_relative_imports_preserved=True,
        original_namespace_unchanged=True, old_identity_rejected=True,
        actual_CLI_help_exit_code=completed.returncode, CLI_command=command,
        source_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob('*.py'))},
        native_search_SHA256=search.NATIVE_SEARCH_SHA256,
        scope='Compiled real native worker routing and actual CLI only; no GPU/model worker, authorization or formal profile executed')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
