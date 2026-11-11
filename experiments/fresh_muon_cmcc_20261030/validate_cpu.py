"""Verify immutable inputs and reused GPU evidence; reject unfinished sources."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b
from .prepare import data_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized()
        and not args.output.exists(), 'Use a fresh CPU-only bitvae validation directory.')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', pid=os.getpid(), argv=sys.argv)
    try:
        b.configure()
        proof = b.verified_implementation()
        dataset, recovery, cache, evidence = b.inputs.load_inputs()
        exposure = data_contract(dataset, recovery)
        b.require(b.mature.STEPS['formal32'] == list(range(32))
            and b.mature.LIMITS == {'qasper':128, 'hotpotqa':32}, 'Changed group sequence or QA generation budget.')
        missing = b.ARTIFACTS/'fresh_fixed2_cmcc_20261030/single_20261030_v2_C0'
        rejected = False
        try:
            b.source(missing/'final_model', missing/'report.json')
        except (ValueError, FileNotFoundError) as error:
            rejected = True
            result['unavailable_C0_reason'] = str(error)
        b.require(rejected, 'The incomplete future parent must not be silently replaced.')
        syntax = []
        for path in b.HERE.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
            syntax.append(str(path))
        result.update(status='verified_protocol_inputs_and_reused_fixed2_GPU_evidence', exit_code=0,
            data=exposure, source_parent_not_available=True, actual_source_rejected=True,
            reused_GPU={key:value for key,value in proof.items() if key != 'record'},
            teacher_cache=cache, M_target_evidence=evidence, source_bindings=b.code_bindings(),
            syntax_checked=syntax, original_schedule_steps=500,
            actual_learning_rates=[1e-6*(1-(step-1)/500) for step in range(1,33)],
            CUDA_initialized=b.torch.cuda.is_initialized(),
            boundary='No C0 parent loaded, no training/export/full9 executed. Existing two-update math/lifecycle is reused; actual new-parent preparation is required before32.')
        b.require(not result['CUDA_initialized'], 'CPU validation initialized CUDA.')
    except BaseException:
        result.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        b.core.write(args.output/'validation.json', result)
    print(json.dumps(dict(status=result['status'], data=exposure)), flush=True)


if __name__ == '__main__':
    main()
