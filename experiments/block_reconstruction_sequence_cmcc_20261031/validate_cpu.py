"""Check new wiring with real CPU children and actual source/CLI contexts only."""
import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

from . import bindings as b
from . import sequence


def cpu_row(root, name, source, *arguments):
    return dict(name=name, argv=[str(b.PYTHON), '-B', '-u', '-c', source, *map(str, arguments)],
        environment=b.executor.environment(''), log=str(root/(name+'.log')),
        exit_file=str(root/(name+'.exit')))


def lifecycle_checks(output):
    evidence = {}
    with tempfile.TemporaryDirectory(prefix='real_children_', dir=output) as temporary:
        root = Path(temporary)
        order = root/'order.txt'
        source = 'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.open("a").write(sys.argv[2]+"\\n")'
        rows = [cpu_row(root, name, source, order, name) for name in ('first', 'second')]
        record = dict(stages={})
        sequence.execute_rows(rows, record, root/'run.json', {})
        b.require(order.read_text() == 'first\nsecond\n'
            and list(record['stages']) == ['first', 'second']
            and record['stages']['first']['finished_unix'] <= record['stages']['second']['started_unix']
            and all(row['exit_code'] == 0 and row['start_ticks'] is not None
                and row['status'] == 'exited' for row in record['stages'].values()),
            'Each next real CPU child must start only after the actual prior child exit0.')
        evidence['two_successful_real_children'] = record
        sentinel = root/'must_not_run'
        rows = [cpu_row(root, 'failure', 'import sys; sys.exit(7)'),
            cpu_row(root, 'dependent', 'from pathlib import Path; import sys; Path(sys.argv[1]).touch()', sentinel)]
        failed = dict(stages={})
        try:
            sequence.execute_rows(rows, failed, root/'failed.json', {})
        except ValueError as error:
            b.require('exited 7' in str(error), 'Unexpected CPU failure propagation: '+str(error))
        else:
            raise ValueError('The real failing CPU child did not stop its dependent.')
        b.require(list(failed['stages']) == ['failure'] and not sentinel.exists()
            and failed['stages']['failure']['exit_code'] == 7, 'A failed stage reached its dependent.')
        evidence['nonzero_real_child_stops_dependency'] = failed
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    b.executor.cpu_environment()
    output = args.output.resolve()
    b.require(output.is_relative_to(b.ROOT) and not output.exists(), 'Use a fresh CPU evidence directory in this task.')
    output.mkdir(parents=True)
    result = dict(schema='BR_sequence_CPU_wiring_validation_v1', status='running',
        python=sys.executable, argv=[sys.executable, *sys.argv], pid=os.getpid(),
        start_ticks=b.executor.proc_start(os.getpid()), started_unix=time.time())
    try:
        syntax = []
        for path in sorted(b.HERE.glob('*.py')):
            ast.parse(path.read_text(), filename=str(path))
            syntax.append(str(path))
        pins, inputs, proofs = b.proofs_and_pins()
        parent = b.bind_parent()
        b.require(sequence.b.events.wait_formal is b.events.wait_formal
            and sequence.b.events.claim_once is b.events.claim_once,
            'The new runner must directly reuse the existing tested kernel wait and claim.')
        result['actual_bound_parent'] = parent
        result['real_child_lifecycle'] = lifecycle_checks(output)
        rows = b.requests()
        b.require([row['name'] for row in rows] == ['C0_source', 'BR36', 'BR_fresh', 'BR_source',
            'E_prepare', 'E32', 'E_fresh', 'BR_full9', 'E_full9', 'comparison']
            and [row['environment']['CUDA_VISIBLE_DEVICES'] for row in rows]
                == ['', '3', '3', '', '', '3', '3', '', '', '']
            and [row['argv'][-1] for row in rows if row['name'].endswith('full9')] == ['3,4', '3,4'],
            'Keep the fixed order, environments and one GPU3/4 complete-nine pair per endpoint.')
        # Existing formal CLI and numerical proof already cover their own paths.
        # Exercise every new parser without constructing a successful model fixture.
        help_rows = []
        for module in ('sequence', 'audit', 'summarize'):
            argv = [str(b.PYTHON), '-B', '-u', '-m', b.PACKAGE+module, '--help']
            completed = subprocess.run(argv, cwd=b.REPO, env={**os.environ, **b.executor.environment('')},
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            help_rows.append(dict(argv=argv, exit_code=completed.returncode, stdout=completed.stdout))
        # The current formal C0 is genuinely unfinished; the new actual reader
        # must reject it before publishing any audit or allowing a GPU stage.
        b.require(b.events.formal_record(parent)['status'] == 'running',
            'This negative check applies only to the actual currently unfinished C0.')
        rejection = deepcopy(rows[0])
        rejection['argv'][-1] = str(output/'unfinished_C0_must_not_publish.json')
        rejected = subprocess.run(rejection['argv'], cwd=b.REPO,
            env={**os.environ, **rejection['environment']}, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        b.require(rejected.returncode != 0 and not Path(rejection['argv'][-1]).exists()
            and 'Require the actual completed mixed C0' in rejected.stdout,
            'The actual unfinished C0 was accepted or rejected for an unrelated reason.')
        result['actual_unfinished_C0_rejected'] = dict(argv=rejection['argv'],
            exit_code=rejected.returncode, stdout_stderr=rejected.stdout, no_audit_published=True)
        protected = {}
        rice = b.TOPIC/'fresh_rice_cmcc_20261031/single_20261031/plan.json'
        for path, count in ((b.CONTROL_PLAN, 1003), (b.events.DIRECTORY/'plan.json', 1021), (rice, 1019)):
            value = b.read(path)
            b.require(len(value['source_sha256']) == count, 'The active source pin inventory unexpectedly changed.')
            b.executor.check_pins(value['source_sha256'])
            protected[str(path)] = dict(sha256=b.sha(path), source_count=count, all_exact=True)
        b.executor.check_pins(pins)
        b.executor.check_pins(inputs)
        from experiments.block_reconstruction_formal_cmcc_20261031 import bindings as formal
        b.require(not formal.torch.cuda.is_initialized(), 'CPU wiring checks initialized CUDA.')
        result.update(status='passed', exit_code=0, syntax_checked=syntax, new_CLI_context=help_rows,
            fixed_stage_requests=rows, budget=b.BUDGET, reused_proofs=proofs,
            source_sha256=pins, input_sha256=inputs, protected_active_source_pins=protected,
            CUDA_initialized=False, actual_C0_BR_or_E_model_loaded=False,
            fabricated_formal_success_fixture=False, GPU_or_actual_waiter_started=False,
            boundary='New real CPU stage sequencing/failure propagation and actual unfinished-source refusal only. '
                'Existing nine-test event and fourteen-module BR evidence reused; no new kernel wait fixture or GPU work.')
    except BaseException:
        result.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        result['finished_unix'] = time.time()
        b.write(output/'validation.json', result)
    print(json.dumps(dict(status=result['status'], report=str(output/'validation.json'))), flush=True)


if __name__ == '__main__':
    main()
