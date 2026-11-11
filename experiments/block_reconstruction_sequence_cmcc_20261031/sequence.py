"""Prepare or execute the one reviewed fixed continuation; no retry or polling."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b


def prepare():
    b.executor.cpu_environment()
    b.require(not os.path.lexists(b.DIRECTORY), 'This fixed plan already exists; never reprepare or overwrite.')
    parent = b.bind_parent()
    pins, inputs, proofs = b.proofs_and_pins()
    plan = dict(schema=b.SCHEMA, created_unix=time.time(), output=str(b.DIRECTORY), parent=parent,
        control_plan=str(b.CONTROL_PLAN), stages=b.requests(), budget=b.BUDGET,
        source_sha256=pins, input_sha256=inputs, reused_proofs=proofs,
        python=str(b.PYTHON), conda_env='bitvae', cwd=str(b.REPO),
        package_versions={name:importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'accelerate', 'trl', 'peft', 'datasets', 'lm_eval')},
        automatic_retry=False, automatic_resume=False, resource_wait=False,
        boundary='One bound worker successful closed publication; actual sources and real exit0 dependencies only. '
            'BR midpoint quality never gates E. One BR36, one E32 and at most two complete-nine results; '
            'other-chain comparison is one snapshot, with no polling or reservation.')
    b.DIRECTORY.mkdir(parents=True)
    b.write(b.DIRECTORY/'plan.json', plan)
    print(json.dumps(dict(status='prepared_not_started', plan=str(b.DIRECTORY/'plan.json'),
        stages=[row['name'] for row in plan['stages']], source_pins=len(pins), input_pins=len(inputs))), flush=True)


def check_plan(plan):
    b.require(plan['schema'] == b.SCHEMA and plan['output'] == str(b.DIRECTORY)
        and plan['control_plan'] == str(b.CONTROL_PLAN) and plan['budget'] == b.BUDGET
        and plan['stages'] == b.requests() and plan['python'] == str(b.PYTHON)
        and plan['conda_env'] == 'bitvae' and plan['cwd'] == str(b.REPO)
        and plan['automatic_retry'] is plan['automatic_resume'] is plan['resource_wait'] is False,
        'The reviewed fixed plan, GPU assignments, budget or commands changed.')
    b.require(plan['parent'] == b.bind_parent(), 'The actual C0 event dependency changed after preparation.')
    b.executor.check_pins(plan['source_sha256'])
    b.executor.check_pins(plan['input_sha256'])


def execute_rows(rows, record, record_path, pins):
    """Use the tested real-child launcher once for each fixed dependency."""
    def changed(stage):
        record['stages'][stage['name']] = stage
        record['status'] = stage['name']+'_'+stage['status']
        b.write(record_path, record)

    for row in rows:
        b.executor.check_pins(pins)
        b.executor.launch_child(row, changed)


def run():
    b.executor.cpu_environment()
    path = b.DIRECTORY/'plan.json'
    plan = b.read(path)
    check_plan(plan)
    record = dict(schema=b.SCHEMA, status='waiting_bound_C0_formal_publication', pid=os.getpid(),
        start_ticks=b.executor.proc_start(os.getpid()), argv=[sys.executable, *sys.argv],
        started_unix=time.time(), plan_sha256=b.sha(path), stages={}, events=[])
    # Reuse the existing flock + exclusive run publication contract verbatim.
    with b.events.claim_once(b.DIRECTORY, record) as record_path:
        code = 1
        try:
            for target in (b.BR, b.E, b.BR_EVAL, b.E_EVAL):
                b.require(not os.path.lexists(target), 'A fixed stage output already exists: '+str(target))

            def observe(event, evidence):
                row = dict(event=event, unix=time.time(), evidence=evidence)
                record['events'].append(row)
                b.write(record_path, record)
                print(json.dumps(row), flush=True)

            record['formal_exit'] = b.events.wait_formal(plan['parent'], observe)
            check_plan(plan)
            execute_rows(plan['stages'], record, record_path,
                {**plan['source_sha256'], **plan['input_sha256']})
            record.update(status='completed_BR36_E32_and_two_complete_nine', exit_code=0,
                comparison=b.read(b.DIRECTORY/'comparison.json'))
            code = 0
        except BaseException:
            record.update(status='failed', exit_code=1, error=traceback.format_exc())
            raise
        finally:
            record['finished_unix'] = time.time()
            b.write(record_path, record)
            (b.DIRECTORY/'run.exit').write_text(str(code)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()


if __name__ == '__main__':
    main()
