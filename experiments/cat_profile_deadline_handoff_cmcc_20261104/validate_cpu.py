"""Actual subprocess ownership/timeout fixtures, never a formal model proof."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time

from .handoff import adopt, draft, process, write


def start(root, *, duration, old_seconds, handler=False):
    done, ready = root / 'done.json', root / 'ready.json'
    worker_code = ('import json,time;from pathlib import Path;time.sleep(' + str(duration) + ');'
        'Path(' + repr(str(done)) + ').write_text(json.dumps(dict(actual_end_epoch=time.time(),scope="CPU_subprocess_fixture")))')
    controller_code = ('import subprocess,sys,json,os,signal,time;from pathlib import Path;'
        + ('signal.signal(signal.SIGTERM,lambda a,b:None);' if handler else '')
        + 'p=subprocess.Popen([sys.executable,"-c",' + repr(worker_code) + '],start_new_session=True);'
        'Path(' + repr(str(ready)) + ').write_text(json.dumps(dict(worker=p.pid,deadline=time.time()+' + str(old_seconds) + ')));'
        '\ntry:p.wait(timeout=' + str(old_seconds) + ')\nexcept subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()\n')
    controller = subprocess.Popen([sys.executable, '-c', controller_code], start_new_session=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    limit = time.monotonic() + 3
    while not ready.exists():
        if time.monotonic() > limit: raise RuntimeError('CPU fixture controller failed to publish actual worker.')
        time.sleep(.005)
    info = json.loads(ready.read_text())
    entry = dict(controller=process(controller.pid), worker=process(info['worker']), old_deadline=info['deadline'],
        new_deadline=info['deadline'] + .8)
    return controller, entry, done


def cleanup(controller, entry):
    for pid in (entry['worker']['pid'], controller.pid):
        try: os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError: pass
    controller.wait(timeout=3)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--report', required=True); parser.add_argument('--draft-output', required=True)
    args = parser.parse_args(); began = time.monotonic(); evidence = {}
    with tempfile.TemporaryDirectory(prefix='profile_adoption_CPU_') as temporary:
        root = Path(temporary)
        case = root / 'natural'; case.mkdir()
        controller, entry, done = start(case, duration=.7, old_seconds=.35)
        result = adopt(entry, case / 'adoption.json', margin=0, verify=lambda _entry: json.loads(done.read_text()))
        assert result['status'] == 'completed_artifacts_verified' and result['OS_exit_code'] is None
        assert result['proof']['actual_end_epoch'] > entry['old_deadline'] and result['proof']['actual_end_epoch'] < entry['new_deadline']
        assert controller.wait(timeout=3) == -signal.SIGTERM
        evidence['natural_after_original_deadline'] = dict(passed=True, old_controller_exit=-15, non_parent_worker_OS_exit_code=None,
            proof_scope=result['proof']['scope'], elapsed_after_old_deadline=result['proof']['actual_end_epoch'] - entry['old_deadline'])
        case = root / 'timeout'; case.mkdir()
        controller, entry, done = start(case, duration=8., old_seconds=.3)
        entry['new_deadline'] = entry['old_deadline'] + .25
        try:
            adopt(entry, case / 'adoption.json', margin=0, verify=lambda _entry: json.loads(done.read_text()))
        except TimeoutError:
            assert not done.exists() and json.loads((case / 'adoption.json').read_text())['status'] == 'failed_after_handoff'
        else: raise AssertionError('Amended expiry failed to stop its actual owned worker.')
        controller.wait(timeout=3)
        evidence['exact_worker_timeout_cleanup'] = True
        for label, wrong, handler in (('wrong_start', True, False), ('TERM_handler', False, True)):
            case = root / label; case.mkdir()
            controller, entry, done = start(case, duration=8., old_seconds=4., handler=handler)
            altered = deepcopy(entry)
            if wrong: altered['worker']['start'] += 1
            try:
                adopt(altered, case / 'adoption.json', margin=0, verify=lambda _entry: json.loads(done.read_text()))
            except ValueError:
                assert controller.poll() is None
            else: raise AssertionError('Unsafe identity/TERM premise was accepted.')
            cleanup(controller, entry)
            evidence[label + '_rejected_without_transfer'] = True
    plan = draft()
    assert plan['user_approved'] is False and all(row['new_deadline'] == row['old_deadline'] + 600 for row in plan['entries'].values())
    write(args.draft_output, plan)
    report = dict(status='CPU_actual_subprocess_handoff_verified', exit_code=0, elapsed_seconds=time.monotonic() - began,
        scope='Actual OS subprocess mechanism only, no CUDA/model/GPU ownership changed', evidence=evidence,
        actual_three_chain_default_TERM_and_independent_PG_verified=True, draft_user_approved=False,
        no_actual_controller_signalled=True)
    write(args.report, report); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
