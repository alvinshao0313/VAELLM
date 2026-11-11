"""PID-bound deadline adoption. Never edits an active request or Python source."""
import argparse
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import select
import signal
import time

BASE = Path('/home/cmcc/VAELLM/artifacts/cat_up_rate_profiles_cmcc_20261104/batch_20261104_1538').resolve()
LABELS = ('k32', 'k40', 'k48')


def _pidfd_call(name, *args):
    # This bitvae Python omits pidfd wrappers; use the same Linux syscalls.
    require(platform.machine() == 'x86_64', 'This local fallback binds the actual x86_64 syscall ABI only.')
    header = Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    expected = {'pidfd_open': 434, 'pidfd_send_signal': 424}[name]
    require(('#define __NR_' + name + ' ' + str(expected)) in header, 'Actual kernel syscall header differs.')
    libc = ctypes.CDLL(None, use_errno=True); libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(expected), *args)
    if result == -1:
        number = ctypes.get_errno(); raise OSError(number, os.strerror(number))
    return int(result)


def pidfd_open(pid):
    return _pidfd_call('pidfd_open', ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_signal(fd, number):
    return _pidfd_call('pidfd_send_signal', ctypes.c_int(fd), ctypes.c_int(number), ctypes.c_void_p(0), ctypes.c_uint(0))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def process(pid):
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text(); fields = stat[stat.rfind(')') + 2:].split()
    status = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines() if ':' in line)
    return dict(pid=pid, start=int(fields[19]), ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]),
        command_sha256=hashlib.sha256((root / 'cmdline').read_bytes()).hexdigest(),
        command=(root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
        term_default=not any(int(status[key].strip(), 16) & (1 << (signal.SIGTERM - 1)) for key in ('SigCgt', 'SigIgn', 'SigBlk')))


def bind(expected):
    fd = pidfd_open(expected['pid'])
    try:
        actual = process(expected['pid'])
        require(all(actual[key] == expected[key] for key in ('pid', 'start', 'command_sha256')), 'Exact process identity changed.')
        poller = select.poll(); poller.register(fd, select.POLLIN)
        require(not poller.poll(0), 'The exact process already exited.')
        return fd, actual
    except BaseException:
        os.close(fd); raise


def validate_entry(label, entry):
    request_path = Path(entry['request']).resolve()
    require(request_path == BASE / label / 'formal_request.json' and sha(request_path) == entry['request_sha256'], 'Wrong frozen formal registry.')
    request = read(request_path)
    require(request['phase'] == 'formal' and request['technical_steps'] is None
        and request['physical_GPU'] == {'k32': 1, 'k40': 3, 'k48': 4}[label]
        and Path(request['output_dir']).resolve() == Path(entry['output']).resolve() == BASE / label / 'formal'
        and entry['old_deadline'] == request['absolute_deadline_epoch']
        and entry['new_deadline'] == entry['old_deadline'] + 600, 'Actual formal phase/GPU/output/old deadline does not match its exact amendment.')


def stop_worker(entry, fd):
    alive = select.poll(); alive.register(fd, select.POLLIN)
    if alive.poll(0): return
    try:
        current = process(entry['worker']['pid'])
        require(current['start'] == entry['worker']['start'] and current['pgid'] == current['pid'], 'Refuse a different worker process group.')
        try:
            os.killpg(current['pgid'], signal.SIGKILL)
        except ProcessLookupError:
            require(alive.poll(5000), 'Missing worker PG did not correspond to exact process exit.')
        try:
            pidfd_signal(fd, signal.SIGKILL)
        except ProcessLookupError:
            require(alive.poll(5000), 'Missing exact worker did not become exited.')
    except (FileNotFoundError, ProcessLookupError):
        require(alive.poll(5000), 'Process disappeared without exact pidfd exit evidence.')
    require(alive.poll(5000), 'Exact worker did not exit after bounded cleanup.')


def draft():
    entries = {}
    for label in LABELS:
        chain = read(BASE / label / 'chain.json')
        stage = chain['stages'][-1]
        require(chain['status'] == 'running' and stage['phase'] == 'formal', 'All three fixed profiles must be in their own formal phase.')
        controller, worker = process(chain['controller_pid']), process(stage['pid'])
        require(controller['term_default'] and worker['pgid'] == worker['pid'] and worker['pgid'] != controller['pgid']
            and worker['ppid'] == controller['pid'], 'Actual default-TERM/independent-worker premise is absent.')
        request = Path(stage['request']).resolve()
        require(request == BASE / label / 'formal_request.json' and sha(request) == stage['request_sha256'], 'Wrong or changed original formal registry.')
        entries[label] = dict(controller=controller, worker=worker, request=str(request), request_sha256=sha(request),
            output=str(BASE / label / 'formal'), old_deadline=stage['deadline_epoch'], new_deadline=stage['deadline_epoch'] + 600)
        validate_entry(label, entries[label])
    return dict(schema='fixed_three_profile_budget_amendment_v0', user_approved=False, user_authorization=None,
        created_epoch=time.time(), old_total_GPU_hours=2.75, new_total_GPU_hours=3.25,
        extra_seconds_per_formal=600, entries=entries, handoff_source_sha256=sha(__file__))


def verify_formal(entry):
    request = read(entry['request']); completed = read(Path(entry['output']) / 'completed.json')
    require(sha(entry['request']) == entry['request_sha256'], 'The original registry changed.')
    require(completed['status'] == 'completed' and completed['exit_code'] == 0 and completed['phase'] == 'formal'
        and completed['actual_technical_steps'] is None, 'No actual formal completion record.')
    require(len(completed['candidates']) == 1, 'Unexpected profile count.')
    result = completed['candidates'][0]; fitted = result['fit_report']; recipe = request['candidates'][0]['recipe']
    require(result['candidate_id'] == fitted['candidate_id'] == request['candidates'][0]['candidate_id']
        and fitted['resolved_recipe'] == recipe and fitted['parent_identity'] == request['run_identity']
        and completed['fit_code_identity'] == request['run_identity']['code_identity']
        and completed['consumer_identity'] == request['run_identity']['consumer_identity']
        and completed['model_identity'] == request['run_identity']['model']
        and completed['data_identity'] == request['run_identity']['data_identity'], 'Completion source/parent/model/data identity mismatch.')
    require(fitted['formal_eligible'] and result['formal_result_eligibility'] and fitted['actual_stage_steps'] == 10000
        and fitted['technical_steps'] is None and len(fitted['stages']) == recipe['S']
        and all(stage['actual_steps'] == 10000 for stage in fitted['stages']), 'Not every actual residual stage completed10000.')
    fresh = result['fresh_evidence']
    require(fresh['state_exact'] and fresh['gpu_validation'] and fresh['same_input_same_BF16_path_logits_exact']
        and fitted['frozen_prefix_before_digest'] == fitted['frozen_prefix_after_digest'], 'Full actual fresh/frozen contract missing.')
    require(all(sha(path) == value for path, value in request['source_sha256'].items()), 'An active pinned source changed.')
    return dict(artifact_completed_verified=True, completed_sha256=sha(Path(entry['output']) / 'completed.json'),
        checkpoint_id=fresh['checkpoint_id'], application_report_exit_code=completed['exit_code'],
        actual_end_epoch=completed['actual_end_epoch'], verification_scope=completed['verification_scope'])


def adopt(entry, report_path, *, verify=verify_formal, margin=10):
    """Internal OS mechanism; actual CLI authorization is checked separately."""
    report_path = Path(report_path)
    require(not report_path.exists() and time.time() + margin < entry['old_deadline'], 'No safe time remains or this handoff already has a record.')
    worker_fd = controller_fd = None
    transferred = False
    report = dict(status='preparing', original_binding=entry, original_request_unchanged=True,
        OS_exit_code=None, OS_exit_reason='Non-parent observer cannot obtain waitpid exit status', started_epoch=time.time())
    previous = {}
    try:
        worker_fd, worker = bind(entry['worker'])
        controller_fd, controller = bind(entry['controller'])
        require(controller['term_default'] and worker['ppid'] == controller['pid'] and worker['pgid'] == worker['pid']
            and worker['pgid'] != controller['pgid'], 'Default TERM or independent worker process group is not established.')
        poller = select.poll(); poller.register(worker_fd, select.POLLIN)
        # The new bounded watch is armed and recorded before the old timer exits.
        report.update(status='armed', armed_epoch=time.time(), adopter_pid=os.getpid())
        write(report_path, report)
        def interrupted(number, _frame):
            raise InterruptedError('Adopter interrupted by signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[number] = signal.signal(number, interrupted)
        pidfd_signal(controller_fd, signal.SIGTERM)
        transferred = True
        report.update(status='watching', controller_TERM_epoch=time.time())
        write(report_path, report)
        old = select.poll(); old.register(controller_fd, select.POLLIN)
        require(old.poll(5000), 'Exact old controller did not exit after default TERM.')
        while True:
            remaining = entry['new_deadline'] - time.time()
            if poller.poll(max(0, min(1000, int(remaining * 1000)))):
                proof = verify(entry)
                require(proof.get('actual_end_epoch', time.time()) <= entry['new_deadline'], 'Artifacts finished beyond the amended authorization.')
                report.update(status='completed_artifacts_verified', proof=proof, ended_epoch=time.time())
                break
            if remaining <= 0:
                raise TimeoutError('The explicitly amended absolute deadline expired.')
    except BaseException as error:
        report.update(status='failed_after_handoff' if transferred else 'handoff_not_performed',
            error=type(error).__name__ + ': ' + str(error), ended_epoch=time.time())
        try:
            if transferred and worker_fd is not None:
                stop_worker(entry, worker_fd)
        except BaseException as cleanup_error:
            report['cleanup_error'] = type(cleanup_error).__name__ + ': ' + str(cleanup_error)
            write(report_path, report)
            raise
        write(report_path, report)
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        if worker_fd is not None: os.close(worker_fd)
        if controller_fd is not None: os.close(controller_fd)
    write(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--draft', action='store_true'); parser.add_argument('--output', required=True)
    parser.add_argument('--amendment'); parser.add_argument('--label', choices=LABELS); args = parser.parse_args()
    if args.draft:
        require(not Path(args.output).exists(), 'Do not overwrite a draft/authorization record.')
        write(args.output, draft()); return
    amendment = read(args.amendment)
    require(amendment['schema'] == 'fixed_three_profile_budget_amendment_v0' and amendment['user_approved'] is True
        and amendment['user_authorization'] and amendment['new_total_GPU_hours'] == 3.25
        and amendment['old_total_GPU_hours'] == 2.75 and amendment['handoff_source_sha256'] == sha(__file__),
        'No explicit matching user authorization for this fixed amendment.')
    require(set(amendment['entries']) == set(LABELS) and args.label, 'Only the three bound profile chains may transfer.')
    for label, entry in amendment['entries'].items():
        validate_entry(label, entry)
    adopt(amendment['entries'][args.label], args.output)


if __name__ == '__main__':
    main()
