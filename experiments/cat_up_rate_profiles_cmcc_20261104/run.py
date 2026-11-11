"""One finite owned worker chain; no retries, scheduler daemon or extra slots."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    started = time.time()
    parser = argparse.ArgumentParser(); parser.add_argument('--label', choices=('k32', 'k40', 'k48'), required=True)
    parser.add_argument('--batch-root', required=True); args = parser.parse_args()
    declared_GPU = os.environ.get('CUDA_VISIBLE_DEVICES')
    # This finite coordinator owns no CUDA context; each actual core worker gets
    # the original full caller environment with its explicit assigned GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['ACCELERATE_USE_CPU'] = 'true'
    from . import bindings as b
    from experiments.cat_auto_protocol_cmcc_20261104.search import validate_request
    from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
    card = b.CARDS[args.label]
    require(declared_GPU == str(card['gpu']) and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae', 'Caller must inherit actual bitvae and the assigned physical GPU.')
    batch = Path(args.batch_root).resolve()
    manifest = b.read(batch / 'batch.json')
    require(manifest['source_sha256'] == b.pins(), 'Batch sources changed after CPU preparation.')
    root = batch / args.label
    require(not root.exists(), 'Use a unique independent new rate chain.')
    root.mkdir()
    deadline = started + card['technical_seconds'] + card['formal_seconds']
    state = dict(status='running', label=args.label, gpu=card['gpu'], controller_pid=os.getpid(),
        started_epoch=started, absolute_deadline_epoch=deadline, maximum_GPU_hours=(deadline - started) / 3600, stages=[])
    b.write(root / 'chain.json', state)
    phases = ('formal',) if args.label == 'k32' else ('technical_validation', 'formal')
    receipt = None
    code = 1
    try:
        for phase in phases:
            output = root / phase
            request = b.make_request(args.label, phase, output=output, chain_deadline=deadline, technical_receipt=receipt)
            validate_request(request)
            registry = root / (phase + '_request.json')
            b.write(registry, request)
            output.mkdir()
            log = (output / 'worker.log').open('w')
            command = [sys.executable, '-m', 'experiments.cat_auto_protocol_cmcc_20261104.search', '--request', str(registry), '--worker']
            worker_environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(card['gpu']), ACCELERATE_USE_CPU='false')
            worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=worker_environment)
            stage = dict(phase=phase, pid=worker.pid, controller_pid=os.getpid(), started_epoch=time.time(), command=command,
                deadline_epoch=request['absolute_deadline_epoch'], request=str(registry), request_sha256=b.adapter.file_sha(registry))
            state['stages'].append(stage); b.write(root / 'chain.json', state); b.write(output / 'process.json', stage)
            try:
                code = worker.wait(timeout=max(0., request['absolute_deadline_epoch'] - time.time()))
            except BaseException as error:
                _kill_group(worker.pid); worker.wait()
                stage['termination_reason'] = type(error).__name__ + ': ' + str(error)
                code = worker.returncode
                raise
            finally:
                log.close()
                if worker.returncode:
                    _kill_group(worker.pid)
                stage.update(exit_code=worker.returncode, ended_epoch=time.time())
                b.write(output / 'exit.json', dict(exit_code=worker.returncode, end_epoch=time.time(), actual_completed=(output / 'completed.json').is_file()))
                b.write(root / 'chain.json', state)
            require(code == 0, 'The actual worker failed; this rate has no dependent formal/retry.')
            completed = b.read(output / 'completed.json')
            require(completed['status'] == 'completed' and completed['exit_code'] == 0 and completed['phase'] == phase,
                'A missing or incomplete actual receipt cannot enable the next stage.')
            if phase == 'technical_validation':
                receipt = output / 'completed.json'
        state.update(status='completed', exit_code=0, ended_epoch=time.time())
        code = 0
    except BaseException as error:
        state.update(status='failed', exit_code=code if code else 1, ended_epoch=time.time(), error=type(error).__name__ + ': ' + str(error))
        b.write(root / 'chain.json', state)
        raise
    finally:
        state['elapsed_seconds'] = time.time() - started
        b.write(root / 'chain.json', state)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
