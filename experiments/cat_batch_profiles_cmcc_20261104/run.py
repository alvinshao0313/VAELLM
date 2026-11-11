"""Exactly one approved finite chain; reuse existing wait/PG cleanup mechanics."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.cat_up_rate_profiles_cmcc_20261104.run import _kill_group


def main():
    started = time.time()
    parser = argparse.ArgumentParser(); parser.add_argument('--label', choices=('k32', 'k40', 'k48'), required=True)
    parser.add_argument('--batch-root', required=True); parser.add_argument('--authorization', required=True)
    args = parser.parse_args(); declared_gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['ACCELERATE_USE_CPU'] = 'true'
    from . import bindings as b
    from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and declared_gpu == str(b.CARDS[args.label]['gpu']),
        'Caller must use confirmed bitvae and this label actual assigned GPU.')
    batch = Path(args.batch_root).resolve(); proposal_path = batch/'proposal.json'; proposal = b.read(proposal_path)
    b.authorization(proposal, proposal_path, args.authorization)
    occupied = subprocess.check_output(['nvidia-smi', '--id='+str(b.CARDS[args.label]['gpu']),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    require(not occupied.strip(), 'The assigned GPU currently has an existing compute process; stop before creating a chain or worker.')
    root = batch/args.label
    root.mkdir()  # Exclusive claim: existing label means no retry or duplicate chain.
    state = dict(status='running', label=args.label, gpu=b.CARDS[args.label]['gpu'], controller_pid=os.getpid(),
        started_epoch=started, absolute_deadline_epoch=started+b.CHAIN_SECONDS, maximum_GPU_hours=1.25,
        proposal=b.bound(proposal_path), authorization=b.bound(args.authorization), stages=[])
    b.write(root/'chain.json', state); code = 1; receipt = None
    try:
        for phase in ('technical_validation', 'formal'):
            request = b.make_request(proposal, proposal_path, args.authorization, args.label, phase,
                chain_started=started, technical_receipt=receipt)
            registry = root/(phase+'_request.json'); b.write_new(registry, request)
            output = Path(request['output_dir']); output.mkdir()
            command = [sys.executable, '-B', '-u', '-m', 'experiments.cat_batch_gather_cmcc_20261104.search',
                '--request', str(registry), '--worker']
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(b.CARDS[args.label]['gpu']),
                ACCELERATE_USE_CPU='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (output/'worker.log').open('x') as log:
                worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env=environment)
                stage = dict(phase=phase, pid=worker.pid, controller_pid=os.getpid(), started_epoch=time.time(),
                    command=command, deadline_epoch=request['absolute_deadline_epoch'], request=str(registry), request_sha256=b.sha(registry))
                state['stages'].append(stage); b.write(root/'chain.json', state); b.write(output/'process.json', stage)
                try:
                    code = worker.wait(timeout=max(0., request['absolute_deadline_epoch']-time.time()))
                except BaseException as error:
                    _kill_group(worker.pid); worker.wait(); code = worker.returncode
                    stage['termination_reason'] = type(error).__name__+': '+str(error)
                    raise
                finally:
                    if worker.returncode:
                        _kill_group(worker.pid)
                    stage.update(exit_code=worker.returncode, ended_epoch=time.time())
                    b.write(output/'exit.json', dict(exit_code=worker.returncode, end_epoch=time.time(),
                        actual_completed=(output/'completed.json').is_file()))
                    b.write(root/'chain.json', state)
            require(code == 0, 'Actual worker failed; stop only this chain, with no retry or extension.')
            completed = b.read(output/'completed.json')
            require(completed['status'] == 'completed' and completed['exit_code'] == 0 and completed['phase'] == phase
                and completed['request_identity'] == b.digest(request), 'Unknown/incomplete receipt cannot advance or complete this chain.')
            if phase == 'technical_validation':
                receipt = output/'completed.json'
                b.technical_evidence(proposal, args.label, receipt)
        state.update(status='completed', exit_code=0, ended_epoch=time.time()); code = 0
    except BaseException as error:
        state.update(status='failed', exit_code=code if code else 1, ended_epoch=time.time(), error=type(error).__name__+': '+str(error))
        raise
    finally:
        state['elapsed_seconds'] = time.time()-started; b.write(root/'chain.json', state)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
