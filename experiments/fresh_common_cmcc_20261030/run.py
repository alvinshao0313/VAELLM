"""Run only the common BF16 -> after_o -> B gate raw2 prerequisite."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from bindings import HERE, HF, gpu_available, recipe, s, source_pins


def execute(output_dir, gpu, validation_only, validation_report):
    s.require(Path(sys.executable).resolve() == s.PYTHON.resolve(), 'Activate bitvae in the caller shell.')
    output = Path(output_dir).resolve()
    s.require(not output.exists(), 'Use a new output directory; never resume or overwrite prior evidence.')
    pins = source_pins()
    hf_identity = s.hf_identity(HF)
    if not validation_only:
        s.require(validation_report is not None, 'Bind the completed new-path GPU validation before formal execution.')
        proof = s.read(validation_report)
        s.require(proof['status'] == 'common_prerequisite_complete' and proof['validation_only']
            and proof['exit_code'] == 0 and proof['source_sha256'] == pins,
            'The GPU validation is incomplete or its bound source changed.')
        verify = s.read(Path(validation_report).parent / 'boundary_verification.json')
        s.require(verify['status'] == 'verified' and verify['non_gate_after_o_complement_exact'],
            'The two real saved boundaries have not been verified.')
        s.require(s.read(Path(validation_report).parent / 'request.json')['hf_identity'] == hf_identity,
            'The actual HF identity changed since the short validation.')
    resource = gpu_available(gpu)
    environment = s.worker_environment(gpu)
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / 'request.json'
    request = dict(schema='cmcc_fresh_common_request_v1', base_model_path=str(HF), output_dir=str(output),
        arm='B', gpu=gpu, validation_only=validation_only, formal_parent_eligible=not validation_only,
        hf_identity=hf_identity, source_sha256=pins, effective_contract=s.policy('B'),
        initial_checkpoint=None, stop='Canonical B gate round_base save returns, before gate recovery.',
        process_protocol='prefix real after_o exit0 -> separate seed31 process -> canonical B raw2 exit0',
        gpu_validation=None if validation_only else dict(path=str(validation_report), sha256=s.sha(validation_report)))
    s.write(request_path, request)
    report_path = output / 'run.json'
    report = dict(status='running', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), resources=resource,
        validation_only=validation_only, source_sha256=pins, request_sha256=s.sha(request_path), phases=[])
    s.write(report_path, report)
    try:
        for phase in ('prefix', 'gate_raw2'):
            if phase == 'gate_raw2':
                s.own_after_o(request)
            command = [str(s.PYTHON), '-B', '-u', str(HERE / 'worker.py'), '--request', str(request_path), '--phase', phase]
            with (output / (phase + '.log')).open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=s.SNAPSHOT, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                event = dict(phase=phase, pid=process.pid, command=command, started_unix=time.time())
                report['phases'].append(event)
                s.write(report_path, report)
                code = process.wait()
            event.update(exit_code=code, elapsed_seconds=time.time() - event['started_unix'])
            (output / (phase + '.exit')).write_text(str(code) + '\n', encoding='utf-8')
            s.write(report_path, report)
            s.require(code == 0, f'{phase} failed; no retry or next phase. See {phase}.log.')
            actual = s.read(output / (phase + '.json'))
            s.require(actual['status'] == ('prefix_completed' if phase == 'prefix' else 'raw2_completed')
                and actual['pid'] == process.pid and actual['seed_calls'] == [31], 'Missing genuine native phase completion.')
        s.require(report['phases'][0]['pid'] != report['phases'][1]['pid'], 'Require two independent native processes.')
        environment['CUDA_VISIBLE_DEVICES'] = ''
        with (output / 'boundary_verification.log').open('x', encoding='utf-8') as log:
            subprocess.run([str(s.PYTHON), '-B', '-u', str(HERE / 'verify_boundaries.py'), '--output', str(output)],
                cwd=s.SNAPSHOT, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        report.update(status='common_prerequisite_complete', exit_code=0,
            after_o=s.read(output / 'prefix.json')['saved_boundaries'][-1],
            gate_raw2=s.read(output / 'gate_raw2.json')['saved_boundaries'][-1],
            scope='Common prerequisite only; no gate recovery, C1 selection, suffix, storage audit or full9.')
    except BaseException as error:
        report.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        s.write(report_path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--validation-report', type=Path)
    args = parser.parse_args()
    execute(args.output_dir, args.gpu, args.validation_only, args.validation_report)
