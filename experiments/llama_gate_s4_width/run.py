"""Run one explicitly bound width arm: native CAT, assembly/audit, full8."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.llama_gate_small_rate_pair.common import ROOT, BASE_ID, check, read, _require
from experiments.llama_gate_small_rate_pair.formal_run import same_protocol
from experiments.llama_gate_s4_width.audit import audit
from experiments.full_rht_gate_pair.common import SHARDS, eval_argv
from experiments.full_rht_gate_pair.compare import score, delta
from experiments.longbench_v1.common import sha256, write_json


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1

    def stage(name, argv):
        log = output / (name + '.log'); _require(not log.exists(), 'Do not overwrite an executed stage.')
        record['status'] = name
        with log.open('w') as handle:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            record['stages'][name] = {'argv': argv, 'pid': process.pid, 'started_unix': time.time()}
            write_json(path, record); value = process.wait()
        (output / (name + '.exit')).write_text(str(value) + '\n')
        record['stages'][name].update(exit_code=value, finished_unix=time.time()); write_json(path, record)
        _require(value == 0, f'{name} exited {value}; inspect {log}')

    try:
        _require(record['schema'] == 'llama_gate_s4_width_v1' and record['status'] == 'bound_ready'
                 and record['decoder_hidden_dim'] in (128, 256) and record['base_checkpoint_id'] == BASE_ID
                 and record['steps_per_stage'] == 10000 and record['total_stage_updates'] == 40000
                 and record['residual_stages'] == 4 and record['codebook_bits'] == 16 and record['transpose'] is False
                 and record['full_evaluation'] and not record['queued'] and not record['automatic_retry']
                 and Path.cwd().resolve() == ROOT and Path(record['output_dir']) == output
                 and type(record['gpu']) is int and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Require one explicitly bound native S4 asymmetric width arm.')
        prepared = read(output / 'launch.prepared.json')
        _require(record.get('preparation_sha256') == sha256(output / 'launch.prepared.json')
                 and all(record.get(k) == v for k, v in prepared.items() if k not in ('status', 'gpu')),
                 'Bound record differs from CPU preparation.')
        check(record); record.update(worker_pid=os.getpid(), worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat')
            .read_text().rsplit(')', 1)[1].split()[19]), started_unix=time.time(), stages={}); write_json(path, record)
        stage('train', record['training_argv'])
        finals = list((output / 'donor').glob('*/final_model')); _require(len(finals) == 1, 'Expected one native donor.')
        native = finals[0].resolve(); record['donor_checkpoint_dir'] = str(native); write_json(path, record)
        stage('assembly', [sys.executable, '-m', 'experiments.w25_eval.assemble_categories', '--base', record['base_dir'],
            '--donor', str(native), '--categories', 'gate_proj', '--output', str(output / 'final_model')])
        proof = audit(record, native); write_json(output / 'audit.json', proof)
        for name, tasks, batch in SHARDS:
            stage(name, eval_argv(output / 'final_model', output / (name + '.json'), tasks=tasks, batch=batch, python=sys.executable))
        current, raw = score([output / (n + '.json') for n, _, _ in SHARDS], record['architecture'], output / 'final_model')
        base, br = score(record['control_result_paths'], record['architecture'], Path(record['base_dir']))
        _require(base['checkpoint_id'] == BASE_ID, 'Wrong contextual mother checkpoint.'); same_protocol(raw, br)
        summary = {'status': 'complete_full8', **current, 'bit_budget': proof['bit_budget'],
            'decoder_hidden_dim': record['decoder_hidden_dim'], 'context_only_vs_e767': delta(current, base),
            'matched_width_comparison': 'Compare both newly completed asymmetric arms; no historical symmetric substitution.',
            'comparison_boundary': record['comparison_boundary'], 'longbench_status': 'not_evaluated',
            'nine_item_goal_status': 'not_evaluated', 'audit_sha256': sha256(output / 'audit.json')}
        write_json(output / 'full8_summary.json', summary); check(record)
        record.update(status='completed', checkpoint_id=current['checkpoint_id'], mean_percent=current['mean_percent'],
                      full8_summary_sha256=sha256(output / 'full8_summary.json')); code = 0
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()); raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch-record', type=Path, required=True)
    run(parser.parse_args().launch_record)
