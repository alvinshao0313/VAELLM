"""One bound long-S2 native donor, source union, and complete matched full8."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.llama_gate_small_rate_pair.common import ROOT, BASE_ID, check, read, _require
from experiments.llama_gate_small_rate_pair.formal_run import same_protocol
from experiments.llama_gate_s2_long.audit import audit, CONTROL_ID, S4_ID
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
        _require(record['schema'] == 'llama_gate_s2_long_v1' and record['status'] == 'bound_ready'
                 and record['steps_per_stage'] == 20000 and record['total_stage_updates'] == 40000
                 and record['residual_stages'] == 2 and record['codebook_bits'] == 32 and record['transpose'] is False
                 and record['full_evaluation'] and not record['queued'] and not record['automatic_retry']
                 and Path.cwd().resolve() == ROOT and Path(record['output_dir']) == output
                 and type(record['gpu']) is int and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Require one explicitly bound complete long-S2 candidate.')
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
        cd = Path(record['historical_control_dir']); sd = Path(record['equal_update_s4_dir'])
        control, cr = score([cd / (n + '.json') for n, _, _ in SHARDS], record['architecture'], cd / 'final_model')
        s4, sr = score([sd / (n + '.json') for n, _, _ in SHARDS], record['architecture'], sd / 'final_model')
        base, br = score(record['control_result_paths'], record['architecture'], Path(record['base_dir']))
        _require(control['checkpoint_id'] == CONTROL_ID and s4['checkpoint_id'] == S4_ID
                 and base['checkpoint_id'] == BASE_ID, 'Wrong matched or secondary source.')
        same_protocol(raw, cr); same_protocol(raw, sr); same_protocol(raw, br)
        summary = {'status': 'complete_full8', **current, 'bit_budget': proof['bit_budget'],
            'primary_vs_s2_10k': delta(current, control), 'equal_updates_vs_s4_10k': delta(current, s4),
            'secondary_vs_e767': delta(current, base), 'historical_control_checkpoint_id': CONTROL_ID,
            'equal_update_s4_checkpoint_id': S4_ID, 'comparison_boundary': record['comparison_boundary'],
            'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated', 'audit_sha256': sha256(output / 'audit.json')}
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
