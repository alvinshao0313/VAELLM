"""Bound native train process, followed by single-card diagnostic export/audit."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.step_export_callback import run_logged_stage

from experiments.llama_decoder_lora_validation.binding import ROOT, check_inputs, read, require, verify_audit
from experiments.w25_eval.lora_audit import load_state


def verify(record):
    output = Path(record['output_dir'])
    training, audited = read(output / 'training.json'), read(output / 'step_0004/audit.json')
    exported = read(output / 'step_0004/final_model/export_record.json')
    verify_audit(audited, record)
    require(training['status'] == 'trained' and training['training_result']['global_step'] == 4
            and exported['source_global_step'] == 4 and exported['source_max_steps'] == 2000
            and exported['restored_decoder_tensors'] == record['geometry']['decoder_tensors']
            and exported['restored_lora_tensors'] == record['geometry']['lora_tensors']
            and exported['optimizer_steps_executed'] == 0, 'Diagnostic training/export schedule differs.')
    native = load_state(training['training_result']['saved_model_dir'])
    independent = load_state(output / 'step_0004/final_model')
    compared = _assert_state_equal(native, independent, 'Native joint final versus independently exported step4')
    require(compared == record['expected_inventory']['deployed_tensors'], 'Wrong saved joint tensor count.')
    write_json(output / 'verification.json', {'status': 'verified', 'diagnostic_not_quality': True,
        'source_checkpoint_id': record['source_checkpoint_id'], 'checkpoint_id': audited['checkpoint_id'],
        'expected_inventory': record['expected_inventory'], 'native_vs_independent_tensors_compared': compared,
        'bit_budget': audited['bit_budget'], 'gradient_update_steps': [1, 2, 3, 4], 'native_save_steps': [2, 4],
        'training_peak_allocated_bytes': training['peak_allocated_bytes'],
        'training_peak_reserved_bytes': training['peak_reserved_bytes'], 'batches': training['observations']['batches'],
        'quality_claim': 'No full8/LongBench; actual joint backward, update, native finalization and independent export only.',
        'evidence_sha256': {str(output / p): sha256(output / p) for p in
            ('training.json', 'step_0004/audit.json', 'step_0004/final_model/export_record.json')}})


def run(path):
    path = Path(path).resolve(); record, code = read(path), 1
    output = Path(record['output_dir'])
    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                and not record['queued'] and not record['automatic_retry']
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path.cwd() == ROOT == Path(record['cwd']) and path.parent == output
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not (output / 'training.json').exists(), 'Require one explicitly bound fresh execution in S.')
        require(record['kind'] == 'validate_s4' and record['export_gpu'] is None, 'Only one-card four-step validation is implemented.')
        check_inputs(record)
        record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(), stages={},
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        def stage(name, argv, target):
            record['stages'][name] = {'argv': argv}; write_json(path, record)
            record['stages'][name]['exit_code'] = run_logged_stage(argv, name=name, output_dir=target, env=os.environ.copy())
            write_json(path, record)
        stage('training', record['argv'], output)
        checkpoint = Path(read(output / 'training.json')['training_result']['run_output_dir']) / 'trainer_state/checkpoint-4'
        final_root = output / 'step_0004'; final_root.mkdir()
        stage('export', [sys.executable, '-m', 'experiments.llama_decoder_lora_validation.export',
            '--checkpoint_dir', str(checkpoint), '--output_dir', str(final_root / 'final_model')], final_root)
        stage('audit', [sys.executable, '-m', 'experiments.llama_decoder_lora_validation.audit',
            '--source_dir', record['source_checkpoint'], '--checkpoint_dir', str(final_root / 'final_model'),
            '--training_checkpoint_dir', str(checkpoint), '--output', str(final_root / 'audit.json')], final_root)
        verify(record)
        check_inputs(record)
        record.update(status='verified')
        code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(dispatcher_exit_code=code, finished_unix=time.time())
        write_json(path, record); (output / 'dispatcher.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch_record', required=True, type=Path)
    sys.exit(run(parser.parse_args().launch_record))
