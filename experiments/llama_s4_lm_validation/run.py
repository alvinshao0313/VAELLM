"""Single-GPU native train-exit/export/audit sequence using existing stage runners."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.llama_s4_lm_validation.config import ROOT, check_inputs, read


def verify(record):
    output = Path(record['output_dir'])
    training, audit = read(output / 'training.json'), read(output / 'step_0004/audit.json')
    exported = read(output / 'step_0004/final_model/export_record.json')
    require(training['status'] == 'trained' and training['training_result']['global_step'] == 4
            and audit['status'] == 'verified' and audit['source_checkpoint_id'] == record['source_checkpoint_id']
            and audit['train_mode'] == 'decoder' and audit['selected_global_step'] == 4
            and audit['selected_modules'] == record['selected_targets']
            and audit['frozen_tensors_compared'] == 1699
            and audit['frozen_code_bank_count'] == audit['code_bank_count'] == 512
            and audit['selected_decoder_tensor_count'] == 1344 and audit['changed_code_bits'] == 0
            and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3043
            and audit['selected_decoder_fp32_changed_elements'] > 0
            and audit['selected_decoder_matches_fp32_cast_exactly'] and audit['frozen_bytes_unchanged']
            and audit['selected_codes_match_frozen_source_exactly'] and audit['selected_exact_sidecar_sha256'] is None
            and audit['packed_padding_unchanged'] and audit['fresh_strict_reload_state_exact']
            and audit['rank_zero'], 'Native saved-state audit is incomplete.')
    require(exported['source_global_step'] == 4 and exported['source_max_steps'] == 2000
            and exported['restored_decoder_tensors'] == 1344 and exported['optimizer_steps_executed'] == 0
            and exported['sparse_bit_restore'] is None, 'Independent export changed the schedule or mutable scope.')
    for key in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components', 'code_padding_bits'):
        require(audit['bit_budget'][key] == record['source_bit_budget'][key], 'Persistent budget changed: ' + key)
    native = _load_state(Path(training['training_result']['saved_model_dir']))
    independent = _load_state(output / 'step_0004/final_model')
    compared = _assert_state_equal(native, independent, 'Native S4 final versus independent step4')
    require(compared == 3043, 'Native versus independent deployment inventory differs.')
    del native, independent
    report = {'status': 'verified', 'diagnostic_not_quality': True,
        'source_checkpoint_id': record['source_checkpoint_id'], 'checkpoint_id': audit['checkpoint_id'],
        'native_vs_independent_tensors_compared': compared, 'bit_budget': audit['bit_budget'],
        'gate_projection_count': 32, 'gate_stage_count': 128, 'gradient_and_update_steps': [1, 2, 3, 4],
        'step1_zero_lr_update_allowed': True, 'expected_inventory': record['expected_inventory'],
        'training_peak_allocated_bytes': training['peak_allocated_bytes'],
        'training_peak_reserved_bytes': training['peak_reserved_bytes'],
        'export_peak_allocated_bytes': exported['peak_cuda_allocated_bytes'],
        'export_peak_reserved_bytes': exported['peak_cuda_reserved_bytes'],
        'batches': training['observations']['batches'],
        'evidence_sha256': {str(output / p): sha256(output / p) for p in
            ('training.json', 'step_0004/audit.json', 'step_0004/final_model/export_record.json')},
        'quality_claim': 'No downstream or LongBench evaluation; four-step compatibility/resource evidence only.'}
    write_json(output / 'verification.json', report)


def run(path):
    path = Path(path).resolve()
    record, code = read(path), 1
    output = Path(record['output_dir'])
    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int and not record['queued']
                and not record['automatic_retry'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path.cwd() == ROOT == Path(record['cwd'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Root must bind the unchanged environment and one GPU in S.')
        require(path.parent == output and not (output / 'training.json').exists()
                and not (output / 'step_0004').exists(), 'Fresh output required; no automatic retry.')
        check_inputs(record)
        record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(), stages={},
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        def stage(name, argv):
            record['stages'][name] = {'argv': argv, 'started_unix': time.time()}
            write_json(path, record)
            status = run_logged_stage(argv, name=name, output_dir=output, env=os.environ.copy())
            record['stages'][name].update(exit_code=status, finished_unix=time.time(), log_sha256=sha256(output / f'{name}.log'))
            write_json(path, record)
        stage('training', record['argv'])
        step = Path(read(output / 'training.json')['training_result']['run_output_dir']) / 'trainer_state/checkpoint-4'
        final = output / 'step_0004/final_model'
        final.parent.mkdir()
        stage('export', [sys.executable, '-u', '-m', 'experiments.llama_s4_lm_validation.export',
            '--checkpoint_dir', str(step), '--output_dir', str(final), '--device', 'cuda:0'])
        stage('audit', [sys.executable, '-u', '-m', 'experiments.llama_s4_lm_validation.audit', '--source_dir', record['source_checkpoint'],
            '--checkpoint_dir', str(final), '--training_checkpoint_dir', str(step), '--output', str(final.parent / 'audit.json')])
        verify(record)
        check_inputs(record)
        record.update(status='verified', verification_sha256=sha256(output / 'verification.json'))
        code = 0
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(dispatcher_exit_code=code, finished_unix=time.time())
        write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    sys.exit(run(parser.parse_args().launch_record))
