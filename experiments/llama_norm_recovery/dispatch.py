"""One explicitly bound GPU: native training exits, then independent norm export/audit."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def now():
    return datetime.now().astimezone().isoformat()


def pins(record):
    for key in ('source_sha256', 'input_evidence_sha256'):
        for path, expected in record[key].items():
            require(sha(Path(record['cwd']) / path) == expected, 'Pinned bytes changed: ' + path)
    for entry in [record['source_weight_stat'], *record['input_data_stat'], *record['hf_source']['weight_files']]:
        stat = Path(entry['path']).stat()
        require((stat.st_size, stat.st_mtime_ns, stat.st_ino) == (entry['size'], entry['mtime_ns'], entry['inode']),
                'Pinned weight/data stat changed: ' + entry['path'])
    for path, expected in record['hf_source']['metadata_sha256'].items():
        require(sha(path) == expected, 'HF metadata changed: ' + path)


def run(path):
    path = Path(path).resolve()
    record, output = read(path), path.parent

    def save():
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)

    def free_gpu():
        apps = subprocess.check_output(['nvidia-smi', '--id=' + str(record['gpu']),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        require(not apps, 'Assigned GPU occupied: ' + apps)

    def stage(name, argv):
        record['status'] = name
        item = {'argv': argv, 'started_at': now()}
        record['stages'][name] = item
        with (output / (name + '.log')).open('x') as log:
            child = subprocess.Popen(argv, cwd=record['cwd'], env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            item.update(pid=child.pid, start_ticks=ticks(child.pid))
            save()
            status = child.wait()
        item.update(exit_code=status, completed_at=now())
        (output / (name + '.exit')).write_text(str(status) + '\n')
        save()
        require(status == 0, f'{name} failed with exit {status}; no retry.')

    code = 1
    try:
        require(record['status'] == 'prepared' and not record['queued']
                and record['readiness']['cpu_configuration'] == 'passed', 'A fresh CPU-prepared run is required.')
        require(type(record['gpu']) is int and 0 <= record['gpu'] < 8
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']), 'Root must bind a single physical GPU.')
        require(Path(sys.executable).resolve() == Path(record['python_executable']).resolve(), 'Use pinned bitvae.')
        for key, value in record['environment'].items():
            require(os.environ.get(key) == value, 'Environment mismatch: ' + key)
        require(not any(key in os.environ for key in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Do not inherit diagnostic precision overrides.')
        pins(record)
        diagnostic = record['kind'] == 'validate_s4'
        require(diagnostic or record['kind'] == 'formal_s400', 'Unknown bounded run kind.')
        if not diagnostic:
            proof = record['readiness']['norm_validation']
            require(proof['sha256'] and sha(proof['path']) == proof['sha256'], 'Bind the successful short validation first.')
            verified = read(proof['path'])
            require(verified['status'] == 'verified' and verified['source_checkpoint_id'] == record['source_checkpoint_id']
                    and verified['expected_inventory'] == record['expected_inventory'], 'Short validation source/inventory differs.')
        step = 4 if diagnostic else 400
        destination = output / f'step_{step:04d}'
        require(not list(output.glob('home_*')) and not destination.exists(), 'Fresh independent output required.')
        free_gpu()
        record.update(worker_pid=os.getpid(), worker_start_ticks=ticks(os.getpid()), started_at=now(), stages={})
        save()
        stage('train', record['argv'])
        free_gpu()
        candidates = list(output.glob(f'*/trainer_state/checkpoint-{step}'))
        require(len(candidates) == 1, 'Exactly one original native saved step is required.')
        checkpoint = candidates[0]
        native = read(checkpoint.parent.parent / 'run_meta.json')
        require(native['global_step'] == step and native['round_base_checkpoint_id'] == record['source_checkpoint_id'],
                'The saved native step/source differs.')
        destination.mkdir()
        final = destination / 'final_model'
        python = record['python_executable']
        stage('export', [python, '-m', 'experiments.w25_eval.norm_export', '--checkpoint_dir', str(checkpoint),
                         '--output_dir', str(final)])
        stage('audit', [python, '-m', 'experiments.w25_eval.norm_audit', '--source_dir', record['source_checkpoint'],
                        '--checkpoint_dir', str(final), '--training_checkpoint_dir', str(checkpoint),
                        '--output', str(destination / 'audit.json')])
        if diagnostic:
            stage('verify', [python, '-m', 'experiments.llama_norm_recovery.validate', '--launch_record', str(path),
                             '--phase', 'verify'])
            require(read(output / 'verification.json')['status'] == 'verified', 'Short validation did not verify.')
            record.update(status='verified', verification=str(output / 'verification.json'),
                          verification_sha256=sha(output / 'verification.json'))
        else:
            for name, tasks, batch in [('mmlu', 'mmlu', '64'),
                ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', '32')]:
                stage(name, [python, 'tools/cat_eval.py', '--checkpoint_dir', str(final), '--strict', '--eval_lm_eval',
                    '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0', '--lm_batch_size', batch,
                    '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8',
                    '--eval_log_dir', str(destination / (name + '_logs')), '--output_json', str(destination / (name + '.json'))])
            from experiments.w25_eval.acceptance import collect_results, validate_checkpoint
            directory, meta, budget = validate_checkpoint(final, record['architecture'])
            raw = collect_results([destination / 'mmlu.json', destination / 'seven.json'],
                                  record['architecture'], checkpoint=(directory, meta))
            audit, expected = read(destination / 'audit.json'), record['expected_inventory']
            require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == record['source_checkpoint_id']
                and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['selected_global_step'] == 400
                and audit['mutable_tensor_count'] == expected['mutable_tensors'] and audit['norm_elements'] == expected['norm_elements']
                and audit['frozen_tensors_compared'] == expected['frozen_tensors']
                and audit['code_bank_count'] == expected['all_code_banks'] and audit['decoder_tensor_count'] == expected['all_decoder_tensors']
                and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == expected['serialized_tensors']
                and audit['changed_code_bits'] == audit['changed_decoder_elements'] == 0 and audit['changed_norm_fp32_elements'] > 0
                and audit['fresh_strict_reload_state_exact'] and audit['norm_matches_selected_fp32_cast_exactly']
                and audit['actual_tensor_storage_unchanged'] and audit['rank_zero'], 'Norm export/freeze contract failed.')
            require(meta['train_mode'] == 'none' and meta['norm_train_mode'] == 'all'
                and meta['lm_head_train_mode'] == 'none' and meta['lora_config'] is None
                and budget['user_linear_bpw'] == record['source_bpw'], 'Topology or storage budget changed.')
            for name in ('mmlu', 'seven'):
                observed = read(destination / (name + '.json'))['evals']['lm_eval']
                for reference in record['protocol_references'][name]:
                    baseline = read(reference['path'])
                    if reference['format'] == 'cat_eval':
                        baseline = baseline['evals']['lm_eval']
                    for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                        other = key.replace('_', '-') if reference['format'] == 'hf' and key in ('n_shot', 'n_samples') else key
                        require(observed[key] == baseline[other], f'Full protocol differs: {name}/{key}/{reference["path"]}')
            summary = {'status': 'complete_eight_only', 'checkpoint_id': meta['checkpoint_id'],
                'checkpoint_dir': str(final), 'source_checkpoint_id': record['source_checkpoint_id'], 'global_step': 400,
                'bit_budget': budget, 'audit_sha256': sha(destination / 'audit.json'), 'source_pins_unchanged': True,
                'longbench': 'not_evaluated', 'nine_item_target': 'not_evaluated', 'quality_target_evaluated': False, **raw}
            (destination / 'eight_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
            record.update(status='completed', mean_percent=raw['mean_percent'], checkpoint_id=meta['checkpoint_id'],
                          final_model=str(final), eight_summary_sha256=sha(destination / 'eight_summary.json'))
        pins(record)
        code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        record.update(exit_code=code, completed_at=now())
        save()
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    sys.exit(run(parser.parse_args().launch_record))


if __name__ == '__main__':
    main()
