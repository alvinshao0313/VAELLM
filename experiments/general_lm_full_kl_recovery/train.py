"""Fresh full-KL FineWeb recovery with the existing every400 full-eight callback."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.general_lm_full_kl_probe.train import configuration, training_arguments
from experiments.general_lm_recovery.train import SAVE_STEPS
from experiments.llama_ce_reproduction.callback import DecoderStepCallback
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from train_utils.runtime_snapshot import to_runtime_jsonable


def check_short_proof(record):
    proof = record['full_kl_short_proof']
    verification, review, training = (read(proof[key]) for key in ('verification', 'review', 'training'))
    require(verification['status'] == review['status'] == 'verified' and training['status'] == 'trained'
            and verification['source_checkpoint_id'] == review['source_checkpoint_id'] == record['source_checkpoint_id']
            and training['training_result']['global_step'] == review['steps'] == 4
            and verification['native_vs_step4_tensors_compared'] == record['expected_inventory']['serialized_tensors']
            and verification['same_gpu_sequential_training_export']
            and verification['training_report_sha256'] == sha256(proof['training'])
            and review['evidence_sha256'][proof['verification']] == sha256(proof['verification']),
            'Reuse the completed real full-KL resource/export proof from the same fresh source.')
    for key in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components'):
        require(verification['bit_budget'][key] == record['source_bit_budget'][key], 'Short proof budget differs: ' + key)


def run_record(path):
    path = Path(path).resolve()
    record, output = read(path), path.parent
    code = 1
    try:
        require(record['kind'] == 'formal_s2000' and record['status'] == 'bound_ready'
                and not record['queued'] and not record['automatic_retry']
                and type(record['gpu']) is int and type(record['export_gpu']) is int
                and record['gpu'] != record['export_gpu']
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and Path.cwd().resolve() == Path(record['cwd']).resolve()
                and output == Path(record['output_dir']).resolve(), 'Explicitly bind two GPUs, bitvae and the fresh output.')
        require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Use the native prepared training environment.')
        require(not list(output.glob('home_*')) and not list(output.glob('step_*'))
                and not (output / 'training.json').exists(), 'No resume, overwrite, or short-state continuation.')
        check_inputs(record); check_short_proof(record)
        cfg, hf, args, names, architecture = configuration(record)
        require(cfg.resume_from_checkpoint is None and cfg.stop_after_step is None
                and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == 'none'
                and tuple(cfg.opt.distill_fp32_components) == ('decoder',)
                and len(names) == architecture['projection_count'] == 252, 'Only fresh f927 decoders may train.')
        callback = DecoderStepCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=cfg.run_root_dir, export_cuda_visible_devices=str(record['export_gpu']))
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), actual_environment={k: os.environ.get(k) for k in record['environment']})
        write_json(path, record)
        configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(int(args.seed))
        torch.cuda.reset_peak_memory_stats()
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(callback,)))
        require(result['global_step'] == 2000 and result['round_base_checkpoint_id'] == record['source_checkpoint_id']
                and callback.completed_steps == SAVE_STEPS, 'Native training or a complete save/evaluation boundary is missing.')
        check_inputs(record)
        report = {'status': 'completed', 'training_result': result, 'callback_completed_steps': callback.completed_steps,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'source_checkpoint_id': record['source_checkpoint_id'], 'expected_inventory': record['expected_inventory'],
            'teacher_output_offload': 'none', 'loss_type': 'kl', 'longbench_status': 'not_evaluated',
            'current_nine_item_goal_status': 'not_evaluated'}
        write_json(output / 'training.json', report)
        record.update(status='completed', training_result=result, completed_steps=callback.completed_steps,
            training_report_sha256=sha256(output / 'training.json'),
            longbench_status='not_evaluated', current_nine_item_goal_status='not_evaluated')
        code = 0
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', type=Path, required=True)
    sys.exit(run_record(parser.parse_args().launch_record))
