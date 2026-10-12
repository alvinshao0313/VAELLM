"""Matched native joint rank2 recovery; reuse the verified model-derived observer."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.llama_decoder_lora_validation.binding import check_inputs, read, require, training_arguments as short_arguments
from experiments.llama_decoder_lora_validation.state import geometry
from experiments.llama_decoder_lora_validation.train import JointObservations as VerifiedObservations
from experiments.llama_decoder_lora_recovery.callback import JointStepCallback
from experiments.longbench_v1.common import sha256, write_json
from experiments.w25_eval.lora_state import validate_source
from train_utils.runtime_snapshot import to_runtime_jsonable


def training_arguments(control, output):
    argv = short_arguments(control, output)
    argv[argv.index('--save_steps') + 1] = '400'
    return argv


def configuration(record):
    control = read(record['control_launch'])
    require(record['kind'] == 'formal_s2000'
            and record['training_argv'] == training_arguments(control, record['output_dir']), 'Wrong matched formal recipe.')
    cfg, hf, args = parse_args(record['training_argv'])
    expected = deepcopy(control['cpu_configuration']['canonical_config'])
    expected.update(run_root_dir=record['output_dir'], train_mode='decoder_lora')
    expected['opt']['distill_fp32_components'] = ['lora', 'decoder']
    expected['lora'] = {'rank': 2, 'alpha': 4.0, 'dropout': 0.0,
                       'rank_explicit': True, 'alpha_explicit': True, 'dropout_explicit': True}
    expected['explicit_cli_flags'] = sorted(expected['explicit_cli_flags'] + ['--lora_rank', '--lora_alpha', '--lora_dropout'])
    require(to_runtime_jsonable(cfg) == expected, 'Parser changed beyond joint mode/rank/output.')
    meta = read(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == record['source_checkpoint_id'] == control['source_checkpoint_id']
            and cfg.student_checkpoint_dir == record['source_checkpoint'] == control['source_checkpoint']
            and validate_source(meta) == record['architecture'] and geometry(meta, record['architecture']) == record['geometry'],
            'Fresh source or native architecture differs.')
    require(args.max_steps == 2000 and args.save_steps == 400 and args.save_total_limit == 1
            and args.get_warmup_steps(2000) == 20 and args.bf16 and not args.fp16
            and args.learning_rate == cfg.opt.decoder_lr == cfg.opt.learning_rate == 3e-6
            and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Original FineWeb schedule, optimizer or full shape differs.')
    return cfg, hf, args


class JointObservations(VerifiedObservations):
    def __init__(self, record):
        super().__init__(record)
        self.short = False


def train(path):
    path = path.resolve(); record = read(path); output = Path(record['output_dir'])
    code, observer, callback = 1, None, None
    started = time.time()
    report = {'status': 'failed', 'source_checkpoint_id': record['source_checkpoint_id'], 'diagnostic_not_quality': False}
    try:
        require(record['status'] == 'bound_ready' and not record['queued'] and not record['automatic_retry']
                and type(record['gpu']) is int and type(record['export_gpu']) is int and record['gpu'] != record['export_gpu']
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']) and torch.cuda.device_count() == 1
                and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve(),
                'Root must bind distinct training/export GPUs and the prepared S/bitvae environment.')
        require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Runtime environment differs from the native control.')
        require(not (output / 'training.json').exists() and not list(output.glob('home_*')) and not list(output.glob('step_*')),
                'Use a fresh output, never short weights or a previous run.')
        check_inputs(record)
        proof = record['readiness']['short_validation']; verified = read(proof['path'])
        require(sha256(proof['path']) == proof['sha256'] and verified['status'] == 'verified'
                and verified['source_checkpoint_id'] == record['source_checkpoint_id']
                and verified['expected_inventory'] == record['expected_inventory'], 'Completed joint proof differs.')
        cfg, hf, args = configuration(record)
        observer = JointObservations(record)
        callback = JointStepCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=cfg.run_root_dir, export_cuda_visible_devices=str(record['export_gpu']))
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=started, actual_environment={k: os.environ.get(k) for k in record['environment']})
        write_json(path, record)
        configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(observer, callback)))
        require(result['global_step'] == 2000 and result['round_base_checkpoint_id'] == record['source_checkpoint_id']
                and [v['step'] for v in observer.report['native_saves']] == callback.completed_steps == [400, 800, 1200, 1600, 2000],
                'Native final source, saves or complete eight-task callbacks differ.')
        check_inputs(record)
        report.update(status='completed', training_result=result)
        record.update(status='completed', training_result=result, completed_steps=callback.completed_steps)
        code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        report.update(observations=observer.report if observer else None,
            callback_completed_steps=callback.completed_steps if callback else [], elapsed_seconds=time.time() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None,
            longbench_status='not_evaluated', current_nine_item_goal_status='not_evaluated')
        write_json(output / 'training.json', report)
        record.update(exit_code=code, finished_unix=time.time(), training_report_sha256=sha256(output / 'training.json'))
        write_json(path, record); (output / 'run.exit').write_text(str(code) + '\n')
        if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch_record', type=Path, required=True)
    train(parser.parse_args().launch_record)
