"""Fresh f927 FineWeb decoder/full-head training through the existing native runtime."""
from __future__ import annotations

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
from experiments.general_lm_head_recovery.callback import UnionStepCallback
from experiments.general_lm_head_recovery.train import HeadObservations, training_arguments
from experiments.general_lm_recovery.train import ROOT, SAVE_STEPS
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable

SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'


def configuration(record):
    require(record['kind'] in ('validate_s4', 'formal_s2000')
            and record['training_argv'] == training_arguments(record), 'Unexpected Qwen head recipe.')
    cfg, hf, args = parse_args(record['training_argv'])
    expected = deepcopy(read(record['control_normalized'])['canonical_config'])
    expected['run_root_dir'] = record['output_dir']
    expected['opt']['distill_fp32_components'] = ['decoder', 'lm_head']
    expected['aux'].update(lm_head_train_mode='full', lm_head_lr=3e-6)
    expected['explicit_cli_flags'] = sorted([*expected['explicit_cli_flags'], '--lm_head_lr'])
    require(to_runtime_jsonable(cfg) == expected,
            'Only output and declared full-head settings may differ from the actual Qwen FineWeb control.')
    meta = read(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    architecture = model_inventory(meta['base_model_path']); validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    require(architecture == record['architecture'] and architecture['model_type'] == 'qwen3'
            and architecture['num_hidden_layers'] == 36 and len(names) == 252
            and set(names) == set(architecture['projection_shapes'])
            and meta['checkpoint_id'] == record['source_checkpoint_id'] == SOURCE_ID
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Require fresh complete f927, never a short-validation endpoint.')
    require(cfg.train_mode == 'decoder' and tuple(cfg.opt.distill_fp32_components) == ('decoder', 'lm_head')
            and cfg.aux.norm_train_mode == cfg.aux.residual_lora_mode == 'none'
            and cfg.aux.lm_head_train_mode == 'full' and cfg.aux.lm_head_lr == cfg.opt.decoder_lr == 3e-6
            and cfg.loss.loss_type == 'kl_top_mass' and cfg.loss.alpha == 0 and cfg.loss.top_k == 100
            and cfg.loss.temperature == 1 and cfg.loss.prompt_loss_weight == 0,
            'Require all native decoders/head, pure top100 mass KL, frozen norms/codes and no LoRA.')
    require(args.max_steps == 2000 and args.save_steps == 400 and args.get_warmup_steps(2000) == 20
            and args.bf16 and not args.fp16 and args.per_device_train_batch_size == 4
            and args.gradient_accumulation_steps == 1 and cfg.data.model_max_length == 1024
            and cfg.data.dataset_task == 'lm' and cfg.data.seed == cfg.data.data_seed == 0
            and cfg.runtime.teacher_output_offload == cfg.runtime.teacher_model_offload == 'none',
            'Keep the native2000/warm20/B4/max1024/FineWeb/no-offload control.')
    return cfg, hf, args, names, architecture


def train_short(record):
    require(record['kind'] == 'validate_s4' and record['status'] == 'training'
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']),
            'The four-update child requires its explicitly bound same-GPU dispatcher.')
    check_inputs(record)
    cfg, hf, args, names, architecture = configuration(record)
    callback = HeadObservations(names, architecture)
    configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(int(args.seed))
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    report = {'status': 'failed', 'diagnostic_not_quality': True}
    try:
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(callback,)))
        observed, expected = callback.diagnostic, record['expected_inventory']
        require(result['global_step'] == 4 and result['round_base_checkpoint_id'] == SOURCE_ID,
                'Require four actual updates from the original f927 source.')
        for key in ('optimizer_steps', 'gradients', 'parameter_updates', 'head_updates', 'losses'):
            require([row['step'] for row in observed[key]] == [1, 2, 3, 4], 'Incomplete observations: ' + key)
        require([row['step'] for row in observed['native_saves']] == [4]
                and {row['step'] for row in observed['batches']} == {1, 2, 3, 4}
                and all(row['shape'][0] == 4 and row['shape'][1] <= 1024 for row in observed['batches'])
                and all(row['main_moment_dtypes'] == ['torch.float32'] for row in observed['optimizer_steps'])
                and sum(row['fp32_delta_l2'] for row in observed['parameter_updates']) > 0,
                'Native batch, optimizer update or save boundary differs.')
        actual = observed['trainable_inventory']
        require(actual['decoder_tensors'] == expected['selected_decoder_tensors']
                and actual['decoder_elements'] == expected['selected_decoder_elements']
                and actual['head_shape'] == expected['head_shape'] and actual['head_elements'] == expected['head_elements']
                and actual['frozen_norm_tensors'] == expected['norm_tensors'],
                'Actual trainable inventory differs from the source-derived Qwen inventory.')
        check_inputs(record)
        report.update(status='trained', training_result=result, observations=observed,
            resource_coverage='Four actual native dynamic B4<=1024 FineWeb batches; not a full-data worst-case bound.',
            full8_and_longbench='not_evaluated')
    finally:
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.time() - started)
        write_json(Path(record['output_dir']) / 'training.json', report)
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


def run_formal(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1
    try:
        require(record['kind'] == 'formal_s2000' and record['status'] == 'bound_ready'
                and type(record['gpu']) is int and type(record['export_gpu']) is int
                and record['gpu'] != record['export_gpu'] and not record['queued'] and not record['automatic_retry']
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and Path.cwd().resolve() == Path(record['cwd']).resolve()
                and output == Path(record['output_dir']).resolve(),
                'Explicitly bind fresh Qwen training and independent export GPUs.')
        require(all(os.environ.get(key) == value for key, value in record['environment'].items())
                and not any(key in os.environ for key in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Native training environment differs.')
        require(not list(output.glob('home_*')) and not list(output.glob('step_*'))
                and not (output / 'training.json').exists(), 'No resume or overwrite.')
        check_inputs(record); proof = read(record['validation_proof'])
        require(proof['status'] == 'verified' and proof['source_checkpoint_id'] == SOURCE_ID
                and proof['expected_inventory'] == record['expected_inventory']
                and proof['source_sha256'] == record['source_sha256'], 'Require the same-source head validation proof.')
        cfg, hf, args, _, _ = configuration(record)
        callback = UnionStepCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=cfg.run_root_dir, export_cuda_visible_devices=str(record['export_gpu']))
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time()); write_json(path, record)
        configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(int(args.seed))
        torch.cuda.reset_peak_memory_stats()
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(callback,)))
        require(result['global_step'] == 2000 and result['round_base_checkpoint_id'] == SOURCE_ID
                and callback.completed_steps == SAVE_STEPS, 'Incomplete native trajectory or full8 callback.')
        check_inputs(record)
        report = {'status': 'completed', 'training_result': result, 'callback_completed_steps': callback.completed_steps,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'source_checkpoint_id': SOURCE_ID, 'expected_inventory': record['expected_inventory'],
            'teacher_output_offload': 'none', 'longbench_status': 'not_evaluated',
            'current_nine_item_goal_status': 'not_evaluated'}
        write_json(output / 'training.json', report)
        record.update(status='completed', training_result=result, completed_steps=callback.completed_steps,
                      training_report_sha256=sha256(output / 'training.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    parser.add_argument('--short-training-child', action='store_true')
    args = parser.parse_args()
    if args.short_training_child: train_short(read(args.launch_record))
    else: sys.exit(run_formal(args.launch_record))
