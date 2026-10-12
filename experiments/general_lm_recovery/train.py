"""Fresh all-decoder recovery on canonical plain-text LM documents."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.llama_ce_reproduction.callback import DecoderStepCallback
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable

ROOT = Path(__file__).resolve().parents[2]
SAVE_STEPS = [400, 800, 1200, 1600, 2000]


def training_arguments(source, hf_model, output, train_file):
    return [
        '--student_checkpoint_dir', str(source), '--teacher_model_path', str(hf_model),
        '--run_root_dir', str(output), '--train_mode', 'decoder',
        '--distill_fp32_components', 'decoder', '--target_layers', 'all', '--target_modules', 'all',
        '--seed', '0', '--data_seed', '0', '--train_file', str(train_file),
        '--dataset_task', 'lm', '--text_field', 'text', '--dynamic_padding', 'true',
        '--model_max_length', '1024', '--group_by_length', 'false',
        '--loss_type', 'kl_top_mass', '--alpha', '0', '--top_k', '100', '--temperature', '1',
        '--prompt_loss_weight', '0', '--hidden_loss_weight', '0', '--pre_mlp_hidden_loss_weight', '0',
        '--selective_student_topk', 'false', '--steps', '2000', '--batch_size', '4',
        '--gradient_accumulation_steps', '1', '--decoder_lr', '3e-6', '--learning_rate', '3e-6',
        '--lr_scheduler_type', 'cosine', '--warmup_ratio', '0.01', '--weight_decay', '0',
        '--max_grad_norm', '1.5', '--logging_steps', '10', '--gradient_checkpointing', 'true',
        '--gradient_checkpointing_kwargs', '{"use_reentrant": false}',
        '--residual_lora_mode', 'none', '--norm_train_mode', 'none', '--lm_head_train_mode', 'none',
        '--parallel_mode', 'dp', '--teacher_output_offload', 'none', '--teacher_model_offload', 'none',
        '--vae_decoder_checkpoint', 'true', '--offload_mode', 'none', '--eval_after_save', 'false',
        '--eval_hif4_act', 'false', '--eval_tasks', '', '--skip_ppl_eval', 'true',
        '--save_tokenizer', 'true', '--full_determinism', 'false', '--bf16', 'true',
        '--eval_strategy', 'no', '--save_strategy', 'steps', '--save_steps', '400', '--save_total_limit', '1',
        '--report_to', 'none',
    ]


def configuration(record):
    require(record['kind'] == 'formal_s2000', 'This entry only runs the full native 2000-step schedule.')
    expected = training_arguments(record['source_checkpoint'], record['architecture']['base_model_path'],
                                  record['output_dir'], record['data']['train_file'])
    require(record['training_argv'] == expected, 'Training arguments differ from the fixed general-LM recipe.')
    cfg, hf, args = parse_args(expected)
    meta = read(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    architecture = model_inventory(meta['base_model_path'])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    require(architecture == record['architecture'] and set(names) == set(architecture['projection_shapes'])
            and meta['checkpoint_id'] == record['source_checkpoint_id'], 'Source or full-projection inventory differs.')
    require(cfg.train_mode == 'decoder' and tuple(cfg.opt.distill_fp32_components) == ('decoder',)
            and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == 'none'
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None, 'Only fresh decoder parameters may train.')
    require(cfg.data.dataset_mix is None and cfg.data.dataset_task == 'lm' and cfg.data.text_field == 'text'
            and cfg.data.train_file == record['data']['train_file'] and cfg.data.dynamic_padding
            and cfg.data.model_max_length == 1024 and not cfg.data.group_by_length and not args.group_by_length
            and cfg.data.seed == cfg.data.data_seed == args.seed == args.data_seed == 0
            and args.accelerator_config.use_seedable_sampler and args.world_size == 1,
            'Require unchanged raw-text LM encoding and the single-process seeded native sampler.')
    require(cfg.loss.loss_type == 'kl_top_mass' and cfg.loss.alpha == 0 and cfg.loss.top_k == 100
            and cfg.loss.temperature == 1 and cfg.loss.prompt_loss_weight == 0
            and cfg.loss.hidden_loss_weight == cfg.loss.pre_mlp_hidden_loss_weight == 0
            and not cfg.loss.selective_student_topk, 'Require pure teacher top-mass KL over all valid LM labels.')
    require(args.max_steps == 2000 and args.save_steps == 400 and args.save_total_limit == 1
            and args.get_warmup_steps(2000) == 20 and args.lr_scheduler_type == 'cosine'
            and args.bf16 and not args.fp16 and cfg.opt.decoder_lr == cfg.opt.learning_rate == args.learning_rate == 3e-6
            and cfg.opt.batch_size == args.per_device_train_batch_size == 4
            and cfg.opt.gradient_accumulation_steps == args.gradient_accumulation_steps == 1
            and cfg.runtime.parallel_mode == 'dp'
            and cfg.runtime.teacher_output_offload == cfg.runtime.teacher_model_offload == cfg.runtime.offload_mode == 'none',
            'Require native2000/warm20/LR3e-6/B4/GAS1/BF16/DP without offload.')
    return cfg, hf, args, names, architecture


def run_record(path):
    path = Path(path).resolve()
    record, output = read(path), path.parent
    code = 1
    try:
        require(record['status'] == 'bound_ready' and not record['queued'] and not record['automatic_retry']
                and type(record['gpu']) is int and type(record['export_gpu']) is int and record['gpu'] != record['export_gpu']
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and Path.cwd().resolve() == Path(record['cwd']).resolve()
                and output == Path(record['output_dir']).resolve(), 'Bind two distinct GPUs, bitvae, cwd and this fresh output first.')
        require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Actual environment differs from the CPU-prepared runtime.')
        require(not list(output.glob('home_*')) and not list(output.glob('step_*')) and not (output / 'training.json').exists(),
                'No overwrite or resume is supported by this experiment.')
        check_inputs(record)
        cfg, hf, args, _names, _architecture = configuration(record)
        callback = DecoderStepCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=cfg.run_root_dir, export_cuda_visible_devices=str(record['export_gpu']))
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), actual_environment={k: os.environ.get(k) for k in record['environment']})
        write_json(path, record)
        configure_e2e_determinism(bool(args.full_determinism))
        set_e2e_seed(int(args.seed))
        torch.cuda.reset_peak_memory_stats()
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(callback,)))
        require(result['global_step'] == 2000 and result['round_base_checkpoint_id'] == record['source_checkpoint_id']
                and callback.completed_steps == SAVE_STEPS, 'Training or a synchronous full8 save boundary is incomplete.')
        check_inputs(record)
        report = {'status': 'completed', 'training_result': result, 'callback_completed_steps': callback.completed_steps,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'source_checkpoint_id': record['source_checkpoint_id'], 'expected_inventory': record['expected_inventory'],
            'teacher_output_offload': 'none', 'longbench_status': 'not_evaluated',
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
        record.update(exit_code=code, finished_unix=time.time())
        write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    sys.exit(run_record(parser.parse_args().launch_record))
