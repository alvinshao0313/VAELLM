"""Fresh decoder/all-norm recovery on the unchanged canonical FineWeb LM control."""
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
from experiments.general_lm_norm_recovery.callback import UnionStepCallback
from experiments.general_lm_norm_recovery.validation import UnionObservations
from experiments.general_lm_recovery.train import training_arguments as decoder_arguments
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable

ROOT = Path(__file__).resolve().parents[2]
SAVE_STEPS = [400, 800, 1200, 1600, 2000]


def training_arguments(source, hf_model, output, train_file, *, diagnostic=False):
    argv = decoder_arguments(source, hf_model, output, train_file)
    for key, value in (('--distill_fp32_components', 'decoder,norm'), ('--norm_train_mode', 'all')):
        argv[argv.index(key) + 1] = value
    argv.extend(['--norm_lr', '1e-5'])
    if diagnostic:
        argv[argv.index('--save_steps') + 1] = '2'
        argv[argv.index('--logging_steps') + 1] = '1'
    return argv


def configuration(record):
    diagnostic = record['kind'] == 'validate_s4'
    require(diagnostic or record['kind'] == 'formal_s2000', 'Unknown native run kind.')
    expected = training_arguments(record['source_checkpoint'], record['architecture']['base_model_path'],
                                  record['output_dir'], record['data']['train_file'], diagnostic=diagnostic)
    require(record['training_argv'] == expected, 'Training arguments differ from the fixed general-LM recipe.')
    cfg, hf, args = parse_args(expected)
    meta = read(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    architecture = model_inventory(meta['base_model_path'])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    require(architecture == record['architecture'] and set(names) == set(architecture['projection_shapes'])
            and meta['checkpoint_id'] == record['source_checkpoint_id'], 'Source or full-projection inventory differs.')
    require(cfg.train_mode == 'decoder' and tuple(cfg.opt.distill_fp32_components) == ('decoder', 'norm')
            and cfg.aux.norm_train_mode == 'all' and cfg.aux.norm_lr == 1e-5
            and cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == 'none'
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None, 'Only fresh decoder plus backbone norm parameters may train.')
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
    require(args.max_steps == 2000 and args.save_steps == (2 if diagnostic else 400) and args.save_total_limit == 1
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
    code, observation = 1, None
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
        diagnostic = record['kind'] == 'validate_s4'
        if not diagnostic:
            proof = record['validation_proof']
            require(isinstance(proof, dict) and sha256(proof['path']) == proof['sha256'],
                    'Bind this model’s completed union short validation first.')
            verified = read(proof['path'])
            require(verified['status'] == 'verified' and verified['source_checkpoint_id'] == record['source_checkpoint_id']
                    and verified['expected_inventory'] == record['expected_inventory']
                    and verified['callback_completed_steps'] == [2, 4], 'Union validation source/inventory/callback differs.')
        check_inputs(record)
        cfg, hf, args, _names, _architecture = configuration(record)
        callback = UnionStepCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=cfg.run_root_dir, export_cuda_visible_devices=str(record['export_gpu']))
        callbacks = []
        if diagnostic:
            observation = UnionObservations(_architecture)
            callbacks.append(observation)
        callbacks.append(callback)
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), actual_environment={k: os.environ.get(k) for k in record['environment']})
        write_json(path, record)
        configure_e2e_determinism(bool(args.full_determinism))
        set_e2e_seed(int(args.seed))
        torch.cuda.reset_peak_memory_stats()
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=tuple(callbacks)))
        expected_steps = [2, 4] if diagnostic else SAVE_STEPS
        require(result['global_step'] == expected_steps[-1] and result['round_base_checkpoint_id'] == record['source_checkpoint_id']
                and callback.completed_steps == expected_steps, 'Training or a synchronous full8 save boundary is incomplete.')
        check_inputs(record)
        report = {'status': 'verified' if diagnostic else 'completed', 'training_result': result, 'callback_completed_steps': callback.completed_steps,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'source_checkpoint_id': record['source_checkpoint_id'], 'expected_inventory': record['expected_inventory'],
            'teacher_output_offload': 'none', 'longbench_status': 'not_evaluated',
            'current_nine_item_goal_status': 'not_evaluated'}
        if diagnostic:
            observed = observation.diagnostic
            require([x['step'] for x in observed['optimizer_steps']] == [1, 2, 3, 4]
                    and [x['step'] for x in observed['native_saves']] == [2, 4]
                    and [x['step'] for x in observed['gradients']] == [1, 2, 3, 4]
                    and [x['step'] for x in observed['updates']] == [1, 2, 3, 4]
                    and {x['step'] for x in observed['batches']} == {1, 2, 3, 4}
                    and all(x['shape'][0] == 4 and x['shape'][1] <= 1024 for x in observed['batches'])
                    and all(x['main_moment_dtypes'] == ['torch.float32'] for x in observed['optimizer_steps']),
                    'Actual four updates, resumed forwards or independent save observations are incomplete.')
            inv = observed['trainable_inventory']
            require(inv['component_counts'] == {'decoder': record['expected_inventory']['selected_decoder_tensors'],
                                               'norm': record['expected_inventory']['norm_tensors']}
                    and inv['rope_dtypes'] == ['torch.float32'], 'Actual union inventory or RoPE precision differs.')
            from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
            native = _load_state(Path(result['saved_model_dir']))
            independent = _load_state(output / 'step_0004/final_model')
            compared = _assert_state_equal(native, independent, 'Union native final versus independent step4')
            require(compared == record['expected_inventory']['serialized_tensors'], 'Incomplete final/step4 persistent-state comparison.')
            del native, independent
            report.update(native_vs_step4_tensors_compared=compared, native_vs_step4_state_exact=True,
                observations=observed, diagnostic_not_quality=True,
                resource_coverage='Four real native B4 dynamic batches, export/audit at2 and4; no full8 in this validation.')
        write_json(output / 'training.json', report)
        record.update(status='completed', training_result=result, completed_steps=callback.completed_steps,
            training_report_sha256=sha256(output / 'training.json'),
            longbench_status='not_evaluated', current_nine_item_goal_status='not_evaluated')
        code = 0
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        if observation is not None:
            observation.close()
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
