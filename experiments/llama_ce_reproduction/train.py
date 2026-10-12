"""Run one fresh, model-adaptive reproduction of the original two CE2000 recipes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.reproduction_recipe import training_command
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def training_arguments(recipe, source, hf_model, output, *, diagnostic=False):
    require(recipe in ('ce10', 'ce50') and (not diagnostic or recipe == 'ce10'), 'Only the two exact CE recipes and CE10 diagnostic are supported.')
    command = training_command(recipe, source_dir=source, hf_model=hf_model, output_dir=output, export_gpu='unbound')
    require(command[3:5] == ['--export_cuda_visible_devices', 'unbound'], 'Unexpected original recipe entry arguments.')
    argv = command[5:]
    argv[argv.index('--target_layers') + 1] = 'all'
    if diagnostic:
        argv[argv.index('--save_steps') + 1] = '2'
    return argv


def check_inputs(record):
    check_hashes(record['source_sha256'])
    check_hashes(record['input_evidence_sha256'])
    for item in [record['source_weight_stat'], *record['input_data_stat'], *record['hf_source']['weight_files']]:
        check_stat(item)
    check_hashes(record['hf_source']['metadata_sha256'])


def configuration(record):
    diagnostic = record['kind'] == 'validate_s4'
    require(diagnostic or record['kind'] == 'formal_s2000', 'Unknown native run kind.')
    expected = training_arguments(record['recipe'], record['source_checkpoint'], record['architecture']['base_model_path'],
                                  record['output_dir'], diagnostic=diagnostic)
    require(record['training_argv'] == expected, 'Native arguments differ from the exact Qwen recipe plus model/output/diagnostic adaptation.')
    cfg, hf, args = parse_args(expected)
    meta = read(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    architecture = model_inventory(meta['base_model_path'])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    require(architecture == record['architecture'] and set(names) == set(architecture['projection_shapes'])
            and meta['checkpoint_id'] == record['source_checkpoint_id'], 'Source identity or complete projection selection differs.')
    require(cfg.train_mode == 'decoder' and tuple(cfg.opt.distill_fp32_components) == ('decoder',)
            and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == 'none'
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None, 'Fresh decoder-only training is required.')
    require(args.max_steps == 2000 and args.save_steps == (2 if diagnostic else 400) and args.save_total_limit == 1
            and args.get_warmup_steps(2000) == 20 and args.bf16 and not args.fp16
            and cfg.opt.batch_size == args.per_device_train_batch_size == 4
            and cfg.opt.gradient_accumulation_steps == args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and cfg.runtime.parallel_mode == 'dp'
            and cfg.runtime.teacher_output_offload == cfg.runtime.teacher_model_offload == cfg.runtime.offload_mode == 'none',
            'Keep native2000/warm20/B4/max1024/BF16/DP with no offload.')
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
                and Path.cwd().resolve() == Path(record['cwd']).resolve(), 'Root must bind two disjoint GPUs and bitvae in the repository.')
        require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Actual runtime environment differs from the prepared recipe.')
        require(not list(output.glob('home_*')) and not list(output.glob('step_*')) and not (output / 'training.json').exists(),
                'Use a fresh output; no resume or overwrite is implemented.')
        check_inputs(record)
        cfg, hf, args, names, architecture = configuration(record)
        diagnostic = record['kind'] == 'validate_s4'
        if not diagnostic:
            proof = record['readiness']['short_validation']
            require(proof['sha256'] and sha256(proof['path']) == proof['sha256'], 'Bind the successful new callback/resource diagnostic first.')
            verified = read(proof['path'])
            require(verified['status'] == 'verified' and verified['source_checkpoint_id'] == record['source_checkpoint_id']
                    and verified['expected_inventory'] == record['expected_inventory']
                    and verified['callback_completed_steps'] == [2, 4] and verified['teacher_output_offload'] == 'none',
                    'The short proof differs from the native source/inventory/loading path.')
        from experiments.llama_ce_reproduction.callback import DecoderStepCallback
        callback = DecoderStepCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=cfg.run_root_dir, export_cuda_visible_devices=str(record['export_gpu']))
        callbacks = [callback]
        if diagnostic:
            from experiments.w25_eval.validate_decoder_subset_training import DecoderObservations
            observation = DecoderObservations(names, architecture)
            callbacks.insert(0, observation)
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), actual_environment={k: os.environ.get(k) for k in record['environment']})
        write_json(path, record)
        configure_e2e_determinism(bool(args.full_determinism))
        set_e2e_seed(int(args.seed))
        torch.cuda.reset_peak_memory_stats()
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=tuple(callbacks)))
        expected_steps = [2, 4] if diagnostic else [400, 800, 1200, 1600, 2000]
        require(result['global_step'] == expected_steps[-1] and result['round_base_checkpoint_id'] == record['source_checkpoint_id']
                and callback.completed_steps == expected_steps, 'Native training or synchronous save boundaries are incomplete.')
        report = {'status': 'verified' if diagnostic else 'completed', 'recipe': record['recipe'], 'training_result': result,
            'callback_completed_steps': callback.completed_steps, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'source_checkpoint_id': record['source_checkpoint_id'],
            'expected_inventory': record['expected_inventory'], 'teacher_output_offload': cfg.runtime.teacher_output_offload,
            'longbench_status': 'not_evaluated', 'current_nine_item_goal_status': 'not_evaluated'}
        if diagnostic:
            observed = observation.diagnostic
            require([x['step'] for x in observed['optimizer_steps']] == [1, 2, 3, 4]
                    and [x['step'] for x in observed['native_saves']] == [2, 4]
                    and {x['step'] for x in observed['batches']} == {1, 2, 3, 4}
                    and all(x['shape'][0] == 4 and x['shape'][1] <= 1024 for x in observed['batches'])
                    and all(x['main_moment_dtypes'] == ['torch.float32'] for x in observed['optimizer_steps']),
                    'Real update/save/resumed-forward observations are incomplete.')
            inv = observed['trainable_inventory']
            require(inv['decoder_tensors'] == record['expected_inventory']['selected_decoder_tensors']
                    and inv['decoder_elements'] == record['expected_inventory']['selected_decoder_elements']
                    and inv['score_elements'] == 0, 'Wrong active decoder inventory.')
            report.update(observations=observed, diagnostic_not_quality=True,
                resource_coverage='Four actual native dynamic batches, plus both paused independent exports/audits; no full8 in the diagnostic.')
        check_inputs(record)
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
