"""Native FineWeb recovery with only the72 full-rotation Q/K decoders selected."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen_fullqk_lm_recovery.train import completed_proof
from experiments.qwen_fullqk_lm_validation.config import ROOT, R2, SOURCE, SOURCE_ID, check_inputs, read
from experiments.qwen_fullqk_qkonly_lm.callback import QKStepCallback, SAVE_STEPS
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable

CONTROL = R2 / 'general_lm_qwen_fullqk_s2000/launch.verified.prepared.json'
OUTPUT = R2 / 'general_lm_qwen_fullqk_qkonly_s2000'


def training_arguments(output):
    argv = list(read(CONTROL)['training_argv'])
    for option, value in (('--target_modules', 'q_proj,k_proj'), ('--run_root_dir', str(output))):
        require(argv.count(option) == 1, 'Ambiguous matched control option: ' + option)
        argv[argv.index(option) + 1] = value
    return argv


def configuration(record):
    require(record['kind'] == 'formal_s2000' and record['training_argv'] == training_arguments(record['output_dir'])
            and record['source_checkpoint'] == str(SOURCE) and record['source_checkpoint_id'] == SOURCE_ID,
            'Only target_modules/output may differ from the matched fresh43b0 all-decoder recipe.')
    cfg, hf, args = parse_args(record['training_argv'])
    expected = deepcopy(read(CONTROL)['cpu_configuration']['canonical_config'])
    expected.update(target_modules=to_runtime_jsonable(cfg.target_modules), run_root_dir=record['output_dir'])
    require(to_runtime_jsonable(cfg) == expected, 'Unexpected native parser difference from the matched control.')
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Always use the original43b0, never a short-run endpoint.')
    validate_recipe(meta, record['architecture'])
    names = selected_names(cfg.target_layers, cfg.target_modules, record['architecture'])
    require(len(names) == 72 and all(n.endswith(('.q_proj', '.k_proj')) for n in names)
            and list(names) == record['selected_targets'] and args.max_steps == 2000 and args.save_steps == 400
            and args.get_warmup_steps(2000) == 20 and cfg.resume_from_checkpoint is None,
            'Wrong selected modules, native schedule, or fresh boundary.')
    return cfg, hf, args, names


def train(path):
    path = Path(path).resolve()
    record, code = read(path), 1
    output = Path(record['output_dir'])
    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int and type(record['export_gpu']) is int
                and record['gpu'] != record['export_gpu'] and not record['queued'] and not record['automatic_retry']
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']) and Path.cwd() == ROOT
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and path.parent == output and all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Root must bind disjoint training/export GPUs and the unchanged S environment.')
        require(not list(output.glob('home_*')) and not list(output.glob('step_*'))
                and not (output / 'training.json').exists(), 'Fresh output only; no automatic retry.')
        check_inputs(record)
        require(completed_proof() == record['resource_validation'], 'Full-QK backward proof changed.')
        cfg, hf, args, _ = configuration(record)
        callback = QKStepCallback(record=record, source_dir=SOURCE, run_root_dir=output,
            export_cuda_visible_devices=str(record['export_gpu']))
        record.update(status='training', worker_pid=os.getpid(), started_unix=time.time(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        configure_e2e_determinism(bool(args.full_determinism))
        set_e2e_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        result = to_runtime_jsonable(run(cfg, hf, args, extra_callbacks=(callback,)))
        require(result['global_step'] == 2000 and result['round_base_checkpoint_id'] == SOURCE_ID
                and callback.completed_steps == SAVE_STEPS, 'Incomplete native training or periodic full8 saves.')
        check_inputs(record)
        write_json(output / 'training.json', {'status': 'completed', 'training_result': result,
            'callback_completed_steps': callback.completed_steps, 'source_checkpoint_id': SOURCE_ID,
            'expected_inventory': record['expected_inventory'], 'actual_scope_sha256': sha256(output / 'actual_scope.json'),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'longbench_status': 'not_evaluated', 'current_nine_item_goal_status': 'not_evaluated'})
        record.update(status='completed', training_result=result, completed_steps=callback.completed_steps,
            training_report_sha256=sha256(output / 'training.json'))
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
    sys.exit(train(parser.parse_args().launch_record))
