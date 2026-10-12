"""Independent Gen native-chat LM recovery from the unchanged e767 source."""
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
from experiments.general_lm_recovery.train import ROOT, SAVE_STEPS, configuration as control_configuration
from experiments.llama_ce_reproduction.callback import DecoderStepCallback
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from train_utils.runtime_snapshot import to_runtime_jsonable

R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
CONTROL = R2 / 'general_lm_llama_d16_combo1600_s2000/launch.prepared.json'
SOURCE_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
DATASET_MIX = 'edgerazor_ii_gen=1'


def training_arguments(control, output):
    argv = list(control['training_argv'])
    argv[argv.index('--run_root_dir') + 1] = str(output)
    index = argv.index('--train_file')
    argv[index:index + 2] = ['--dataset_mix', DATASET_MIX]
    return argv


def configuration(record):
    require(record['kind'] == 'formal_s2000' and Path(record['control_preparation']) == CONTROL
            and record['source_checkpoint_id'] == SOURCE_ID, 'Require the fixed fresh e767 Gen2000 recipe.')
    control = read(CONTROL)
    old_cfg, old_hf, old_args, names, architecture = control_configuration(control)
    expected_argv = training_arguments(control, record['output_dir'])
    require(record['training_argv'] == expected_argv and record['source_checkpoint'] == control['source_checkpoint']
            and record['architecture'] == architecture and architecture['model_type'] == 'llama',
            'Only the dataset selection and independent output may change from e767 FineWeb.')
    cfg, hf, args = parse_args(expected_argv)
    expected = deepcopy(to_runtime_jsonable(old_cfg))
    expected['run_root_dir'] = record['output_dir']
    expected['data'].update(train_file=None, dataset_mix=DATASET_MIX,
                            dataset_mix_sources=['edgerazor_ii_gen'], dataset_mix_weights=[1.0])
    expected['explicit_cli_flags'] = sorted((set(expected['explicit_cli_flags']) - {'--train_file'}) | {'--dataset_mix'})
    actual_training, expected_training = to_runtime_jsonable(args), to_runtime_jsonable(old_args)
    # HF creates its default logging directory from wall time on each parse.
    # It does not change training, and report_to=none does not use that path.
    actual_training.pop('logging_dir'); expected_training.pop('logging_dir')
    require(to_runtime_jsonable(cfg) == expected and to_runtime_jsonable(hf) == to_runtime_jsonable(old_hf)
            and actual_training == expected_training,
            'A native training setting changed beyond the declared data/output delta.')
    require(cfg.data.dataset_task == 'lm' and not cfg.data.group_by_length and not args.group_by_length
            and args.accelerator_config.use_seedable_sampler and args.world_size == 1,
            'Gen LM must retain the native map dataset and seeded Trainer sampler.')
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
                and output == Path(record['output_dir']).resolve(), 'Bind two distinct GPUs, bitvae, cwd and the fresh output.')
        require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Actual environment differs from the preparation.')
        require(not list(output.glob('home_*')) and not list(output.glob('step_*'))
                and not (output / 'training.json').exists(), 'No overwrite or resume is supported.')
        check_inputs(record)
        cfg, hf, args, _names, _architecture = configuration(record)
        callback = DecoderStepCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=cfg.run_root_dir, export_cuda_visible_devices=str(record['export_gpu']))
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), actual_environment={k: os.environ.get(k) for k in record['environment']})
        write_json(path, record)
        configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(int(args.seed))
        torch.cuda.reset_peak_memory_stats()
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(callback,)))
        require(result['global_step'] == 2000 and result['round_base_checkpoint_id'] == SOURCE_ID
                and callback.completed_steps == SAVE_STEPS, 'Native training or a synchronous full8 save boundary is incomplete.')
        check_inputs(record)
        report = {'status': 'completed', 'training_result': result, 'callback_completed_steps': callback.completed_steps,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'source_checkpoint_id': SOURCE_ID, 'expected_inventory': record['expected_inventory'],
            'data': record['data'], 'teacher_output_offload': 'none', 'longbench_status': 'not_evaluated',
            'current_nine_item_goal_status': 'not_evaluated'}
        write_json(output / 'training.json', report)
        record.update(status='completed', training_result=result, completed_steps=callback.completed_steps,
            training_report_sha256=sha256(output / 'training.json'), longbench_status='not_evaluated',
            current_nine_item_goal_status='not_evaluated')
        code = 0
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        record.update(exit_code=code, finished_unix=time.time())
        write_json(path, record); (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', type=Path, required=True)
    sys.exit(run_record(parser.parse_args().launch_record))
