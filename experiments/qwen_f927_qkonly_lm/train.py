"""Fresh f927 Q/K-only FineWeb recovery; reuse the completed full-QK save callback."""
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
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback
from experiments.qwen_fullqk_lm_validation.config import ROOT, R2, check_inputs, read
from experiments.qwen_fullqk_qkonly_lm.callback import QKStepCallback, SAVE_STEPS
from experiments.w25_eval.subset_state import persistent_inventory, selected_names, validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable

CONTROL = R2 / 'general_lm_qwen_fullqk_qkonly_s2000'
SOURCE = Path('/home/star/VAELLM/result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model')
SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
OUTPUT = R2 / 'general_lm_qwen_f927_qkonly_s2000'


def training_arguments(output):
    argv = list(read(CONTROL / 'launch.prepared.json')['training_argv'])
    for option, value in (('--student_checkpoint_dir', str(SOURCE)), ('--run_root_dir', str(output))):
        require(argv.count(option) == 1, 'Ambiguous matched option: ' + option)
        argv[argv.index(option) + 1] = value
    return argv


def configuration(record):
    require(record['kind'] == 'formal_s2000' and record['training_argv'] == training_arguments(record['output_dir'])
            and record['source_checkpoint'] == str(SOURCE) and record['source_checkpoint_id'] == SOURCE_ID,
            'Only fresh source/output may differ from the completed43b0 QK-only recipe.')
    cfg, hf, args = parse_args(record['training_argv'])
    expected = deepcopy(read(CONTROL / 'launch.prepared.json')['cpu_configuration']['canonical_config'])
    expected.update(student_checkpoint_dir=str(SOURCE), run_root_dir=record['output_dir'])
    require(to_runtime_jsonable(cfg) == expected, 'Unexpected native parser difference.')
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID and all(s['weight_rotation']['block_size'] == 32
            for s in meta['converted_modules']), 'Require the original f927 block32 source.')
    validate_recipe(meta, record['architecture'])
    names = selected_names(cfg.target_layers, cfg.target_modules, record['architecture'])
    require(list(names) == record['selected_targets'] and len(names) == 72
            and args.max_steps == 2000 and args.save_steps == 400 and args.get_warmup_steps(2000) == 20
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None, 'Wrong fresh scope or schedule.')
    return cfg, hf, args


class F927QKCallback(QKStepCallback):
    # Only the startup rotation assertion differs; on_save/export/audit/full8 remain inherited.
    def on_train_begin(self, args, state, control, **kwargs):
        AuxStepExportCallback.on_train_begin(self, args, state, control, **kwargs)
        require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == 400
                and args.get_warmup_steps(2000) == 20, 'Keep fresh2000/warm20/save400.')
        model, selected = kwargs['model'], set(self.record['selected_targets'])
        inventory, banks = persistent_inventory(model, self.record['architecture'])
        expected = {key for name in selected for key in inventory[name]}
        active = {key: p for key, p in model.named_parameters() if p.requires_grad}
        require(len(selected) == 72 and len(expected) == len(active) == 432 and set(active) == expected
                and all(p.dtype == torch.float32 for p in active.values())
                and sum(p.numel() for p in active.values()) == 1239552 and not hasattr(model, 'sparse_bit_tuning'),
                'Only432 Q/K decoder leaves may train.')
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        require(not hasattr(optimizer, 'bit_optimizer') and {id(p) for g in optimizer.param_groups for p in g['params']}
                == {id(p) for p in active.values()}, 'Optimizer scope differs.')
        serialized = model.state_dict()
        require(len(serialized) == 3423 and len(set(serialized) - expected) == 2991
                and sum(len(v) for v in banks.values()) == 504, 'Persistent inventory differs.')
        modules = dict(model.named_modules())
        for name in inventory:
            module, chosen = modules[name], name in selected
            require(module.trainable_decode == chosen and module.cache_decoded_weight == (not chosen)
                    and module.weight_rotation.block_size == 32, 'Decode/cache/block32 differs: ' + name)
        write_json(self.run_root_dir / 'actual_scope.json', {'status': 'verified', 'selected_targets': sorted(selected),
            'decoder_keys': sorted(active), 'decoder_tensors': 432, 'decoder_elements': 1239552,
            'frozen_tensors': 2991, 'all_code_banks': 504, 'serialized_tensors': 3423,
            'selected_trainable_decode_no_cache': 72, 'frozen_cached_projections': 180,
            'optimizer_exact': True, 'persistent_values_checked_later_by_native_step_audit': True})
        return control


def train(path):
    path = Path(path).resolve()
    record, code = read(path), 1
    output = Path(record['output_dir'])
    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int and type(record['export_gpu']) is int
                and record['gpu'] != record['export_gpu'] and not record['queued'] and not record['automatic_retry']
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']) and Path.cwd() == ROOT
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve() and path.parent == output
                and all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Bind disjoint training/export GPUs with the unchanged S environment.')
        require(not list(output.glob('home_*')) and not list(output.glob('step_*'))
                and not (output / 'training.json').exists(), 'Fresh output only; no automatic retry.')
        check_inputs(record)
        cfg, hf, args = configuration(record)
        callback = F927QKCallback(record=record, source_dir=SOURCE, run_root_dir=output,
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
