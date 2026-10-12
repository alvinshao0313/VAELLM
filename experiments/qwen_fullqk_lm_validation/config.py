"""Fixed full-QK endpoint and unchanged native FineWeb four-step prefix."""
from copy import deepcopy
from pathlib import Path

from compressed_e2e_fintuning.args import parse_args
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
ROOT = R2 / 'rotation_factor_probe/source_snapshot_20261005_01'
SOURCE = R2 / 'qwen_qk_rotation_pair/full/final_model'
SOURCE_ID = '43b0c5a2-4ef7-4a00-a6d1-ed887be18d01'
CONTROL = R2 / 'general_lm_recovery_s2000/qwen3/launch.json'
OUTPUT = R2 / 'qwen_fullqk_lm_validate_s4'


def training_arguments(output):
    argv = list(read(CONTROL)['training_argv'])
    for option, value in (('--student_checkpoint_dir', str(SOURCE)),
                          ('--run_root_dir', str(output)), ('--save_steps', '2')):
        require(argv.count(option) == 1, 'Ambiguous original option: ' + option)
        argv[argv.index(option) + 1] = value
    return argv


def configuration(record):
    argv = training_arguments(record['output_dir'])
    require(record['kind'] == 'validate_s4' and record['training_argv'] == argv,
            'Only source/output/save2 may differ from the original FineWeb recipe.')
    cfg, hf, args = parse_args(argv)
    canonical = to_runtime_jsonable(cfg)
    reference = deepcopy(read(CONTROL)['cpu_configuration']['canonical_config'])
    reference['student_checkpoint_dir'] = str(SOURCE)
    reference['run_root_dir'] = record['output_dir']
    # save_steps is a native TrainingArguments passthrough, recorded in remaining_argv.
    rest = reference['remaining_argv']
    require(rest.count('--save_steps') == 1, 'Original canonical save interval is ambiguous.')
    rest[rest.index('--save_steps') + 1] = '2'
    require(canonical == reference, 'Actual canonical parser differs beyond the three declared substitutions.')
    meta = read(SOURCE / 'checkpoint_meta.json')
    architecture = model_inventory(meta['base_model_path'])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    require(meta['checkpoint_id'] == record['source_checkpoint_id'] == SOURCE_ID
            and cfg.student_checkpoint_dir == record['source_checkpoint'] == str(SOURCE)
            and architecture == record['architecture'] and len(names) == 252,
            'Fresh full-QK source or all-decoder architecture differs.')
    for spec in meta['converted_modules']:
        qk = spec['name'].endswith(('.q_proj', '.k_proj'))
        require(spec['weight_rotation']['block_size'] == (0 if qk else 32),
                'Source rotation layout differs: ' + spec['name'])
        if qk:
            require(spec['compressed_in_features'] == 4032 and spec['parallel_parts'] == 1
                    and spec['residual_stages'] == 2 and spec['parallel_stage_decode'],
                    'QK full-rotation geometry differs.')
    require(args.max_steps == 2000 and args.save_steps == 2 and args.get_warmup_steps(2000) == 20
            and args.save_total_limit == 1 and args.bf16 and not args.fp16
            and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and args.learning_rate == cfg.opt.decoder_lr == 3e-6
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Require the original B4/max1024/2000 schedule and fresh decoder-only prefix.')
    return cfg, hf, args, names, architecture
