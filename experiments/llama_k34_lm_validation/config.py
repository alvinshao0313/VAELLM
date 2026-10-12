"""Fixed K34 gate-True endpoint and unchanged native FineWeb four-step prefix."""
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
SOURCE = R2 / 'llama_gate_k34_transpose/true/final_model'
SOURCE_ID = '98a80541-fd2b-4ce4-b88d-0809f82d4ef3'
CONTROL = R2 / 'general_lm_llama_d16_combo1600_s2000/launch.json'
OUTPUT = R2 / 'llama_k34_lm_validate_s4'


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
            and architecture == record['architecture'] and len(names) == 224,
            'Fresh K34 gate-True source or all-decoder architecture differs.')
    gates = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    require(len(gates) == 32 and all(s['transpose'] and s['stage_codebook_dims'] == [32, 32]
            and s['compressed_in_features'] == 4032 and s['compressed_out_features'] == 14336
            and s['weight_rotation']['block_size'] == 32 and s['protected_input_indices']['shape'] == [64]
            and all(d['in_dim'] == 34 for d in s['stage_decoders'])
            and all(v['logical_shape'] == [1806336, 1, 34] and v['shape'] == [1806336, 1, 5]
                    for v in s['stage_vq_weights']) for s in gates), 'Wrong K34/True source geometry.')
    require(args.max_steps == 2000 and args.save_steps == 2 and args.get_warmup_steps(2000) == 20
            and args.save_total_limit == 1 and args.bf16 and not args.fp16
            and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and args.learning_rate == cfg.opt.decoder_lr == 3e-6
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Require the original B4/max1024/2000 schedule and fresh decoder-only prefix.')
    return cfg, hf, args, names, architecture
