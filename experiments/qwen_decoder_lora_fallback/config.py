"""One matched fallback: f927/FineWeb/all decoder + ordinary all-projection rank2."""
from copy import deepcopy
from pathlib import Path

from compressed_e2e_fintuning.args import parse_args
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.lora_state import validate_source
from experiments.w25_eval.subset_state import validate_recipe, selected_names
from train_utils.runtime_snapshot import to_runtime_jsonable

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
ROOT = R2 / 'rotation_factor_probe/source_snapshot_20261005_01'
CONTROL = R2 / 'general_lm_recovery_s2000/qwen3/launch.json'
SHORT = R2 / 'qwen_decoder_lora_rank2_validate_s4'
FORMAL = R2 / 'general_lm_qwen_decoder_lora_rank2_s2000'


def training_arguments(output, diagnostic):
    argv = list(read(CONTROL)['training_argv'])
    changes = [('--run_root_dir', str(output)), ('--train_mode', 'decoder_lora'),
               ('--distill_fp32_components', 'decoder,lora')]
    if diagnostic: changes.append(('--save_steps', '2'))
    for key, value in changes:
        require(argv.count(key) == 1, 'Ambiguous original argument: ' + key)
        argv[argv.index(key) + 1] = value
    require(not any(k in argv for k in ('--lora_rank', '--lora_alpha', '--lora_dropout')), 'Control unexpectedly specifies LoRA.')
    return argv + ['--lora_rank', '2', '--lora_alpha', '4', '--lora_dropout', '0']


def configuration(record):
    diagnostic = record['kind'] == 'validate_s4'
    require(diagnostic or record['kind'] == 'formal_s2000', 'Unknown scope.')
    argv = training_arguments(record['output_dir'], diagnostic)
    require(argv == record['training_argv'], 'Prepared argv differs.')
    cfg, hf, args = parse_args(argv)
    expected = deepcopy(read(CONTROL)['cpu_configuration']['canonical_config'])
    expected.update(run_root_dir=record['output_dir'], train_mode='decoder_lora')
    expected['opt']['distill_fp32_components'] = ['lora', 'decoder']
    expected['lora'] = {'rank': 2, 'alpha': 4.0, 'dropout': 0.0,
                        'rank_explicit': True, 'alpha_explicit': True, 'dropout_explicit': True}
    expected['explicit_cli_flags'] = sorted(expected['explicit_cli_flags'] + ['--lora_rank', '--lora_alpha', '--lora_dropout'])
    if diagnostic:
        rest = expected['remaining_argv']; rest[rest.index('--save_steps') + 1] = '2'
    require(to_runtime_jsonable(cfg) == expected, 'Native parser changed beyond output/joint-mode/rank and diagnostic save2.')
    meta = read(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    architecture = validate_source(meta); validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    require(meta['checkpoint_id'] == record['source_checkpoint_id'] == 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
            and cfg.student_checkpoint_dir == record['source_checkpoint'] and architecture == record['architecture']
            and len(names) == 252 and set(names) == set(architecture['projection_shapes']), 'Fresh source/scope differs.')
    require(args.max_steps == 2000 and args.save_steps == (2 if diagnostic else 400) and args.save_total_limit == 1
            and args.get_warmup_steps(2000) == 20 and args.bf16 and not args.fp16
            and args.learning_rate == cfg.opt.decoder_lr == cfg.opt.learning_rate == 3e-6
            and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Keep original schedule, learning rates, source, batch and length.')
    return cfg, hf, args, names, architecture


def verify_audit(report, record, step):
    require(report['status'] == 'verified' and report['train_mode'] == 'decoder_lora'
            and report['source_checkpoint_id'] == record['source_checkpoint_id'] and report['selected_global_step'] == step
            and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 3927
            and report['selected_decoder_tensor_count'] == 1512 and report['frozen_tensors_compared'] == 1911
            and report['frozen_code_bank_count'] == report['code_bank_count'] == 504 and report['changed_code_bits'] == 0
            and report['lora_rank'] == 2 and report['lora_tensor_count'] == 504
            and report['selected_decoder_matches_fp32_cast_exactly'] and report['selected_lora_matches_native_bf16_cast_exactly']
            and report['fresh_strict_reload_state_exact'] and report['frozen_bytes_unchanged']
            and report['selected_codes_match_frozen_source_exactly'], 'Incomplete union/frozen native audit.')
    require(report['bit_budget']['budget_passed'] and report['bit_budget']['user_linear_bits'] == 17152966656
            and report['bit_budget']['packed_linear_bits'] == 17152966656, 'Rank2 measured budget differs.')
