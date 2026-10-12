"""Fixed cd338 all-gate S4 decoder capacity comparison; scheduling belongs to root."""
from copy import deepcopy
import json
from pathlib import Path

from compressed_e2e_fintuning.args import parse_args
from experiments.general_lm_recovery.train import training_arguments
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from train_utils.runtime_snapshot import to_runtime_jsonable

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
ROOT = Path(__file__).resolve().parents[2]
SOURCE = R2 / 'general_lm_llama_gate_s4_s2000/step_1600/final_model'
SOURCE_ID = 'cd33814b-d5db-4f7e-b6b8-dbde1a06b964'
CONTROL = SOURCE.parents[1] / 'launch.prepared.json'
OUTPUT = R2 / 'llama_gate_s4_identity_r1'
DESIGN = R2 / 'llama_l16_gate_identity_r1_s128/all_gate_s4_next_design.json'
GATE_NAMES = tuple(f'model.layers.{i}.mlp.gate_proj' for i in range(32))
PREFIXES = tuple(name + '._parallel_stage_decoder.' for name in GATE_NAMES)


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    check_hashes(record['source_sha256'])
    check_hashes(record['input_evidence_sha256'])
    check_hashes(record['hf_source']['metadata_sha256'])
    for item in [record['source_weight_stat'], *record['input_data_stat'], *record['hf_source']['weight_files']]:
        check_stat(item)


def arguments(record, arm):
    source = SOURCE if arm == 'r0' else Path(record['output_dir']) / 'r1_initial'
    argv = training_arguments(source, record['hf_source']['path'],
                              Path(record['output_dir']) / arm, record['data']['train_file'])
    for key, value in {'--target_modules': 'gate_proj', '--save_steps': str(record['schedule']['save_steps']),
                       '--save_total_limit': str(record['schedule']['save_total_limit'])}.items():
        argv[argv.index(key) + 1] = value
    return argv


def configuration(record, arm):
    require(arm in record['schedule']['arms'], 'Arm is outside this prepared run.')
    argv = arguments(record, arm)
    require(argv == record['training_argv'][arm], 'Training argv differs from CPU preparation.')
    cfg, hf, args = parse_args(argv)
    require(cfg.train_mode == 'decoder' and tuple(cfg.opt.distill_fp32_components) == ('decoder',)
            and cfg.aux.residual_lora_mode == cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == 'none'
            and cfg.target_layers == 'all' and tuple(cfg.target_modules) == ('gate_proj',)
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Fresh all-gate decoder-only scope is required; no external branch or resume.')
    require(args.max_steps == 2000 and args.get_warmup_steps(2000) == 20
            and args.save_steps == record['schedule']['save_steps']
            and args.save_total_limit == record['schedule']['save_total_limit']
            and args.lr_scheduler_type == 'cosine' and args.learning_rate == cfg.opt.decoder_lr == 3e-6
            and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and cfg.data.seed == cfg.data.data_seed == 0
            and cfg.loss.loss_type == 'kl_top_mass' and cfg.loss.top_k == 100 and cfg.loss.alpha == 0
            and cfg.loss.temperature == 1 and args.bf16 and not args.fp16,
            'Keep the real first4/400 updates of the existing2000 cosine/warm20 B4x1024 recipe.')
    return cfg, hf, args


def expected_specs(meta, arm):
    require(arm in ('r0', 'r1'), 'Unknown arm.')
    specs = deepcopy(meta['converted_modules'])
    for spec in specs:
        if spec['name'] in GATE_NAMES:
            require(spec['residual_stages'] == 4 and not spec['transpose'] and spec['parallel_parts'] == 1,
                    'Only the fixed four-stage untransposed gate source is supported.')
            for field in ('decoders', 'stage_decoders'):
                for stage in spec[field]:
                    require(stage['num_res_blocks'] == 0, 'Source must have R0 gate decoders.')
                    stage['num_res_blocks'] = int(arm == 'r1')
    return specs


def state_inventory(state):
    selected = {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'numel': value.numel()}
                for key, value in state.items() if key.startswith(PREFIXES)}
    return {'selected': selected, 'selected_elements': sum(x['numel'] for x in selected.values()),
            'state_tensors': len(state), 'frozen_tensors': len(state) - len(selected),
            'banks': sum('vq_weight' in key for key in state)}


def budget_check(actual, record, arm):
    source = record['source_bit_budget']
    extra = 0 if arm == 'r0' else record['expansion']['added_bits']
    for key in ('user_linear_bits', 'packed_linear_bits'):
        require(actual[key] == source[key] + extra, 'Unexpected decoder budget: ' + key)
    require(actual['code_padding_bits'] == source['code_padding_bits']
            and actual['outside_linear_bits'] == source['outside_linear_bits']
            and actual['original_linear_weights'] == source['original_linear_weights']
            and actual['budget_passed'], 'Padding/outside state/rank/logic budget changed.')


def actual_configuration(cfg, hf, args):
    return {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
            'training_args': to_runtime_jsonable(args)}
