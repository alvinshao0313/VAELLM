"""Binding for the single L16 identity-R1 capacity diagnostic; no GPU scheduling."""
from copy import deepcopy
import json
from pathlib import Path

import torch

from compressed_e2e_fintuning.args import parse_args
from experiments.general_lm_recovery.train import training_arguments
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from train_utils.runtime_snapshot import to_runtime_jsonable

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
ROOT = Path(__file__).resolve().parents[2]
SOURCE = R2 / 'llama_gate_k34_transpose/true/final_model'
SOURCE_ID = '98a80541-fd2b-4ce4-b88d-0809f82d4ef3'
CONTROL = R2 / 'general_lm_llama_k34true_s2000/launch.prepared.json'
HOLD = R2 / 'llama_column_feedback_fineweb128'
OUTPUT = R2 / 'llama_l16_gate_identity_r1_s128'
TARGET = 'model.layers.16.mlp.gate_proj'
PREFIX = TARGET + '._parallel_stage_decoder.'
DOMAINS = ('alpaca_observe32', 'fineweb_observe32')
STEPS = 128


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
    for key, value in {'--target_layers': '16', '--target_modules': 'gate_proj', '--save_steps': str(STEPS)}.items():
        argv[argv.index(key) + 1] = value
    return argv


def configuration(record, arm):
    require(arm in ('r0', 'r1'), 'Only the fixed two arms are defined.')
    argv = arguments(record, arm)
    require(argv == record['training_argv'][arm], 'Training argv differs from CPU preparation.')
    cfg, hf, args = parse_args(argv)
    require(cfg.train_mode == 'decoder' and tuple(cfg.opt.distill_fp32_components) == ('decoder',)
            and cfg.aux.residual_lora_mode == cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == 'none'
            and tuple(cfg.target_layers) == (16,) and tuple(cfg.target_modules) == ('gate_proj',)
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Fresh one-gate decoder-only scope; no external branch or native pause path.')
    require(args.max_steps == 2000 and args.get_warmup_steps(2000) == 20 and args.save_steps == STEPS
            and args.lr_scheduler_type == 'cosine' and args.learning_rate == cfg.opt.decoder_lr == 3e-6
            and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and cfg.data.seed == cfg.data.data_seed == 0
            and cfg.loss.loss_type == 'kl_top_mass' and cfg.loss.top_k == 100 and cfg.loss.alpha == 0
            and cfg.loss.temperature == 1 and args.bf16 and not args.fp16,
            'Keep the actual first128 steps of the original2000 cosine/warm20 B4x1024 recipe.')
    return cfg, hf, args


def expanded_specs(meta):
    specs = deepcopy(meta['converted_modules'])
    for spec in specs:
        if spec['name'] == TARGET:
            for field in ('decoders', 'stage_decoders'):
                for stage in spec[field]:
                    require(stage['num_res_blocks'] == 0, 'Only R0 can be expanded.')
                    stage['num_res_blocks'] = 1
    return specs


def state_inventory(state):
    selected = {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'numel': value.numel()}
                for key, value in state.items() if key.startswith(PREFIX)}
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
            and actual['budget_passed'], 'Padding/outside state/rank/logic budget changed.')


def actual_configuration(cfg, hf, args):
    return {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
            'training_args': to_runtime_jsonable(args)}
