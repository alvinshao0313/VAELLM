"""Shared, fixed recipe for the approved Qwen/Llama full-RHT gate comparison."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.longbench_v1.common import check_hashes, check_stat, sha256
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = Path('/home/star/VAELLM')
RESULT = ORIGINAL / 'result/qwen3_8b_mmlu65_20261004'
Q_TEMPLATE = RESULT / 'best1200_fresh_gate_up_k32_32/launch.json'
L_TEMPLATE = ORIGINAL / 'result/llama31_8b_w25_20261004/exact_ce2000_reproduction/ce10_task50/launch.json'
EXPECTED_IDS = {'qwen3': 'f927ffda-dd5b-4d00-b63a-5e839a182d70',
                'llama': '35776715-2707-45a9-9fb4-2da1efef3f0e'}
EXPECTED_BITS = {'qwen3': 17065672704, 'llama': 17100439552}
SHARDS = (('mmlu', 'mmlu', 64),
          ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', 32))


def read(path):
    return json.loads(Path(path).read_text())


def recipe(model_path, output, *, layers, block_size, steps):
    """Use the reviewed native CAT argv; only the documented pair fields change."""
    _require(block_size in (0, 32) and steps in (2, 10000), 'Use the fixed short or complete recipe.')
    argv = read(Q_TEMPLATE)['training_argv'][2:]
    changes = {'--model_path': str(model_path), '--output_dir': str(output),
               '--compression_categories': 'gate_proj', '--linear_group_size': str(layers),
               '--weight_rotation_block_size': str(block_size), '--vae_steps': f'default={steps}'}
    if steps == 2:
        changes['--log_every'] = '1'
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Unexpected template argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, hf, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    _require(cat.target_layers == 'all' and cat.batch_size == 8192
             and cat.after_category_mode == 'none' and training.bf16 and not training.fp16,
             'Native parser changed the approved full-category recipe.')
    return argv, normalized


def check(record):
    _require(Path(record['cwd']).resolve() == ROOT and ROOT != ORIGINAL,
             'Execute the real snapshot entry, never the active public source tree.')
    check_hashes(record['source_sha256'])
    check_hashes(record['input_sha256'])
    for value in record['protected_stats']:
        check_stat(value)


def eval_argv(checkpoint, output, *, tasks, batch, python):
    return [python, 'tools/cat_eval.py', '--checkpoint_dir', str(checkpoint), '--strict',
            '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks,
            '--num_fewshot', '0', '--lm_batch_size', str(batch), '--tokenizer_use_fast', 'true',
            '--prewarm_group_size', '8', '--eval_log_dir', str(output.parent / (output.stem + '_logs')),
            '--output_json', str(output)]
