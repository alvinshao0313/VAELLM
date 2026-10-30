"""Bind the fixed old-best recovery stages to the actual G09 native source."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'
SNAPSHOT = RESULT / 'rotation_factor_probe/source_snapshot_20261005_01'
REFERENCE = RESULT / 'general_lm_qwen_k38_qkonly_s2000'
PROFILE = os.environ.get('VAELLM_ALIGNED_PROFILE', 'G')
if PROFILE not in ('C', 'G'):
    raise ValueError('Use the fixed C or G alignment profile.')
SOURCE_AUDIT = RESULT / f'qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027/{PROFILE}/09_joint1200/native_audit.json'
SOURCE_ID = {'G': 'e8e5efd4-655e-462c-b6e0-fedf4e2e0346', 'C': '726b7b36-bb95-4418-b626-cc029b4f0aec'}[PROFILE]
GPU = {'G': 0, 'C': 1}[PROFILE]
OUTPUT = Path('/data/star/VAELLM/cat_aligned_recovery_20261028') / PROFILE / '01_qk_fineweb2000'
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT))

import torch
from compressed_e2e_fintuning.args import parse_args
from experiments.llama_ce_reproduction.train import check_inputs
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source():
    audit = read(SOURCE_AUDIT)
    directory = Path(audit['checkpoint_dir']).resolve()
    meta = read(directory / 'checkpoint_meta.json')
    require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id'] == SOURCE_ID
            and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
            and audit['fresh_strict_reload_state_exact'] and audit['bit_budget']['budget_passed'],
            'Use only the actual verified G09 native endpoint.')
    return directory, meta, audit


def training_arguments():
    directory, _, _ = source()
    argv = list(read(REFERENCE / 'launch.prepared.json')['training_argv'])
    for key, value in (('--student_checkpoint_dir', str(directory)), ('--run_root_dir', str(OUTPUT))):
        require(argv.count(key) == 1, 'Ambiguous old training argument: ' + key)
        argv[argv.index(key) + 1] = value
    return argv


def configuration(record):
    directory, meta, _ = source()
    require(record['training_argv'] == training_arguments()
            and record['source_checkpoint'] == str(directory) and record['source_checkpoint_id'] == SOURCE_ID
            and record['output_dir'] == str(OUTPUT), 'Source/output are the only CLI changes from old QK2000.')
    cfg, hf, args = parse_args(record['training_argv'])
    expected = deepcopy(read(REFERENCE / 'launch.prepared.json')['cpu_configuration']['canonical_config'])
    expected.update(student_checkpoint_dir=str(directory), run_root_dir=str(OUTPUT))
    require(to_runtime_jsonable(cfg) == expected, 'Canonical training configuration differs beyond source/output.')
    validate_recipe(meta, record['architecture'])
    names = selected_names(cfg.target_layers, cfg.target_modules, record['architecture'])
    require(list(names) == record['selected_targets'] and len(names) == 72
            and all(d['in_dim'] == 38 for item in meta['converted_modules'] if item['name'] in names
                    for d in item['stage_decoders'])
            and record['rotation_blocks'] == {item['name']: item['weight_rotation']['block_size']
                                             for item in meta['converted_modules']},
            'Keep actual G09 geometry and all72 full-RHT/K38 Q/K projections.')
    require(args.max_steps == 2000 and args.save_steps == 400 and args.get_warmup_steps(2000) == 20
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Use fresh native2000/cosine/warm20, without short-run limits.')
    return cfg, hf, args


def read_prepared():
    record = read(OUTPUT / 'prepared.json')
    require(record['schema'] == 'CG09_aligned_QK_FineWeb2000_v1' and record['status'] == 'CPU_prepared'
            and record['profile'] == PROFILE and record['gpu'] == GPU and record['automatic_retry'] is False,
            'Use this source-bound fixed stage only.')
    check_inputs(record)
    return record
