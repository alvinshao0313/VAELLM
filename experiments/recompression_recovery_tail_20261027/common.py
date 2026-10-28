"""Fixed same-source C/G recovery tail and its mature native runtime bindings."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOPIC = ROOT / 'result/qwen3_8b_mmlu65_20261004'
PREVIOUS = TOPIC / 'qwen_recompression_followup_20261026'
OUTPUT = PREVIOUS / 'qwen_recompression_recovery_tail_20261027'
DATA = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027')
SNAPSHOT = TOPIC / 'rotation_factor_probe/source_snapshot_20261005_01'
HF = Path('/home/star/models/Qwen/Qwen3-8B')
ARCHIVE = ROOT / 'result/qwen3_8b_w25_20260930/format_recovery_data'
SOURCES = {profile: Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/repaired_recovery') / profile / 'assembly/final_model'
           for profile in ('C', 'G')}
SOURCE_IDS = dict(C='d3d0a300-64d6-4e6f-8dd6-30ac4df67828', G='3e1e256f-aef0-4728-8762-6c520bb3c4f1')
GPUS = {'C': {'decoder': 1, 'chatml': 6}, 'G': {'decoder': 3, 'chatml': 7}}
TARGET = 'model.layers.0.mlp.down_proj'
PREPARED = OUTPUT / 'prepared.json'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SNAPSHOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stat(path):
    path = Path(path).resolve()
    info = path.stat()
    return dict(path=str(path), size=info.st_size, mtime_ns=info.st_mtime_ns, inode=info.st_ino)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def environment(gpu='', *, format_recovery=False):
    from experiments.qwen3_w25.reproduce_from_bf16 import environment as original
    env = original(str(gpu), format_recovery=format_recovery)
    env.update(PYTHONPATH=str(SNAPSHOT) + os.pathsep + str(ROOT), PYTHONDONTWRITEBYTECODE='1', MKL_NUM_THREADS='4')
    for key in tuple(env):
        if key in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK',
                   'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
            del env[key]
    return env


def check_gpu(gpu):
    apps = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    require(not apps.strip(), f'Assigned GPU{gpu} is occupied: {apps.strip()}')


def native_context():
    names = ('rotation.hadamard_utils', 'rotation.hadamard_extended', 'rotation.weight_preprocess',
             'litebsq.vae_linear', 'train_utils.checkpoint_v6', 'train_utils.v6_model_loader',
             'compressed_e2e_fintuning.runtime_v6', 'compressed_e2e_fintuning.runtime_v6_pipeline',
             'sparse_bit_tuning.manager', 'e2e_common.data')
    context = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in names}
    require(all(Path(path).is_relative_to(SNAPSHOT) for path in context.values()), 'Use the actual mature snapshot runtime throughout.')
    had, factor = importlib.import_module('rotation.hadamard_utils').get_hadK(4032)
    require(factor == 252 and tuple(had.shape) == (252, 252) and had.device.type == 'cpu', 'Actual fullRHT4032/H252 support is required.')
    return context


def load_cpu(directory):
    helper = module(TOPIC / 'qwen_c1_dual_down_pv32_20261019/pv_cpu_loader.py', '_recompression_tail_cpu_native')
    return helper.load_cpu_native(directory)


def checkpoint(directory):
    from experiments.qwen3_w25.summarize_results import _validate_checkpoint
    from train_utils.checkpoint_bpw import audit_checkpoint
    meta, _old_budget = _validate_checkpoint(Path(directory))
    budget = audit_checkpoint(directory, max_bpw=2.6, max_lora_rank=0)
    require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys'], 'Complete deployed native state must pass rank0/logical2.6.')
    return meta, budget


def training_argv(profile, recipe, source=None):
    from experiments.qwen3_w25.reproduction_recipe import training_command
    command = training_command(recipe, source_dir=SOURCES[profile] if source is None else source,
        hf_model=HF, output_dir=DATA / profile / ('06_decoder_kl' if recipe == 'decoder' else '09_joint1200'),
        export_gpu=GPUS[profile]['decoder'])
    args = command[3:]
    index = args.index('--export_cuda_visible_devices')
    del args[index:index + 2]
    return args


def check_prepared():
    record = read(PREPARED)
    require(record['status'] == 'cpu_verified' and record['schema'] == 'fixed_CG_same_source_recovery_tail_v1', 'Use the verified fixed two-source preparation.')
    for path, digest in record['source_sha256'].items():
        require(sha(path) == digest, 'Pinned implementation/input changed: ' + path)
    for item in record['input_stats']:
        require(stat(item['path']) == item, 'Bound source/data input changed: ' + item['path'])
    require(native_context() == record['native_context'], 'Imported runtime context changed.')
    return record
