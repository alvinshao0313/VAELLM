"""Bindings shared by the two fixed raw-C/raw-G abs005 Rice7 candidates."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'
OUTPUT = RESULT / 'qwen_recompression_followup_20261026/sparse'
ARTIFACT = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/sparse')
SNAPSHOT = RESULT / 'rotation_factor_probe/source_snapshot_20261005_01'
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
HF = Path('/home/star/models/Qwen/Qwen3-8B')
RICE = RESULT / 'qwen_sparse005_lossless_index_rice7_20261024/rice7.py'
REPORTS = {name: RESULT / 'qwen_recompression_20261026' / relative for name, relative in
           [('C', 'C_D/C_down.json'), ('G', 'G/G_full.json')]}
RECIPE = dict(rank_metric='sparse_residual_abs', top_p=.005, min_abs=0., codec='blocked_quantized',
              index_bits=8, value_bits=8, block_shape=[256, 256])
FORMAT = 'vaellm_raw_abs005_rice7_bundle_v1'
FIELDS = ('active_block_ids', 'block_ptr', 'qvalues', 'scales', 'zero_points',
          'rice7_header', 'rice7_block_offsets', 'rice7_stream')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT))

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def stamp(path):
    path = Path(path).resolve()
    info = path.stat()
    return dict(path=str(path), bytes=info.st_size, inode=info.st_ino, device=info.st_dev, mtime_ns=info.st_mtime_ns)


def check_prepared(path):
    value = read(path)
    require(value['status'] == 'CPU_prepared_GPU_not_started' and value['recipe'] == RECIPE,
            'Use the fixed CPU source preparation.')
    for record in value['bindings']:
        require(binding(record['path']) == record, 'Bound implementation/input changed: ' + record['path'])
    for record in value['weight_stats']:
        require(stamp(record['path']) == record, 'Bound source weight changed: ' + record['path'])
    return value


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy()).hexdigest())


def exact(actual, expected, label):
    require(actual.keys() == expected.keys(), label + ': persistent keys differ.')
    for name, value in expected.items():
        found = actual[name]
        require(found.dtype == value.dtype and found.shape == value.shape
                and torch.equal(found.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)), label + ': ' + name)
    return len(expected)


def compact_bytes(value):
    require(value.device.type == 'cpu' and value.is_contiguous()
            and value.untyped_storage().nbytes() == value.numel() * value.element_size(),
            'Rice fields must be compact CPU tensors.')
    return value.numel() * value.element_size()


def environment(gpu=''):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(SNAPSHOT), PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key in tuple(env):
        if key in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK',
                   'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
            del env[key]
    return env


def cpu_only():
    require(Path(sys.executable).resolve() == PYTHON.resolve()
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Use caller-activated bitvae with CUDA hidden for CPU work.')
    native_context()


def native_context():
    names = ('rotation.hadamard_utils', 'rotation.hadamard_extended', 'rotation.weight_preprocess',
             'train_utils.checkpoint_v6', 'train_utils.checkpoint_bpw', 'train_utils.v6_model_loader',
             'litebsq.vae_linear', 'litebsq.vae_linear_prewarm', 'litebsq.sparse_residual', 'tools.cat_eval')
    paths = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in names}
    require(all(Path(path).is_relative_to(SNAPSHOT) for path in paths.values()),
            'Every native codec/rotation/loader must come from the mature snapshot.')
    had, factor = importlib.import_module('rotation.hadamard_utils').get_hadK(4032)
    require(factor == 252 and had.shape == (252, 252) and had.device.type == 'cpu',
            'The real full-RHT4032 path requires the existing252 Hadamard family.')
    return paths


def aux32():
    # This existing panel uses dataset[1024:1280]; use its first unchanged batch only.
    manifest = read(RESULT / 'qwen_hot_m4096_s128_20261013/teacher_targets.json')
    record = manifest['dataset_binding']
    require(binding(record['path']) == record, 'The existing auxiliary_train dataset changed.')
    dataset = torch.load(record['path'], map_location='cpu', weights_only=True)
    cases = dataset['training_cases'][1024:1056]
    require(len(cases) == 32 and dataset['train_ids'][1024:1056] == [case['case_id'] for case in cases],
            'Use the original fixed auxiliary32 order.')
    for case in cases:
        info = case['source']
        require(info['repository'] == 'cais/mmlu' and info['config'] == 'auxiliary_train'
                and info['official_split'] == info['source_split'] == 'train'
                and case['answer_query_positions'].tolist() == [len(case['input_ids']) - 1],
                'Only the existing auxiliary_train/train native inputs are permitted.')
    return cases, dict(dataset=record, case_ids=[case['case_id'] for case in cases],
                       input_records=[tensor_record(case['input_ids']) for case in cases],
                       source='cais/mmlu', config='auxiliary_train', split='train',
                       provenance_manifest=str(RESULT / 'qwen_hot_m4096_s128_20261013/teacher_targets.json'))
