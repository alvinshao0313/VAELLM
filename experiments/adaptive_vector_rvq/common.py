"""Fixed fresh-HF down-matrix experiment; no dependencies on active experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HF = Path('/home/star/models/Qwen/Qwen3-8B')
G_REPORT = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_recompression_20261026/G/G_full.json'
OUTPUT = Path('/data/star/VAELLM/adaptive_vector_rvq_20261027')
LAYERS = (0, 18, 35)
DIM, BITS, HIDDEN = 32, 48, 128
SEED, BATCH, CHUNK = 31, 8192, 8192
HEADER_BYTES = 16384
ACTIVATIONS = ROOT / 'result/qwen_cat_biaxial_20261022/prepared/activations.pt'
ACTIVATION_MANIFEST = ACTIVATIONS.with_name('manifest.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stat(path):
    path = Path(path).resolve()
    value = path.stat()
    return dict(path=str(path), bytes=value.st_size, inode=value.st_ino, mtime_ns=value.st_mtime_ns)


def digest(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        value.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return value.hexdigest()


def compact(tensor):
    require(tensor.device.type == 'cpu' and tensor.is_contiguous()
        and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(), 'Persist only compact CPU tensors.')
    return tensor.numel() * tensor.element_size()


def arguments(dtype='bf16'):
    return SimpleNamespace(codebook_dim=DIM, codebook_bits=BITS, base_ch=HIDDEN, num_res_blocks=0,
        norm_type='layer', activation_type='swish', decoder_type='symmetric', parallel_layers=1,
        vae_decoder_checkpoint=True, vae_weight_dtype=dtype, vae_autocast_dtype=dtype,
        quantizer_type='BSQ', new_quant=True, gamma0=1., gamma=1., zeta=1., inv_temperature=100.,
        l1_weight=1., lfq_weight=2.5, commitment_loss_weight=.25, entropy_loss_weight=.01,
        recon_loss_type='mse', optimizer='adamw', beta1=.9, beta2=.95, weight_decay=0., lr=.003)


def check_prepared(directory):
    record = read(Path(directory) / 'prepared.json')
    require(record['status'] == 'completed' and record['schema'] == 'adaptive_vector_rvq_inputs_v1', 'Use the completed real-matrix preparation.')
    for path, digest in record['source_sha256'].items():
        require(sha(path) == digest, 'Pinned implementation or metadata changed: ' + path)
    for item in record['input_stats']:
        require(stat(item['path']) == item, 'Bound source/prepared matrix changed: ' + item['path'])
    return record
