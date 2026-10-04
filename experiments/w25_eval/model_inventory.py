"""Derive the two supported architectures from real local HF configuration."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, Qwen3ForCausalLM

from train_utils.config.targets import discover_cat_projection_name_inventory

CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def model_inventory(base_model_path):
    base = Path(base_model_path).resolve()
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    classes = {'qwen3': Qwen3ForCausalLM, 'llama': LlamaForCausalLM}
    if config.model_type not in classes:
        raise ValueError(f'Unsupported model_type: {config.model_type}')
    config._attn_implementation = 'sdpa'  # Shape discovery only, never a forward.
    with torch.device('meta'):
        model = classes[config.model_type](config)
    found = discover_cat_projection_name_inventory(model, compression_categories=CATEGORIES)
    expected = {(i, category) for i in range(config.num_hidden_layers) for category in CATEGORIES}
    if set(found) != expected:
        raise ValueError('Actual model does not expose every expected CAT projection.')
    modules = dict(model.named_modules())
    shapes = {name: list(modules[name].weight.shape) for name in found.values()}
    if not all(t.device.type == 'meta' for t in model.state_dict().values()):
        raise ValueError('Shape discovery unexpectedly allocated real model state.')
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('The evaluation protocol requires the actual fast tokenizer.')
    return {
        'base_model_path': str(base), 'model_type': config.model_type,
        'model_class': type(model).__name__, 'num_hidden_layers': config.num_hidden_layers,
        'projection_shapes': shapes, 'projection_count': len(shapes),
        'original_linear_weights': sum(modules[name].weight.numel() for name in shapes),
        'dense_state_tensor_count': len(model.state_dict()),
        'dense_parameter_elements': sum(p.numel() for p in model.parameters()),
        'config_sha256': sha256(base / 'config.json'),
        'tokenizer_class': type(tokenizer).__name__, 'tokenizer_is_fast': True,
        'tokenizer_bos_token_id': tokenizer.bos_token_id,
        'tokenizer_eos_token_id': tokenizer.eos_token_id,
        'tokenizer_pad_token_id': tokenizer.pad_token_id,
        'weights_loaded_for_inventory': False,
    }


def verify_source_manifest(path, inventory):
    """Verify the GET downloader's completed official size/Git/LFS evidence.

    The same descriptor fields can describe an already downloaded Qwen snapshot.
    Every local file is rehashed; this is an identity check, not a model load.
    """
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    base = Path(inventory['base_model_path'])
    repo = {'qwen3': 'Qwen/Qwen3-8B', 'llama': 'meta-llama/Llama-3.1-8B-Instruct'}[inventory['model_type']]
    if (report.get('status') != 'completed_verified' or report.get('repo_id') != repo
            or Path(report.get('local_dir', '')).resolve() != base
            or not re.fullmatch(r'[0-9a-f]{40}', report.get('revision', ''))):
        raise ValueError('Source manifest must identify the completed fixed official snapshot.')
    files = report.get('files', {})
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    required.update(name for name in ('special_tokens_map.json', 'added_tokens.json', 'vocab.json',
                                      'merges.txt', 'tokenizer.model') if (base / name).exists())
    if not required <= files.keys():
        raise ValueError('Source manifest omits required model/tokenizer metadata.')
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    if not shards or not shards <= files.keys():
        raise ValueError('Source manifest omits model index weight shards.')
    verified = {}
    for name, entry in files.items():
        if Path(name).name != name or entry.get('status') != 'verified':
            raise ValueError(f'Invalid/unverified source file entry: {name}')
        actual = base / name
        size = actual.stat().st_size
        if size != entry.get('expected_size') or size != entry.get('received_bytes'):
            raise ValueError(f'Source size changed: {name}')
        sha = sha256(actual)
        if sha != entry.get('sha256'):
            raise ValueError(f'Source SHA256 changed: {name}')
        lfs = entry.get('official_lfs_sha256')
        if name in shards and not lfs:
            raise ValueError(f'Weight shard lacks official LFS SHA256: {name}')
        if lfs:
            if sha != lfs:
                raise ValueError(f'Official LFS identity mismatch: {name}')
        else:
            git = hashlib.sha1(f'blob {size}\0'.encode() + actual.read_bytes()).hexdigest()
            if git != entry.get('official_git_blob_sha1'):
                raise ValueError(f'Official Git blob identity mismatch: {name}')
        verified[name] = {'size': size, 'sha256': sha}
    return {'manifest': str(path), 'manifest_sha256': sha256(path), 'repo_id': repo,
            'revision': report['revision'], 'files': verified,
            'verification': 'Actual local bytes match completed official Git/LFS pins.'}
