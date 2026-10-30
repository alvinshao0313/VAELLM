"""Verify and independently load the deployed models in a research handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest(package):
    package = Path(package).resolve()
    value = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    if value.get('schema') != 'vaellm_research_handoff_v1':
        raise ValueError('Unsupported handoff manifest.')
    return package, value


def verify(package):
    package, record = manifest(package)
    total = 0
    for entry in record['files']:
        path = package / entry['path']
        if not path.resolve().is_relative_to(package) or path.is_symlink():
            raise ValueError('File must remain inside the extracted package: ' + entry['path'])
        if path.stat().st_size != entry['bytes'] or file_sha256(path) != entry['sha256']:
            raise ValueError('File content differs: ' + entry['path'])
        total += entry['bytes']
    return {'status': 'verified', 'files': len(record['files']), 'bytes': total}


def load_model(package, name):
    """Construct from the included config and restore all native persistent state."""
    package, record = manifest(package)
    entry = record['models'][name]
    checkpoint = package / entry['path']
    sys.path.insert(0, str(package / record['runtime']['path']))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import no_init_weights
    from rotation.model_utils import temporarily_disable_torch_weight_init
    from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    # Native strict loading below replaces the entire initialized state.
    # CPU construction needs no original BF16 HF weights or CUDA allocation.
    with no_init_weights(), temporarily_disable_torch_weight_init():
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.bfloat16, attn_implementation='eager')
    model.seqlen = 2048
    model.config._attn_implementation = entry['attention_implementation']
    model, meta, loaded = load_v6_full_checkpoint_into_model(
        model, str(checkpoint), expected_kind='final_model', map_location='cpu', strict=True)
    if meta['checkpoint_id'] != entry['checkpoint_id'] or loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('Wrong model identity or incomplete native checkpoint.')
    if len(model.state_dict()) != entry['persistent_tensors']:
        raise ValueError('Persistent tensor inventory differs from the selected deployment.')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model.eval().requires_grad_(False)
    return model, tokenizer, meta


def check_load(package, name):
    package, record = manifest(package)
    model, tokenizer, meta = load_model(package, name)
    import torch
    from train_utils.checkpoint_bpw import audit_state_dict

    entry = record['models'][name]
    checkpoint = package / entry['path']
    saved = torch.load(checkpoint / meta['state_dict_file'], map_location='cpu',
                       weights_only=True, mmap=True)
    actual = model.state_dict()
    if set(actual) != set(saved):
        raise ValueError('Loaded and saved state keys differ.')
    for key, value in saved.items():
        if actual[key].dtype != value.dtype or not torch.equal(actual[key], value):
            raise ValueError('Native loading changed a persistent tensor: ' + key)
    budget = audit_state_dict(meta, actual, max_bpw=2.6, max_lora_rank=0)
    if not budget['budget_passed'] or budget['contract_violations']:
        raise ValueError('Deployment no longer meets its recorded storage contract.')
    for key in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights'):
        if budget[key] != entry['bit_budget'][key]:
            raise ValueError('Storage ledger differs: ' + key)
    if torch.cuda.is_initialized():
        raise ValueError('CPU loading unexpectedly initialized CUDA.')
    return {'status': 'verified', 'model': name, 'checkpoint_id': meta['checkpoint_id'],
            'persistent_tensors_exact': len(saved), 'tokenizer_size': len(tokenizer),
            'logical_bpw': budget['user_linear_bits'] / budget['original_linear_weights'],
            'packed_bpw': budget['packed_linear_bits'] / budget['original_linear_weights'],
            'attention_implementation': model.config._attn_implementation,
            'cuda_initialized': False, 'forward_or_training_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    load = sub.add_parser('check-load')
    load.add_argument('--model', choices=('qwen', 'llama'), required=True)
    args = parser.parse_args()
    result = verify(args.package) if args.command == 'verify' else check_load(args.package, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
