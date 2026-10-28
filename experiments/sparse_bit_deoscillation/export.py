"""Export an actual eight-tensor BSQ endpoint with its frozen native source."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'
SNAPSHOT = RESULT / 'rotation_factor_probe/source_snapshot_20261005_01'
SOURCE = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_b32_sparse_mlp005_probe_20261024/native_export/final_model')
SOURCE_ID = 'e01b09b7-a85d-4534-a5d2-5ab3b91312e6'
TARGET = 'model.layers.18.mlp.down_proj'
CPU_LOADER = RESULT / 'qwen_c1_dual_down_pv32_20261019/pv_cpu_loader.py'
META_KEY = 'sparse_bit_deoscillation'
sys.path.insert(0, str(SNAPSHOT))

import torch
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from train_utils.checkpoint_bpw import audit_checkpoint, audit_state_dict
from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load, save_v6_full_checkpoint
from transformers import AutoTokenizer


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_keys(state):
    keys = {key for key in state if key.startswith(TARGET + '._parallel_stage_decoder.')}
    require(len(keys) == 6, 'The fixed target must have exactly six decoder tensors.')
    return keys | {TARGET + '.vq_weight', TARGET + '.vq_weight_s1'}


@torch.no_grad()
def export(endpoint, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Use a CPU-only writer in the caller-activated bitvae environment.')
    endpoint, output = endpoint.resolve(), output.resolve()
    audit_path = endpoint / 'native_audit.json'
    require(not output.exists() and not audit_path.exists(), 'Do not overwrite an exported endpoint.')
    payload_path, report_path = endpoint / 'payload.pt', endpoint / 'endpoint.json'
    proof = read(report_path)
    payload_binding, report_binding = binding(payload_path), binding(report_path)
    package = torch.load(payload_path, map_location='cpu', weights_only=True, mmap=True)
    require(package['format'] == 'sparse_bit_deoscillation_endpoint_v1'
            and package['source_checkpoint_id'] == SOURCE_ID
            and Path(package['source_dir']).resolve() == SOURCE and package['target'] == TARGET,
            'Only an actual endpoint of the declared current best source is supported.')
    require(proof['payload'] == payload_binding and proof['global_step'] == package['global_step']
            and proof['arm'] == package['arm'] and proof['source_checkpoint_id'] == SOURCE_ID,
            'Endpoint report and serialized payload identities must agree.')
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Source UUID changed.')
    source_meta_binding = binding(SOURCE / 'checkpoint_meta.json')
    source_path = SOURCE / meta['state_dict_file']
    source_stat = (source_path.stat().st_ino, source_path.stat().st_size, source_path.stat().st_mtime_ns)
    original = torch.load(source_path, map_location='cpu', weights_only=True, mmap=True)
    replacement = package['tensors']
    require(len(original) == 4071 and set(replacement) == expected_keys(original),
            'Replace only the six decoder tensors and two packed banks; retain the other 4063 fields.')
    for key, value in replacement.items():
        require(value.shape == original[key].shape and value.dtype == original[key].dtype
                and value.is_contiguous() and value.device.type == 'cpu'
                and bool(torch.isfinite(value).all()), 'Invalid native replacement tensor: ' + key)
    records = {key: tensor_record(value) for key, value in replacement.items()}
    require(records == proof['tensor_records'], 'Saved endpoint tensor records differ.')
    expected = {**original, **replacement}
    source_budget = audit_state_dict(meta, original, max_bpw=2.6, max_lora_rank=0)
    expected_budget = audit_state_dict(meta, expected, max_bpw=2.6, max_lora_rank=0)
    require(source_budget['budget_passed'] and expected_budget['budget_passed']
            and all(expected_budget[key] == source_budget[key]
                    for key in ('user_linear_bits', 'packed_linear_bits')), 'Endpoint altered storage geometry.')
    provenance = {'source_checkpoint_id': SOURCE_ID, 'source_dir': str(SOURCE),
                  'arm': package['arm'], 'global_step': package['global_step'], 'target': TARGET,
                  'endpoint': report_binding, 'payload': payload_binding,
                  'replaced_tensor_keys': sorted(replacement), 'frozen_tensors': 4063,
                  'inference_only': True, 'exact_resume_claimed': False,
                  'source_metadata': source_meta_binding, 'export_source': binding(__file__)}
    audit = {'status': 'invalid', 'checkpoint_dir': str(output), 'sources': provenance,
             'source_checkpoint_id': SOURCE_ID, 'persistent_tensors': 4071,
             'updated_tensors': 8, 'frozen_tensors': 4063, 'frozen_sparse_buffers': 648}
    try:
        loader = module_from(CPU_LOADER, '_deoscillation_cpu_loader')
        model, actual_meta, loaded = loader.load_cpu_native(SOURCE)
        require(actual_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys,
                'Source strict native loading failed.')
        live = model.state_dict()
        for key, value in replacement.items():
            live[key].copy_(value)
        model.eval().requires_grad_(False)
        refresh_vae_linear_runtime_after_state_load(model)
        _assert_state_equal(model.state_dict(), expected, 'Eight replacements and frozen source')
        tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
        saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
            compressed_targets=meta['compressed_targets'], pending_dense_targets=(), skip_targets=(),
            train_mode='decoder', after_category_mode='none', norm_train_mode='none', lm_head_train_mode='none',
            lora_config=None, base_model_path=meta['base_model_path'], tokenizer=tokenizer,
            extra_meta={META_KEY: provenance})
        del live, model
        gc.collect()
        saved_meta = read(output / 'checkpoint_meta.json')
        serialized = torch.load(output / saved_meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
        audit['serialized_tensors_compared'] = _assert_state_equal(serialized, expected, 'Serialized endpoint')
        del serialized
        fresh, fresh_meta, loaded = loader.load_cpu_native(output)
        audit['fresh_strict_tensors_compared'] = _assert_state_equal(fresh.state_dict(), expected, 'Fresh strict endpoint')
        require(not loaded.missing_keys and not loaded.unexpected_keys and fresh_meta == saved_meta
                and saved['checkpoint_id'] == saved_meta['checkpoint_id'] != SOURCE_ID
                and saved_meta['converted_modules'] == meta['converted_modules']
                and saved_meta['lora_config'] is None and saved_meta['extra_meta'][META_KEY] == provenance,
                'Export changed topology, provenance or UUID semantics.')
        del fresh
        gc.collect()
        budget = audit_checkpoint(output, max_bpw=2.6, max_lora_rank=0)
        require(budget['budget_passed'] and all(budget[key] == source_budget[key]
                    for key in ('user_linear_bits', 'packed_linear_bits')), 'Actual saved bit ledger changed.')
        require(binding(SOURCE / 'checkpoint_meta.json') == source_meta_binding
                and (source_path.stat().st_ino, source_path.stat().st_size, source_path.stat().st_mtime_ns) == source_stat
                and not torch.cuda.is_initialized(), 'Writer modified its source or initialized CUDA.')
        audit.update(status='verified', checkpoint_id=saved['checkpoint_id'], bit_budget=budget,
                     native_weight_record=proof['native_weight_record'], strict_reload_state_exact=True)
    except BaseException as error:
        audit['error'] = repr(error)
        raise
    finally:
        write(audit_path, audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.endpoint, args.output), ensure_ascii=False), flush=True)
