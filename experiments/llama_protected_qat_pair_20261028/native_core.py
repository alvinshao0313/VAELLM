"""Explicit same-a1 matched64 endpoint binding for the proven native2.6 loader."""
from pathlib import Path
from types import SimpleNamespace
import qat_common as c

torch, require, read, write, sha, binding = c.torch, c.require, c.read, c.write, c.sha, c.binding
HF, S, PUBLIC, PYTHON = c.HF, c.S, c.PUBLIC, c.PYTHON
HERE = c.CODE
RUNTIME = c.previous.SPARSE / 'native_export/runtime.py'
MAX_BPW, BPW = 2.6, c.BPW
b = SimpleNamespace(c=c)
_arm = None


def bind(arm):
    global _arm
    require(_arm is None and arm in c.ARMS, 'Bind one actual matched64 endpoint per process.')
    _arm = arm


def check_budget(budget, record):
    expected = record['bit_budget']
    keys = ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components',
        'outside_linear_bits', 'code_padding_bits')
    require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
        and budget['compressed_target_count'] == 224 and budget['user_linear_bpw'] == budget['packed_linear_bpw'] == BPW
        and all(budget[key] == expected[key] for key in keys)
        and not budget['components'].get('residual_lora', {}).get('storage_bits', 0),
        'Require unchanged actual3619 sparse005 ledger under authorized logical2.6/rank0.')


def check_export():
    require(_arm in c.ARMS, 'Bind this process before loading an endpoint.')
    record = c.prepared(); checkpoint = c.DATA / _arm / 'formal64/native_model'
    report = read(c.HERE / _arm / 'formal64/report.json')
    meta = read(checkpoint / 'checkpoint_meta.json')
    audit_path = checkpoint / 'assembly_report.json'; audit = read(audit_path)
    c.check_binding(audit['weight'])
    require(report['status'] == 'complete' and report['exit_code'] == 0
        and report['source_checkpoint_id'] == c.SOURCE_ID and report['training']['steps'] == 64
        and report['training']['group_indices'] == list(range(32)) * 2 and report['training']['source_restored_exact']
        and report['prepared_sha256'] == c.sha(c.HERE / 'prepared.json')
        and audit == report['export'] and audit['status'] == 'verified'
        and audit['checkpoint_id'] == meta['checkpoint_id'] != c.SOURCE_ID
        and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3619
        and audit['sources']['frozen_tensors_compared'] == record['frozen_state'][_arm]['tensors']
        and audit['sources']['sparse_tensors_frozen'] == 576 and audit['frozen_complement_exact'] and audit['sparse576_exact'],
        'Require completed actual matched64 native3619 export with its exact frozen complement.')
    check_budget(audit['bit_budget'], record)
    return checkpoint, audit_path, audit, meta, record


def config(arm):
    require(arm in c.ARMS, 'Use the declared matched64 arm.')
    result = read(c.HERE / arm / 'longbench_config.json')
    checkpoint = c.DATA / arm / 'formal64/native_model'
    meta = read(checkpoint / 'checkpoint_meta.json')
    require(result['model']['checkpoint'] == str(checkpoint) and result['model']['checkpoint_id'] == meta['checkpoint_id'],
        'Full21 configuration belongs to another endpoint.')
    require(result['model']['Llama_QAT_authorized_budget'] == {'max_linear_bpw': 2.6, 'max_lora_rank': 0,
        'native_tensors': 3619, 'native_projections': 224, 'native_main_banks': 512,
        'export_audit': str(checkpoint / 'assembly_report.json'), 'export_audit_sha256': sha(checkpoint / 'assembly_report.json')},
        'Require explicit authorized native2.6/rank0 loader binding.')
    return result
