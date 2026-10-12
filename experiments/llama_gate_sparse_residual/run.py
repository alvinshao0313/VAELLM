"""Native residual construction, independent strict reload, then optional full eight tasks."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from experiments.full_rht_gate_pair.common import SHARDS, eval_argv, read
from experiments.full_rht_gate_pair.compare import delta, score
from experiments.llama_gate_sparse_residual.audit import frozen_and_budget, local_forward, state, verify
from experiments.llama_gate_sparse_residual.prepare import BASE, HF, PROBE, ROOT, check
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from litebsq.vae_linear import VAELinear
from litebsq.fused_multistage_decoder import extract_packed_symmetric_stage_weights
from train_utils.cat_residual_from_base import _ResidualTarget, _decode_base_reconstruction, _install_sparse_residual_payload
from train_utils.cat_train_residual_protection import build_sparse_residual_payload
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def fp32_copy_reconstruction(name, module):
    """Supply matching FP32 operands without mutating the retained BF16 module."""
    module.clear_decoded_weight_cache()
    original = module._parallel_stage_decoder
    _require(all(value.dtype == torch.bfloat16 for value in original.parameters()),
             'The original decoder must retain exact BF16 storage: ' + name)
    temporary = deepcopy(module)
    try:
        decoder = temporary._parallel_stage_decoder.float()
        before, copied = original.state_dict(), decoder.state_dict()
        _require(len(before) == len(copied) == 6 and set(before) == set(copied)
                 and all(value.dtype == torch.float32 and value.data_ptr() != before[key].data_ptr()
                         and torch.equal(value, before[key].float()) for key, value in copied.items()),
                 'Temporary decoder must be an independent exact FP32 view of the BF16 values.')
        operands = extract_packed_symmetric_stage_weights(decoder)
        _require(len(operands) == 6 and all(value.dtype == torch.float32 for value in operands),
                 'Every native fused decoder weight, bias and norm operand must match FP32 input.')
        target = _ResidualTarget(name=name, category='gate_proj', module=temporary, transpose=temporary.transpose)
        result = _decode_base_reconstruction(target)
        _require(result.dtype == torch.float32 and result.device.type == 'cpu'
                 and bool(torch.isfinite(result).all()), 'Invalid native FP32 scratch reconstruction.')
        return result
    finally:
        temporary.clear_decoded_weight_cache()
        del temporary


def build(record):
    check(record); output = Path(record['output_dir']); started = time.time()
    model, meta, _ = load_v6_model_checkpoint(str(BASE), strict=True, map_location='cpu')
    model.eval().requires_grad_(False)
    old_meta, original = state(BASE)
    _assert_state_equal(model.state_dict(), original, 'loaded complete e767')
    index = read(HF / 'model.safetensors.index.json')['weight_map']; rows = {}; probe = None
    with torch.inference_mode():
        for name in record['targets']:
            tick = time.time(); module = model.get_submodule(name)
            _require(isinstance(module, VAELinear), 'Expected native VAELinear: ' + name)
            module.to('cuda:0')
            if name == PROBE:
                generator = torch.Generator(device='cpu').manual_seed(31)
                x = torch.randn((4, 4096), generator=generator).to(torch.bfloat16)
                before = local_forward(module, x)
            reconstructed = fp32_copy_reconstruction(name, module)
            key = name + '.weight'
            with safe_open(str(HF / index[key]), framework='pt', device='cpu') as handle:
                reference = handle.get_tensor(key)
            _require(reference.dtype == torch.bfloat16 and tuple(reference.shape) == (14336, 4096),
                     'Exact HF reference tensor differs: ' + name)
            reference = reference.float().contiguous()
            payload, nnz, storage = build_sparse_residual_payload(linear_name=name,
                target_weight=reference, reconstructed_weight=reconstructed, activation_weight=None,
                activation_mean=None, **record['recipe'])
            _require(payload is not None and nnz > 0, 'Native selection produced no sparse residual: ' + name)
            _install_sparse_residual_payload(module, payload)
            if name == PROBE:
                after = local_forward(module, x)
                _require(not torch.equal(before, after), 'Nonzero installed residual had no BF16 forward effect.')
                probe = {'input': x, 'before': before, 'after': after}
            module.clear_decoded_weight_cache(); module.to('cpu')
            rows[name] = {'nnz': nnz, **storage, 'seconds': time.time() - tick,
                'temporary_decoder_fp32_tensors': 6, 'exact_bf16_values_promoted': True,
                'original_decoder_dtype': 'torch.bfloat16', 'temporary_copy_released': True}
            print({'constructed': name, **rows[name]}, flush=True)
            del reconstructed, reference, payload; torch.cuda.empty_cache()
    _require(probe is not None, 'Both scopes must include the real L16 forward probe.')
    tokenizer = AutoTokenizer.from_pretrained(HF, local_files_only=True, use_fast=True)
    checkpoint = output / 'final_model'
    _require(not checkpoint.exists(), 'Do not overwrite a constructed checkpoint.')
    save_v6_full_checkpoint(model, str(checkpoint), checkpoint_kind='final_model',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
        skip_targets=meta['skip_targets'], legacy_original_only_sources=meta.get('legacy_original_only_sources', []),
        train_mode='none', after_category_mode='none', lora_config=None,
        completed_categories=meta['completed_categories'], compression_categories=meta['compression_categories'],
        target_layers=meta['target_layers'], target_modules=meta['target_modules'], base_model_path=str(HF),
        tokenizer=tokenizer, extra_meta={'stage': 'gate_sparse_residual', 'source_base_vae_checkpoint': str(BASE),
            'source_base_checkpoint_id': record['base_checkpoint_id'], 'sparse_recipe': record['recipe'],
            'sparse_targets': record['targets'], 'residual_target': record['residual_target'],
            'decode_policy': record['decode_policy']})
    saved_meta, saved = state(checkpoint)
    _assert_state_equal(model.state_dict(), saved, 'native save')
    audit = frozen_and_budget(saved_meta, saved, record)
    torch.save(probe, output / 'local_probe.pt')
    change = probe['after'].float() - probe['before'].float()
    report = {'status': 'saved', 'checkpoint_id': saved_meta['checkpoint_id'], 'scope': record['scope'],
        'base_checkpoint_id': record['base_checkpoint_id'], 'residual_target': record['residual_target'],
        'decode_policy': record['decode_policy'],
        'per_gate': rows, 'seconds': time.time() - started, **audit,
        'local_forward': {'module': PROBE, 'shape': [4, 14336], 'dtype': 'torch.bfloat16',
            'before_finite': True, 'after_finite': True, 'changed_elements': int(torch.count_nonzero(change)),
            'rms_change': float(change.square().mean().sqrt()), 'quality_evidence': False}}
    write_json(output / 'build_report.json', report)
    print({'status': 'saved', 'checkpoint_id': saved_meta['checkpoint_id'], 'seconds': report['seconds'],
           'frozen': audit['original_frozen_tensors'], 'added': audit['added_tensors']}, flush=True)


def run(path):
    record = read(path); output = path.parent; code = 1

    def stage(name, argv):
        record['status'] = name; log = output / (name + '.log')
        _require(not log.exists(), 'Do not overwrite an executed stage.')
        with log.open('w') as handle:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            record['stages'][name] = {'argv': argv, 'pid': child.pid, 'started_unix': time.time()}
            write_json(path, record); value = child.wait()
        (output / (name + '.exit')).write_text(str(value) + '\n')
        record['stages'][name].update(exit_code=value, finished_unix=time.time()); write_json(path, record)
        _require(value == 0, f'{name} exited {value}; inspect {log}')

    try:
        _require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                 and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path.cwd().resolve() == ROOT and not record['queued'] and not record['automatic_retry']
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and all(os.environ.get(key) == value for key, value in record['environment'].items()),
                 'Require the explicitly bound environment and fresh reviewed invocation.')
        check(record)
        record.update(worker_pid=os.getpid(), started_unix=time.time(), stages={},
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        for phase in ('build', 'verify'):
            stage(phase, [sys.executable, '-u', '-m', 'experiments.llama_gate_sparse_residual.run',
                          '--launch-record', str(path), '--phase', phase])
        audit = read(output / 'audit.json')
        if record['scope'] == 'all32':
            for name, tasks, batch in SHARDS:
                stage(name, eval_argv(output / 'final_model', output / (name + '.json'),
                                     tasks=tasks, batch=batch, python=sys.executable))
            current, raw = score([output / (name + '.json') for name, _, _ in SHARDS],
                                 record['architecture'], output / 'final_model')
            control, baseline_raw = score([BASE.parent / (name + '.json') for name, _, _ in SHARDS],
                                         record['architecture'], BASE)
            for left, right in zip(raw, baseline_raw, strict=True):
                for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                    _require(left[key] == right[key], 'Evaluation protocol differs from e767: ' + key)
            current.update(vs_e767=delta(current, control), user_linear_bpw=audit['bit_budget']['user_linear_bpw'],
                           comparison_boundary=record['comparison_boundary'], longbench_status='not_evaluated')
            write_json(output / 'full8_summary.json', current)
            record.update(mean_percent=current['mean_percent'], mmlu_correct=current['integer_counts']['mmlu']['correct'])
        check(record)
        record.update(status='completed', checkpoint_id=audit['checkpoint_id'], audit_sha256=sha256(output / 'audit.json'))
        code = 0
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    parser.add_argument('--phase', choices=('build', 'verify'))
    args = parser.parse_args(); path = args.launch_record.resolve()
    if args.phase:
        record = read(path)
        _require(type(record['gpu']) is int and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']),
                 'A phase requires the parent explicitly bound GPU.')
        (build if args.phase == 'build' else verify)(record)
    else:
        run(path)
