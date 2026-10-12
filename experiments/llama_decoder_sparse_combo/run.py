"""Native patch-only assembly, fresh strict reload and complete eight-task evaluation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from transformers import AutoTokenizer

from experiments.full_rht_gate_pair.common import SHARDS, eval_argv, read
from experiments.full_rht_gate_pair.compare import delta, score
from experiments.llama_decoder_sparse_combo.audit import frozen_and_budget, verify
from experiments.llama_decoder_sparse_combo.prepare import BASE, PATCH, E767, HF, ROOT, SCALARS, check
from experiments.llama_gate_sparse_residual.audit import FIELDS, PROBE, local_forward, state
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from litebsq.vae_linear import VAELinear
from train_utils.cat_residual_from_base import _install_sparse_residual_payload
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def build(record):
    check(record); output = Path(record['output_dir']); started = time.time()
    model, meta, _ = load_v6_model_checkpoint(str(BASE), strict=True, map_location='cpu')
    model.eval().requires_grad_(False)
    _, original = state(BASE); donor_meta, donor = state(PATCH)
    _assert_state_equal(model.state_dict(), original, 'loaded unchanged D1200')
    specs = {s['name']: s for s in donor_meta['converted_modules']}; probe = None
    with torch.inference_mode():
        for name in record['targets']:
            module = model.get_submodule(name)
            _require(isinstance(module, VAELinear) and not module.has_sparse_residual(),
                     'Expected a native D1200 gate without sparse residual: ' + name)
            if name == PROBE:
                module.to('cuda:0')
                x = torch.randn((4, 4096), generator=torch.Generator(device='cpu').manual_seed(31)).to(torch.bfloat16)
                before = local_forward(module, x)
            payload = {'sparse_residual_' + k: specs[name]['sparse_residual_' + k] for k in SCALARS}
            payload.update({'sparse_residual_' + k: donor[name + '.sparse_residual_' + k].clone() for k in FIELDS})
            _install_sparse_residual_payload(module, payload)
            if name == PROBE:
                module.to('cuda:0')
                after = local_forward(module, x)
                _require(not torch.equal(before, after), 'Nonzero copied patch had no native BF16 probe effect.')
                probe = {'input': x, 'before': before, 'after': after}
                module.clear_decoded_weight_cache(); module.to('cpu')
    _require(probe is not None, 'The complete32 gate union must include L16.')
    checkpoint = output / 'final_model'; _require(not checkpoint.exists(), 'Do not overwrite a combination.')
    tokenizer = AutoTokenizer.from_pretrained(HF, local_files_only=True, use_fast=True)
    save_v6_full_checkpoint(model, str(checkpoint), checkpoint_kind='final_model',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
        skip_targets=meta['skip_targets'], legacy_original_only_sources=meta.get('legacy_original_only_sources', []),
        train_mode='none', after_category_mode='none', lora_config=None,
        completed_categories=meta['completed_categories'], compression_categories=meta['compression_categories'],
        target_layers=meta['target_layers'], target_modules=meta['target_modules'], base_model_path=str(HF),
        tokenizer=tokenizer, extra_meta={'stage': 'decoder1200_fixed_sparse_patch',
            'source_base_vae_checkpoint': str(BASE), 'source_base_checkpoint_id': record['base_checkpoint_id'],
            'sparse_patch_checkpoint': str(PATCH), 'sparse_patch_checkpoint_id': record['patch_checkpoint_id'],
            'sparse_residual_source_checkpoint_id': record['residual_source_checkpoint_id'],
            'sparse_targets': record['targets'], 'residual_target': record['residual_target'],
            'decode_policy': record['decode_policy'], 'hf_residual_recomputed': False,
            'comparison_boundary': record['comparison_boundary']})
    saved_meta, saved = state(checkpoint)
    _assert_state_equal(model.state_dict(), saved, 'native combination save')
    audit = frozen_and_budget(saved_meta, saved, record)
    torch.save(probe, output / 'local_probe.pt')
    diff = probe['after'].float() - probe['before'].float()
    report = {'status': 'saved', 'checkpoint_id': saved_meta['checkpoint_id'],
        'base_checkpoint_id': record['base_checkpoint_id'], 'patch_checkpoint_id': record['patch_checkpoint_id'],
        'hf_residual_recomputed': False, 'seconds': time.time() - started, **audit,
        'local_forward': {'module': PROBE, 'shape': [4, 14336], 'dtype': 'torch.bfloat16',
            'before_finite': True, 'after_finite': True, 'changed_elements': int(torch.count_nonzero(diff)),
            'rms_change': float(diff.square().mean().sqrt()), 'quality_evidence': False}}
    write_json(output / 'build_report.json', report)
    print({'status': 'saved', 'checkpoint_id': saved_meta['checkpoint_id'], 'seconds': report['seconds'],
           'original_frozen': 2979, 'sparse_copied_exactly': 192}, flush=True)


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
                 and all(os.environ.get(k) == v for k, v in record['environment'].items()),
                 'Require the explicitly bound environment and fresh reviewed invocation.')
        check(record)
        record.update(worker_pid=os.getpid(), started_unix=time.time(), stages={},
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        for phase in ('build', 'verify'):
            stage(phase, [sys.executable, '-u', '-m', 'experiments.llama_decoder_sparse_combo.run',
                          '--launch-record', str(path), '--phase', phase])
        audit = read(output / 'audit.json')
        for name, tasks, batch in SHARDS:
            stage(name, eval_argv(output / 'final_model', output / (name + '.json'),
                                 tasks=tasks, batch=batch, python=sys.executable))
        current, raw = score([output / (name + '.json') for name, _, _ in SHARDS],
                             record['architecture'], output / 'final_model')
        comparisons = {}
        for label, source in (('d1200', BASE), ('e767_sparse', PATCH), ('e767', E767)):
            control, control_raw = score([source.parent / (name + '.json') for name, _, _ in SHARDS],
                                         record['architecture'], source)
            for left, right in zip(raw, control_raw, strict=True):
                for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                    _require(left[key] == right[key], 'Full8 protocol differs from ' + label + ': ' + key)
            comparisons[label] = delta(current, control)
        current.update(comparisons=comparisons, user_linear_bpw=audit['bit_budget']['user_linear_bpw'],
            comparison_boundary=record['comparison_boundary'], longbench_status='not_evaluated',
            nine_item_goal_status='not_evaluated')
        write_json(output / 'full8_summary.json', current); check(record)
        record.update(status='completed', checkpoint_id=audit['checkpoint_id'],
            mean_percent=current['mean_percent'], mmlu_correct=current['integer_counts']['mmlu']['correct'],
            audit_sha256=sha256(output / 'audit.json')); code = 0
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
