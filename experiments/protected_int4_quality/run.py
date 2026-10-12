"""One sidecar-expanded protected-INT4 quality diagnostic using unchanged native inference."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from experiments.protected_int4_quality.common import check, read, SHARDS, verify_shard, summarize
from experiments.protected_int4_quality.protection import immutable_inventory, replay, verify_native_cache
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and output == Path(record['output_dir'])
            and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and not (output / 'report.json').exists() and not (output / 'run.exit').exists(),
            'Use a fresh root-bound GPU, bitvae and output; no resume.')
    model = None; code = 1
    report = {'status': 'starting', 'source_checkpoint_id': record['source_checkpoint_id'], 'arm': record['arm'],
        'model_label': record['model_label'], 'protocol': record['protocol'], 'started_unix': time.time(),
        'model_exported': False, 'goal_acceptance': False, 'accounting': record['accounting']}
    try:
        check(record)
        require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Environment or single GPU differs.')
        for name, version in record['package_versions'].items():
            require(importlib.metadata.version(name) == version, 'Package changed: ' + name)
        record.update(status='running', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]), started_unix=time.time())
        write_json(path, record); write_json(output / 'report.json', report)
        from train_utils.v6_model_loader import load_v6_model_checkpoint
        from tools.cat_eval import _prepare_model_for_eval
        from train_utils.eval_utils import run_lm_eval
        model, meta, loaded = load_v6_model_checkpoint(record['source_checkpoint'], strict=True, map_location='cpu')
        require(meta['checkpoint_id'] == record['source_checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys
                and len(model.state_dict()) == record['expected_serialized_tensors'], 'Fresh source strict load differs.')
        model.eval().requires_grad_(False); model.config.use_cache = False
        report['replay'], frozen_cpu = replay(model, record)
        report['warmup'] = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
        policy = record['model_policy']
        require(model.config.model_type == policy['model_type'] and model.config._attn_implementation == policy['attn_implementation']
                and model.config.rope_scaling == policy['rope_scaling'] and model.config.rope_theta == policy['rope_theta']
                and model.get_input_embeddings().weight.dtype == torch.bfloat16, 'Native model-specific precision/backend changed.')
        rope = {name: str(value.dtype) for name, value in model.named_buffers() if 'inv_freq' in name}
        require(rope and set(rope.values()) == {'torch.float32'}, 'Native FP32 RoPE changed.')
        frozen_gpu = immutable_inventory(model, record['selected_targets'])
        require(frozen_cpu['state'] == frozen_gpu['state'] and frozen_cpu['buffers'] == frozen_gpu['buffers'],
                'GPU transfer/prewarm changed a frozen non-protection tensor.')
        report['native_cache_validation'] = verify_native_cache(model, record)
        expected_bits = record['accounting']['expanded_simulation_linear_bits'] - record['accounting']['source_user_linear_bits'] + record['accounting']['replaced_native_qvalues_scales_bits']
        require(report['native_cache_validation']['expanded_protected_weight_storage_bits'] == expected_bits,
                'Actual expanded simulation storage differs from accounting.')
        tokenizer = AutoTokenizer.from_pretrained(record['hf_path'], use_fast=True, local_files_only=True)
        ids = tokenizer('A short test sentence.', return_tensors='pt', add_special_tokens=False)['input_ids'][:, :4]
        require(ids.shape == (1, 4), 'Require the actual four-token native integration probe.')
        with torch.inference_mode():
            logits = model(input_ids=ids.to('cuda:0'), use_cache=False).logits
        require(logits.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all()), 'Native integration forward is not finite BF16.')
        report['integration_probe'] = {'input_shape': list(ids.shape), 'output_shape': list(logits.shape),
            'output_dtype': str(logits.dtype), 'finite': True, 'model_policy': policy, 'rope_buffers': rope}
        del logits, ids, frozen_cpu
        write_json(output / 'report.json', report); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for shard, tasks, batch in SHARDS:
            args = SimpleNamespace(tasks=tasks, num_fewshot=0, batch_size=batch, lm_limit=None, model_path=record['hf_path'],
                eval_log_dir=str(output), eval_run_ts=shard, mmlu_debug_samples=0)
            metrics = run_lm_eval(model, tokenizer, args)
            verify_shard(metrics, read(record['control_result_paths'][shard])['evals']['lm_eval'], batch)
            report.setdefault('completed_shards', []).append(shard); write_json(output / 'report.json', report)
        require(immutable_inventory(model, record['selected_targets']) == frozen_gpu,
                'Inference changed frozen state, buffers or parameter identities.')
        report['final_native_cache_validation'] = verify_native_cache(model, record)
        check(record)
        report.update(status='diagnostic_complete', **summarize(output, record), all_non_protection_state_and_identity_unchanged=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            longbench_status='not_evaluated', nine_item_goal_status='not_evaluated')
        code = 0
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        report.update(exit_code=code, finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        write_json(output / 'report.json', report)
        record.update(status=report['status'], exit_code=code, finished_unix=report['finished_unix'], report_sha256=sha256(output / 'report.json'))
        write_json(path, record); (output / 'run.exit').write_text(str(code) + '\n')
        del model; gc.collect()
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(run(parser.parse_args().launch_record))
