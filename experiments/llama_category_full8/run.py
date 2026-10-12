"""One in-memory original-BF16 category replacement, followed by native complete eight tasks."""
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

from experiments.llama_category_full8.common import SHARDS, check, read, summarize, verify_shard
from experiments.longbench_attribution.restore import frozen_inventory, restore_targets, tensor_digest
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.mmlu_attribution.run import load_model


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and output == Path(record['output_dir'])
            and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and not (output / 'report.json').exists() and not (output / 'run.exit').exists()
            and not list(output.glob('lm_eval_results_*.json')), 'Use a fresh explicitly bound single-GPU arm.')
    code, model = 1, None
    report = {'status': 'starting', 'arm': record['arm'], 'source_checkpoint_id': record['source_checkpoint_id'],
              'protocol': record['protocol'], 'diagnostic_only': True, 'model_exported': False,
              'goal_acceptance': False, 'started_unix': time.time()}
    try:
        require(all(os.environ.get(k) == v for k, v in record['environment'].items()), 'Environment differs.')
        check(record)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Bind one root-assigned GPU.')
        for name, version in record['package_versions'].items():
            require(importlib.metadata.version(name) == version, 'Evaluation package changed: ' + name)
        record.update(status='running', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time())
        write_json(path, record); write_json(output / 'report.json', report)
        tokenizer = AutoTokenizer.from_pretrained(record['hf_path'], use_fast=True, local_files_only=True)
        model, report['load'] = load_model('compressed', record, logging.getLogger(__name__))
        targets = record['selected_targets']
        restoration = restore_targets(model, record['hf_path'], targets)
        before = restoration.pop('frozen_inventory')
        report['restoration'] = restoration
        require(restoration['selected_count'] == 32 and restoration['non_target_state_and_identity_exact'],
                'Category restoration did not preserve its complete complement.')
        dense_bits = sum(value.untyped_storage().nbytes() * 8
                         for name in targets for value in model.get_submodule(name).state_dict().values())
        require(dense_bits == record['diagnostic_budget']['restored_bf16_bits'], 'Actual restored BF16 storage differs.')
        report['diagnostic_budget'] = {**record['diagnostic_budget'], 'actual_restored_bf16_storage_bits': dense_bits}
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        from train_utils.eval_utils import run_lm_eval
        for shard, tasks, batch_size in SHARDS:
            args = SimpleNamespace(tasks=tasks, num_fewshot=0, batch_size=batch_size, lm_limit=None,
                model_path=record['hf_path'], eval_log_dir=str(output), eval_run_ts=shard, mmlu_debug_samples=0)
            metrics = run_lm_eval(model, tokenizer, args)
            reference = read(record['control_result_paths'][shard])['evals']['lm_eval']
            verify_shard(metrics, reference, batch_size)
            report.setdefault('completed_shards', []).append(shard)
            write_json(output / 'report.json', report)
        require(frozen_inventory(model, targets) == before, 'Evaluation changed non-target state or identity.')
        for name, values in restoration['restored_hf_tensors'].items():
            for suffix, identity in values.items():
                require(tensor_digest(getattr(model.get_submodule(name), suffix)) == identity['sha256'],
                        'Evaluation changed a restored HF parameter.')
        summary = summarize(output, record)
        check(record)
        report.update(status='diagnostic_complete', **summary, non_target_state_and_identity_unchanged=True,
            restored_hf_tensors_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), longbench_status='not_evaluated',
            nine_item_goal_status='not_evaluated')
        code = 0
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        report.update(exit_code=code, finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        write_json(output / 'report.json', report)
        record.update(status=report['status'], exit_code=code, finished_unix=report['finished_unix'],
                      report_sha256=sha256(output / 'report.json'))
        write_json(path, record); (output / 'run.exit').write_text(str(code) + '\n')
        del model; gc.collect()
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(run(parser.parse_args().launch_record))
