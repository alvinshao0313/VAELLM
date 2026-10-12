"""Deploy one calibration-code plus decoder500 gate and verify native f927 reload."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from transformers import AutoTokenizer

from experiments.gate_kl_calibration_decoder_evaluation.prepare import (
    SOURCE_ID, TARGET, budget_equal, patched_state, read, require, require_state,
)
from experiments.gate_student_context_evaluation.run import check
from experiments.longbench_v1.common import file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


from experiments.gate_student_context_evaluation.deploy import native_policy, decoded_gate


@torch.no_grad()
def deploy(record_path):
    record = read(record_path)
    output, source = Path(record['output_dir']), Path(record['source_checkpoint'])
    audit = {'status': 'deploying', 'arm': record['arm'], 'started_unix': time.time(),
             'worker_pid': os.getpid(), 'worker_start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19])}
    try:
        require(record['status'] == 'deploy' and type(record['gpu']) is int
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and torch.cuda.device_count() == 1, 'Deploy only through the bound single-GPU dispatcher.')
        require(not (output / 'final_model').exists() and not (output / 'audit.json').exists(), 'Never overwrite an existing deployment.')
        check(record)
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        meta = read(source / 'checkpoint_meta.json')
        require(meta['checkpoint_id'] == record['source_checkpoint_id'] == SOURCE_ID, 'Fresh f927 source identity differs.')
        original = torch.load(source / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
        endpoint = torch.load(record['endpoint_path'], map_location='cpu', weights_only=True, mmap=True)
        require({k: tensor_record(v) for k, v in endpoint.items()} == record['endpoint_state'], 'Endpoint tensor proof differs.')
        expected, patch = patched_state(original, endpoint, record['arm'])
        require(patch == record['patch_contract'], 'The selected local patch differs from CPU preparation.')
        budget_equal(audit_state_dict(meta, original, max_bpw=2.5, max_lora_rank=0), record['source_bit_budget'])
        model, loaded_meta, loaded = load_v6_model_checkpoint(str(source), strict=True, map_location='cpu', expected_kind='final_model')
        require(loaded_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys, 'Original full model strict load failed.')
        model.eval().requires_grad_(False)
        require_state(model.state_dict(), original, 'Fresh f927 full model')
        rope_before = native_policy(model, record['source_native_policy'])
        gate = model.get_submodule(TARGET)
        require(isinstance(gate, VAELinear), 'Original target is not a native VAELinear.')
        incompatible = gate.load_state_dict(endpoint, strict=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys, 'The local endpoint did not load strictly.')
        gate.clear_decoded_weight_cache()
        gate._build_parallel_stage_decode_plan()
        require_state(model.state_dict(), expected, 'Patched full model')
        reference = decoded_gate(model)
        require_state(model.state_dict(), expected, 'Native decode preserves persistent state')
        tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
        require(tokenizer.is_fast, 'The original evaluation tokenizer must be fast.')
        provenance = {'source_checkpoint_id': SOURCE_ID, 'source_checkpoint': str(source), 'target': TARGET,
            'arm': record['arm'], 'endpoint_path': record['endpoint_path'], 'endpoint_sha256': record['endpoint_sha256'],
            'patch_contract': patch, 'purpose': 'Single-layer isolated full8 evaluation, not a resume checkpoint.'}
        saved = save_v6_full_checkpoint(model, str(output / 'final_model'), checkpoint_kind='final_model',
            compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
            skip_targets=meta['skip_targets'], legacy_original_only_sources=meta.get('legacy_original_only_sources', []),
            train_mode='none', norm_train_mode='none', lm_head_train_mode='none', lora_config=None,
            completed_categories=meta.get('completed_categories', []), compression_categories=meta.get('compression_categories', []),
            target_layers=[18], target_modules=['gate_proj'],
            finalized_status={'decoder_finalized': True, 'runtime_clean': True, 'isolated_endpoint_persisted': True}, runtime_audit=provenance,
            base_model_path=meta['base_model_path'], tokenizer=tokenizer, save_config=True,
            extra_meta={'isolated_gate_endpoint': provenance})
        final_meta = saved['meta_payload']
        require(final_meta['checkpoint_id'] != SOURCE_ID and final_meta['converted_modules'] == meta['converted_modules'],
                'Export must receive a new UUID and preserve every compressed module spec.')
        persisted = torch.load(saved['state_dict'], map_location='cpu', weights_only=True, mmap=True)
        serialized_count = require_state(persisted, expected, 'Serialized full model')
        budget = audit_state_dict(final_meta, persisted, max_bpw=2.5, max_lora_rank=0)
        budget_equal(budget, record['source_bit_budget'])
        del model, gate, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        reloaded, reloaded_meta, restored = load_v6_model_checkpoint(str(output / 'final_model'), strict=True,
                                                                   map_location='cpu', expected_kind='final_model')
        require(reloaded_meta == final_meta and not restored.missing_keys and not restored.unexpected_keys,
                'Independent native reconstruction failed strict loading.')
        reloaded.eval().requires_grad_(False)
        reloaded_count = require_state(reloaded.state_dict(), expected, 'Independently reconstructed full model')
        require(native_policy(reloaded, record['source_native_policy']) == rope_before, 'RoPE buffers changed on reconstruction.')
        actual = decoded_gate(reloaded)
        require(torch.equal(actual, reference), 'Saved/reloaded native BF16 full gate weight differs.')
        require_state(reloaded.state_dict(), expected, 'Reload decode preserves all persistent state')
        check(record)
        audit.update(status='verified', source_checkpoint_id=SOURCE_ID, checkpoint_id=final_meta['checkpoint_id'],
            endpoint_sha256=record['endpoint_sha256'], **patch,
            serialized_tensors_compared=serialized_count, reloaded_tensors_compared=reloaded_count,
            fresh_strict_reload_state_exact=True, endpoint_state_exact=True, frozen_outside_target_exact=True,
            native_full_weight_reload_exact=True, integer_budget_unchanged=True, rank_zero=True,
            native_full_weight=tensor_record(actual), native_rope_buffers=rope_before,
            bit_budget=budget, source_bit_budget=record['source_bit_budget'],
            checkpoint_meta_sha256=sha256(saved['meta']), checkpoint_state_stat=file_stat(saved['state_dict']),
            checkpoint_state_sha256=sha256(saved['state_dict']),
            source_weight_stat=record['source_weight_stat'], source_sha256=record['source_sha256'],
            input_sha256=record['input_sha256'], maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            maximum_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
            reload_scope='Fresh native model reconstruction in this deployment process; full8 subsequently loads in independent processes.')
    except BaseException as error:
        audit.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        audit['finished_unix'] = time.time()
        write_json(output / 'audit.json', audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    deploy(parser.parse_args().launch_record)

