"""Deploy one fixed-code gate/down endpoint into fresh f927 and verify native reload."""
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

from experiments.gate_down_student_context_evaluation.prepare import (
    SOURCE_ID, PREFIX, budget_equal, local_endpoint, patched_state, read, require, require_state,
)
from experiments.gate_down_student_context_evaluation.run import check
from experiments.gate_student_context_evaluation.deploy import native_policy
from experiments.longbench_v1.common import file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


@torch.no_grad()
def decoded_pair(model):
    """Native decode already includes transpose, inverse rotation and INT8 restoration."""
    result = {}
    for kind, shape in [('gate_proj', (12288, 4096)), ('down_proj', (4096, 12288))]:
        layer = model.get_submodule(PREFIX + kind)
        require(isinstance(layer, VAELinear), 'Keep both native VAELinear projections.')
        layer.clear_decoded_weight_cache()
        layer.to('cuda:0')
        layer._build_parallel_stage_decode_plan()
        weight = layer._decode_weight(dtype=torch.bfloat16).detach().cpu()
        require(weight.dtype == torch.bfloat16 and tuple(weight.shape) == shape
                and bool(torch.isfinite(weight).all()), 'Invalid native restored projection: ' + kind)
        result[kind] = weight
        layer.clear_decoded_weight_cache()
        layer.to('cpu')
        layer._build_parallel_stage_decode_plan()
    return result


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
        require(not (output / 'final_model').exists() and not (output / 'audit.json').exists(), 'Never overwrite a deployment.')
        check(record)
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        meta = read(source / 'checkpoint_meta.json')
        require(meta['checkpoint_id'] == record['source_checkpoint_id'] == SOURCE_ID, 'Wrong fresh source UUID.')
        original = torch.load(source / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
        endpoint = torch.load(record['endpoint_path'], map_location='cpu', weights_only=True, mmap=True)
        require({k: tensor_record(v) for k, v in endpoint.items()} == record['endpoint_state'], 'The saved endpoint differs from its proof.')
        expected, patch = patched_state(original, endpoint, record['arm'])
        require(patch == record['patch_contract'] and patch['selected_projections'] == record['selected_projections'], 'CPU patch contract changed.')
        budget_equal(audit_state_dict(meta, original, max_bpw=2.5, max_lora_rank=0), record['source_bit_budget'])
        model, loaded_meta, loaded = load_v6_model_checkpoint(str(source), strict=True, map_location='cpu', expected_kind='final_model')
        require(loaded_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys, 'Original full model strict load failed.')
        model.eval().requires_grad_(False)
        require_state(model.state_dict(), original, 'Fresh full f927 model')
        rope_before = native_policy(model, record['source_native_policy'])
        local = local_endpoint(endpoint, record['arm'])
        for kind in record['selected_projections']:
            layer = model.get_submodule(PREFIX + kind)
            require(isinstance(layer, VAELinear), 'Selected target must remain native VAELinear.')
            state = {key[len(kind) + 1:]: value for key, value in local.items() if key.startswith(kind + '.')}
            incompatible = layer.load_state_dict(state, strict=True)
            require(not incompatible.missing_keys and not incompatible.unexpected_keys, 'Selected endpoint failed strict local load.')
            layer.clear_decoded_weight_cache()
            layer._build_parallel_stage_decode_plan()
        require_state(model.state_dict(), expected, 'Patched full model')
        reference = decoded_pair(model)
        require_state(model.state_dict(), expected, 'Native pair decode preserves persistent state')
        tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
        require(tokenizer.is_fast, 'Use the original fast evaluation tokenizer.')
        provenance = {'source_checkpoint_id': SOURCE_ID, 'source_checkpoint': str(source),
            'targets': [PREFIX + kind for kind in record['selected_projections']], 'arm': record['arm'],
            'endpoint_path': record['endpoint_path'], 'endpoint_sha256': record['endpoint_sha256'],
            'patch_contract': patch, 'purpose': 'Isolated fixed-code L18 full8 evaluation; not a resume checkpoint.'}
        saved = save_v6_full_checkpoint(model, str(output / 'final_model'), checkpoint_kind='final_model',
            compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
            skip_targets=meta['skip_targets'], legacy_original_only_sources=meta.get('legacy_original_only_sources', []),
            train_mode='none', norm_train_mode='none', lm_head_train_mode='none', lora_config=None,
            completed_categories=meta.get('completed_categories', []), compression_categories=meta.get('compression_categories', []),
            target_layers=[18], target_modules=record['selected_projections'],
            finalized_status={'decoder_finalized': True, 'runtime_clean': True, 'isolated_endpoint_persisted': True}, runtime_audit=provenance,
            base_model_path=meta['base_model_path'], tokenizer=tokenizer, save_config=True,
            extra_meta={'isolated_gate_down_endpoint': provenance})
        final_meta = saved['meta_payload']
        require(final_meta['checkpoint_id'] != SOURCE_ID and final_meta['converted_modules'] == meta['converted_modules'],
                'Export requires a new UUID and identical compressed module specifications.')
        persisted = torch.load(saved['state_dict'], map_location='cpu', weights_only=True, mmap=True)
        serialized_count = require_state(persisted, expected, 'Serialized full model')
        budget = audit_state_dict(final_meta, persisted, max_bpw=2.5, max_lora_rank=0)
        budget_equal(budget, record['source_bit_budget'])
        del model, layer, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        reloaded, reloaded_meta, restored = load_v6_model_checkpoint(str(output / 'final_model'), strict=True,
                                                                   map_location='cpu', expected_kind='final_model')
        require(reloaded_meta == final_meta and not restored.missing_keys and not restored.unexpected_keys, 'Fresh native reconstruction failed strict loading.')
        reloaded.eval().requires_grad_(False)
        reloaded_count = require_state(reloaded.state_dict(), expected, 'Independently reconstructed full model')
        require(native_policy(reloaded, record['source_native_policy']) == rope_before, 'Native FP32 RoPE changed on reconstruction.')
        actual = decoded_pair(reloaded)
        require_state(actual, reference, 'Saved/reloaded native BF16 full gate/down matrices')
        require_state(reloaded.state_dict(), expected, 'Reload decode preserves every persistent tensor')
        check(record)
        audit.update(status='verified', source_checkpoint_id=SOURCE_ID, checkpoint_id=final_meta['checkpoint_id'],
            endpoint_sha256=record['endpoint_sha256'], **patch,
            serialized_tensors_compared=serialized_count, reloaded_tensors_compared=reloaded_count,
            fresh_strict_reload_state_exact=True, endpoint_state_exact=True, frozen_outside_target_exact=True,
            native_full_weight_reload_exact=True, integer_budget_unchanged=True, rank_zero=True,
            native_full_weights={kind: tensor_record(value) for kind, value in actual.items()}, native_rope_buffers=rope_before,
            bit_budget=budget, source_bit_budget=record['source_bit_budget'],
            checkpoint_meta_sha256=sha256(saved['meta']), checkpoint_state_stat=file_stat(saved['state_dict']),
            checkpoint_state_sha256=sha256(saved['state_dict']), source_weight_stat=record['source_weight_stat'],
            source_sha256=record['source_sha256'], input_sha256=record['input_sha256'],
            maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0), maximum_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
            reload_scope='Fresh native model reconstruction within deployment; full8 subsequently loads in independent processes.')
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
