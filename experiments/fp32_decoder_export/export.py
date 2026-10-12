"""Isolated QCE7200 export retaining native FP32 decoder values and BF16 compute."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.e2e_subset_export import _sha256
from experiments.qwen3_w25.export_e2e_selected import restore_decoder_state
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.subset_export import CODE_FILES as EXISTING_CODE_FILES
from experiments.w25_eval.subset_state import (
    checkpoint_inventory, inspect_step, persistent_inventory, selected_layers, selected_modules,
)

REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO / 'result/qwen3_8b_w25_20260930/e2e_mlpbest_ce50_s8000'
SOURCE_FILES = tuple(dict.fromkeys((*EXISTING_CODE_FILES,
    'experiments/fp32_decoder_export/export.py', 'experiments/fp32_decoder_export/audit.py',
    'experiments/rotation_factor_probe/capture.py', 'tools/cat_eval.py',
    'train_utils/distill_precision.py', 'litebsq/vae_linear_prewarm.py',
    'litebsq/fused_multistage_decoder.py')))


def inspect_scope(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    _require(checkpoint.is_relative_to(RUN_ROOT) and checkpoint.name == 'checkpoint-7200',
             'This isolated comparison requires the actual QCE8000 checkpoint-7200.')
    result = inspect_step(checkpoint)
    meta, state, snapshot, _source, _base, architecture = result
    _require(state['global_step'] == 7200 and state['max_steps'] == 8000
             and meta['train_mode'] == 'decoder' and architecture['model_type'] == 'qwen3'
             and architecture['num_hidden_layers'] == 36 and architecture['projection_count'] == 252,
             'Wrong native step, schedule, train mode, or actual architecture.')
    _require(len(selected_layers(meta['target_layers'], architecture)) == 36
             and len(selected_modules(meta['target_modules'], architecture)) == 7
             and snapshot['training_args']['bf16'] and not snapshot['training_args']['fp16'],
             'Require all original decoders with unchanged BF16 computation.')
    return result


def rope_record(model):
    values = {name: value for name, value in model.named_buffers() if 'inv_freq' in name}
    _require(values and all(value.dtype == torch.float32 for value in values.values()),
             'Native RoPE buffers must remain FP32.')
    return {name: tensor_record(value) for name, value in values.items()}


def precision_inventory(model, decoder_keys):
    parameters = dict(model.named_parameters())
    _require(decoder_keys <= parameters.keys(), 'Missing persistent decoder parameters.')
    _require({name for name, value in parameters.items() if value.dtype == torch.float32} == decoder_keys,
             'FP32 parameter storage must be exactly the native decoder inventory.')
    _require(all(value.dtype == torch.bfloat16 for name, value in parameters.items() if name not in decoder_keys),
             'The original non-decoder backbone must remain BF16.')
    _require(model.get_input_embeddings().weight.dtype == torch.bfloat16,
             'Normal evaluation must select BF16 activation/cache dtype.')
    return {'decoder_parameters': len(decoder_keys), 'decoder_elements': sum(parameters[k].numel() for k in decoder_keys),
            'decoder_dtype': 'torch.float32', 'other_parameter_dtype': 'torch.bfloat16',
            'activation_dtype': 'torch.bfloat16', 'rope': rope_record(model)}


def export_checkpoint(checkpoint, output_dir, *, device='cuda:0'):
    from compressed_e2e_fintuning.mid_eval import temporary_inference_decode_mode
    from compressed_e2e_fintuning.runtime_v6 import (
        _assert_final_runtime_clean, _build_v6_tokenizer, _finalize_decoders, _sync_model_padding_config,
    )
    from compressed_e2e_fintuning.runtime_v6_pipeline import (
        _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe,
    )
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from litebsq.vae_linear import clear_model_vae_linear_cache
    from train_utils.checkpoint_v6 import _atomic_write_json, save_v6_full_checkpoint
    from train_utils.config.targets import collect_e2e_compressed_targets
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    checkpoint, output = Path(checkpoint).resolve(), Path(output_dir).resolve()
    _require(not output.exists() and not output.is_relative_to(RUN_ROOT), 'Use a new isolated output outside the live run.')
    _require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'Use one process.')
    meta, state, snapshot, source, base, architecture = inspect_scope(checkpoint)
    training = SimpleNamespace(**snapshot['training_args'])
    device = torch.device(device)
    _require(torch.cuda.is_available() and device.type == 'cuda' and torch.cuda.device_count() == 1,
             'Real export probes require one explicitly assigned CUDA device.')
    started = time.perf_counter()
    configure_e2e_determinism(bool(training.full_determinism)); set_e2e_seed(training.seed)
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    if training.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(training.tf32)
        torch.backends.cudnn.allow_tf32 = bool(training.tf32)
    _source_meta, source_budget, _ = checkpoint_inventory(source, architecture)
    files = [checkpoint / name for name in ('checkpoint_meta.json', 'trainer_state.json', 'training_model_state.pt')]
    files += [checkpoint.parent.parent / 'normalized_e2e_runtime_args.json', source / 'checkpoint_meta.json']
    source_stat = (source / 'pytorch_model.bin').stat()
    audit = {'runtime': 'experiments.fp32_decoder_export.export', 'source_checkpoint': str(checkpoint),
        'source_checkpoint_id': meta['checkpoint_id'], 'round_base_dir': str(source),
        'round_base_checkpoint_id': base['checkpoint_id'], 'source_global_step': state['global_step'],
        'source_max_steps': state['max_steps'], 'train_mode': 'decoder', 'architecture': architecture,
        'optimizer_steps_executed': 0, 'trainer_or_main_optimizer_constructed': False,
        'teacher_or_training_data_loaded': False, 'source_files_sha256': {str(p): _sha256(p) for p in files},
        'round_base_weight_stat': {'path': str(source / 'pytorch_model.bin'), 'size': source_stat.st_size,
                                  'mtime_ns': source_stat.st_mtime_ns, 'inode': source_stat.st_ino},
        'code_sha256': {str(REPO / name): _sha256(REPO / name) for name in SOURCE_FILES},
        'source_bit_budget': source_budget,
        'precision_policy': 'Retain actual native FP32 decoders. All other original storage unchanged; BF16 compute. No prepare_model_export cast.'}
    model, loaded, _ = load_v6_model_checkpoint(str(source), strict=True, map_location='cpu')
    _require(loaded['checkpoint_id'] == meta['round_base_checkpoint_id'], 'Source UUID changed.')
    original_rope = rope_record(model)
    inventory, banks = persistent_inventory(model, architecture)
    selected = collect_e2e_compressed_targets(model, target_layers=selected_layers(meta['target_layers'], architecture),
        target_modules=selected_modules(meta['target_modules'], architecture), num_layers=36)
    keys = {key for values in inventory.values() for key in values}
    _require(len(selected) == len(inventory) == 252 and len(keys) == 1512
             and sum(len(v) for v in banks.values()) == 504, 'Unexpected native decoder/code inventory.')
    audit['restored_decoder_tensors'] = restore_decoder_state(model, selected, checkpoint=checkpoint, snapshot=snapshot)
    _require(audit['restored_decoder_tensors'] == 1512, 'Incomplete native FP32 restore.')
    audit['precision'] = precision_inventory(model, keys)
    _require(audit['precision']['rope'] == original_rope, 'Restore changed native RoPE.')
    tokenizer = _build_v6_tokenizer(str(base['base_model_path']), SimpleNamespace(**snapshot['hf_args']))
    _sync_model_padding_config(model, tokenizer); model.config.use_cache = False
    model.eval().to(device)
    inputs = _build_finalization_probe_inputs(tokenizer)
    trained, dtype = _run_finalization_probe(model, inputs)
    with temporary_inference_decode_mode(model, cache_decoded_weight=True):
        deployed, deployment_dtype = _run_finalization_probe(model, inputs)
    _require(dtype == deployment_dtype == torch.bfloat16, 'Decoder storage must not change BF16 output precision.')
    delta = deployed - trained
    audit['decoder_compute_path_difference'] = {'max_abs': float(delta.abs().max()),
        'relative_l2': float(delta.norm() / trained.norm().clamp_min(1e-12)), 'output_dtype': str(dtype)}
    _require(_finalize_decoders(selected) == 252, 'Incomplete native finalization.')
    _assert_final_runtime_clean(model)
    finalized, final_dtype = _run_finalization_probe(model, inputs)
    _require(final_dtype == dtype, 'Finalization changed compute precision.')
    audit['core_structural_finalization_forward_parity'] = _assert_finalization_probe_close(
        deployed, finalized, output_dtype=dtype, label='FP32-storage native finalization')
    audit['precision_after_finalization'] = precision_inventory(model, keys)
    _require(audit['precision_after_finalization'] == audit['precision'], 'Finalization changed parameter/RoPE storage.')
    clear_model_vae_linear_cache(model); _assert_final_runtime_clean(model)
    model.requires_grad_(False).to('cpu')
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        **{key: tuple(base.get(key) or ()) for key in ('compressed_targets', 'pending_dense_targets', 'skip_targets',
            'legacy_original_only_sources', 'completed_categories', 'compression_categories')},
        train_mode='decoder', norm_train_mode='none', lm_head_train_mode='none', lora_config=None,
        resolved_learning_rates=meta['resolved_learning_rates'], target_layers=meta['target_layers'],
        target_modules=meta['target_modules'], immutable_resume_contract=meta['immutable_resume_contract'],
        base_model_path=base['base_model_path'], tokenizer=tokenizer, save_config=True,
        finalized_status={'sparse_bit_committed': False, 'decoder_finalized': True, 'lora_finalized': False,
                          'aux_finalized': True, 'runtime_clean': True, 'inference_forward_parity': True}, runtime_audit=audit)
    _require(saved['checkpoint_id'] not in (meta['checkpoint_id'], base['checkpoint_id']), 'Export reused source UUID.')
    probe_path = output / 'deployment_probe.pt'
    torch.save({'inputs': inputs, 'logits': finalized, 'output_dtype': str(dtype)}, probe_path)
    record = {**audit, 'status': 'exported', 'checkpoint_id': saved['checkpoint_id'], 'output_dir': str(output),
        'deployment_probe_sha256': _sha256(probe_path), 'export_decoder_dtype': 'torch.float32',
        'elapsed_seconds': time.perf_counter() - started,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'fresh_strict_state_audit_pending': True, 'full_downstream_evaluation_pending': True}
    _atomic_write_json(record, str(output / 'export_record.json'))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    result = export_checkpoint(args.checkpoint_dir, args.output_dir, device=args.device)
    print({key: result[key] for key in ('status', 'checkpoint_id', 'source_global_step', 'export_decoder_dtype')})
