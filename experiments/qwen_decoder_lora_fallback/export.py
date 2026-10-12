"""Native decoder-then-LoRA finalization of one independently saved joint step."""
import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from compressed_e2e_fintuning.mid_eval import temporary_inference_decode_mode
from compressed_e2e_fintuning.runtime_v6 import (
    _assert_final_runtime_clean, _build_v6_tokenizer, _finalize_decoders, _sync_model_padding_config)
from compressed_e2e_fintuning.runtime_v6_pipeline import (
    _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe)
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from e2e_common.full_lora import finalize_model_level_lora
from experiments.longbench_v1.common import sha256, file_stat, check_stat, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen_decoder_lora_fallback.state import inspect_step, restore_joint
from experiments.w25_eval.subset_state import persistent_inventory, selected_layers, selected_modules
from litebsq.vae_linear import clear_model_vae_linear_cache
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.config.targets import collect_e2e_compressed_targets
from train_utils.distill_precision import prepare_model_export
from train_utils.v6_model_loader import load_v6_model_checkpoint


def export_checkpoint(checkpoint, output_dir, device='cuda:0'):
    checkpoint, output = Path(checkpoint).resolve(), Path(output_dir).resolve()
    require(not output.exists() and int(os.environ.get('WORLD_SIZE', '1')) == 1, 'Fresh single-process export required.')
    meta, state, snapshot, source, base, architecture = inspect_step(checkpoint)
    training, device = SimpleNamespace(**snapshot['training_args']), torch.device(device)
    require(torch.cuda.is_available() and device.type == 'cuda', 'Native structural probes require CUDA.')
    started = time.time()
    configure_e2e_determinism(bool(training.full_determinism)); set_e2e_seed(training.seed)
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    if training.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(training.tf32)
    files = [checkpoint / k for k in ('checkpoint_meta.json', 'trainer_state.json', 'training_model_state.pt')]
    files += [checkpoint.parent.parent / 'normalized_e2e_runtime_args.json', source / 'checkpoint_meta.json']
    report = {'source_checkpoint': str(checkpoint), 'source_checkpoint_id': meta['checkpoint_id'],
        'round_base_checkpoint_id': base['checkpoint_id'], 'source_global_step': state['global_step'],
        'source_max_steps': state['max_steps'], 'optimizer_steps_executed': 0,
        'trainer_or_main_optimizer_constructed': False, 'teacher_or_training_data_loaded': False,
        'source_weight_stat': file_stat(source / 'pytorch_model.bin'),
        'source_files_sha256': {str(p): sha256(p) for p in files},
        'restored_decoder_tensors': 1512, 'restored_lora_tensors': 504}
    model, loaded, _ = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True)
    require(loaded['checkpoint_id'] == base['checkpoint_id'], 'Loaded source UUID differs.')
    persistent_inventory(model, architecture)
    selected = collect_e2e_compressed_targets(model, target_layers=selected_layers(meta['target_layers'], architecture),
        target_modules=selected_modules(meta['target_modules'], architecture), num_layers=architecture['num_hidden_layers'])
    require(len(selected) == 252, 'Expected all252 projection objects.')
    model, decoders, factors = restore_joint(model, selected, checkpoint, meta, snapshot, architecture)
    del decoders, factors
    tokenizer = _build_v6_tokenizer(base['base_model_path'], SimpleNamespace(**snapshot['hf_args']))
    _sync_model_padding_config(model, tokenizer); model.config.use_cache = False
    model.eval().to(device)
    inputs = _build_finalization_probe_inputs(tokenizer)
    before, dtype = _run_finalization_probe(model, inputs)
    with temporary_inference_decode_mode(model, cache_decoded_weight=True):
        reference, reference_dtype = _run_finalization_probe(model, inputs)
    require(reference_dtype == dtype, 'Deployment reference changed output dtype.')
    delta = reference - before
    report['decoder_compute_path_difference'] = {'max_abs': float(delta.abs().max()),
        'relative_l2': float(delta.norm() / before.norm().clamp_min(1e-12))}
    require(_finalize_decoders(selected) == 252, 'Incomplete decoder finalization.')
    model = finalize_model_level_lora(model, compressed_proxy_names=[name for name, _ in selected])
    _assert_final_runtime_clean(model)
    core, core_dtype = _run_finalization_probe(model, inputs)
    require(core_dtype == dtype, 'Joint native finalization changed output dtype.')
    report['core_structural_finalization_forward_parity'] = _assert_finalization_probe_close(
        reference, core, output_dtype=dtype, label='Joint decoder and ordinary LoRA structural finalization')
    report['export_dtype'] = str(prepare_model_export(model, training))
    exported, exported_dtype = _run_finalization_probe(model, inputs)
    require(exported_dtype == dtype, 'BF16 export output dtype differs.')
    delta = exported - core
    report['export_precision_probe'] = {'max_abs': float(delta.abs().max()),
        'relative_l2': float(delta.norm() / core.norm().clamp_min(1e-12))}
    clear_model_vae_linear_cache(model); _assert_final_runtime_clean(model)
    model.requires_grad_(False).to('cpu')
    result = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        **{k: tuple(base.get(k) or ()) for k in ('compressed_targets', 'pending_dense_targets', 'skip_targets',
            'legacy_original_only_sources', 'completed_categories', 'compression_categories')},
        train_mode='decoder_lora', norm_train_mode='none', lm_head_train_mode='none', lora_config=None,
        resolved_learning_rates=meta['resolved_learning_rates'], target_layers=meta['target_layers'],
        target_modules=meta['target_modules'], immutable_resume_contract=meta['immutable_resume_contract'],
        finalized_status={'sparse_bit_committed': False, 'decoder_finalized': True, 'lora_finalized': True,
            'aux_finalized': True, 'runtime_clean': True, 'inference_forward_parity': True},
        runtime_audit=report, base_model_path=base['base_model_path'], tokenizer=tokenizer, save_config=True)
    require(all(sha256(p) == report['source_files_sha256'][str(p)] for p in files), 'Export modified a saved input.')
    check_stat(report['source_weight_stat'])
    report.update(status='exported', checkpoint_id=result['checkpoint_id'], elapsed_seconds=time.time() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device), fresh_strict_state_audit_pending=True)
    write_json(output / 'export_record.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_dir', required=True); parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    export_checkpoint(args.checkpoint_dir, args.output_dir, args.device)
