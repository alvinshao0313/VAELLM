"""Export all native FP32 backbone norms after training exits, retaining all compressed bytes."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.w25_eval.norm_state import assert_norm_cast, inspect_step, restore_norm_state, norm_parameters
from experiments.w25_eval.subset_state import checkpoint_inventory, persistent_inventory, selected_names

REPO = Path(__file__).resolve().parents[2]
CODE_FILES = (
    'experiments/w25_eval/norm_state.py', 'experiments/w25_eval/norm_export.py',
    'experiments/w25_eval/norm_audit.py', 'experiments/w25_eval/model_inventory.py',
    'experiments/w25_eval/subset_state.py', 'experiments/w25_eval/acceptance.py',
    'experiments/qwen3_w25/assemble_categories.py',
    'experiments/qwen3_w25/audit_e2e_decoder.py',
    'compressed_e2e_fintuning/runtime_v6.py', 'compressed_e2e_fintuning/runtime_v6_pipeline.py',
    'compressed_e2e_fintuning/v6_runtime_state.py', 'train_utils/model_level_trainables.py',
    'train_utils/model_level_checkpoint_state.py', 'train_utils/distill_precision.py',
    'train_utils/checkpoint_v6.py', 'train_utils/v6_model_loader.py',
    'litebsq/vae_linear.py', 'litebsq/fused_multistage_decoder.py',
)


def export_checkpoint(checkpoint, output_dir, *, device='cuda:0'):
    from compressed_e2e_fintuning.runtime_v6 import _assert_final_runtime_clean, _build_v6_tokenizer, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from litebsq.vae_linear import clear_model_vae_linear_cache
    from train_utils.checkpoint_v6 import _atomic_write_json, save_v6_full_checkpoint
    from train_utils.distill_precision import prepare_model_export
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    checkpoint, output = Path(checkpoint).resolve(), Path(output_dir).resolve()
    _require(not output.exists() and int(os.environ.get('WORLD_SIZE', '1')) == 1,
             'Export requires one process and a new output directory.')
    meta, state, snapshot, source, base_meta, architecture = inspect_step(checkpoint)
    training = SimpleNamespace(**snapshot['training_args'])
    device = torch.device(device)
    _require(device.type == 'cuda' and torch.cuda.is_available(), 'Native finalization probes require CUDA.')
    started = time.perf_counter()
    configure_e2e_determinism(bool(training.full_determinism))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_e2e_seed(training.seed)
    if training.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(training.tf32)
        torch.backends.cudnn.allow_tf32 = bool(training.tf32)
    source_meta, source_budget, _ = checkpoint_inventory(source, architecture)
    _require(source_budget['budget_passed'], 'Source violates the rank-zero 2.5 bpw budget.')
    files = [checkpoint / name for name in ('checkpoint_meta.json', 'trainer_state.json', 'training_model_state.pt')]
    files += [checkpoint.parent.parent / 'normalized_e2e_runtime_args.json',
              source / 'pytorch_model.bin', source / 'checkpoint_meta.json']
    audit = {'runtime': 'experiments.w25_eval.norm_export', 'source_checkpoint': str(checkpoint),
        'source_checkpoint_id': meta['checkpoint_id'], 'round_base_dir': str(source),
        'round_base_checkpoint_id': source_meta['checkpoint_id'], 'source_global_step': state['global_step'],
        'source_max_steps': state['max_steps'], 'architecture': architecture,
        'train_mode': 'none', 'norm_train_mode': 'all', 'lm_head_train_mode': 'none',
        'optimizer_steps_executed': 0, 'trainer_or_main_optimizer_constructed': False,
        'teacher_or_training_data_loaded': False,
        'source_files_sha256': {str(p): _sha256(p) for p in files},
        'code_sha256': {name: _sha256(REPO / name) for name in CODE_FILES}}
    model, loaded, _ = load_v6_model_checkpoint(str(source), strict=True, map_location='cpu')
    _require(loaded['checkpoint_id'] == meta['round_base_checkpoint_id'], 'Loaded source UUID changed.')
    inventory, banks = persistent_inventory(model, architecture)
    names = selected_names(meta['target_layers'], meta['target_modules'], architecture)
    selected = tuple((name, model.get_submodule(name)) for name in names)
    audit['mutable_restore'] = restore_norm_state(model, selected, checkpoint=checkpoint, snapshot=snapshot)
    audit.update(deployed_projection_count=len(inventory), frozen_decoder_tensors=sum(map(len, inventory.values())),
                 frozen_code_banks=sum(map(len, banks.values())), decoder_finalization_performed=False,
                 head_fusion_performed=False)
    tokenizer = _build_v6_tokenizer(base_meta['base_model_path'], SimpleNamespace(**snapshot['hf_args']))
    _sync_model_padding_config(model, tokenizer)
    model.config.use_cache = False
    model.eval().to(device)
    _assert_final_runtime_clean(model)
    probe = _build_finalization_probe_inputs(tokenizer)
    before, dtype = _run_finalization_probe(model, probe)
    # Native train_mode=none does not finalize/repack any decoder or fuse a head.
    core, core_dtype = _run_finalization_probe(model, probe)
    _require(core_dtype == dtype, 'Native structural finalization changed compute dtype.')
    audit['core_structural_finalization_forward_parity'] = _assert_finalization_probe_close(
        before, core, output_dtype=dtype, label='Norm-only native no-op structural finalization')
    selected_norms = {key: value.detach().cpu().clone() for key, value in norm_parameters(model).items()}
    audit['export_dtype'] = str(prepare_model_export(model, training))
    after, export_dtype = _run_finalization_probe(model, probe)
    _require(export_dtype == dtype, 'Export changed output compute dtype.')
    assert_norm_cast(norm_parameters(model), selected_norms)
    delta = after - core
    audit['export_precision_probe'] = {'output_dtype': str(dtype), 'max_abs': float(delta.abs().max()),
        'relative_l2': float(delta.norm() / core.norm().clamp_min(1e-12))}
    audit['norm_matches_selected_fp32_cast_exactly'] = True
    del selected_norms
    clear_model_vae_linear_cache(model)
    _assert_final_runtime_clean(model)
    model.requires_grad_(False).to('cpu')
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        **{key: tuple(base_meta.get(key) or ()) for key in ('compressed_targets', 'pending_dense_targets', 'skip_targets',
             'legacy_original_only_sources', 'completed_categories', 'compression_categories')},
        train_mode='none', norm_train_mode='all', lm_head_train_mode='none', lora_config=None,
        resolved_learning_rates=meta['resolved_learning_rates'], target_layers=meta['target_layers'],
        target_modules=meta['target_modules'], immutable_resume_contract=meta['immutable_resume_contract'],
        base_model_path=base_meta['base_model_path'], tokenizer=tokenizer, save_config=True,
        finalized_status={'sparse_bit_committed': False, 'decoder_finalized': False, 'lora_finalized': False,
                          'aux_finalized': True, 'runtime_clean': True, 'inference_forward_parity': True},
        runtime_audit=audit)
    _require(saved['checkpoint_id'] not in (meta['checkpoint_id'], base_meta['checkpoint_id']), 'Export reused an input UUID.')
    result = {'status': 'exported', 'checkpoint_id': saved['checkpoint_id'], 'output_dir': str(output),
        'elapsed_seconds': time.perf_counter() - started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'fresh_strict_state_audit_pending': True, 'full_downstream_evaluation_pending': True, **audit}
    _atomic_write_json(result, str(output / 'export_record.json'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_dir', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    report = export_checkpoint(args.checkpoint_dir, args.output_dir, device=args.device)
    print(json.dumps({key: report[key] for key in ('status', 'checkpoint_id', 'source_global_step', 'output_dir')}))


if __name__ == '__main__':
    main()
