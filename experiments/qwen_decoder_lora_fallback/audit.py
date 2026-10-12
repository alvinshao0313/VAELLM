"""Compose exact native decoder, LoRA and complement audits without training."""
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import time

import torch

from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
from experiments.qwen_decoder_lora_fallback.state import inspect_step, read, restore_joint
from experiments.w25_eval.lora_audit import load_state
from experiments.w25_eval.subset_audit import audit_subset_training_state
from experiments.w25_eval.subset_state import persistent_inventory, selected_layers, selected_modules
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.config.targets import collect_e2e_compressed_targets
from train_utils.v6_model_loader import load_v6_model_checkpoint


def audit(source_dir, checkpoint_dir, training_checkpoint_dir, output):
    source, final, checkpoint, output = map(lambda p: Path(p).resolve(),
        (source_dir, checkpoint_dir, training_checkpoint_dir, output))
    require(source != final and not output.exists(), 'Distinct source/final and fresh audit required.')
    started = time.time()
    report = {'status': 'failed', 'audit_script_sha256': sha256(Path(__file__))}
    try:
        meta, step, snapshot, actual_source, base, architecture = inspect_step(checkpoint)
        require(source == actual_source, 'Wrong original round base.')
        original, actual = load_state(source), load_state(final)
        final_meta = read(final / 'checkpoint_meta.json')
        require(final_meta['train_mode'] == 'decoder_lora' and final_meta['lora_config'] is None
                and final_meta['immutable_resume_contract'] == meta['immutable_resume_contract']
                and final_meta['target_layers'] == meta['target_layers'] and final_meta['target_modules'] == meta['target_modules']
                and final_meta['norm_train_mode'] == final_meta['lm_head_train_mode'] == 'none'
                and not final_meta.get('post_norm_head_linear') and not (final_meta.get('extra_meta') or {}).get('residual_lora'),
                'Deployment contract/topology differs.')
        stripped = deepcopy(final_meta['converted_modules'])
        for spec in stripped: spec['low_rank_a'] = spec['low_rank_b'] = None
        require(_inference_spec(stripped) == _inference_spec(base['converted_modules']), 'Non-LoRA inference specs changed.')
        source_budget = audit_state_dict(base, original, max_bpw=2.5, max_lora_rank=0)
        budget = audit_state_dict(final_meta, actual, max_bpw=2.5, max_lora_rank=2)
        require(source_budget['budget_passed'] and budget['budget_passed']
                and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17152966656
                and budget['original_linear_weights'] == 6945767424
                and budget['components']['lora']['storage_bits'] == 87293952, 'Actual rank2 deployment budget differs.')
        for component, value in source_budget['components'].items():
            if component != 'lora': require(budget['components'].get(component) == value, 'Component storage changed: ' + component)
        require(budget['full_model_tensor_bytes'] - source_budget['full_model_tensor_bytes'] == 10911744,
                'Added storage is not precisely504 BF16 LoRA factors.')
        model, loaded, _ = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True)
        require(loaded['checkpoint_id'] == base['checkpoint_id'], 'Source reload UUID differs.')
        inventory, banks = persistent_inventory(model, architecture)
        selected = collect_e2e_compressed_targets(model, target_layers=selected_layers(meta['target_layers'], architecture),
            target_modules=selected_modules(meta['target_modules'], architecture), num_layers=architecture['num_hidden_layers'])
        model, decoders, factors = restore_joint(model, selected, checkpoint, meta, snapshot, architecture)
        require(len(original) == 3423 and len(actual) == 3927 and not set(original) & set(factors)
                and set(actual) == set(original) | set(factors), 'Saved union has missing/extra tensors.')
        report.update(audit_subset_training_state(original, {k: actual[k] for k in original}, inventory, banks,
            names=tuple(architecture['projection_shapes']), mutable=decoders, packed_snapshot=None, train_mode='decoder'))
        _assert_state_equal({k: actual[k] for k in factors}, factors, 'Native LoRA scaling and BF16 projection')
        require(all(v.dtype == torch.bfloat16 and v.untyped_storage().nbytes() == v.numel() * 2 for v in factors.values()),
                'LoRA projection has unexpected precision/backing storage.')
        nonzero = sum(int(torch.count_nonzero(v)) for k, v in factors.items() if k.endswith('.low_rank_a'))
        require(nonzero > 0, 'No LoRA B update survives BF16 export.')
        del decoders, factors, selected, model, original
        gc.collect()
        reloaded, reloaded_meta, _ = load_v6_model_checkpoint(str(final), map_location='cpu', strict=True, expected_kind='final_model')
        require(reloaded_meta['checkpoint_id'] == final_meta['checkpoint_id']
                and persistent_inventory(reloaded, architecture) == (inventory, banks), 'Fresh topology/UUID differs.')
        compared = _assert_state_equal(reloaded.state_dict(), actual, 'Every fresh strict joint deployment tensor')
        report.update(status='verified', train_mode='decoder_lora', source_checkpoint_id=base['checkpoint_id'],
            checkpoint_id=final_meta['checkpoint_id'], selected_checkpoint_id=meta['checkpoint_id'], selected_global_step=step['global_step'],
            selected_training_state_sha256=sha256(checkpoint / 'training_model_state.pt'),
            bit_budget=budget, source_bit_budget=source_budget, lora_rank=2, lora_tensor_count=504,
            nonzero_bf16_low_rank_a_elements=nonzero, selected_lora_matches_native_bf16_cast_exactly=True,
            serialized_tensors_compared=len(actual), reloaded_tensors_compared=compared,
            fresh_strict_reload_state_exact=True, optimizer_steps_executed_by_export=0)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.time() - started
        write_json(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source_dir', 'checkpoint_dir', 'training_checkpoint_dir', 'output'): parser.add_argument('--' + key, required=True)
    audit(**vars(parser.parse_args()))
