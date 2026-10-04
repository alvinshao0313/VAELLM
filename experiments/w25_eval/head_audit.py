"""Audit the sole updated head, unchanged compressed storage and complete fresh reload."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import torch
from transformers import AutoConfig

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec, _validate_export_storage
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.w25_eval.head_state import HEAD_KEY, audit_head_state, inspect_step, load_head_state
from experiments.w25_eval.subset_state import checkpoint_inventory, persistent_inventory


def audit_checkpoint(source_dir, checkpoint_dir, output, *, training_checkpoint_dir):
    from train_utils.checkpoint_v6 import _atomic_write_json
    from train_utils.model_level_trainables import iter_named_backbone_norm_modules
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    source, final, output, training = map(lambda p: Path(p).resolve(),
                                        (source_dir, checkpoint_dir, output, training_checkpoint_dir))
    _require(source != final and not output.exists(), 'Use distinct source/final and a new audit file.')
    report = {'status': 'failed', 'source_dir': str(source), 'checkpoint_dir': str(final),
        'training_checkpoint_dir': str(training), 'audit_script_sha256': _sha256(Path(__file__)),
        'scope': 'CPU full-state/storage, saved FP32 head projection and strict reload; no quality claim'}
    started = time.perf_counter()
    try:
        source_meta, source_budget, architecture = checkpoint_inventory(source)
        final_meta, final_budget, _ = checkpoint_inventory(final, architecture)
        meta, state, _snapshot, step_source, _base, step_architecture = inspect_step(training)
        _require(step_source == source and step_architecture == architecture
                 and meta['round_base_checkpoint_id'] == source_meta['checkpoint_id'], 'Selected source identity differs.')
        _require(final_meta['immutable_resume_contract'] == meta['immutable_resume_contract']
                 and final_meta['target_layers'] == meta['target_layers']
                 and final_meta['target_modules'] == meta['target_modules'], 'Final/step native contracts differ.')
        _require(final_meta['checkpoint_id'] not in (source_meta['checkpoint_id'], meta['checkpoint_id'])
                 and final_meta['train_mode'] == 'none' and final_meta['lm_head_train_mode'] == 'full'
                 and final_meta['norm_train_mode'] == 'none' and final_meta['lora_config'] is None
                 and not final_meta.get('post_norm_head_linear'), 'Unexpected final identity/head topology.')
        _require({x['name']: _inference_spec(x) for x in source_meta['converted_modules']}
                 == {x['name']: _inference_spec(x) for x in final_meta['converted_modules']},
                 'Frozen projection inference specifications changed.')
        original, actual = _load_state(source), _load_state(final)
        config = AutoConfig.from_pretrained(architecture['base_model_path'], local_files_only=True)
        _require(list(original[HEAD_KEY].shape) == [config.vocab_size, config.hidden_size]
                 and not config.tie_word_embeddings, 'Head differs from the actual untied model architecture.')
        # Keep CUDA visible for the existing FA2-aware CPU map loader.
        model, loaded, _ = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True, expected_kind='final_model')
        _require(loaded['checkpoint_id'] == source_meta['checkpoint_id'], 'Source changed during strict load.')
        inventory, banks = persistent_inventory(model, architecture)
        norm_keys = {name + '.' + key for name, module in iter_named_backbone_norm_modules(model) for key in module.state_dict()}
        report['source_reloaded_tensors_compared'] = _assert_state_equal(model.state_dict(), original, 'Source strict load')
        del model
        gc.collect()
        mutable, _manifest = load_head_state(training)
        report.update(audit_head_state(original, actual, inventory, banks, mutable[HEAD_KEY]))
        report['frozen_backbone_norm_tensors_compared'] = _assert_state_equal(
            {key: actual[key] for key in norm_keys}, {key: original[key] for key in norm_keys}, 'All norms frozen')
        _validate_export_storage(source_budget, final_budget, report)
        _require(source_budget['components'] == final_budget['components']
                 and source_budget['user_linear_bpw'] == final_budget['user_linear_bpw'], 'Head-only storage budget changed.')
        reloaded, loaded, _ = load_v6_model_checkpoint(str(final), map_location='cpu', strict=True, expected_kind='final_model')
        _require(loaded['checkpoint_id'] == final_meta['checkpoint_id'] and type(reloaded.lm_head) is torch.nn.Linear
                 and not reloaded.config.tie_word_embeddings
                 and reloaded.lm_head.weight.data_ptr() != reloaded.get_input_embeddings().weight.data_ptr(),
                 'Fresh final head topology/identity differs.')
        _require(persistent_inventory(reloaded, architecture) == (inventory, banks), 'Fresh compressed inventory changed.')
        compared = _assert_state_equal(reloaded.state_dict(), actual, 'Fresh strict reload versus every saved tensor')
        report.update(status='verified', architecture=architecture, source_checkpoint_id=source_meta['checkpoint_id'],
            checkpoint_id=final_meta['checkpoint_id'], selected_checkpoint_id=meta['checkpoint_id'],
            selected_global_step=state['global_step'], selected_training_state_sha256=_sha256(training / 'training_model_state.pt'),
            source_bit_budget=source_budget, bit_budget=final_budget, projection_count=len(inventory),
            serialized_tensors_compared=len(actual), reloaded_tensors_compared=compared,
            fresh_strict_reload_state_exact=True, rank_zero=True, actual_tensor_storage_unchanged=True,
            inference_specs_unchanged_except_activation_checkpointing=True)
        return report
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        _atomic_write_json(report, str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source_dir', 'checkpoint_dir', 'output', 'training_checkpoint_dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    audit_checkpoint(args.source_dir, args.checkpoint_dir, args.output, training_checkpoint_dir=args.training_checkpoint_dir)


if __name__ == '__main__':
    main()
