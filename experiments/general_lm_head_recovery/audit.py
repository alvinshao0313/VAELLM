"""Audit native all-decoder/full-head exports against their selected FP32 union."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import torch

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec, _validate_export_storage
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.w25_eval.norm_state import norm_parameters
from experiments.qwen3_w25.e2e_fullhead_audit import audit_fullhead_state
from experiments.general_lm_head_recovery.state import HEAD_KEY, load_union_state
from experiments.w25_eval.subset_audit import audit_subset_training_state
from experiments.w25_eval.subset_state import checkpoint_inventory, persistent_inventory, selected_names


def audit_union_state(original, actual, inventory, banks, mutable, norm_keys):
    decoder_keys = {key for values in inventory.values() for key in values}
    _require(set(mutable) == decoder_keys | {HEAD_KEY}, 'Unexpected mutable decoder/head union.')
    # Preserve the existing selected-FP32 decoder audit while the existing dense
    # head audit verifies its complete row-chunked cast, storage and changes.
    report = audit_subset_training_state(
        {key: value for key, value in original.items() if key != HEAD_KEY},
        {key: value for key, value in actual.items() if key != HEAD_KEY},
        inventory, banks, names=tuple(inventory),
        mutable={key: mutable[key] for key in decoder_keys}, packed_snapshot=None, train_mode='decoder')
    report.update(audit_fullhead_state(original, actual, inventory, banks, mutable[HEAD_KEY]))
    count = _assert_state_equal({key: original[key] for key in norm_keys},
                               {key: actual[key] for key in norm_keys}, 'Every backbone norm remains frozen')
    _require(report['frozen_tensors_compared'] == len(original) - len(mutable)
             and report['changed_code_bits'] == 0, 'Frozen complement or code state differs.')
    report.update(norm_train_mode='none', lm_head_train_mode='full', mutable_tensor_count=len(mutable),
                  frozen_norm_tensors=count, all_packed_banks_match_source_exactly=True)
    return report


def audit_checkpoint(source_dir, checkpoint_dir, output, *, training_checkpoint_dir):
    from experiments.general_lm_head_recovery.state import inspect_step
    from train_utils.checkpoint_v6 import _atomic_write_json
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    source, final, output, training = (Path(path).resolve() for path in
        (source_dir, checkpoint_dir, output, training_checkpoint_dir))
    _require(source != final and not output.exists(), 'Use distinct source/final and a new audit output.')
    report = {'status': 'failed', 'source_dir': str(source), 'checkpoint_dir': str(final),
        'training_checkpoint_dir': str(training), 'audit_script_sha256': _sha256(Path(__file__)),
        'scope': 'CPU saved FP32 decoder/head union, BF16 deployment, frozen complement and fresh strict reload'}
    started = time.perf_counter()
    try:
        source_meta, source_budget, architecture = checkpoint_inventory(source)
        final_meta, final_budget, _ = checkpoint_inventory(final, architecture)
        meta, state, _snapshot, step_source, _base, step_architecture = inspect_step(training)
        _require(Path(step_source).resolve() == source and step_architecture == architecture
                 and meta['round_base_checkpoint_id'] == source_meta['checkpoint_id'], 'Saved source identity differs.')
        _require(final_meta['immutable_resume_contract'] == meta['immutable_resume_contract']
                 and final_meta['target_layers'] == meta['target_layers']
                 and final_meta['target_modules'] == meta['target_modules'], 'Final and native step contracts differ.')
        _require(final_meta['checkpoint_id'] not in (source_meta['checkpoint_id'], meta['checkpoint_id'])
                 and Path(final_meta['base_model_path']).resolve() == Path(source_meta['base_model_path']).resolve()
                 and final_meta['train_mode'] == 'decoder' and final_meta['norm_train_mode'] == 'none'
                 and final_meta['lm_head_train_mode'] == 'full' and final_meta['lora_config'] is None
                 and not final_meta.get('post_norm_head_linear'), 'Final identity or trainable topology differs.')
        _require({item['name']: _inference_spec(item) for item in source_meta['converted_modules']}
                 == {item['name']: _inference_spec(item) for item in final_meta['converted_modules']},
                 'Projection rotation/protection/decoder specification changed.')
        original, actual = _load_state(source), _load_state(final)
        model, loaded, _ = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True,
                                                    expected_kind='final_model')
        _require(loaded['checkpoint_id'] == source_meta['checkpoint_id'], 'Source changed during strict load.')
        inventory, banks = persistent_inventory(model, architecture)
        shapes = {key: list(value.shape) for key, value in norm_parameters(model).items()}
        names = selected_names(meta['target_layers'], meta['target_modules'], architecture)
        _require(set(names) == set(inventory), 'The selected projection inventory is not complete.')
        report['source_reloaded_tensors_compared'] = _assert_state_equal(model.state_dict(), original, 'Source strict load')
        del model
        gc.collect()
        decoder_shapes = {key: list(original[key].shape) for values in inventory.values() for key in values}
        mutable, manifest = load_union_state(training, decoder_shapes, list(original[HEAD_KEY].shape))
        report.update(audit_union_state(original, actual, inventory, banks, mutable, shapes))
        embedding_key = 'model.embed_tokens.weight'
        _assert_state_equal({embedding_key: original[embedding_key]}, {embedding_key: actual[embedding_key]}, 'Frozen native embedding')
        report['embedding_frozen_exact'] = True
        report['outside_head_embedding_storage_bytes'] = {key: actual[key].untyped_storage().nbytes()
                                                        for key in (HEAD_KEY, embedding_key)}
        _validate_export_storage(source_budget, final_budget, report)
        _require(source_budget['components'] == final_budget['components']
                 and source_budget['user_linear_bits'] == final_budget['user_linear_bits']
                 and source_budget['user_linear_bpw'] == final_budget['user_linear_bpw'],
                 'The decoder/head update changed deployment storage or the bit budget.')
        reloaded, loaded, _ = load_v6_model_checkpoint(str(final), map_location='cpu', strict=True,
                                                      expected_kind='final_model')
        _require(loaded['checkpoint_id'] == final_meta['checkpoint_id']
                 and persistent_inventory(reloaded, architecture) == (inventory, banks)
                 and {key: list(value.shape) for key, value in norm_parameters(reloaded).items()} == shapes,
                 'Fresh deployment identity or native inventories changed.')
        _require(type(reloaded.lm_head) is torch.nn.Linear and not reloaded.config.tie_word_embeddings,
                 'The original untied Linear head topology changed.')
        compared = _assert_state_equal(reloaded.state_dict(), actual, 'Fresh strict reload versus every saved tensor')
        report.update(status='verified', architecture=architecture,
            source_checkpoint_id=source_meta['checkpoint_id'], checkpoint_id=final_meta['checkpoint_id'],
            selected_checkpoint_id=meta['checkpoint_id'], selected_global_step=state['global_step'],
            selected_training_state_sha256=_sha256(training / 'training_model_state.pt'),
            component_counts={'decoder': 6 * len(inventory), 'lm_head': 1},
            source_bit_budget=source_budget, bit_budget=final_budget, projection_count=len(inventory),
            serialized_tensors_compared=len(actual), reloaded_tensors_compared=compared,
            fresh_strict_reload_state_exact=True, rank_zero=True, native_decoder_dtype='torch.bfloat16',
            native_head_dtype='torch.bfloat16', actual_tensor_storage_unchanged=True,
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
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit_checkpoint(args.source_dir, args.checkpoint_dir, args.output,
                     training_checkpoint_dir=args.training_checkpoint_dir)


if __name__ == '__main__':
    main()
