"""Native all-D step projection/frozen-state audit for the fixed13bd geometry."""
import argparse
import gc
from pathlib import Path
import time

import torch

from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from experiments.qwen3_w25.audit_e2e_decoder import _decoder_storage, _inference_spec, _validate_export_storage
from experiments.llama_s4_lm_validation.state import SOURCE, SOURCE_ID, checkpoint_inventory, inspect_step, persistent_inventory
from train_utils.checkpoint_v6 import load_v6_training_model_state
from train_utils.v6_model_loader import load_v6_model_checkpoint


def compare_state(original, actual, inventory, banks, mutable, manifest):
    decoder = {k for keys in inventory.values() for k in keys}
    codes = {k for rows in banks.values() for k, _ in rows}
    require(len(original) == len(actual) == 3043 and original.keys() == actual.keys()
            and len(decoder) == 1344 and len(codes) == 512 and not decoder & codes
            and decoder | codes <= original.keys(), 'Complete mixed-stage inventory differs.')
    require(len(manifest) == len(decoder) and set(mutable) == decoder
            and {r['name']: r['component_class'] for r in manifest} == {k: 'decoder' for k in decoder},
            'Native mutable state is not exactly1344 decoder leaves.')
    for key in original:
        require(original[key].shape == actual[key].shape and original[key].dtype == actual[key].dtype,
                'Saved schema changed: ' + key)
    frozen = {k: v for k, v in original.items() if k not in decoder}
    count = _assert_state_equal({k: actual[k] for k in frozen}, frozen, 'Every1699 non-decoder tensor including512 banks')
    fp32, bf16 = {}, {}
    for key in sorted(decoder):
        value = mutable[key]
        require(value.dtype == torch.float32 and value.shape == original[key].shape
                and original[key].dtype == actual[key].dtype == torch.bfloat16
                and bool(torch.isfinite(value).all()) and bool(torch.isfinite(original[key]).all()),
                'Expected finite original BF16 and mutable FP32 decoder: ' + key)
        projected = value.bfloat16()
        require(bool(torch.isfinite(projected).all()), 'BF16 decoder overflow: ' + key)
        _assert_state_equal({key: actual[key]}, {key: projected}, 'Exact native FP32-to-BF16 projection')
        fp32[key] = int(torch.count_nonzero(value != original[key].float()))
        bf16[key] = int(torch.count_nonzero(actual[key] != original[key]))
    require(sum(fp32.values()) > 0, 'No real FP32 decoder update was saved.')
    before, after = (_decoder_storage(s, decoder) for s in (original, actual))
    require(before == after, 'Grouped decoder storage changed.')
    return {'selected_modules': list(inventory), 'selected_module_count': len(inventory),
        'decoder_tensor_count': len(decoder), 'selected_decoder_tensor_count': len(decoder),
        'selected_decoder_fp32_changed_elements': sum(fp32.values()),
        'selected_decoder_fp32_changed_elements_by_key': fp32, 'changed_decoder_elements': sum(bf16.values()),
        'changed_decoder_tensor_count': sum(v > 0 for v in bf16.values()),
        'selected_decoder_matches_fp32_cast_exactly': True, 'selected_decoder_update_survived_bf16': sum(bf16.values()) > 0,
        'frozen_tensors_compared': count, 'frozen_bytes_unchanged': True, 'code_bank_count': len(codes),
        'frozen_code_bank_count': len(codes), 'selected_projection_code_bank_count': len(codes), 'selected_code_bank_count': 0,
        'changed_code_bits': 0, 'stage_changed_bits': {n: [0] * len(rows) for n, rows in banks.items()},
        'packed_padding_unchanged': True, 'selected_codes_match_frozen_source_exactly': True,
        'decoder_storage_source': before, 'decoder_storage_target': after, 'decoder_storage_saved_bits': 0,
        'decoder_storage_source_bits': before['storage_bits'], 'decoder_storage_target_bits': after['storage_bits']}


def audit(source_dir, checkpoint_dir, output, *, training_checkpoint_dir):
    source, final, output, step = (Path(p).resolve() for p in (source_dir, checkpoint_dir, output, training_checkpoint_dir))
    require(source == SOURCE and source != final and not output.exists(), 'Use original13bd and a new independent audit.')
    start = time.time(); report = {'status': 'failed', 'source_dir': str(source), 'checkpoint_dir': str(final),
        'training_checkpoint_dir': str(step), 'audit_script_sha256': sha256(__file__)}
    try:
        sm, sb, architecture = checkpoint_inventory(source); fm, fb, _ = checkpoint_inventory(final, architecture)
        tm, ts, _snapshot, base, _bm, step_arch = inspect_step(step)
        require(sm['checkpoint_id'] == SOURCE_ID and base == source and architecture == step_arch
                and fm['checkpoint_id'] not in (SOURCE_ID, tm['checkpoint_id'])
                and fm['immutable_resume_contract'] == tm['immutable_resume_contract']
                and fm['target_layers'] == tm['target_layers'] and fm['target_modules'] == tm['target_modules']
                and fm['train_mode'] == tm['train_mode'] == 'decoder' and fm['norm_train_mode'] == fm['lm_head_train_mode'] == 'none'
                and fm['lora_config'] is None and not fm.get('post_norm_head_linear'), 'Source/step/final contract differs.')
        require([_inference_spec(s) for s in sm['converted_modules']] == [_inference_spec(s) for s in fm['converted_modules']],
                'Decoder or protection/rotation deployment geometry changed.')
        original, actual = _load_state(source), _load_state(final)
        model, loaded, _ = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True, expected_kind='final_model')
        require(loaded['checkpoint_id'] == SOURCE_ID, 'Wrong loaded source identity.')
        inventory, banks = persistent_inventory(model, architecture)
        report['source_reloaded_tensors_compared'] = _assert_state_equal(model.state_dict(), original, 'Source strict load')
        mutable, manifest = load_v6_training_model_state(str(step), map_location='cpu')
        del model
        gc.collect()
        report.update(compare_state(original, actual, inventory, banks, mutable, manifest))
        require(report['frozen_tensors_compared'] == 1699, 'Wrong complete frozen complement.')
        _validate_export_storage(sb, fb, report)
        reloaded, loaded, _ = load_v6_model_checkpoint(str(final), map_location='cpu', strict=True, expected_kind='final_model')
        require(loaded['checkpoint_id'] == fm['checkpoint_id'] and type(reloaded.lm_head) is torch.nn.Linear
                and not reloaded.config.tie_word_embeddings and persistent_inventory(reloaded, architecture) == (inventory, banks),
                'Fresh deployment topology differs.')
        count = _assert_state_equal(reloaded.state_dict(), actual, 'Fresh strict saved state')
        report.update(status='verified', source_checkpoint_id=SOURCE_ID, checkpoint_id=fm['checkpoint_id'],
            selected_checkpoint_id=tm['checkpoint_id'], selected_global_step=ts['global_step'], train_mode='decoder',
            serialized_tensors_compared=len(actual), reloaded_tensors_compared=count, source_bit_budget=sb, bit_budget=fb,
            selected_training_state_sha256=sha256(step / 'training_model_state.pt'), selected_exact_sidecar_sha256=None,
            fresh_strict_reload_state_exact=True, rank_zero=True, sparse_bit_restore=None,
            native_decoder_dtype='torch.bfloat16', inference_specs_unchanged_except_activation_checkpointing=True)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.time() - start; write_json(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source_dir', 'checkpoint_dir', 'output', 'training_checkpoint_dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); audit(args.source_dir, args.checkpoint_dir, args.output, training_checkpoint_dir=args.training_checkpoint_dir)
