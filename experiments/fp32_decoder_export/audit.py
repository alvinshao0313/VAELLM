"""Audit real FP32 decoder storage and normal BF16 inference against the same native step."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from experiments.fp32_decoder_export.export import inspect_scope, precision_inventory, rope_record
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
from experiments.qwen3_w25.e2e_subset_export import _sha256
from experiments.w25_eval.subset_state import checkpoint_inventory, persistent_inventory


def read(path):
    return json.loads(Path(path).read_text())


def check_export_inputs(record):
    for key in ('code_sha256', 'source_files_sha256'):
        for path, digest in record[key].items():
            _require(_sha256(path) == digest, 'Bound source/input changed: ' + path)
    item = record['round_base_weight_stat']
    value = Path(item['path']).stat()
    _require((value.st_size, value.st_mtime_ns, value.st_ino) ==
             (item['size'], item['mtime_ns'], item['inode']), 'Round-base weight identity changed.')


def audit_checkpoint(checkpoint, training_checkpoint, bf16_checkpoint, output, *, device='cuda:0'):
    from compressed_e2e_fintuning.runtime_v6_pipeline import _assert_finalization_probe_close, _run_finalization_probe
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from tools.cat_eval import _prepare_model_for_eval
    from train_utils.checkpoint_v6 import _atomic_write_json, load_v6_training_model_state
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    checkpoint, training_checkpoint, bf16_checkpoint, output = (
        Path(p).resolve() for p in (checkpoint, training_checkpoint, bf16_checkpoint, output))
    _require(not output.exists() and checkpoint != bf16_checkpoint, 'Use independent variants and a new audit record.')
    meta, state, snapshot, source, base, architecture = inspect_scope(training_checkpoint)
    device = torch.device(device)
    _require(torch.cuda.is_available() and device.type == 'cuda' and torch.cuda.device_count() == 1,
             'The new save/load/prewarm path requires one explicitly assigned CUDA device.')
    started = time.perf_counter()
    report = {'status': 'failed', 'scope': 'CPU exact state/storage plus fresh v6 normal BF16 group8 prewarm and four-token inference; no training or quality claim.',
        'checkpoint_dir': str(checkpoint), 'training_checkpoint_dir': str(training_checkpoint),
        'bf16_checkpoint_dir': str(bf16_checkpoint), 'source_dir': str(source),
        'source_global_step': state['global_step'], 'optimizer_steps_executed': 0}
    try:
        export = read(checkpoint / 'export_record.json')
        control_export = read(bf16_checkpoint / 'export_record.json')
        _require(export['status'] == control_export['status'] == 'exported'
                 and Path(export['source_checkpoint']).resolve() == training_checkpoint
                 and Path(control_export['source_checkpoint']).resolve() == training_checkpoint
                 and export['source_checkpoint_id'] == control_export['source_checkpoint_id'] == meta['checkpoint_id']
                 and export['source_global_step'] == control_export['source_global_step'] == 7200,
                 'Both storage variants must come from exactly the same real native step.')
        check_export_inputs(export)
        for name in ('checkpoint_meta.json', 'trainer_state.json', 'training_model_state.pt'):
            path = str(training_checkpoint / name)
            _require(export['source_files_sha256'][path] == control_export['source_files_sha256'][path],
                     'The BF16 control binds a different native input: ' + name)
        final_meta, final_budget, _ = checkpoint_inventory(checkpoint, architecture)
        control_meta, control_budget, _ = checkpoint_inventory(bf16_checkpoint, architecture)
        source_meta, source_budget, _ = checkpoint_inventory(source, architecture)
        _require(final_meta['checkpoint_id'] == export['checkpoint_id']
                 and control_meta['checkpoint_id'] == control_export['checkpoint_id']
                 and len({final_meta['checkpoint_id'], control_meta['checkpoint_id'], meta['checkpoint_id'], base['checkpoint_id']}) == 4,
                 'Storage variants must have independent v6 identities.')
        for row in (final_meta, control_meta):
            _require(row['immutable_resume_contract'] == meta['immutable_resume_contract']
                     and row['train_mode'] == 'decoder' and row['norm_train_mode'] == row['lm_head_train_mode'] == 'none'
                     and row['lora_config'] is None and not row.get('post_norm_head_linear')
                     and row['target_layers'] == meta['target_layers'] and row['target_modules'] == meta['target_modules'],
                     'Export changed the native topology or immutable training contract.')
        normalized_specs = lambda value: {row['name']: _inference_spec(row, export_decoder_dtype=True)
                                           for row in value['converted_modules']}
        _require(normalized_specs(final_meta) == normalized_specs(control_meta) == normalized_specs(source_meta),
                 'Only decoder storage dtype and activation-checkpoint policy may differ.')
        actual, original, control = (_load_state(p) for p in (checkpoint, source, bf16_checkpoint))
        _require(actual.keys() == original.keys() == control.keys() and len(actual) == 3423, 'Changed complete tensor inventory.')
        mutable, manifest = load_v6_training_model_state(str(training_checkpoint), map_location='cpu')
        keys = set(mutable)
        _require(len(keys) == len(manifest) == 1512
                 and {row['name']: row['component_class'] for row in manifest} == {k: 'decoder' for k in keys}
                 and all('._parallel_stage_decoder.' in k for k in keys), 'Native mutable set must contain all and only real decoder tensors.')
        _require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in mutable.values()),
                 'Native decoder values must really be finite FP32, not promoted BF16 exports.')
        count = _assert_state_equal({k: actual[k] for k in keys}, mutable, 'Saved FP32 decoder versus true native mutable state')
        retained_elements = sum(int(torch.count_nonzero(value != value.bfloat16().float())) for value in mutable.values())
        _require(retained_elements > 0, 'Native FP32 state contains no extra precision; there is no retention contrast.')
        _assert_state_equal({k: control[k] for k in keys}, {k: value.bfloat16() for k, value in mutable.items()},
                            'Same-step BF16 control versus native projection')
        frozen = {k: value for k, value in original.items() if k not in keys}
        frozen_count = _assert_state_equal({k: actual[k] for k in frozen}, frozen, 'All non-decoder source bytes')
        _assert_state_equal({k: control[k] for k in frozen}, frozen, 'BF16 control non-decoder source bytes')
        _require(frozen_count == 1911, 'Incomplete complement including all 504 uint8 code banks.')
        _require(all(original[k].dtype == control[k].dtype == torch.bfloat16 for k in keys), 'This contrast requires the BF16 source/control inventory.')
        elements = sum(mutable[k].numel() for k in keys)
        _require(elements == 4633344 and final_budget['original_linear_weights'] == 6945767424,
                 'Unexpected actual Qwen decoder/Linear element inventory.')
        extra = elements * 16
        _require(control_budget['user_linear_bits'] == source_budget['user_linear_bits']
                 and final_budget['user_linear_bits'] == control_budget['user_linear_bits'] + extra
                 and final_budget['packed_linear_bits'] == control_budget['packed_linear_bits'] + extra,
                 'Integer storage delta differs from exactly BF16-to-FP32 decoder storage.')
        for component in set(final_budget['components']) | set(control_budget['components']):
            if component != 'decoder':
                _require(final_budget['components'][component] == control_budget['components'][component],
                         'Non-decoder storage changed: ' + component)
        _require(final_budget['full_model_tensor_bytes'] == control_budget['full_model_tensor_bytes'] + extra // 8
                 and final_budget['budget_passed'], 'Full serialized storage delta or 2.5bit budget failed.')
        training = SimpleNamespace(**snapshot['training_args'])
        configure_e2e_determinism(bool(training.full_determinism)); set_e2e_seed(training.seed)
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        if training.tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = bool(training.tf32)
            torch.backends.cudnn.allow_tf32 = bool(training.tf32)
        model, loaded, result = load_v6_model_checkpoint(str(checkpoint), map_location='cpu', strict=True, expected_kind='final_model')
        _require(loaded['checkpoint_id'] == final_meta['checkpoint_id'] and not result.missing_keys and not result.unexpected_keys,
                 'Fresh v6 strict reconstruction failed.')
        reload_count = _assert_state_equal(model.state_dict(), actual, 'All serialized tensors versus fresh strict model')
        inventory, banks = persistent_inventory(model, architecture)
        _require({key for values in inventory.values() for key in values} == keys
                 and sum(len(v) for v in banks.values()) == 504, 'Loaded decoder/code topology changed.')
        precision = precision_inventory(model, keys)
        _require(precision == export['precision_after_finalization'], 'Fresh loading changed parameter or FP32 RoPE precision/state.')
        model.eval().requires_grad_(False); model.config.use_cache = False
        _require(getattr(model, '_distill_model_compute_dtype', None) is None,
                 'Use the real loader/evaluation activation policy, not a diagnostic override.')
        warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
        _require(warmup['total'] == warmup['warmed'] == 252 and warmup['failed'] == 0
                 and warmup['prewarm_dtype'] == 'torch.bfloat16', 'Normal group8 BF16 prewarm did not cover the model.')
        probe_path = checkpoint / 'deployment_probe.pt'
        _require(_sha256(probe_path) == export['deployment_probe_sha256'], 'Saved pre-export inference probe changed.')
        probe = torch.load(probe_path, map_location='cpu', weights_only=True)
        logits, dtype = _run_finalization_probe(model, probe['inputs'])
        _require(dtype == torch.bfloat16 and probe['output_dtype'] == str(dtype), 'Normal inference output dtype changed.')
        parity = _assert_finalization_probe_close(probe['logits'], logits, output_dtype=dtype,
                                                 label='Saved FP32 decoder versus fresh normal prewarm inference')
        _require(rope_record(model) == precision['rope'], 'Normal inference changed original FP32 RoPE.')
        model.to('cpu')
        _assert_state_equal(model.state_dict(), actual, 'Persistent bytes after normal prewarm/inference')
        check_export_inputs(export)
        report.update(status='verified', checkpoint_id=final_meta['checkpoint_id'], bf16_checkpoint_id=control_meta['checkpoint_id'],
            selected_checkpoint_id=meta['checkpoint_id'], source_checkpoint_id=base['checkpoint_id'],
            fp32_native_decoder_tensors_exact=count, fp32_decoder_elements=elements, frozen_source_tensors_exact=frozen_count,
            native_values_distinct_from_bf16_projection=retained_elements,
            frozen_code_banks=504, fresh_strict_tensors_exact=reload_count, same_native_bf16_cast_exact=True,
            decoder_dtype='torch.float32', compute_dtype='torch.bfloat16', precision=precision,
            source_bit_budget=source_budget, bf16_bit_budget=control_budget, bit_budget=final_budget,
            added_storage_bits=extra, added_storage_bytes=extra // 8, prewarm=warmup,
            fresh_normal_inference_parity=parity, persistent_state_after_inference_exact=True,
            optimizer_or_training_executed=False, full_downstream_evaluation_pending=True,
            export_record_sha256=_sha256(checkpoint / 'export_record.json'),
            bf16_export_record_sha256=_sha256(bf16_checkpoint / 'export_record.json'),
            native_training_model_state_sha256=_sha256(training_checkpoint / 'training_model_state.pt'),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        return report
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        _atomic_write_json(report, str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint_dir', 'training_checkpoint_dir', 'bf16_checkpoint_dir', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    result = audit_checkpoint(args.checkpoint_dir, args.training_checkpoint_dir, args.bf16_checkpoint_dir,
                              args.output, device=args.device)
    print({key: result[key] for key in ('status', 'checkpoint_id', 'fp32_native_decoder_tensors_exact', 'added_storage_bits')})
