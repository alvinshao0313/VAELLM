"""Build a fresh raw-source abs005 payload, then save its actual native/Rice bundle."""
from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
import shutil
import time

import sparse_common as c
import sparse_bundle as bundle

t = c.torch


def prepare(profile, output, artifact):
    c.cpu_only()
    t.set_num_threads(4)
    c.require(output == c.OUTPUT / profile and artifact == c.ARTIFACT / profile
              and not output.exists() and not artifact.exists(), 'Use one new fixed-profile output.')
    report_path = c.REPORTS[profile]
    report = c.read(report_path)
    c.require(report['status'] == 'completed' and report['exit_code'] == 0 and report['profile'] == profile,
              'The actual raw source must have completed successfully.')
    entries = [row for row in report['saved_boundaries'] if row['kind'] == 'final_model']
    c.require(len(entries) == 1, 'Require the single actual final-model source from the native report.')
    source = Path(entries[0]['path']).resolve()
    meta = c.read(source / 'checkpoint_meta.json')
    c.require(meta['checkpoint_id'] == entries[0]['checkpoint_id']
              and meta['lora_config'] is None and not meta['pending_dense_targets'] and not meta['skip_targets']
              and len(meta['converted_modules']) == 252, 'Raw source is not a complete rank0 native model.')
    original = t.load(source / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    banks = [key for key, value in original.items() if '.vq_weight' in key and value.dtype == t.uint8]
    from train_utils.checkpoint_bpw import audit_state_dict
    budget = audit_state_dict(meta, original, max_bpw=2.6, max_lora_rank=0)
    c.require(len(original) == 3423 and len(banks) == 504 and budget['budget_passed'] and not budget['contract_violations']
              and not budget['unclassified_target_keys'], 'Require all3423 actual frozen source tensors.')
    names = sorted(row['name'] for row in meta['converted_modules'] if '.mlp.' in row['name'])
    index_path = c.HF / 'model.safetensors.index.json'
    donors = c.read(index_path)['weight_map']
    hf_donors = {name: dict(key=name + '.weight', shard=donors[name + '.weight']) for name in names}
    c.require(len(names) == 108 and all(original[name + '.protected_input_indices'].numel()
              in (64, 256) for name in names), 'Fixed108 protected-input MLP topology differs.')
    _, aux_evidence = c.aux32()
    rice = c.module(c.RICE, '_raw_sparse_cpu_rice7')
    # Verify the existing lossless codec on a real251659-entry supported payload, without using it as a candidate.
    fixture = c.RESULT / 'qwen_b32_sparse_mlp005_probe_20261024/run/payload.pt'
    old = t.load(fixture, map_location='cpu', weights_only=True, mmap=True)
    prefix = names[0] + '.sparse_residual_'
    local, ptr = old[prefix + 'local_indices'], old[prefix + 'block_ptr']
    encoded, _ = rice.encode(local, ptr)
    c.require(t.equal(rice.decode({**encoded, 'block_ptr': ptr}), local), 'Real existing Rice roundtrip failed.')
    paths = {report_path, source / 'checkpoint_meta.json', source / 'config.json', index_path, c.RICE, fixture,
             Path(aux_evidence['provenance_manifest']), *(c.HERE.glob('sparse*.py'))}
    context = c.native_context()
    paths.update(Path(path) for path in context.values())
    paths.update(c.SNAPSHOT / name for name in ('tools/cat_eval.py', 'train_utils/checkpoint_v6.py',
        'train_utils/checkpoint_bpw.py', 'train_utils/v6_model_loader.py', 'litebsq/vae_linear.py',
        'litebsq/vae_linear_prewarm.py', 'litebsq/sparse_residual.py', 'train_utils/cat_train_residual_protection.py'))
    record = dict(schema='raw_CG_abs005_Rice7_prepared_v1', status='CPU_prepared_GPU_not_started',
        profile=profile, source_dir=str(source), source_checkpoint_id=meta['checkpoint_id'],
        source_bit_budget=budget, source_tensors=3423, recipe=c.RECIPE, names=names, HF_donors=hf_donors,
        auxiliary32=aux_evidence, bindings=[c.binding(path) for path in sorted(paths)],
        weight_stats=[c.stamp(source / meta['state_dict_file']),
                      *(c.stamp(c.HF / name) for name in sorted({row['shard'] for row in hf_donors.values()}))],
        native_context=context, CPU_validation=dict(actual_source_mmap_audit=True, real_Rice_encode_decode_exact=True,
            actual_full_RHT4032_factor252=True,
            fixture_is_not_candidate_payload=True, CUDA_initialized=False),
        pending_validation=['actual native GPU BF16 residual construction', 'new payload exact Rice coordinate/value roundtrip',
            'native save/fresh strict3423+864', 'actual full storage accounting',
            'independent108 cache and one auxiliary32 consumer forward', 'full8 and LongBench21'],
        training=False, external_LoRA=False, created_unix=time.time())
    c.write(output / 'prepared.json', record)
    c.require(not t.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    return record


@t.no_grad()
def build(prepared_path, artifact):
    prepared = c.check_prepared(prepared_path)
    c.require(c.native_context() == prepared['native_context'], 'Actual native build context changed.')
    c.require(t.cuda.is_available() and t.cuda.device_count() == 1 and not artifact.exists(),
              'Build only on the explicitly assigned GPU into a fresh artifact directory.')
    t.set_num_threads(4)
    t.manual_seed(137)
    t.backends.cuda.matmul.allow_tf32 = False
    t.backends.cudnn.allow_tf32 = False
    t.set_float32_matmul_precision('highest')
    logging.basicConfig(level=logging.INFO)
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    from tools.cat_eval import _prepare_model_for_eval
    from train_utils.cat_train_residual_protection import build_sparse_residual_payload
    from litebsq.sparse_residual import decode_blocked_quantized_sparse_residual
    from safetensors import safe_open
    source = Path(prepared['source_dir'])
    model, meta, loaded = load_v6_model_checkpoint(str(source), base_model_path=str(c.HF), map_location='cpu', strict=True)
    c.require(meta['checkpoint_id'] == prepared['source_checkpoint_id'] and not loaded.missing_keys
              and not loaded.unexpected_keys and len(model.state_dict()) == 3423, 'Raw native strict load differs.')
    model.eval().requires_grad_(False)
    warm = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    c.require(warm['warmed'] == warm['total'] == 252 and not warm['failed'] and not warm['skipped'],
              'Build from all actual native group8 caches.')
    frozen = {key: c.tensor_record(value) for key, value in model.state_dict().items()}
    rice = c.module(c.RICE, '_raw_sparse_build_rice7')
    modules, proof = {}, {}
    started = time.monotonic()
    for name in prepared['names']:
        layer = model.get_submodule(name)
        original = layer._cached_weight
        c.require(original.dtype == t.bfloat16 and original.shape.numel() == 50331648
                  and layer.cache_decoded_weight and layer.temporary and not layer.trainable_decode,
                  'Use actual original-coordinate native BF16 cache: ' + name)
        native = original.detach().cpu()
        donor = prepared['HF_donors'][name]
        with safe_open(str(c.HF / donor['shard']), framework='pt', device='cpu') as stream:
            target = stream.get_tensor(donor['key']).clone()
        protected = layer.protected_input_indices.detach().cpu()
        c.require(target.dtype == native.dtype == t.bfloat16 and target.shape == native.shape,
                  'Original HF BF16 donor geometry differs: ' + name)
        target[:, protected] = native[:, protected]
        payload, nnz, storage = build_sparse_residual_payload(linear_name=name, target_weight=target,
            reconstructed_weight=native, activation_weight=None, **c.RECIPE)
        c.require(payload is not None and nnz == 251659, 'Do not change the fixedceil(N*.005) density: ' + name)
        fields = {key: payload['sparse_residual_' + key] for key in (*rice.UNCHANGED, 'local_indices')}
        encoded, bits = rice.encode(fields['local_indices'], fields['block_ptr'])
        decoded_local = rice.decode({**encoded, 'block_ptr': fields['block_ptr']})
        c.require(t.equal(decoded_local, fields['local_indices']), 'Actual new Rice coordinates differ: ' + name)
        restore = {key: fields[key] for key in rice.UNCHANGED}
        restore['local_indices'] = decoded_local
        args = dict(out_features=layer.out_features, in_features=layer.in_features, block_rows=256, block_cols=256,
                    index_bits=8, value_bits=8, value_dtype=t.bfloat16, device='cpu')
        expected = decode_blocked_quantized_sparse_residual(**fields, **args)
        found = decode_blocked_quantized_sparse_residual(**restore, **args)
        c.require(all(t.equal(left, right) for left, right in zip(expected, found, strict=True))
                  and not t.isin(found[1], protected).any().item(), 'Rice global coordinates/BF16 values changed: ' + name)
        rows, cols, patch = (value.to('cuda:0') for value in found)
        value = original.clone()
        value.index_put_((rows, cols), patch, accumulate=True)
        c.require(t.equal(value[:, protected.to('cuda:0')], original[:, protected.to('cuda:0')]),
                  'Actual BF16 addition changed protected columns: ' + name)
        modules[name] = {**{key: fields[key] for key in rice.UNCHANGED}, **encoded}
        proof[name] = dict(nnz=nnz, original_blocked_codec_bytes=storage['codec_bytes'],
            Rice_tensor_bytes=sum(c.compact_bytes(tensor) for tensor in modules[name].values()),
            Rice_unpadded_bits=bits, coordinate_and_BF16_patch_exact=True, protected_columns_exact=True,
            expected_native_sparse_cache=c.tensor_record(value))
        print(dict(event='raw_abs005_Rice7_built', module=name, nnz=nnz,
                   tensor_bytes=proof[name]['Rice_tensor_bytes'], seconds=time.monotonic() - started), flush=True)
        del target, native, payload, fields, restore, expected, found, rows, cols, patch, value
    c.require({key: c.tensor_record(value) for key, value in model.state_dict().items()} == frozen,
              'Payload construction modified an original source tensor.')
    values = dict(schema=c.FORMAT + '_build', source_checkpoint_id=meta['checkpoint_id'],
                  source_dir=str(source), recipe=c.RECIPE, modules=modules)
    budget = bundle.ledger(meta, t.load(source / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True),
                           values, meta['checkpoint_id'])
    artifact.mkdir(parents=True)
    t.save(values, artifact / 'payload.pt')
    reloaded = t.load(artifact / 'payload.pt', map_location='cpu', weights_only=True, mmap=True)
    c.exact(bundle.rice_state(reloaded, prepared['names']), bundle.rice_state(values, prepared['names']), 'Saved new Rice864')
    report = dict(status='built', source_checkpoint_id=meta['checkpoint_id'], profile=prepared['profile'],
        payload=c.binding(artifact / 'payload.pt'), modules=proof, frozen_source_tensors=3423,
        actual_Rice_fields=864, recipe=c.RECIPE, bit_budget=budget, prepared=c.binding(prepared_path),
        actual_native_group8=True, actual_whole_model_forwards=0, elapsed_seconds=time.monotonic() - started)
    c.write(prepared_path.parent / 'build.json', report)
    del model
    gc.collect()
    t.cuda.empty_cache()
    c.check_prepared(prepared_path)
    return report


@t.no_grad()
def export(prepared_path, artifact):
    c.cpu_only()
    t.set_num_threads(4)
    prepared = c.check_prepared(prepared_path)
    build_report = c.read(prepared_path.parent / 'build.json')
    c.require(build_report['status'] == 'built' and build_report['prepared'] == c.binding(prepared_path)
              and build_report['payload'] == c.binding(artifact / 'payload.pt'), 'Use the actual new raw-source payload.')
    values = t.load(artifact / 'payload.pt', map_location='cpu', weights_only=True, mmap=True)
    source = Path(prepared['source_dir'])
    meta = c.read(source / 'checkpoint_meta.json')
    original = t.load(source / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    banks = [key for key, value in original.items() if '.vq_weight' in key and value.dtype == t.uint8]
    c.require(len(banks) == 504, 'All504 original packed banks must remain in the new native component.')
    output = artifact / 'final_model'
    c.require(not output.exists() and values['source_checkpoint_id'] == meta['checkpoint_id'], 'Never overwrite or relabel a source.')
    loader = c.module(c.RESULT / 'qwen_c1_dual_down_pv32_20261019/pv_cpu_loader.py', '_raw_sparse_cpu_native_loader')
    model, actual, loaded = loader.load_cpu_native(source)
    c.require(actual == meta and not loaded.missing_keys and not loaded.unexpected_keys, 'Actual CPU native load failed.')
    c.exact(model.state_dict(), original, 'Original frozen3423 before native save')
    from transformers import AutoTokenizer
    from train_utils.checkpoint_v6 import save_v6_full_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(str(c.HF), use_fast=True, local_files_only=True)
    provenance = dict(profile=prepared['profile'], source_dir=str(source), source_checkpoint_id=meta['checkpoint_id'],
                      recipe=c.RECIPE, prepared=c.binding(prepared_path), build=c.binding(prepared_path.parent / 'build.json'),
                      inference_only=True, exact_resume_claimed=False)
    saved = save_v6_full_checkpoint(model, str(output / 'base'), checkpoint_kind='final_model',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=(), skip_targets=(), train_mode='none',
        after_category_mode='none', norm_train_mode='none', lm_head_train_mode='none', lora_config=None,
        base_model_path=meta['base_model_path'], tokenizer=tokenizer, extra_meta={'raw_abs005_Rice7_base': provenance})
    del model
    gc.collect()
    base_meta = c.read(output / 'base/checkpoint_meta.json')
    local_base = t.load(output / 'base/pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    c.exact(local_base, original, 'Native serialized frozen3423')
    c.require(base_meta['converted_modules'] == meta['converted_modules']
              and saved['checkpoint_id'] == base_meta['checkpoint_id'] != meta['checkpoint_id'],
              'Native save changed original topology or reused source identity.')
    serialized = dict(schema=c.FORMAT + '_payload', checkpoint_id=saved['checkpoint_id'],
                      source_checkpoint_id=meta['checkpoint_id'], recipe=c.RECIPE, modules=values['modules'])
    t.save(serialized, output / 'rice7_payload.pt')
    shutil.copyfile(c.RICE, output / 'rice7.py')
    files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'vocab.json', 'merges.txt', 'added_tokens.json']
    for name in files:
        shutil.copyfile(output / 'base' / name, output / name)
    all_files = ['base/checkpoint_meta.json', 'base/pytorch_model.bin', 'rice7_payload.pt', 'rice7.py', *files]
    file_records = {name: c.binding(output / name) for name in all_files}
    outer = {**base_meta, 'format': c.FORMAT, 'state_dict_file': 'base/pytorch_model.bin',
        'source_checkpoint_id': meta['checkpoint_id'], 'base_tensors': 3423, 'rice_tensors': 864, 'native_tensors': 4287,
        'recipe': c.RECIPE, 'rice_module_names': prepared['names'], 'rice_fields': list(c.FIELDS),
        'unpacked_local_indices_persisted': False,
        'files': {name: {key: record[key] for key in ('bytes', 'sha256')} for name, record in file_records.items()}}
    c.write(output / 'checkpoint_meta.json', outer)
    actual_rice = bundle.package(output, outer)
    c.exact(bundle.rice_state(actual_rice, prepared['names']), bundle.rice_state(values, prepared['names']), 'Serialized new Rice864')
    fresh, fresh_meta, audit = bundle.load_cpu_bundle(output)
    expected = {**original, **{'rice7_payload.' + key: value
        for key, value in bundle.rice_state(values, prepared['names']).items()}}
    compared = c.exact(fresh.state_dict(), expected, 'Independent fresh strict4287')
    budget = audit['bit_budget']
    c.require(fresh_meta == outer and budget['user_linear_bits'] == build_report['bit_budget']['user_linear_bits']
              and budget['packed_linear_bits'] == build_report['bit_budget']['packed_linear_bits'],
              'Saved full native/Rice accounting differs from actual build.')
    report = dict(status='verified' if budget['budget_passed'] else 'over_budget', checkpoint_dir=str(output),
        checkpoint_id=saved['checkpoint_id'], source_checkpoint_id=meta['checkpoint_id'], sources=provenance,
        base_frozen_tensors_compared=3423, rice_tensors_compared=864, serialized_tensors_compared=4287,
        fresh_strict_tensors_compared=compared, source_banks_compared=len(banks), bit_budget=budget,
        export_used_cuda=False, no_LoRA_optimizer=True, unpacked_local_indices_persisted=False,
        actual_native_save=True, independent_fresh_strict=True, full_state_key_value_dtype_shape_exact=True,
        payload_tensor_bytes=sum(c.compact_bytes(tensor) for fields in values['modules'].values() for tensor in fields.values()),
        payload_file_bytes=(output / 'rice7_payload.pt').stat().st_size, native_base_file_bytes=(output / 'base/pytorch_model.bin').stat().st_size)
    c.write(prepared_path.parent / 'audit.json', report)
    c.write(output / 'assembly_report.json', report)
    del fresh
    gc.collect()
    c.check_prepared(prepared_path)
    c.require(not t.cuda.is_initialized(), 'CPU writer initialized CUDA.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'build', 'export'])
    parser.add_argument('--profile', choices=['C', 'G'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact_root.resolve()
    if args.command == 'prepare':
        c.require(args.profile and args.output, 'Prepare requires profile/output.')
        result = prepare(args.profile, args.output.resolve(), artifact)
    else:
        c.require(args.prepared, 'Build/export requires prepared.json.')
        result = (build if args.command == 'build' else export)(args.prepared.resolve(), artifact)
    print(dict(status=result['status']), flush=True)


if __name__ == '__main__':
    main()
