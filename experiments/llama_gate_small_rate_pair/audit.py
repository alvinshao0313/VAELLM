"""Check actual K34 native storage, padding, and the unchanged evaluation path."""
import argparse
import logging
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from experiments.llama_gate_small_rate_pair.common import (
    BASE_ID, DENOMINATOR, GROUPS, LOGICAL_BITS, PACKED_BITS, check, read, _require)
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from litebsq.bitpack import unpack_uint8_tensor_to_bool, pack_bool_tensor_to_uint8
from litebsq.fused_multistage_decoder import fused_decode_batched_same_shape_decoders
from litebsq.packed_bit_linear import packed_u8_parallel_linear
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state
from train_utils.v6_model_loader import load_v6_model_checkpoint


def audit(path):
    record = read(path); output = Path(record['output_dir']); check(record); started = time.time()
    native = Path(record['donor_checkpoint_dir']); dm = load_v6_meta(str(native))
    actual = read(native.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == native.parent, 'Wrong native output.')
    actual['cat_args']['output_dir'] = str(output / 'donor')
    _require(actual == record['normalized_expected'], 'Native settings differ from preparation.')
    specs = {s['name']: s for s in dm['converted_modules']}
    names = {n for n in record['architecture']['projection_shapes'] if n.endswith('.gate_proj')}
    _require(len(names) == 32 and set(specs) == set(dm['compressed_targets']) == names
             and dm['compression_categories'] == dm['completed_categories'] == ['gate_proj']
             and dm['after_category_mode'] == 'none' and dm['lora_config'] is None
             and dm['base_model_path'] == record['architecture']['base_model_path'], 'Wrong native donor.')
    for name, s in specs.items():
        _require(not s['transpose'] and [s['out_features'], s['in_features']] == [14336, 4096]
                 and [s['compressed_out_features'], s['compressed_in_features']] == [14336, 4032]
                 and s['codebook_dim'] == 32 and s['stage_codebook_dims'] == [32, 32]
                 and s['residual_stages'] == 2 and s['parallel_parts'] == 1, 'Wrong gate geometry: ' + name)
        _require(len(s['stage_vq_weights']) == len(s['stage_decoders']) == 2
                 and all(b == {'shape': [GROUPS, 1, 5], 'dtype': 'uint8', 'storage_format': 'bitpack_u8',
                      'logical_shape': [GROUPS, 1, 34], 'logical_dtype': 'bool', 'pack_bits': 8}
                         for b in s['stage_vq_weights']), 'Wrong K34 packed geometry: ' + name)
        _require(all(d['in_dim'] == 34 and d['out_dim'] == 32 and d['hidden_dim'] == 128
                 and d['num_res_blocks'] == 0 and d['param_dtype'] == 'bfloat16' and d['norm_type'] == 'layer'
                 and d['activation_type'] == 'swish' and d['decoder_type'] == 'symmetric'
                 for d in s['stage_decoders']), 'Wrong native decoder: ' + name)
        _require(s['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and s['protected_input_qvalues'] == {'shape': [64, 14336], 'dtype': 'uint8'}
                 and s['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                 and s['protected_output_indices'] is None and s['protected_channel_quant_format'] == 'int8'
                 and s['weight_rotation']['type'] == 'two_sided_hadamard'
                 and s['weight_rotation']['block_size'] == 32, 'Wrong protection or rotation: ' + name)
    log = (output / 'train.log').read_text()
    for stage in (1, 2):
        _require(f'[gate_proj.L0-31/stage{stage}] step=2/2 ' in log, 'Missing real two-step stage completion.')
    final = output / 'final_model'; meta = load_v6_meta(str(final)); report = read(final / 'assembly_report.json')
    source, budget = report['sources'], report['bit_budget']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == BASE_ID and source['donor_checkpoint_id'] == dm['checkpoint_id']
             and source['categories'] == ['gate_proj'] and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 2979,
             'Native source union/save/fresh is incomplete.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == LOGICAL_BITS and budget['packed_linear_bits'] == PACKED_BITS
             and budget['code_padding_bits'] == PACKED_BITS - LOGICAL_BITS
             and budget['original_linear_weights'] == DENOMINATOR and budget['compressed_target_count'] == 224
             and budget['components'] == record['expected_budget_components'], 'Actual logical/physical ledger differs.')
    state = torch.load(final / meta['state_dict_file'], weights_only=True, map_location='cpu', mmap=True)
    _require(len(state) == 2979, 'Saved complete inventory differs.')
    banks = 0
    for s in specs.values():
        for key, spec in _code_specs(s):
            value = state[key]; _require(not bool((value[..., -1] & 0xfc).any()), 'Nonzero padding: ' + key)
            for chunk in value.split(65536):
                shape = (len(chunk), 1, 34)
                replay = pack_bool_tensor_to_uint8(unpack_uint8_tensor_to_bool(chunk, logical_shape=shape), logical_shape=shape)
                _require(torch.equal(replay, chunk), 'K34 native pack/unpack changed bytes: ' + key)
            banks += 1
    _require(banks == 64, 'Incomplete K34 bank coverage.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protected = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {f: tensor_record(state[n + '.weight_rotation.' + f])
                 for f in ('input_signs', 'output_signs')} for n in sorted(names)}
    runtime = load_v6_cat_runtime_state(str(native), required=True); activation = runtime['activation_runtime']; cache = activation['cache']
    _require(runtime['global_adaptive_plan'] is None and activation['dataset'] == cache['dataset'] == 'alpaca=1'
             and activation['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
             and activation['seqlen'] == cache['seqlen'] == 256 and activation['seed'] == cache['seed'] == 31
             and names <= set(activation['stats_by_linear']), 'Actual original calibration differs.')
    del state
    model, loaded_meta, loaded = load_v6_model_checkpoint(str(final), map_location='cpu', strict=True, expected_kind='final_model')
    _require(loaded_meta['checkpoint_id'] == meta['checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys,
             'Independent native inference load failed.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    from tools.cat_eval import _prepare_model_for_eval
    torch.cuda.reset_peak_memory_stats()
    warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    _require(warmup['total'] == warmup['warmed'] == 224 and warmup['failed'] == warmup['skipped'] == 0
             and warmup['prewarm_dtype'] == 'torch.bfloat16', 'Actual all224 group8 BF16 prewarm failed.')
    # Real trained L0..7, both stages, first4096 vectors: exercise the non-byte-aligned
    # cold packed input and group8 kernel against native uncompressed 0/1 operands.
    decoders, dense, expected, packed_expected = [], [], [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for layer in range(8):
            module = model.get_submodule(f'model.layers.{layer}.mlp.gate_proj')
            decoder = module._parallel_stage_decoder
            packed = module._get_parallel_stage_grouped_vq_packed(device=torch.device('cuda:0'))[:4096]
            bits = unpack_uint8_tensor_to_bool(packed, logical_shape=(4096, 2, 34)).to(torch.bfloat16)
            hidden = packed_u8_parallel_linear(packed, decoder.linear_in, logical_in_dim=34, activation_dtype=torch.bfloat16)
            packed_expected.append(decoder.linear_out(F.silu(decoder.norm_out(hidden))).sum(dim=1).to(torch.bfloat16))
            expected.append(decoder(bits).sum(dim=1).to(torch.bfloat16)); dense.append(bits); decoders.append(decoder)
        actual = fused_decode_batched_same_shape_decoders(decoders, dense)
    epsilon = 2 * torch.finfo(torch.bfloat16).eps; parity = []
    for i, (p, d, a) in enumerate(zip(packed_expected, expected, actual, strict=True)):
        torch.testing.assert_close(p, d, rtol=epsilon, atol=epsilon)
        torch.testing.assert_close(a, p, rtol=epsilon, atol=epsilon)
        parity.append({'layer': i, 'packed_vs_dense_max_abs': float((p.float() - d.float()).abs().max()),
                       'group8_vs_packed_max_abs': float((a.float() - p.float()).abs().max())})
    tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
    ids = tokenizer('A short test sentence.', return_tensors='pt', add_special_tokens=False)['input_ids'][:, :4]
    _require(ids.shape == (1, 4), 'Require four actual tokenizer tokens.')
    with torch.inference_mode():
        logits = model(input_ids=ids.to('cuda:0'), use_cache=False).logits
    rope = {key: str(value.dtype) for key, value in model.named_buffers() if 'inv_freq' in key}
    _require(logits.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all()) and rope
             and set(rope.values()) == {'torch.float32'}, 'Native BF16 full forward/FP32 RoPE failed.')
    result = {'status': 'verified_native_short', 'checkpoint_id': meta['checkpoint_id'], 'base_checkpoint_id': BASE_ID,
        'donor_checkpoint_id': dm['checkpoint_id'], 'bit_budget': budget, 'source_union_save_fresh_tensors': 2979,
        'replaced_gate_count': 32, 'preserved_projection_count': 192, 'zero_padding_and_pack_replay_banks': banks,
        'protection_tensor_records': protected, 'rotation_sign_records': signs,
        'calibration_cache': tensor_tree(cache), 'calibration_stats': tensor_tree(activation['stats_by_linear']),
        'native_group8_prewarm': warmup, 'actual_trained_kernel_parity': parity, 'parity_rtol_atol': epsilon,
        'integration_forward': {'input_shape': list(ids.shape), 'output_shape': list(logits.shape),
                                'dtype': str(logits.dtype), 'finite': True, 'rope': rope},
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'assembly_report_sha256': sha256(final / 'assembly_report.json'),
        'actual_normalized_sha256': sha256(native.parent / 'normalized_cat_runtime_args.json'),
        'scope': 'Two real training steps per stage and native deployment path only; no downstream quality result.',
        'elapsed_seconds': time.time() - started}
    check(record); write_json(output / 'verification.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    audit(parser.parse_args().launch_record)
