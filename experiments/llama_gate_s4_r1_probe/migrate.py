"""Expand all native S4 gate decoders with zero-output residual blocks."""
import argparse
import gc
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.llama_gate_r1_probe.migrate import error, strict_load
from experiments.llama_gate_s4_r1_probe.common import (
    GATE_NAMES, PREFIXES, SOURCE, SOURCE_ID, budget_check, check, expected_specs,
    read, require, state_inventory as inventory)
from experiments.longbench_v1.common import file_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from litebsq.autoencoder import Decoder
from litebsq.parallel_layers import ResnetBlock1D
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import save_v6_full_checkpoint


@torch.no_grad()
def expand_gate(gate):
    require(isinstance(gate, VAELinear) and gate.parallel_stage_decode
            and gate.parallel_parts == 1 and gate.residual_stages == 4, 'Require a native packed S4 gate.')
    decoder = gate._parallel_stage_decoder
    require(isinstance(decoder, Decoder) and decoder.num_models == gate.residual_stages
            and (decoder.in_dim, decoder.out_dim, decoder.hidden_dim, decoder.num_res_blocks,
                 decoder.norm_type, decoder.activation_type, decoder.decoder_type)
            == (16, 32, 128, 0, 'layer', 'swish', 'symmetric') and not len(decoder.blocks),
            'Only the actual K16/D32/H128/R0 gate profile may be expanded.')
    original_parameters = dict(decoder.named_parameters())
    reference = next(iter(original_parameters.values()))
    require({(p.device, p.dtype, p.requires_grad) for p in original_parameters.values()}
            == {(reference.device, reference.dtype, reference.requires_grad)}
            and reference.dtype == torch.bfloat16, 'Mixed decoder parameter profiles.')
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        block = ResnetBlock1D(in_channels=decoder.hidden_dim, out_channels=decoder.hidden_dim,
            norm_type=decoder.norm_type, activation_type=decoder.activation_type, num_models=decoder.num_models)
    block.to(device=reference.device, dtype=reference.dtype)
    block.norm1.weight.fill_(1)
    block.norm1.bias.zero_()
    block.linear1.conv.weight.zero_()
    block.linear1.conv.bias.zero_()
    block.train(decoder.training).requires_grad_(reference.requires_grad)
    decoder.blocks.append(block)
    decoder.num_res_blocks = 1
    gate.clear_decoded_weight_cache()
    require(all(dict(decoder.named_parameters())[key] is value for key, value in original_parameters.items()),
            'Expansion replaced an existing decoder parameter object.')
    return {'num_models': decoder.num_models, 'added_parameters': sum(p.numel() for p in block.parameters()),
            'added_tensor_bytes': sum(v.numel() * v.element_size() for v in block.state_dict().values()),
            'added_tensor_keys': list(block.state_dict()), 'original_parameter_objects_preserved': True,
            'wholefused_supported': False}


@torch.no_grad()
def migrate(path):
    record = read(path); check(record)
    root = Path(record['output_dir']); output = root / 'r1_initial'
    require(not output.exists() and not (root / 'migration.json').exists(), 'Do not overwrite a migration.')
    require(torch.cuda.is_available(), 'The real S4 migration requires its assigned CUDA GPU.')
    started = time.time()
    original = _load_state(SOURCE)
    model, meta = strict_load(SOURCE)
    require(meta['checkpoint_id'] == SOURCE_ID and inventory(original) == record['inventories']['r0'],
            'Wrong source identity or persistent gate inventory.')
    _assert_state_equal(model.state_dict(), original, 'Original S4 strict reconstruction')
    errors, expansions = {}, {}
    for name in GATE_NAMES:
        gate = model.get_submodule(name).to('cuda:0')
        gate._build_parallel_stage_decode_plan()
        cached = gate._decode_weight(dtype=torch.bfloat16).detach().cpu()
        gate.cache_decoded_weight = False
        packed = gate._decode_weight(dtype=torch.bfloat16).detach().cpu()
        expansions[name] = expand_gate(gate)
        gate._build_parallel_stage_decode_plan()
        after = gate._decode_weight(dtype=torch.bfloat16).detach().cpu()
        errors[name] = {'r0_cached_vs_r0_packed': error(packed, cached),
                       'r1_packed_vs_r0_packed': error(after, packed),
                       'r1_vs_original_inference': error(after, cached)}
        require(all(torch.isfinite(x).all() for x in (cached, packed, after))
                and errors[name]['r1_packed_vs_r0_packed']['relative_l2'] <= 2e-3
                and errors[name]['r1_vs_original_inference']['relative_l2'] <= 2e-3,
                'S4 identity migration exceeds the BF16 decode tolerance: ' + name)
        gate.clear_decoded_weight_cache(); gate.to('cpu'); gate._build_parallel_stage_decode_plan()
        gate.cache_decoded_weight = True
        del cached, packed, after, gate
    expected = model.state_dict()
    _assert_state_equal({key: expected[key] for key in original}, original, 'All original3043 tensors after migration')
    require(inventory(expected) == record['inventories']['r1']
            and sum(x['added_parameters'] for x in expansions.values()) == record['expansion']['added_parameters']
            and len(expected) - len(original) == record['expansion']['added_state_tensors'], 'Wrong all-gate expansion.')
    for key in set(expected) - set(original):
        require(key.startswith(tuple(PREFIXES)) and '.blocks.0.' in key
                and torch.all(expected[key] == (1 if key.endswith('norm1.weight') else 0)),
                'Unexpected new tensor or non-identity initialization: ' + key)
    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True, use_fast=True)
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        **{key: tuple(meta.get(key) or ()) for key in ('compressed_targets', 'pending_dense_targets', 'skip_targets',
            'legacy_original_only_sources', 'completed_categories', 'compression_categories')},
        train_mode='none', after_category_mode='none', lora_config=None,
        base_model_path=meta['base_model_path'], tokenizer=tokenizer,
        extra_meta={'source_checkpoint_id': SOURCE_ID, 'identity_residual_targets': list(GATE_NAMES),
                    'residual_initialization': 'All S4 gate blocks: norm gamma1/beta0, linear weight/bias0.'})
    serialized = _load_state(output); saved_meta = read(output / 'checkpoint_meta.json')
    require(saved_meta['converted_modules'] == expected_specs(meta, 'r1'), 'Migration changed other codec metadata.')
    count = _assert_state_equal(serialized, expected, 'R1 serialized native state')
    budget = audit_state_dict(saved_meta, serialized, max_bpw=2.5, max_lora_rank=0)
    budget_check(budget, record, 'r1')
    del model, expected; gc.collect(); torch.cuda.empty_cache()
    fresh, fresh_meta = strict_load(output)
    require(fresh_meta == saved_meta, 'Fresh metadata differs.')
    _assert_state_equal(fresh.state_dict(), serialized, 'R1 fresh native state')
    check(record)
    report = {'status': 'verified', 'checkpoint_id': saved['checkpoint_id'], 'source_checkpoint_id': SOURCE_ID,
        'expansions': expansions, 'initial_native_bf16_errors': errors, 'original_tensors_exact': len(original),
        'serialized_fresh_exact': count, 'bit_budget': budget, 'elapsed_seconds': time.time() - started,
        'weight_stat': file_stat(output / 'pytorch_model.bin'),
        'metadata_sha256': sha256(output / 'checkpoint_meta.json'),
        'full_model_initial_observations': 'No initial full8 or development-set evaluation; '
            'real per-gate identity plus subsequent saved-step fresh-model forward verification.'}
    write_json(root / 'migration.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', required=True, type=Path)
    migrate(parser.parse_args().launch_record)
