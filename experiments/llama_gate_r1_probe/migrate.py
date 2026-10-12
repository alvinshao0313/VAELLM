"""One native Llama gate expansion, then exact v6 save/fresh-load verification."""
import argparse
import gc
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.llama_gate_r1_probe.common import (
    PREFIX, SOURCE, SOURCE_ID, TARGET, _assert_state_equal, _load_state, budget_check, check,
    expanded_specs, read, require, state_inventory, write_json)
from experiments.qwen3_w25.expand_decoder import expand_packed_decoder_r0_to_r1_
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def strict_load(path):
    model, meta, result = load_v6_model_checkpoint(str(path), map_location='cpu', strict=True, expected_kind='final_model')
    require(not result.missing_keys and not result.unexpected_keys, 'Native strict load is incomplete.')
    model.eval().requires_grad_(False)
    return model, meta


def error(actual, reference):
    difference = actual.float() - reference.float()
    return {'equal': bool(torch.equal(actual, reference)), 'max_abs': float(difference.abs().max()),
            'relative_l2': float(difference.double().norm() / reference.double().norm().clamp_min(1e-30))}


@torch.no_grad()
def migrate(path):
    record = read(path); check(record)
    root = Path(record['output_dir']); output = root / 'r1_initial'
    require(not output.exists(), 'No replacement of an existing migration.')
    started = time.time()
    original = _load_state(SOURCE)
    model, meta = strict_load(SOURCE)
    require(meta['checkpoint_id'] == SOURCE_ID, 'Wrong initial model.')
    _assert_state_equal(model.state_dict(), original, 'Before migration')
    gate = model.get_submodule(TARGET).to('cuda:0')
    gate._build_parallel_stage_decode_plan()
    cached_path = gate._decode_weight(dtype=torch.bfloat16).detach().cpu()
    gate.cache_decoded_weight = False
    packed_path = gate._decode_weight(dtype=torch.bfloat16).detach().cpu()
    before_parameters = {k: p for k, p in gate._parallel_stage_decoder.named_parameters()}
    expansion = expand_packed_decoder_r0_to_r1_(gate)
    require(expansion['added_parameters'] == record['expansion']['added_parameters']
            and all(dict(gate._parallel_stage_decoder.named_parameters())[k] is p for k, p in before_parameters.items()),
            'Migration replaced existing parameter identities or added the wrong residual block.')
    gate._build_parallel_stage_decode_plan()
    after = gate._decode_weight(dtype=torch.bfloat16).detach().cpu()
    errors = {'r0_cached_vs_r0_packed': error(packed_path, cached_path),
              'r1_packed_vs_r0_packed': error(after, packed_path), 'r1_vs_original_inference': error(after, cached_path)}
    # This bound is for BF16 decode, not for frozen bytes or save/reload. With a
    # zero block the same packed path should normally be exactly equal.
    require(all(torch.isfinite(x).all() for x in (cached_path, packed_path, after))
            and errors['r1_packed_vs_r0_packed']['relative_l2'] <= 2e-3
            and errors['r1_vs_original_inference']['relative_l2'] <= 2e-3,
            'Identity migration exceeds the declared BF16 numerical tolerance; stop before training.')
    gate.clear_decoded_weight_cache(); gate.to('cpu'); gate._build_parallel_stage_decode_plan()
    gate.cache_decoded_weight = True
    expected = model.state_dict()
    _assert_state_equal({k: expected[k] for k in original}, original, 'All original bytes after expansion')
    require(state_inventory(expected) == record['inventories']['r1'], 'Unexpected expanded inventory.')
    for key in set(expected) - set(original):
        require(key.startswith(PREFIX + 'blocks.0.') and torch.all(expected[key] == (1 if key.endswith('norm1.weight') else 0)),
                'Wrong residual identity initialization.')
    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True, use_fast=True)
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=(), skip_targets=(), train_mode='none',
        after_category_mode='none', lora_config=None, base_model_path=meta['base_model_path'], tokenizer=tokenizer,
        extra_meta={'source_checkpoint_id': SOURCE_ID, 'identity_residual_target': TARGET,
                    'residual_initialization': expansion, 'purpose': 'Matched128-step decoder capacity probe'})
    serialized = _load_state(output)
    count = _assert_state_equal(serialized, expected, 'R1 serialized state')
    saved_meta = read(output / 'checkpoint_meta.json')
    require(saved_meta['converted_modules'] == expanded_specs(meta), 'Metadata changed beyond this one decoder depth.')
    budget = audit_state_dict(saved_meta, serialized, max_bpw=2.5, max_lora_rank=0)
    budget_check(budget, record, 'r1')
    del model, gate, expected, before_parameters; gc.collect(); torch.cuda.empty_cache()
    fresh, fresh_meta = strict_load(output)
    require(fresh_meta == saved_meta, 'Fresh metadata differs.')
    _assert_state_equal(fresh.state_dict(), serialized, 'R1 fresh native state')
    write_json(root / 'migration.json', {'status': 'verified', 'checkpoint_id': saved['checkpoint_id'],
        'source_checkpoint_id': SOURCE_ID, 'expansion': expansion, 'initial_native_bf16_errors': errors,
        'original_tensors_exact': len(original), 'serialized_fresh_exact': count, 'bit_budget': budget,
        'elapsed_seconds': time.time() - started, 'full_model_initial_observations': 'Separate fixed-domain stage follows.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', required=True, type=Path)
    migrate(parser.parse_args().launch_record)
