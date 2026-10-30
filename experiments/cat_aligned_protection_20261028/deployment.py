"""Explicit2.6/rank0 native save plus independent strict state/geometry/bank audit."""
import gc
from pathlib import Path
import bindings as b
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import persistent_inventory
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


@b.torch.no_grad()
def save_and_verify(model, output, meta, expected, provenance, tokenizer):
    output = Path(output).resolve()
    b.require(not output.exists() and not output.is_symlink(), 'Never replace native deployment output.')
    source_budget = audit_state_dict(meta, expected, max_bpw=2.6, max_lora_rank=0)
    b.require(source_budget['budget_passed'] and not source_budget['contract_violations']
        and not source_budget['unclassified_target_keys'] and not source_budget['components'].get('dense_weights'),
        'Explicit2.6/rank0 audit failed before native save.')
    _assert_state_equal(model.state_dict(), expected, 'In-memory actual aligned deployment')
    architecture = model_inventory(meta['base_model_path'])
    inventory, banks = persistent_inventory(model, architecture)
    bank_count = sum(len(values) for values in banks.values())
    b.require(len(inventory) == len(banks) == 252 and bank_count == 504 and len(expected) == 3423,
        'Incomplete original252/504bank/3423 inventory.')
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=(), skip_targets=(), train_mode='none',
        after_category_mode='none', lora_config=None, base_model_path=meta['base_model_path'], tokenizer=tokenizer,
        extra_meta={'aligned_protection': provenance})
    serialized = b.load_state(output)
    compared = _assert_state_equal(serialized, expected, 'Serialized actual aligned3423')
    fresh, loaded, result = load_v6_model_checkpoint(str(output), map_location='cpu', strict=True, expected_kind='final_model')
    fresh_compared = _assert_state_equal(fresh.state_dict(), expected, 'Independent fresh strict aligned3423')
    parent_ids = provenance['parent_checkpoint_ids']
    b.require(not result.missing_keys and not result.unexpected_keys and saved['checkpoint_id'] == loaded['checkpoint_id']
        and loaded['checkpoint_id'] not in parent_ids and loaded.get('immutable_resume_contract') is None
        and loaded['lora_config'] is None and type(fresh.lm_head) is b.torch.nn.Linear and not fresh.config.tie_word_embeddings
        and {_['name']: _inference_spec(_) for _ in loaded['converted_modules']}
        == {_['name']: _inference_spec(_) for _ in meta['converted_modules']}
        and persistent_inventory(fresh, architecture) == (inventory, banks), 'Native fresh UUID/topology/geometry/banks changed.')
    budget = audit_state_dict(loaded, serialized, max_bpw=2.6, max_lora_rank=0)
    b.require(budget['budget_passed'] and all(budget[key] == source_budget[key] for key in b.BUDGET_FIELDS),
        'Saved native storage differs from actual expected state.')
    del fresh
    gc.collect()
    return dict(status='verified', profile=b.PROFILE, checkpoint_dir=str(output), checkpoint_id=loaded['checkpoint_id'],
        serialized_tensors_compared=compared, fresh_strict_tensors_compared=fresh_compared,
        fresh_strict_reload_state_exact=True, code_bank_count=bank_count, bit_budget=budget,
        source_geometry_unchanged=True, checkpoint_weight_stat=b.file_stat(output / loaded['state_dict_file']),
        checkpoint_meta_sha256=b.sha256(output / 'checkpoint_meta.json'), sources=provenance)
