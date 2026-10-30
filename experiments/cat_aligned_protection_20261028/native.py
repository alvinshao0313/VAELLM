"""The established native student loading path with the current 2.6bit/rank0 bound."""
import logging

import bindings as b
from experiments.longbench_v1.models import model_policy
from experiments.w25_eval.model_inventory import model_inventory
from tools.cat_eval import _prepare_model_for_eval
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.v6_model_loader import load_v6_model_checkpoint


@b.torch.no_grad()
def load_student(config):
    b.qa.check_config(config)
    b.require(b.torch.cuda.is_available() and b.torch.cuda.device_count() == 1, 'Use exactly one assigned visible GPU.')
    b.torch.cuda.reset_peak_memory_stats()
    record = config['model']
    model, meta, loaded = load_v6_model_checkpoint(record['checkpoint'], base_model_path=record['hf_path'], map_location='cpu', strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys and meta['checkpoint_id'] == record['checkpoint_id'],
        'The actual aligned student did not load fresh and strict.')
    architecture = model_inventory(record['hf_path'])
    budget = audit_state_dict(meta, model.state_dict(), max_bpw=2.6, max_lora_rank=0)
    b.require(budget['budget_passed'] and not budget['contract_violations']
        and budget['compressed_target_count'] == architecture['projection_count'] == 252
        and budget['original_linear_weights'] == architecture['original_linear_weights'], 'Invalid native student storage/rank0.')
    model.eval().requires_grad_(False)
    warm = _prepare_model_for_eval(model, config['device'], 8, logging.getLogger(__name__), {})
    policy = model_policy(record['hf_path'])
    b.require(model.config.model_type == policy['model_type'] and model.config.rope_scaling == policy['rope_scaling']
        and model.config.rope_theta == policy['rope_theta'] and model.config._attn_implementation == policy['attn_implementation']
        and model.get_input_embeddings().weight.dtype == b.torch.bfloat16, 'Aligned native student runtime policy changed.')
    return model, dict(strict_missing=[], strict_unexpected=[], checkpoint_id=meta['checkpoint_id'],
        persistent_tensors=len(model.state_dict()), bit_budget=budget, warmup=warm,
        native_policy=b.core.native_policy(model), model_policy=policy,
        load_peak_allocated_bytes=b.torch.cuda.max_memory_allocated(),
        load_peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
