"""Dense PEFT adapters and byte evidence for the isolated after-gate probe."""
from __future__ import annotations

import hashlib

import torch
from peft import LoraConfig, get_peft_model

from e2e_common.full_lora import iter_named_peft_lora_layers, _logical_adapter_target_name
from train_utils.distill_precision import prepare_model_export
from train_utils.model_level_trainables import (
    ModelLevelTrainableSelection, classify_peft_lora_parameters, enable_decoder_trainables,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_records(model_or_state):
    state = model_or_state.state_dict() if hasattr(model_or_state, 'state_dict') else model_or_state
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                   'storage_bytes': value.untyped_storage().nbytes(), 'sha256': tensor_sha(value)}
            for name, value in state.items()}


def select_adapter(model, dense_names, decoder_targets, *, use_dora):
    require(all(type(model.get_submodule(name)) is torch.nn.Linear for name in dense_names),
            'DoRA is supported here only on the remaining real dense Linear weights.')
    model.requires_grad_(False)
    wrapped = get_peft_model(model, LoraConfig(task_type=None, r=12, lora_alpha=24,
        lora_dropout=0.1, target_modules=list(dense_names), bias='none',
        init_lora_weights=True, use_dora=bool(use_dora)))
    require({_logical_adapter_target_name(name) for name, _ in iter_named_peft_lora_layers(wrapped)}
            == set(dense_names), 'Adapter target inventory differs.')
    lora, head = classify_peft_lora_parameters(wrapped)
    require(not head, 'No head adapter is allowed.')
    magnitudes = [p for name, p in lora.items() if '.lora_magnitude_vector.' in name]
    require(len(magnitudes) == (len(dense_names) if use_dora else 0), 'Magnitude inventory differs.')
    decoder = enable_decoder_trainables(wrapped, selected_modules=decoder_targets) if decoder_targets else {}
    return ModelLevelTrainableSelection(peft_model=wrapped, lora_parameters=lora, decoder_parameters=decoder)


def merge_adapter(wrapped, precision):
    # The repository's additive dense finalizer does not implement DoRA.
    merged = wrapped.merge_and_unload(safe_merge=True)
    require(not list(iter_named_peft_lora_layers(merged)), 'Native merge left live adapters.')
    prepare_model_export(merged, precision)
    return merged


def compare_states(before, after, mutable):
    require(set(before) == set(after), 'Persistent topology changed.')
    for name in before:
        require(all(before[name][key] == after[name][key] for key in ('shape', 'dtype', 'storage_bytes')),
                'Persistent storage schema changed: ' + name)
        if name not in mutable:
            require(before[name] == after[name], 'Frozen tensor changed: ' + name)
    changed = [name for name in sorted(mutable) if before[name]['sha256'] != after[name]['sha256']]
    require(changed, 'No permitted weight survived BF16 export.')
    return {'frozen_tensors': len(before) - len(mutable), 'mutable_tensors': len(mutable),
            'changed_mutable_tensors': changed, 'frozen_bytes_exact': True}


def output_difference(before, after):
    delta = after.float() - before.float()
    require(torch.isfinite(delta).all(), 'Nonfinite output difference.')
    return {'relative_l2': float(delta.norm() / before.float().norm().clamp_min(1e-30)),
            'max_absolute': float(delta.abs().max()),
            'cosine': float(torch.nn.functional.cosine_similarity(before.float().reshape(1, -1),
                                                                  after.float().reshape(1, -1)))}
