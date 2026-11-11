"""Load this complete Qwen3 v6 state without redundantly loading HF weights."""
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers.modeling_utils import no_init_weights

from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model, load_v6_meta, validate_v6_meta


def load_complete_checkpoint(directory, *, base_model_path=None):
    directory = Path(directory).resolve()
    meta = validate_v6_meta(load_v6_meta(str(directory)), expected_kind='final_model')
    if (meta['converted_module_count'] != 252 or len(meta['compressed_targets']) != 252
            or meta['pending_dense_targets'] or meta['skip_targets'] or meta['lora_config'] is not None
            or meta.get('shared_protected_residual_decoders') or meta.get('post_norm_head_linear')
            or (meta.get('extra_meta') or {}).get('residual_lora')):
        raise ValueError('This isolated loader requires the actual complete252/rank0 G09 topology.')
    # Match the canonical get_model factory's original HF config and generation
    # policy. The complete v6 state supplies every parameter, including head/norms.
    base = Path(base_model_path or meta['base_model_path']).resolve()
    config = AutoConfig.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    if config.model_type != 'qwen3' or config.num_hidden_layers != 36 or config.tie_word_embeddings:
        raise ValueError('This loader covers only the verified untied36-layer Qwen3-8B architecture.')
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2', trust_remote_code=False)
    model.seqlen = 2048
    model.generation_config = GenerationConfig.from_pretrained(str(base), local_files_only=True)
    model, loaded, result = load_v6_full_checkpoint_into_model(model, str(directory),
        expected_kind='final_model', map_location='cpu', strict=True)
    if loaded != meta or result.missing_keys or result.unexpected_keys or len(model.state_dict()) != 3423:
        raise ValueError('The complete checkpoint must strictly supply all3423 persistent fields.')
    rope = {name: tensor for name, tensor in model.named_buffers() if 'inv_freq' in name}
    if (not rope or any(tensor.dtype != torch.float32 or tensor.device.type != 'cpu'
                       or not torch.isfinite(tensor).all().item() for tensor in rope.values())
            or model.config._attn_implementation != 'flash_attention_2'
            or model.get_input_embeddings().weight.dtype != torch.bfloat16
            or model.config.rope_scaling != config.rope_scaling or model.config.rope_theta != config.rope_theta):
        raise ValueError('Keep original BF16/FA2 policy and constructor-generated FP32 RoPE buffers.')
    return model.eval().requires_grad_(False), loaded, result
