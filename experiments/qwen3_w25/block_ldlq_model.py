"""Native VAELinear adapter for the approved fixed-decoder BlockLDLQ trial."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from litebsq.bitpack import pack_bool_tensor_to_uint8
from litebsq.fused_multistage_decoder import (
    extract_packed_symmetric_stage_weights, fused_multistage_symmetric_decode,
    packed_symmetric_decoder_supports_fuse,
)
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import _transform_last
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load, save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


SOURCE_ID = "4b6da7c8-730c-4876-954b-402bff67258e"
CODE_KEYS = ("vq_weight", "vq_weight_s1")


def validate_layer(layer):
    if not isinstance(layer, VAELinear):
        raise ValueError("BlockLDLQ requires VAELinear.")
    if (layer.out_features, layer.in_features, layer.compressed_in_features) != (4096, 12288, 12032):
        raise ValueError("This approved trial requires the original down geometry and input256 protection.")
    if (layer.transpose or layer.residual_stages != 2 or layer.parallel_parts != 1
            or layer.codebook_dim != 16 or not layer.parallel_stage_decode
            or not layer._parallel_stage_restore_identity
            or list(layer._parallel_stage_layout) != [(0, 0), (1, 0)]):
        raise ValueError("Expected two D16 stages in input-contiguous layout.")
    decoder = layer._parallel_stage_decoder
    if (not packed_symmetric_decoder_supports_fuse(decoder) or decoder.in_dim != 24
            or decoder.out_dim != 16 or decoder.hidden_dim != 128 or decoder.num_models != 2):
        raise ValueError("Expected fixed K24/D16/H128/R0 two-stage symmetric decoder.")
    if any(p.dtype != torch.bfloat16 for p in decoder.parameters()):
        raise ValueError("This trial preserves BF16 decoder storage and compute.")
    if layer.weight_rotation is None or layer.weight_rotation.block_size != 64:
        raise ValueError("Expected the source block64 rotation.")
    layer.weight_rotation.validate_state()
    if (layer.protected_input_indices is None or layer.protected_input_indices.numel() != 256
            or layer.protected_input_qvalues is None or layer.protected_channel_quant_format != "int8"
            or layer.protected_output_indices is not None or layer.protected_residual_indices is not None
            or layer.low_rank_a is not None or layer.low_rank_b is not None or layer.has_sparse_residual()
            or layer.original_weight is not None or layer.always_use_original or layer.trainable_decode or layer.bias is not None
            or getattr(layer, "_sparse_bit_binding", None) is not None):
        raise ValueError("Only fixed input-INT8 protection without trainable/original/residual branches is supported.")
    for stage, name in enumerate(CODE_KEYS):
        spec = layer.get_stage_part_vq_spec(stage)
        if spec["logical_shape"] != [4096 * 752, 1, 24] or layer.get_stage_part_vq_storage(stage).dtype != torch.uint8:
            raise ValueError(f"Unexpected packed bank {name}.")
    return decoder


def read_codes(layer):
    return torch.stack([layer.get_stage_part_vq_weight(stage).reshape(4096, 752, 24)
                        for stage in range(2)])


def apply_codes(layer, codes):
    if codes.dtype != torch.bool or tuple(codes.shape) != (2, 4096, 752, 24):
        raise ValueError("Expected both complete bool code banks.")
    for stage in range(2):
        shape = layer.get_stage_part_vq_spec(stage)["logical_shape"]
        packed = pack_bool_tensor_to_uint8(codes[stage].reshape(shape), logical_shape=shape)
        layer.get_stage_part_vq_storage(stage).copy_(packed)
    # Parallel decoding owns a grouped copy of the two banks. Rebuild it as
    # well as the decoded-weight cache, so native measurements use new codes.
    refresh_vae_linear_runtime_after_state_load(layer)


class NativeStageOracle:
    """Use the same two-model fused kernel, isolating one stage's contribution.

    Keeping two models retains the production multi-model LayerNorm and output
    bias rounding. Zeroing the other output is temporary oracle state only.
    """

    def __init__(self, layer):
        weights = extract_packed_symmetric_stage_weights(validate_layer(layer))
        self.stage_weights = []
        for stage in range(2):
            fixed = list(weights)
            fixed[4], fixed[5] = weights[4].clone(), weights[5].clone()
            fixed[4][1 - stage].zero_()
            fixed[5][1 - stage].zero_()
            self.stage_weights.append(tuple(fixed))

    @torch.no_grad()
    def __call__(self, stage, bits):
        if stage not in (0, 1) or bits.ndim != 2 or bits.shape[1] != 24 or bits.dtype != torch.bool:
            raise ValueError("Invalid stage oracle input.")
        output = torch.empty((len(bits), 16), device=bits.device, dtype=torch.bfloat16)
        for start in range(0, len(bits), 8192):
            selected = bits[start:start + 8192]
            grouped = torch.zeros((len(selected), 2, 24), device=bits.device, dtype=torch.bfloat16)
            grouped[:, stage] = selected
            output[start:start + len(selected)] = fused_multistage_symmetric_decode(
                grouped, *self.stage_weights[stage])
        return output


@torch.no_grad()
def oracle_weight(oracle, codes):
    rows, blocks = codes.shape[1:3]
    a = oracle(0, codes[0].reshape(-1, 24))
    b = oracle(1, codes[1].reshape(-1, 24))
    return (a + b).reshape(rows, blocks * 16)


def rotate_input_rows(matrix, rotation):
    if matrix.dtype != torch.float64 or matrix.shape[-1] != rotation.in_features:
        raise ValueError("Hessian rotation requires FP64 and the compressed input axis.")
    return _transform_last(matrix * rotation.input_signs.to(matrix), block_size=rotation.input_block,
                           factor=rotation.input_factor, had=rotation.input_had, inverse=False)


def rotate_hessian(hessian, rotation):
    return rotate_input_rows(rotate_input_rows(hessian, rotation).T, rotation).T.contiguous()


@torch.no_grad()
def native_output_metrics(layer, reference, inputs, lengths):
    """Measure actual BF16 VAELinear output against the original BF16 Linear."""
    if sum(lengths) != len(inputs) or inputs.dtype != torch.bfloat16:
        raise ValueError("Token/document boundary or dtype mismatch.")
    device = next(layer.parameters()).device
    reference = reference.to(device=device, dtype=torch.bfloat16)
    layer.clear_decoded_weight_cache()
    values, offset = [], 0
    for length in lengths:
        x = inputs[offset:offset + length].to(device)
        actual, expected = layer(x), F.linear(x, reference)
        values.append(float((actual.float() - expected.float()).square().double().mean()))
        offset += length
    layer.clear_decoded_weight_cache()
    return {"valid_tokens": len(inputs), "token_mean_mse": sum(v * n for v, n in zip(values, lengths)) / len(inputs),
            "document_mean_mse": sum(values) / len(values), "per_document_mse": values,
            "metric": "native_bf16_outputs_then_fp32_difference_fp64_mean"}


def save_and_verify(model, output, source_meta, expected, provenance, tokenizer):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    refresh_vae_linear_runtime_after_state_load(model)
    _assert_state_equal(model.state_dict(), expected, "BlockLDLQ before save")
    saved = save_v6_full_checkpoint(
        model, str(output), checkpoint_kind="final_model", compressed_targets=source_meta["compressed_targets"],
        pending_dense_targets=(), skip_targets=(), train_mode="none", after_category_mode="none",
        base_model_path=source_meta["base_model_path"], tokenizer=tokenizer,
        extra_meta={"block_ldlq": provenance})
    serialized = _load_state(output)
    count = _assert_state_equal(serialized, expected, "BlockLDLQ saved state")
    loaded, meta, _ = load_v6_model_checkpoint(str(output), map_location="cpu", strict=True, expected_kind="final_model")
    fresh_count = _assert_state_equal(loaded.state_dict(), expected, "BlockLDLQ independent fresh load")
    if (meta["checkpoint_id"] == source_meta["checkpoint_id"] or meta.get("immutable_resume_contract") is not None
            or (meta.get("extra_meta") or {}).get("cat_runtime_state_file")
            or {s["name"]: s for s in meta["converted_modules"]}
            != {s["name"]: s for s in source_meta["converted_modules"]}):
        raise ValueError("Export changed topology or inherited the source identity/resume state.")
    del loaded, serialized
    gc.collect()
    budget = audit_checkpoint(output, max_bpw=2.5, max_lora_rank=0)
    if not budget["budget_passed"]:
        raise ValueError(f"Exported budget failed: {budget['contract_violations']}")
    return {"checkpoint_id": saved["checkpoint_id"], "serialized_tensors_compared": count,
            "fresh_tensors_compared": fresh_count, "bit_budget": budget}
