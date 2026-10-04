"""Read-only block32 source diagnosis of packed/native versus whole-fused decode.

This measures existing kernels, without training, patching them, or exporting a
model. The 256-row reference uses the actual packed decoder's dtype boundaries.
Complete q/o matrices are needed to call the source's unmodified inverse rotation
and INT8 scatter, followed by four-token Linear forward comparisons.
"""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from litebsq.bitpack import unpack_uint8_tensor_to_bool
from litebsq.fused_multistage_decoder import (
    extract_packed_symmetric_stage_weights,
    fused_decode_packed_symmetric_decoder,
    _fused_forward_triton,
)
from litebsq.packed_bit_linear import packed_u8_parallel_linear
from litebsq.vae_linear import VAELinear
from train_utils.v6_model_loader import load_v6_model_checkpoint


TARGETS = (("model.layers.3.self_attn.q_proj", 32), ("model.layers.3.self_attn.o_proj", 48))
SOURCE_ID = "ab0280cf-3637-47e3-b771-074b74532a54"


def fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def difference(actual, reference):
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
    delta = actual.float() - reference.float()
    norm = reference.float().norm()
    return {"shape": list(actual.shape), "dtype": str(actual.dtype),
            "equal": torch.equal(actual, reference),
            "max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt()),
            "relative_l2": float(delta.norm() / norm.clamp_min(1e-12))}


def row_reference(decoder, packed, logical_bits):
    dense = unpack_uint8_tensor_to_bool(
        packed, logical_shape=(packed.shape[0], 2, logical_bits),
    ).to(torch.bfloat16)
    w_in, b_in, ln_w, ln_b, w_out, b_out = extract_packed_symmetric_stage_weights(decoder)
    # The first packed matmul rounds its weights to BF16 but retains FP32 bias
    # through addition; LayerNorm statistics and SiLU use native PyTorch ops.
    with torch.autocast("cuda", enabled=False):
        hidden = (torch.einsum("bsi,shi->bsh", dense.float(), w_in.bfloat16().float())
                  + b_in.float().unsqueeze(0)).bfloat16()
        normalized = F.layer_norm(hidden, (hidden.shape[-1],), None, None,
                                  decoder.norm_out.norm.eps)
        affine = normalized * ln_w.bfloat16().unsqueeze(0) + ln_b.bfloat16().unsqueeze(0)
        active = F.silu(affine)
        stages = (torch.einsum("bsh,soh->bso", active, w_out.bfloat16())
                  + b_out.bfloat16().unsqueeze(0))
    return dense, hidden, active, stages


def inspect_projection(source_module, *, expected_bits):
    # This copy has the E2E FP32 decoder storage, with unchanged source values.
    module = copy.deepcopy(source_module).cuda().eval().requires_grad_(False)
    module._parallel_stage_decoder.float()
    module._decoder_compute_dtype = torch.bfloat16
    decoder = module._parallel_stage_decoder
    assert module.residual_stages == 2 and module.parallel_parts == 1
    assert module.weight_rotation.block_size == 32
    assert decoder.decoder_type == "symmetric" and len(decoder.blocks) == 0
    assert decoder.num_models == 2 and decoder.in_dim == expected_bits
    assert module.low_rank_a is None and module.low_rank_b is None
    assert all(module.get_stage_part_vq_spec(i, 0)["logical_shape"][-1] == expected_bits for i in range(2))
    before = fingerprint(module.state_dict())
    grouped = module._get_parallel_stage_grouped_vq_packed(device=torch.device("cuda:0"))
    row_ids = torch.linspace(0, grouped.shape[0] - 1, 256, device="cuda").round().long()
    sampled = grouped.index_select(0, row_ids).contiguous()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dense, ref_hidden, ref_active, ref_stages = row_reference(decoder, sampled, expected_bits)
        hidden = packed_u8_parallel_linear(sampled, decoder.linear_in,
                                          logical_in_dim=expected_bits, activation_dtype=torch.bfloat16)
        active = F.silu(decoder.norm_out(hidden))
        stages = decoder.linear_out(active)
        native = stages.sum(dim=1).bfloat16()
        fused = fused_decode_packed_symmetric_decoder(decoder, dense)
        # Existing private entry exposes intermediate activations without changing
        # either kernel or decoder state. It still returns the real fused sum.
        weights = tuple(value.to(dense.dtype) for value in extract_packed_symmetric_stage_weights(decoder))
        aux_sum, fused_hidden, fused_active = _fused_forward_triton(
            dense, *weights, eps=decoder.norm_out.norm.eps, save_aux=True,
        )
        sample_metrics = {
            "rows": 256, "row_ids_sha256": fingerprint({"rows": row_ids}),
            "packed_first_linear_vs_pytorch": difference(hidden, ref_hidden),
            "native_activation_vs_pytorch": difference(active, ref_active),
            "native_stage_vs_pytorch": [difference(stages[:, i], ref_stages[:, i]) for i in range(2)],
            "native_sum_vs_pytorch": difference(native, ref_stages.sum(dim=1).bfloat16()),
            "wholefused_sum_vs_native": difference(fused, native),
            "wholefused_sum_vs_pytorch": difference(fused, ref_stages.sum(dim=1).bfloat16()),
            "wholefused_aux_entry_vs_normal_entry": difference(aux_sum, fused),
            "wholefused_first_linear_vs_native": [difference(fused_hidden[:, i], hidden[:, i]) for i in range(2)],
            "wholefused_activation_vs_native": [difference(fused_active[:, i], active[:, i]) for i in range(2)],
            "wholefused_stage_outputs": "Kernel exposes the stage sum; per-stage evidence above is pre-output activation.",
        }
        del dense, sampled, weights, hidden, active, stages, native, fused, aux_sum, fused_hidden, fused_active
        del ref_hidden, ref_active, ref_stages, grouped
        VAELinear.reset_fuse_stats()
        module.cache_decoded_weight = False
        module.clear_decoded_weight_cache()
        native_compressed = module._decode_compressed_weight(dtype=torch.bfloat16)
        native_inverse = module.weight_rotation(native_compressed, inverse=True)
        native_full = module._finalize_decoded_weight_from_compressed(
            native_compressed, dtype=torch.bfloat16, include_low_rank=False,
        )
        native_stats = VAELinear.get_fuse_stats()
        assert native_stats["packed_u8_linear_hit"] > 0
        VAELinear.reset_fuse_stats()
        module.cache_decoded_weight = True
        module.clear_decoded_weight_cache()
        fused_compressed = module._decode_compressed_weight(dtype=torch.bfloat16)
        fused_inverse = module.weight_rotation(fused_compressed, inverse=True)
        fused_full = module._finalize_decoded_weight_from_compressed(
            fused_compressed, dtype=torch.bfloat16, include_low_rank=False,
        )
        fused_stats = VAELinear.get_fuse_stats()
        assert fused_stats["hit"] > 0 and fused_stats["packed_u8_linear_hit"] == 0
        full_metrics = {
            "compressed_shape": list(native_compressed.shape),
            "wholefused_vs_packed_before_inverse": difference(fused_compressed, native_compressed),
            "wholefused_vs_packed_after_inverse": difference(fused_inverse, native_inverse),
            "wholefused_vs_packed_after_int8_scatter": difference(fused_full, native_full),
            "packed_decode_stats": native_stats, "wholefused_decode_stats": fused_stats,
        }
        del native_compressed, fused_compressed, native_inverse, fused_inverse
        inputs = torch.randn(4, module.in_features, generator=torch.Generator().manual_seed(31),
                             dtype=torch.bfloat16).cuda()
        native_output = F.linear(inputs, native_full, module.bias)
        fused_output = F.linear(inputs, fused_full, module.bias)
        full_metrics["four_token_linear_forward"] = difference(fused_output, native_output)
        # An actual cold-cache VAELinear forward checks that the measured restored
        # fused weight is also the one consumed by the deployment entry point.
        module.clear_decoded_weight_cache()
        actual_output = module(inputs)
        full_metrics["actual_cold_cache_vs_measured_fused_forward"] = difference(actual_output, fused_output)
        torch.testing.assert_close(actual_output, fused_output, rtol=0, atol=0)
        assert torch.equal(module._cached_weight, fused_full)
    after = fingerprint(module.state_dict())
    assert before == after
    result = {"codebook_bits": expected_bits, "stages": 2, "decoder_dim": 32,
              "decoder_hidden": decoder.hidden_dim, "rotation": module.weight_rotation.to_spec(),
              "decoder_storage_dtype": "torch.float32", "activation_dtype": "torch.bfloat16",
              "source_value_promotion_only": True, "sample": sample_metrics, "full_projection": full_metrics,
              "diagnostic_module_state_before": before, "diagnostic_module_state_after": after}
    del module, decoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    destination = Path(args.report)
    if destination.exists():
        raise FileExistsError(destination)
    start = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, meta, loaded = load_v6_model_checkpoint(args.checkpoint_dir, map_location="cpu", strict=True)
    assert meta["checkpoint_id"] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys
    model.eval().requires_grad_(False)
    source_before = fingerprint(model.state_dict())
    torch.cuda.reset_peak_memory_stats()
    results = {}
    for name, bits in TARGETS:
        print(f"Inspecting {name}, K={bits}", flush=True)
        results[name] = inspect_projection(model.get_submodule(name), expected_bits=bits)
        print(json.dumps(results[name], ensure_ascii=False, allow_nan=False), flush=True)
    source_after = fingerprint(model.state_dict())
    assert source_before == source_after
    report = {"status": "complete", "purpose": "existing_kernel_diagnosis_not_training_or_quality_evaluation",
              "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()), "checkpoint_id": SOURCE_ID,
              "source_model_state_before": source_before, "source_model_state_after": source_after,
              "allow_tf32": False, "results": results,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "elapsed_seconds": time.monotonic() - start,
              "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                  (Path(__file__), Path("litebsq/fused_multistage_decoder.py"), Path("litebsq/vae_linear.py"),
                   Path("litebsq/parallel_layers.py"), Path("litebsq/packed_bit_linear.py"),
                   Path("rotation/weight_preprocess.py"))}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"COMPLETE {destination}", flush=True)


if __name__ == "__main__":
    main()
