"""CAT geometry checks with real packed decoders; no training or quality claim."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from litebsq.autoencoder import Decoder
from litebsq.bitpack import unpack_uint8_tensor_to_bool
from litebsq.fused_multistage_decoder import packed_symmetric_decoder_supports_fuse
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear, prime_named_vae_linear_cache


PRESETS = {"dual32_bf16": (32, 2, torch.bfloat16), "single16_fp32": (16, 1, torch.float32),
           "mlp25_bf16": (32, 2, torch.bfloat16)}


def make_module(latent, preset="dual32_bf16", *, device="cuda"):
    dim, stages, storage_dtype = PRESETS[preset]
    hidden = 128 if preset == "mlp25_bf16" else 256
    rows = 256 * 224 // dim
    indices = torch.arange(0, 256, 8)
    qvalues, scales = encode_protected_channel_weight(torch.randn(32, 256).bfloat16(), quant_format="int8")
    bits = [torch.randint(0, 2, (rows, 1, latent), dtype=torch.bool) for _ in range(stages)]
    decoders = [Decoder(in_dim=latent, out_dim=dim, hidden_dim=hidden, num_res_blocks=0,
                        norm_type="layer", activation_type="swish", decoder_type="symmetric",
                        use_checkpoint=True).to(dtype=storage_dtype) for _ in range(stages)]
    # Match CAT's single-stage constructor and its unconditional final packing.
    payload = ({"vq_weight": bits[0], "decoder": decoders[0]} if stages == 1 else
               {"stage_vq_weights": bits, "stage_decoders": decoders})
    module = VAELinear(
        in_features=256, out_features=256, bias=None, codebook_dim=dim,
        compressed_in_features=224, transpose=True,
        protected_input_indices=indices, protected_input_qvalues=qvalues, protected_input_scales=scales,
        protected_channel_quant_format="int8", **payload)
    module.pack_parallel_stage_decoder_(trainable=False)
    module.to(device).eval().requires_grad_(False)
    module._build_parallel_stage_decode_plan()
    assert module.cache_decoded_weight and not module.trainable_decode
    assert module._parallel_stage_layout_is_stage_major and module._parallel_stage_restore_identity
    assert packed_symmetric_decoder_supports_fuse(module._parallel_stage_decoder)
    assert module.residual_stages == module._parallel_stage_decoder.num_models == stages
    assert all(parameter.dtype == storage_dtype for parameter in module.parameters())
    if preset == "mlp25_bf16":
        assert latent == 40
        stored = [module.get_stage_part_vq_storage(stage, 0).cpu() for stage in range(stages)]
        for raw, packed in zip(bits, stored, strict=True):
            assert packed.dtype == torch.uint8 and tuple(packed.shape) == (rows, 1, 5)
            assert packed.stride() == (5, 5, 1)
            assert torch.equal(unpack_uint8_tensor_to_bool(packed, logical_shape=raw.shape), raw)
        grouped = module._get_parallel_stage_grouped_vq_packed(device=next(module.parameters()).device).cpu()
        assert grouped.stride() == (10, 5, 1)
        original_grouped = torch.stack([raw[:, 0] for raw in bits], dim=1)
        assert torch.equal(unpack_uint8_tensor_to_bool(grouped, logical_shape=original_grouped.shape), original_grouped)
        storage_bits = sum(packed.numel() * 8 for packed in stored)
        logical_bits = sum(raw.numel() for raw in bits)
        assert storage_bits == logical_bits
        module._diagnostic_packing_audit = {
            "stage_packed_shapes": [list(packed.shape) for packed in stored],
            "stage_packed_strides": [list(packed.stride()) for packed in stored],
            "grouped_packed_shape": list(grouped.shape), "grouped_packed_stride": list(grouped.stride()),
            "logical_bits": logical_bits, "storage_bits": storage_bits, "padding_bits": 0,
            "pack_unpack_original_bool_exact": True, "grouped_stage_order_exact": True}
    return module


def delta(actual, reference):
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    difference = actual.float() - reference.float()
    return {"equal": torch.equal(actual, reference), "max_absolute_error": difference.abs().max().item(),
            "relative_l2_error": (difference.norm() / reference.float().norm()).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="dual32_bf16",
                        help="H256 D32/S2/BF16 (default), H256 D16/S1/FP32, or MLP K40/H128/D32/S2/BF16; BF16 activations.")
    args = parser.parse_args()
    dim, stages, storage_dtype = PRESETS[args.preset]
    path = Path(args.report)
    if path.exists():
        raise FileExistsError(path)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    report = {"status": "running", "purpose": "fused_deployment_geometry_not_quality_or_full_model_memory",
              "configuration": {"preset": args.preset, "hidden_dim": 128 if args.preset == "mlp25_bf16" else 256,
                                "residual_stages": stages, "codebook_dim": dim,
                                "decoder_type": "symmetric", "num_res_blocks": 0, "dtype": "bfloat16",
                                "decoder_storage_dtype": str(storage_dtype),
                                "linear_shape": [256, 256], "protected_int8_columns": 32,
                                "logical_code_rows": 256 * 224 // dim, "prewarm_group_size": 8},
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": {}}
    try:
        with torch.inference_mode():
            for latent in ((40,) if args.preset == "mlp25_bf16" else (32, 48)):
                report["phase"] = f"K{latent}: native_cold_cache"
                print(report["phase"], flush=True)
                modules = [make_module(latent, args.preset) for _ in range(8)]
                original = [{name: value.cpu().clone() for name, value in module.state_dict().items()}
                            for module in modules]
                sample = torch.randn(1, 8, 256, device="cuda:0", dtype=torch.bfloat16)
                VAELinear.reset_fuse_stats()
                native_output = modules[0](sample)
                native_weight = modules[0]._cached_weight.clone()
                native_stats = VAELinear.get_fuse_stats()
                assert native_stats["hit"] == 1 and native_stats["miss"] == native_stats["packed_u8_linear_hit"] == 0
                report["phase"] = f"K{latent}: grouped_prewarm"
                print(report["phase"], flush=True)
                for module in modules:
                    module.clear_decoded_weight_cache()
                VAELinear.reset_fuse_stats()
                warmed = prime_named_vae_linear_cache(
                    [(f"model.layers.{index}.self_attn.o_proj", module) for index, module in enumerate(modules)],
                    dtype=torch.bfloat16, group_size=8, compute_device="cuda:0")
                fused_stats = VAELinear.get_fuse_stats()
                assert warmed == {"total": 8, "warmed": 8, "skipped": 0, "failed": 0}
                assert fused_stats["hit"] == 8 and fused_stats["miss"] == fused_stats["packed_u8_linear_hit"] == 0
                assert all(module._cached_weight.dtype == torch.bfloat16 for module in modules)
                item = {"native_fuse_stats": native_stats, "grouped_fuse_stats": fused_stats,
                        "packed_decoder_num_models": modules[0]._parallel_stage_decoder.num_models,
                        "decoder_storage_dtypes": sorted({str(parameter.dtype) for parameter in modules[0].parameters()}),
                        "prewarm": warmed, "weight": delta(modules[0]._cached_weight, native_weight),
                        "output": delta(modules[0](sample), native_output)}
                if args.preset == "mlp25_bf16":
                    audit = modules[0]._diagnostic_packing_audit
                    assert all(module._diagnostic_packing_audit == audit for module in modules)
                    item["packing_audit"] = {"checked_modules": len(modules), **audit}
                for module, snapshot in zip(modules, original):
                    assert snapshot.keys() == module.state_dict().keys()
                    for name, value in module.state_dict().items():
                        assert value.dtype == snapshot[name].dtype
                        assert torch.equal(value.cpu().contiguous().reshape(-1).view(torch.uint8),
                                           snapshot[name].contiguous().reshape(-1).view(torch.uint8))
                item["persistent_state_unchanged"] = True
                report["cases"][str(latent)] = item
                print(json.dumps({"K": latent, **item}), flush=True)
                del modules, original, native_weight, native_output, sample
        torch.cuda.synchronize()
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
