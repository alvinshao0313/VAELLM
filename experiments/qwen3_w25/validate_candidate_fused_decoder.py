"""Validate one isolated candidate kernel without replacing the live runtime."""

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

import litebsq.fused_multistage_decoder as live_fused
from litebsq.packed_bit_linear import packed_u8_parallel_linear
from experiments.qwen3_w25.diagnose_decoder_kernel import (
    SOURCE_ID, TARGETS, difference, fingerprint, row_reference,
)
from experiments.qwen3_w25.validate_h256_deployment import make_module
from train_utils.v6_model_loader import load_v6_model_checkpoint


CANDIDATE_SHA = "ac09ece5e46bb71463e6e63de0eb2d05107db4f2bf48029ae2ba4b94f9cd7abd"


def resources(candidate):
    result = {}
    for name in ("_fused_fwd_kernel", "_fused_fwd_kernel_layered"):
        for cache in getattr(candidate, name).cache.values():
            for key, kernel in cache.items():
                digest = hashlib.sha256((name + key).encode()).hexdigest()
                metadata = kernel.metadata
                result[digest] = {"kernel": name, "shared_bytes": metadata.shared,
                    "num_warps": metadata.num_warps, "num_stages": metadata.num_stages,
                    "dot_ir": [line.strip() for line in kernel.asm["ttir"].splitlines() if "tt.dot " in line]}
    return result


def packed_native(decoder, packed):
    hidden = packed_u8_parallel_linear(packed, decoder.linear_in,
        logical_in_dim=decoder.in_dim, activation_dtype=torch.bfloat16)
    return decoder.linear_out(F.silu(decoder.norm_out(hidden))).sum(dim=1).bfloat16()


def row_case(candidate, module, result, *, check_gradients):
    decoder = module._parallel_stage_decoder
    before = fingerprint(module.state_dict())
    decoder.requires_grad_(check_gradients)
    grouped = module._get_parallel_stage_grouped_vq_packed(device=torch.device("cuda:0"))
    ids = torch.linspace(0, grouped.shape[0] - 1, 256, device="cuda").round().long()
    packed = grouped.index_select(0, ids).contiguous()
    with torch.no_grad():
        dense, _, _, reference_stages = row_reference(decoder, packed, decoder.in_dim)
        reference = reference_stages.sum(dim=1).bfloat16()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        native = packed_native(decoder, packed)
        actual = candidate.fused_decode_packed_symmetric_decoder(decoder, dense)
        result["sample_forward_vs_packed"] = difference(actual, native)
        result["sample_forward_vs_pytorch"] = difference(actual, reference)
        result["packed_vs_pytorch"] = difference(native, reference)
        epsilon = torch.finfo(torch.bfloat16).eps
        torch.testing.assert_close(actual, native, atol=2 * epsilon, rtol=2 * epsilon)
        if check_gradients:
            parameters = tuple(decoder.parameters())
            gradient = torch.randn(actual.shape, generator=torch.Generator().manual_seed(47),
                                   dtype=torch.bfloat16).cuda()
            expected_grads = torch.autograd.grad(native, parameters, gradient)
            actual_grads = torch.autograd.grad(actual, parameters, gradient)
            result["gradients"] = {}
            failures = []
            for (name, parameter), expected, actual_gradient in zip(
                decoder.named_parameters(), expected_grads, actual_grads, strict=True,
            ):
                assert actual_gradient.dtype == parameter.dtype
                metrics = difference(actual_gradient, expected)
                limit = 1e-5 if name == "linear_in.conv.weight" else 2 * epsilon
                metrics["relative_l2_limit_from_existing_patch_test"] = limit
                result["gradients"][name] = metrics
                if metrics["relative_l2"] > limit:
                    failures.append((name, metrics["relative_l2"], limit))
            assert not failures, f"Candidate gradients violate existing patch-test limits: {failures}"
    decoder.requires_grad_(False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        single = candidate.fused_decode_packed_symmetric_decoder(decoder, dense)
        grouped_outputs = candidate.fused_decode_batched_same_shape_decoders([decoder] * 8, [dense] * 8)
        assert len(grouped_outputs) == 8
        result["group8_vs_single"] = [difference(output, single) for output in grouped_outputs]
        for output in grouped_outputs:
            torch.testing.assert_close(output, single, atol=2 * epsilon, rtol=2 * epsilon)
    assert before == fingerprint(module.state_dict())
    result["module_state_unchanged"] = True


def full_projection(candidate, module):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        module.cache_decoded_weight = False
        module.clear_decoded_weight_cache()
        native = module._decode_compressed_weight(dtype=torch.bfloat16)
        dense = module._get_parallel_stage_grouped_vq(dtype=torch.bfloat16, device=torch.device("cuda:0"))
        flat = candidate.fused_decode_packed_symmetric_decoder(module._parallel_stage_decoder, dense)
        split_shape = (module.compressed_in_features, module.compressed_out_features) if module.transpose else (
            module.compressed_out_features, module.compressed_in_features)
        split = flat.reshape(split_shape)
        actual = split.t().contiguous() if module.transpose else split.contiguous()
        native_inverse = module.weight_rotation(native, inverse=True)
        actual_inverse = module.weight_rotation(actual, inverse=True)
        native_full = module._finalize_decoded_weight_from_compressed(native, dtype=torch.bfloat16, include_low_rank=False)
        actual_full = module._finalize_decoded_weight_from_compressed(actual, dtype=torch.bfloat16, include_low_rank=False)
        inputs = torch.randn(4, module.in_features, generator=torch.Generator().manual_seed(31),
                             dtype=torch.bfloat16).cuda()
        return {"before_inverse": difference(actual, native),
                "after_inverse": difference(actual_inverse, native_inverse),
                "after_int8_scatter": difference(actual_full, native_full),
                "four_token_linear_forward": difference(F.linear(inputs, actual_full), F.linear(inputs, native_full))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_file", required=True)
    parser.add_argument("--expected_candidate_sha256", default=CANDIDATE_SHA,
                        help="Explicit identity of the isolated candidate; default is the verified 09-24 patch.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    path, candidate_path = Path(args.report), Path(args.candidate_file).resolve()
    if path.exists():
        raise FileExistsError(path)
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    assert candidate_sha == args.expected_candidate_sha256
    live_path = Path(live_fused.__file__).resolve()
    live_sha = hashlib.sha256(live_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("w25_isolated_candidate_fused_decoder", candidate_path)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    assert sys.modules["litebsq.fused_multistage_decoder"] is live_fused
    assert candidate.fused_decode_packed_symmetric_decoder.__module__ == spec.name
    start = time.monotonic()
    report = {"status": "running", "candidate_sha256": candidate_sha, "live_kernel_sha256": live_sha,
              "candidate_file": str(candidate_path), "purpose": "isolated_kernel_math_and_resources_only",
              "cases": {}, "source_checkpoint_id": SOURCE_ID}
    model = None
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        model, meta, loaded = load_v6_model_checkpoint(args.checkpoint_dir, map_location="cpu", strict=True)
        assert meta["checkpoint_id"] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys
        model.eval().requires_grad_(False)
        source_before = fingerprint(model.state_dict())
        report["source_model_state_before"] = source_before
        torch.cuda.reset_peak_memory_stats()
        for target, bits in TARGETS:
            report["phase"] = target
            print(f"CASE {target}", flush=True)
            module = copy.deepcopy(model.get_submodule(target)).cuda().eval().requires_grad_(False)
            module._parallel_stage_decoder.float()
            module._decoder_compute_dtype = torch.bfloat16
            assert module.weight_rotation.block_size == 32
            assert module._parallel_stage_decoder.in_dim == bits
            before = fingerprint(module.state_dict())
            item = {"origin": "block32_source", "K": bits, "H": 128, "D": 32, "S": 2,
                    "storage_dtype": "torch.float32", "activation_dtype": "torch.bfloat16"}
            report["cases"][target] = item
            previous_resources = set(resources(candidate))
            row_case(candidate, module, item, check_gradients=True)
            item["full_projection"] = full_projection(candidate, module)
            assert before == fingerprint(module.state_dict())
            item["compiled_resources"] = {key: value for key, value in resources(candidate).items() if key not in previous_resources}
            print(json.dumps(item, allow_nan=False), flush=True)
            del module
            gc.collect()
            torch.cuda.empty_cache()
        report["phase"] = "H256_K48_FP32"
        print("CASE H256_K48_FP32", flush=True)
        torch.manual_seed(31)
        module = make_module(48)
        module._parallel_stage_decoder.float()
        module._decoder_compute_dtype = torch.bfloat16
        item = {"origin": "existing_H256_CAT_deployment_fixture", "K": 48, "H": 256, "D": 32, "S": 2,
                "storage_dtype": "torch.float32", "activation_dtype": "torch.bfloat16"}
        report["cases"][report["phase"]] = item
        previous_resources = set(resources(candidate))
        row_case(candidate, module, item, check_gradients=False)
        item["compiled_resources"] = {key: value for key, value in resources(candidate).items() if key not in previous_resources}
        print(json.dumps(item, allow_nan=False), flush=True)
        source_after = fingerprint(model.state_dict())
        assert source_before == source_after
        report["source_model_state_after"] = source_after
        torch.cuda.synchronize()
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      compiled_resources=resources(candidate),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        assert hashlib.sha256(live_path.read_bytes()).hexdigest() == live_sha
        assert sys.modules["litebsq.fused_multistage_decoder"] is live_fused
        report["live_kernel_file_and_import_unchanged"] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"RESULT {report['status']} {path}", flush=True)


if __name__ == "__main__":
    main()
