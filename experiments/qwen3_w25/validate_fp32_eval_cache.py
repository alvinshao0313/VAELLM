"""Bounded real-source validation of BF16 eval caches with FP32 decoder storage."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import torch

from litebsq.vae_linear import VAELinear
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint


TARGET = "model.layers.3.self_attn.o_proj"


def decoder_fingerprint(targets):
    digest = hashlib.sha256()
    dtypes = set()
    for module_name, module in targets:
        for name, parameter in module._parallel_stage_decoder.named_parameters():
            dtypes.add(str(parameter.dtype))
            value = parameter.detach().cpu().contiguous()
            digest.update(f"{module_name}.{name}|{value.dtype}|{tuple(value.shape)}".encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest(), sorted(dtypes)


def difference(actual, reference):
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    delta = actual.float() - reference.float()
    rms = reference.float().square().mean().sqrt()
    return {"equal": torch.equal(actual, reference), "dtype": str(actual.dtype),
            "shape": list(actual.shape), "max_absolute_error": delta.abs().max().item(),
            "rmse": delta.square().mean().sqrt().item(), "reference_rms": rms.item(),
            "relative_l2_error": (delta.square().mean().sqrt() / rms).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    destination = Path(args.report)
    if destination.exists():
        raise FileExistsError(destination)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("fp32_eval_cache_validation")
    start = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model, meta, loaded = load_v6_model_checkpoint(args.checkpoint_dir, map_location="cpu", strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    model.eval().requires_grad_(False)
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, VAELinear)]
    assert len(targets) == 252
    decoder_before, storage_dtypes = decoder_fingerprint(targets)
    assert storage_dtypes == ["torch.float32"]
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16
    module = model.get_submodule(TARGET)
    assert module.residual_stages == 2 and module.parallel_parts == 1
    assert all(module.get_stage_part_vq_spec(stage, 0)["logical_shape"][-1] == 48 for stage in range(2))
    assert all(getattr(layer, "_decoder_compute_dtype", None) is None for _, layer in targets)
    original = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    torch.cuda.reset_peak_memory_stats()
    prepared = {}
    with torch.inference_mode():
        stats = _prepare_model_for_eval(model, "cuda:0", 8, logger, prepared)
        assert stats["total"] == stats["warmed"] == 252
        assert stats["failed"] == stats["skipped"] == 0 and stats["prewarm_dtype"] == "torch.bfloat16"
        assert all(layer._cached_weight.dtype == torch.bfloat16 for _, layer in targets)
        cache_pointers = {name: layer._cached_weight.data_ptr() for name, layer in targets}
        grouped_weight = module._cached_weight.clone()
        captured = {}

        def before(_module, inputs):
            captured["input"] = inputs[0].detach().clone()

        def after(_module, _inputs, output):
            captured["output"] = output.detach().clone()

        pre = module.register_forward_pre_hook(before)
        post = module.register_forward_hook(after)
        try:
            model(input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda:0"), use_cache=False)
        finally:
            pre.remove()
            post.remove()
        assert captured["input"].dtype == torch.bfloat16
        assert cache_pointers == {name: layer._cached_weight.data_ptr() for name, layer in targets}
        grouped_output = module(captured["input"])
        assert torch.equal(grouped_output, captured["output"])
        # Clear only the cached tensor. Disabling the cache policy would select
        # packed_u8 decode instead of normal eval's cold-cache full fused path.
        assert module.cache_decoded_weight
        module.clear_decoded_weight_cache()
        native_weight = module._decode_weight(dtype=torch.bfloat16, include_low_rank=False)
        assert module._cached_weight is None
        native_output = module(captured["input"])
        assert torch.equal(module._cached_weight, native_weight)
        weight_difference = difference(grouped_weight, native_weight)
        output_difference = difference(grouped_output, native_output)
    decoder_after, storage_dtypes_after = decoder_fingerprint(targets)
    assert decoder_before == decoder_after and storage_dtypes == storage_dtypes_after
    assert original.keys() == module.state_dict().keys()
    for name, value in module.state_dict().items():
        assert value.dtype == original[name].dtype
        torch.testing.assert_close(value.cpu(), original[name], rtol=0, atol=0)
    torch.cuda.synchronize()
    report = {"status": "complete", "purpose": "cache_dtype_validation_not_quality_evaluation",
              "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
              "checkpoint_id": meta["checkpoint_id"], "target": TARGET,
              "decoder_storage_dtypes": storage_dtypes, "decoder_sha256_before": decoder_before,
              "decoder_sha256_after": decoder_after, "target_persistent_state_unchanged": True,
              "prewarm": stats, "native_input_shape": list(captured["input"].shape),
              "native_input_dtype": str(captured["input"].dtype),
              "all_252_caches_reused_by_forward": True,
              "native_comparison_cache_policy": True,
              "native_comparison_cold_cache": True,
              "grouped_vs_native_weight": weight_difference,
              "grouped_vs_native_output": output_difference,
              "elapsed_seconds": time.monotonic() - start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                              (Path(__file__), Path("tools/cat_eval.py"))}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Validation completed: weight=%s output=%s report=%s", weight_difference,
                output_difference, destination)


if __name__ == "__main__":
    main()
