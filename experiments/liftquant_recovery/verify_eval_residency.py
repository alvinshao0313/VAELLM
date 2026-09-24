"""Bounded real-model verification of streamed vs resident packed-BF16 eval."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.liftquant_recovery.evaluate_pair import resident_inference, streamed_inference
from litebsq.vae_linear import VAELinear
from train_utils.v6_model_loader import load_v6_model_checkpoint


def cached_weights(model):
    result = {}
    for name, module in model.named_modules():
        if not isinstance(module, VAELinear):
            continue
        weight = module._cached_weight
        if (weight is None or weight.device.type != "cuda" or weight.dtype != torch.bfloat16
                or not module.cache_decoded_weight or getattr(module, "trainable_decode", False)):
            raise ValueError(f"Expected a usable packed-BF16 inference cache: {name}")
        result[name] = weight.data_ptr()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ids", required=True, help="actual calibration_ids.pt produced by the recovery sampler")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-memory-gib", type=float, default=64)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.gpu_memory_gib * 2**30 / total, 1), device)
    all_ids = torch.load(args.ids, map_location="cpu", weights_only=True)
    if all_ids.ndim != 2 or all_ids.shape[0] < 2 or all_ids.shape[1] != 2048:
        raise ValueError("Expected real calibration data containing at least 2 x 2048 tokens.")
    ids = all_ids[:2].clone()
    del all_ids
    model, meta, loaded = load_v6_model_checkpoint(args.checkpoint, map_location="cpu", strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Strict native model load failed.")
    model.eval().requires_grad_(False)
    timings = {}
    started = time.perf_counter()
    with torch.no_grad(), streamed_inference(model, device):
        reference = model(input_ids=ids.to(device), use_cache=False).logits.detach().cpu()
    torch.cuda.synchronize()
    timings["streamed_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    with torch.no_grad(), resident_inference(model, device):
        if model.device != device or model.get_input_embeddings().weight.device != device:
            raise ValueError("Resident model and embedding devices must both match HFLM's input device.")
        before_cache = cached_weights(model)
        prediction = model(input_ids=ids.to(device), use_cache=False).logits
        if cached_weights(model) != before_cache:
            raise ValueError("A resident forward missed/replaced a prepared packed-BF16 cache.")
        if len(before_cache) != len(meta["compressed_targets"]):
            raise ValueError("Prepared cache count differs from the native compressed scope.")
        if prediction.shape != reference.shape or prediction.dtype != reference.dtype:
            raise ValueError("Residency changed output shape or precision.")
        torch.cuda.synchronize()
        timings["resident_prepare_and_forward_seconds"] = time.perf_counter() - started
        # Only token chunks become FP32 on GPU. Never hold two full FP32 logits.
        eps = torch.finfo(torch.bfloat16).eps
        max_abs = torch.zeros((), device=device, dtype=torch.float64)
        error_sq, reference_sq = max_abs.clone(), max_abs.clone()
        nll_reference, nll_prediction = max_abs.clone(), max_abs.clone()
        matches = torch.zeros((), device=device, dtype=torch.long)
        for sample in range(len(ids)):
            for start in range(0, ids.shape[1], 32):
                end = min(start + 32, ids.shape[1])
                a = reference[sample, start:end].to(device).float()
                b = prediction[sample, start:end].float()
                # One BF16 relative rounding unit; small absolute guard near zero.
                # Actual error, NLL and decisions are reported separately.
                torch.testing.assert_close(b, a, rtol=eps, atol=1e-4)
                delta = b - a
                max_abs = torch.maximum(max_abs, delta.abs().max().double())
                error_sq += delta.square().sum(dtype=torch.float64)
                reference_sq += a.square().sum(dtype=torch.float64)
                matches += (a.argmax(-1) == b.argmax(-1)).sum()
                usable = min(end, ids.shape[1] - 1) - start
                if usable > 0:
                    labels = ids[sample, start + 1:start + 1 + usable].to(device)
                    nll_reference += F.cross_entropy(a[:usable], labels, reduction="sum").double()
                    nll_prediction += F.cross_entropy(b[:usable], labels, reduction="sum").double()
                del a, b, delta
        scored_tokens = len(ids) * (ids.shape[1] - 1)
        report = dict(
            status="PASS", checkpoint=args.checkpoint, checkpoint_id=meta["checkpoint_id"],
            ids_path=args.ids, ids_shape=list(ids.shape),
            ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            max_abs_logit_error=max_abs.item(),
            relative_l2_logit_error=(error_sq / reference_sq.clamp_min(1e-30)).sqrt().item(),
            argmax_agreement=matches.item() / ids.numel(),
            reference_nll=nll_reference.item() / scored_tokens,
            resident_nll=nll_prediction.item() / scored_tokens,
            nll_delta=(nll_prediction - nll_reference).item() / scored_tokens,
            cached_linears=len(before_cache), cache_pointers_unchanged=True,
            output_dtype=str(prediction.dtype),
            tolerance=dict(logits_rtol=eps, logits_atol=1e-4,
                           rationale="one BF16 relative rounding unit; no dtype or arithmetic change"),
            max_memory_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            timings=timings, downstream_evaluation="separate unchanged evaluate_pair --limit 1 path",
        )
        del prediction
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
