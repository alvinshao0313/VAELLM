"""Read-only full-Gram objective/gradient/timing comparison on the protected base."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import math
from pathlib import Path
import time

import torch
from torch import nn
from transformers import AutoTokenizer

from experiments.linear_output.objectives import linear_output_mse
from experiments.qwen3_w25.bit_recipe_pilot import document_plan, validate_recipe_source
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.full_linear_gram import build_full_linear_gram, full_linear_gram_mse
from experiments.qwen3_w25.local_bits import _code_banks, _digest
from experiments.qwen3_w25.local_decoder import _schema, _storage_bits, _validate
from experiments.qwen3_w25.local_recovery import BASE_MODEL, capture_shard_inputs
from experiments.qwen3_w25.pilot_local_decoder import _write_report
from experiments.qwen3_w25.protected_bit_recovery import (
    CODE_FILES as RECOVERY_CODE_FILES, _file_sha256, load_recipe_selection, source_state_sha256,
    validate_pilot_source_model,
)
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager
from sparse_bit_tuning.triton_kernels import sparse_bit_triton_available
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


TARGETS = ("model.layers.3.self_attn.o_proj", "model.layers.3.mlp.down_proj")
CODE_FILES = (*RECOVERY_CODE_FILES, "experiments/qwen3_w25/full_linear_gram.py",
              "experiments/qwen3_w25/gram_validation.py", "experiments/qwen3_w25/validate_full_linear_gram.py")
CHUNK_SIZE = 256


def tensor_difference(actual, reference):
    """Compare every element, with bounded CPU FP64 reduction workspace."""
    if actual.shape != reference.shape or actual.device.type != "cpu" or reference.device.type != "cpu":
        raise ValueError("Gradient comparisons require equally shaped complete CPU tensors.")
    a, b = actual.reshape(-1), reference.reshape(-1)
    squares = [0.0, 0.0, 0.0]
    dot = absolute_max = 0.0
    for left, right in zip(a.split(1_048_576), b.split(1_048_576)):
        left, right = left.double(), right.double()
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise FloatingPointError("Non-finite complete gradient.")
        difference = left - right
        squares = [old + float(value.square().sum()) for old, value in zip(squares, (difference, left, right))]
        dot += float((left * right).sum())
        absolute_max = max(absolute_max, float(difference.abs().max()))
    norm_diff, norm_a, norm_b = map(math.sqrt, squares)
    return {"shape": list(actual.shape), "numel": actual.numel(), "dtype": str(actual.dtype),
            "max_absolute_difference": absolute_max, "difference_l2": norm_diff,
            "actual_l2": norm_a, "reference_l2": norm_b,
            "relative_l2": norm_diff / max(norm_b, 1e-300),
            "cosine": dot / (norm_a * norm_b) if norm_a and norm_b else None}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_start(device):
    if device.type != "cuda":
        return 0
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def _memory_result(device, baseline):
    if device.type != "cuda":
        return {"baseline_allocated_bytes": 0, "peak_allocated_bytes": 0,
                "peak_increment_bytes": 0, "peak_reserved_bytes": 0}
    peak = torch.cuda.max_memory_allocated(device)
    return {"baseline_allocated_bytes": baseline, "peak_allocated_bytes": peak,
            "peak_increment_bytes": peak - baseline, "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def benchmark_backward(loss_factory, clear_gradients, read_gradients, device):
    """One warmup plus two synchronized forward/backward measurements; no updates."""
    rows, losses = [], []
    for iteration in range(3):
        clear_gradients()
        _sync(device)
        baseline = _memory_start(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=False):
            loss = loss_factory()
        _sync(device)
        forward_done = time.perf_counter()
        loss.backward()
        _sync(device)
        backward_done = time.perf_counter()
        value = float(loss.detach())
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError("Invalid validation loss.")
        losses.append(value)
        if iteration:
            rows.append({"forward_seconds": forward_done - start,
                         "backward_seconds": backward_done - forward_done,
                         "total_seconds": backward_done - start, **_memory_result(device, baseline)})
        del loss
    gradients = []
    for gradient in read_gradients():
        if gradient is None:
            raise RuntimeError("A complete requested gradient is missing.")
        gradient = gradient.detach().cpu().clone()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Non-finite validation gradient.")
        gradients.append(gradient)
    clear_gradients()
    return {"loss": losses[-1], "repeat_losses": losses, "warmup_passes": 1,
            "measured_passes": 2, "timings": rows,
            "mean_forward_seconds": sum(row["forward_seconds"] for row in rows) / 2,
            "mean_backward_seconds": sum(row["backward_seconds"] for row in rows) / 2,
            "mean_total_seconds": sum(row["total_seconds"] for row in rows) / 2}, gradients


def measure_error_gradient(error, objective):
    leaf = error.detach().clone().requires_grad_(True)

    def clear():
        leaf.grad = None

    return benchmark_backward(lambda: objective(leaf), clear, lambda: [leaf.grad], leaf.device)


@contextmanager
def packed_proxy_graph(module, reference, train_x, heldout_x):
    """Temporarily attach the production graph; restore all original state even on error."""
    decoder, parameters, device = _validate(module, reference, train_x, heldout_x, 1, 1.0, CHUNK_SIZE)
    if device.type != "cuda" or not sparse_bit_triton_available():
        raise RuntimeError("Real packed-proxy validation requires CUDA and Triton.")
    banks = _code_banks(module)
    original = {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}
    original_parameters = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    state_before, schema_before, bits_before = _digest(original, set()), _schema(module), _storage_bits(module)
    ids_before = {name: id(parameter) for name, parameter in module.named_parameters()}
    flags = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    modes = [(child, child.training) for child in module.modules()]
    runtime = {key: getattr(module, key) for key in ("trainable_decode", "cache_decoded_weight", "parallel_stage_decode")}
    had_policy = hasattr(module, "_decoder_compute_dtype")
    policy = getattr(module, "_decoder_compute_dtype", None)
    root, manager, successful = nn.Module(), None, False
    root.add_module("layer", module)
    evidence = {"source_module_sha256_before": state_before, "optimizer_updates": 0, "final_commit_called": False,
                "packed_banks": [{"name": name, "logical_shape": list(shape)} for name, shape in banks]}
    try:
        module.eval().requires_grad_(False)
        for parameter in parameters.values():
            parameter.data = parameter.data.float()
        module.enable_sparse_bit_decode_graph(parallel_stage_decode=True)
        module._decoder_compute_dtype = torch.float32
        module.train()
        decoder.eval()
        module._build_parallel_stage_decode_plan()
        # LR and schedule satisfy the manager API; neither optimizer_step nor
        # final_commit is called, so they cannot affect the validation gradients.
        config = SparseBitTuningConfig(enabled=True, active_ratio=1.0, optimizer="adamw", bit_lr=1.0,
                                      weight_decay=0.0, round_steps=128,
                                      proxy_coordinates="decoder_sensitivity").normalized()
        manager = SparseBitTuningManager(root_model=root, targets=[("layer", module)],
            target_devices={"layer": device}, training_seed=31, config=config, streaming=False)
        manager.configure_schedule(total_optimizer_steps=128)
        manager.initialize_scores()
        scales = dict(manager.score_module.coordinate_scales)
        score_digest = _digest(manager.score_module.state_dict(), set())
        yield root, manager, evidence
        if (score_digest != _digest(manager.score_module.state_dict(), set())
                or scales != manager.score_module.coordinate_scales):
            raise RuntimeError("Read-only comparison changed proxy scores or coordinate scales.")
        if any(parameter.grad is not None for parameter, _ in flags):
            raise RuntimeError("A frozen source parameter received a gradient.")
        if any(not torch.equal(parameter.detach(), original_parameters[name].to(device, torch.float32))
               for name, parameter in parameters.items()):
            raise RuntimeError("Frozen FP32 decoder changed during comparison.")
        manager.detach_runtime()
        for name, parameter in parameters.items():
            parameter.data = original_parameters[name].to(device).clone()
        state_after = _digest(module.state_dict(), set())
        if (state_after != state_before or _schema(module) != schema_before or _storage_bits(module) != bits_before
                or ids_before != {name: id(parameter) for name, parameter in module.named_parameters()}):
            raise RuntimeError("Validation changed persistent code/INT8/decoder state or parameter identity.")
        evidence.update(source_module_sha256_after=state_after, complete_state_unchanged=True,
                        proxy_scores_unchanged=True, coordinate_scales_fixed=True,
                        decoder_compute_parameters_unchanged=True, parameter_ids_unchanged=True,
                        persistent_storage_bits_before=bits_before, persistent_storage_bits_after=_storage_bits(module))
        successful = True
    finally:
        if manager is not None:
            manager.detach_runtime()
        if hasattr(module, "_sparse_bit_binding"):
            delattr(module, "_sparse_bit_binding")
        if hasattr(root, "sparse_bit_tuning"):
            delattr(root, "sparse_bit_tuning")
        for name, parameter in parameters.items():
            parameter.data = original_parameters[name].to(device).clone()
        if not successful:
            module.load_state_dict(original, strict=True)
        for parameter, flag in flags:
            parameter.grad = None
            parameter.requires_grad_(flag)
        for child, mode in modes:
            child.training = mode
        for key, value in runtime.items():
            setattr(module, key, value)
        if had_policy:
            module._decoder_compute_dtype = policy
        elif hasattr(module, "_decoder_compute_dtype"):
            delattr(module, "_decoder_compute_dtype")
        module._build_parallel_stage_decode_plan()
        module.clear_decoded_weight_cache()


@torch.no_grad()
def native_endpoint(module, reference, train_x, heldout_x):
    delta = module._decode_weight(dtype=train_x.dtype).float() - reference
    return {"train_mse": float(linear_output_mse(delta, train_x, chunk_size=CHUNK_SIZE, recompute=False)),
            "heldout_mse": float(linear_output_mse(delta, heldout_x, chunk_size=CHUNK_SIZE, recompute=False)),
            "weight_mse": float(delta.square().mean())}


def compare_module_gram(module, bundle):
    device = next(module.parameters()).device
    reference = bundle["reference_weight"].to(device=device, dtype=torch.float32)
    train_x, heldout_x = (bundle[key].to(device) for key in ("train_inputs", "heldout_inputs"))
    native_before = native_endpoint(module, reference, train_x, heldout_x)
    results, baseline_gradients = {}, {}
    with packed_proxy_graph(module, reference, train_x, heldout_x) as (root, manager, evidence):
        with torch.no_grad():
            error = module._decode_weight(dtype=torch.float32) - reference
        specs = sorted(manager.bank_specs, key=lambda spec: spec.stage_idx)
        if len(specs) != 2 or [spec.stage_idx for spec in specs] != [0, 1]:
            raise RuntimeError("Expected the complete two packed proxy banks.")
        coordinates = [{"stage": spec.stage_idx, "scale": manager.score_module.coordinate_scales[spec.canonical_key],
                        "radius": manager.score_module.score_radius(spec), "numel": spec.score_end - spec.score_start}
                       for spec in specs]
        def proxy_gradients():
            return [None if manager.score_module.score_chunks[spec.chunk_id].grad is None else
                    manager.score_module.score_chunks[spec.chunk_id].grad[spec.score_start:spec.score_end] for spec in specs]

        for name, dtype in (("streaming_fp32", None), ("gram_fp32", torch.float32), ("gram_fp64", torch.float64)):
            gram, preprocessing = None, None
            if dtype is not None:
                _sync(device)
                baseline = _memory_start(device)
                start = time.perf_counter()
                gram = build_full_linear_gram(train_x, dtype=dtype)
                _sync(device)
                preprocessing = {"seconds": time.perf_counter() - start,
                    "matrix_shape": list(gram.matrix.shape), "matrix_dtype": str(gram.matrix.dtype),
                    "matrix_bytes": gram.matrix.numel() * gram.matrix.element_size(), "token_count": gram.token_count,
                    **_memory_result(device, baseline)}
            objective = ((lambda delta: linear_output_mse(delta, train_x, chunk_size=CHUNK_SIZE, recompute=True))
                         if gram is None else (lambda delta: full_linear_gram_mse(delta, gram)))
            error_report, error_gradients = measure_error_gradient(error, objective)
            proxy_report, proxy_values = benchmark_backward(
                lambda: objective(module._decode_weight(dtype=torch.float32) - reference),
                lambda: root.zero_grad(set_to_none=True), proxy_gradients, device)
            if any(value.numel() != row["numel"] or not bool(torch.count_nonzero(value))
                   for value, row in zip(proxy_values, coordinates)):
                raise RuntimeError("Both complete packed stages must receive nonzero gradients.")
            current = {"error": error_gradients, "proxy": proxy_values}
            comparisons = {}
            if name == "streaming_fp32":
                baseline_gradients = current
            else:
                for label, values in current.items():
                    comparisons[label] = [tensor_difference(value, original)
                                          for value, original in zip(values, baseline_gradients[label])]
                for label, report in (("error", error_report), ("proxy", proxy_report)):
                    original_loss = results["streaming_fp32"][label]["loss"]
                    report["loss_absolute_difference"] = abs(report["loss"] - original_loss)
                    report["loss_relative_difference"] = abs(report["loss"] - original_loss) / max(abs(original_loss), 1e-300)
            results[name] = {"gram_preprocessing": preprocessing, "error": error_report,
                             "proxy": proxy_report, "complete_gradient_comparison_to_streaming": comparisons}
            print(f"Completed objective/gradient measurements: {name}", flush=True)
            del gram, objective, error_gradients, proxy_values, current
    module.clear_decoded_weight_cache()
    native_after = native_endpoint(module, reference, train_x, heldout_x)
    if native_after != native_before:
        raise RuntimeError("The unchanged source did not reproduce its deterministic native X endpoints.")
    evidence["native_endpoints_exact"] = True
    return {"input_shape": list(train_x.shape), "heldout_shape": list(heldout_x.shape),
            "weight_shape": list(reference.shape), "native_before": native_before, "native_after": native_after,
            "proxy_coordinates": coordinates, "comparisons": results, "evidence": evidence,
            "memory_scope": "Each case retains original BF16 train/heldout X, source graph and FP32 E; only its own Gram is resident. Gradient CPU copies/comparison are outside measured timings.",
            "timing_scope": "error: objective only; proxy: real packed decode plus objective; synchronized wall time; Gram construction reported separately"}


def run_gram_validation(checkpoint_dir, pilot_json, output_dir):
    source, pilot_path, output = (Path(path).resolve() for path in (checkpoint_dir, pilot_json, output_dir))
    if output.is_relative_to(source) or output.exists():
        raise ValueError("Validation output must be new and outside the immutable source checkpoint.")
    meta, audit = _validate_checkpoint(source)
    pilot, pilot_sha, _ = load_recipe_selection(pilot_path, source, meta)
    if not audit["budget_passed"]:
        raise ValueError("Source exceeds the no-LoRA actual bit budget.")
    output.mkdir(parents=True)
    repo, device = Path(__file__).resolve().parents[2], torch.device("cuda:0")
    report = {"status": "preparing", "source_checkpoint_dir": str(source), "source_checkpoint_id": meta["checkpoint_id"],
              "source_state_sha256": source_state_sha256(source), "source_bit_budget": audit,
              "pilot_json": str(pilot_path), "pilot_sha256": pilot_sha,
              "code_sha256": {name: _file_sha256(repo / name) for name in CODE_FILES}, "targets": {},
              "configuration": {"targets": list(TARGETS), "seed": 31, "train_documents": 512,
                  "heldout_documents": 64, "max_length": 256, "chunk_size": CHUNK_SIZE, "capture_passes": 1,
                  "warmup_passes": 1, "measured_passes": 2, "optimizer_updates": 0, "checkpoint_written": False,
                  "allow_tf32": False, "deterministic_algorithms": True,
                  "boundary": "Initial-state objective and complete gradient comparison only; does not verify 128-step hard-code endpoints or downstream accuracy."}}
    path = output / "validation.json"
    _write_report(path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train, heldout, data = build_calibration_batches(tokenizer, seed=31, train_documents=512,
                                                       heldout_documents=64, max_length=256)
        plan = document_plan(train, heldout, data)
        if data != pilot["data"] or plan != pilot["document_plan"]:
            raise ValueError("Gram validation data differs from the completed protected-source pilot.")
        report.update(data=data, document_plan=plan)
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL,
            map_location="cpu", strict=True, expected_kind="final_model")
        model.eval().requires_grad_(False)
        validate_recipe_source(model, loaded_meta)
        validate_pilot_source_model(model, pilot)
        before = _digest(model.state_dict(), set())
        start = time.perf_counter()
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundles = capture_shard_inputs(teacher, TARGETS, train, heldout)
        _sync(device)
        report["capture_seconds"] = time.perf_counter() - start
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        report["status"] = "running"
        _write_report(path, report)
        for target in TARGETS:
            bundle = bundles.pop(target)
            for name, split in (("train_inputs", "train"), ("heldout_inputs", "heldout")):
                if len(bundle[name]) != data[split]["valid_tokens"] or bundle[name].dtype != torch.bfloat16:
                    raise ValueError("Captured input tokens/dtype differ from the fixed data identity.")
            layer = model.get_submodule(target).to(device)
            print(f"Comparing {target}", flush=True)
            try:
                report["targets"][target] = compare_module_gram(layer, bundle)
            finally:
                layer.cpu()
                layer._build_parallel_stage_decode_plan()
                layer.clear_decoded_weight_cache()
            del bundle
            gc.collect()
            torch.cuda.empty_cache()
            _write_report(path, report)
        after = _digest(model.state_dict(), set())
        if before != after:
            raise RuntimeError("The complete source model state changed during validation.")
        report.update(status="complete", model_state_sha256_before=before,
                      model_state_sha256_after=after, complete_model_state_unchanged=True)
        _write_report(path, report)
        return report
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        _write_report(path, report)
        raise
