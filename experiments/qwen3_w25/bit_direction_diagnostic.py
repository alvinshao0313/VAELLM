"""Finite-bit direction probes of a frozen decoder and teacher-input output MSE.

This is an isolated mechanism diagnostic, not model KL, training, or downstream
evaluation. Candidate bits are selected before looking at any gradient or loss.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.linear_output.objectives import linear_output_mse
from experiments.qwen3_w25.local_bits import _code_banks, _digest
from sparse_bit_tuning.config import SparseBitTuningConfig, active_count
from sparse_bit_tuning.manager import SparseBitTuningManager
from sparse_bit_tuning.triton_kernels import sparse_bit_triton_available


SOURCE_ID = "552282f2-868a-4105-90df-46f3c6f6a818"
TARGETS = ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj")


def candidate_coordinates(logical_indices, logical_shape):
    rows, parts, width = map(int, logical_shape)
    values = list(map(int, logical_indices))
    if parts != 1 or rows < 1 or width < 1 or len(set(values)) != len(values):
        raise ValueError("Candidates require unique logical bits in a single-part bank.")
    if any(value < 0 or value >= rows * width for value in values):
        raise ValueError("Candidate logical bit lies outside the bank.")
    return [{"logical_index": value, "block_index": value // width,
             "latent_bit": value % width, "byte_index": (value % width) // 8,
             "byte_bit": value % width % 8} for value in values]


def fixed_probe_groups(count):
    if type(count) is not int or count < 4:
        raise ValueError("At least four predetermined candidates per stage are required.")
    singles = [(f"stage{stage}_bit{ordinal:02d}", [(stage, ordinal)])
               for stage in range(2) for ordinal in range(count)]
    return singles + [
        ("stage0_first4", [(0, i) for i in range(4)]),
        ("stage1_first4", [(1, i) for i in range(4)]),
        ("cross_stage_first2", [(stage, i) for stage in range(2) for i in range(2)]),
    ]


def ste_flip_prediction(score_gradient, scale, hard_bit):
    if not math.isfinite(float(scale)) or scale <= 0 or hard_bit not in (0, 1, False, True):
        raise ValueError("A positive sensitivity scale and binary hard bit are required.")
    return float(score_gradient) * float(scale) * (1 - 2 * int(hard_bit))


def stable_output_delta(before, after):
    """FP64 diagnostic of the SAME already-rounded output tensors, not X @ dW.

    The production local objective still squares/reduces in FP32. This shadow
    measurement avoids subtracting two large scalar losses; it does not remove
    FP32 GEMM rounding or certify an exact-real-arithmetic change.
    """
    if before.shape != after.shape or before.ndim != 2 or before.numel() == 0:
        raise ValueError("Matching nonempty [tokens, output] tensors are required.")
    if before.dtype != after.dtype or before.dtype not in (torch.float32, torch.float64):
        raise ValueError("Residual tensors must have matching FP32/FP64 dtype.")
    left, right = before.double(), after.double()
    difference = right - left
    return {
        "shadow_delta_mse_fp64": float((difference * (right + left)).mean()),
        "output_delta_abs_max": float(difference.abs().max()),
        "output_delta_rms": float(difference.square().mean().sqrt()),
        "output_changed_elements": int(torch.count_nonzero(difference)),
        "output_elements": before.numel(),
    }


@torch.no_grad()
def _output_measurement(error_weight, inputs, chunk_size):
    # Match linear_output_mse's FP32 matmul, per-chunk square/sum and accumulation.
    outputs = []
    total = error_weight.new_zeros((), dtype=torch.float32)
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        for chunk in inputs.split(chunk_size):
            residual = F.linear(chunk.float(), error_weight.float())
            total = total + residual.square().sum()
            outputs.append(residual)
    value = total / (len(inputs) * error_weight.shape[0])
    residuals = torch.cat(outputs)
    if not bool(torch.isfinite(residuals).all()) or not bool(torch.isfinite(value)):
        raise FloatingPointError("Nonfinite local output measurement.")
    ulp = torch.nextafter(value, torch.full_like(value, float("inf"))) - value
    return residuals, {"loss_fp32": float(value), "shadow_loss_fp64": float(residuals.double().square().mean()),
                       "loss_fp32_ulp": float(ulp)}


def _direction_summary(rows):
    singles = [row for row in rows if len(row["bits"]) == 1]
    summaries = {}
    for split in ("train", "heldout"):
        for resolved in (False, True):
            selected = [row for row in singles if row["ste_train_prediction"] != 0
                        and row[split]["shadow_delta_mse_fp64"] != 0
                        and (not resolved or row[split]["above_baseline_fp32_loss_ulp"])]
            summaries[f"{split}_{'above_loss_ulp' if resolved else 'all_nonzero'}"] = {
                "eligible": len(selected),
                "same_sign_count": sum((row["ste_train_prediction"] < 0)
                                       == (row[split]["shadow_delta_mse_fp64"] < 0) for row in selected),
                "predicted_decrease_but_increased": sum(row["ste_train_prediction"] < 0
                    and row[split]["shadow_delta_mse_fp64"] > 0 for row in selected),
                "agreement_fraction": (sum((row["ste_train_prediction"] < 0)
                    == (row[split]["shadow_delta_mse_fp64"] < 0) for row in selected) / len(selected)
                    if selected else None),
            }
    return summaries


def diagnose_bit_directions(module, reference_weight, train_inputs, heldout_inputs, *,
                            module_name, candidates_per_stage=16, seed=31, chunk_size=128):
    """Probe real packed STE and finite flips; restore every persistent source byte.

    Decoder parameters and decoded full weights are FP32 in this local proxy,
    including the original reconstruction of protection/rotation/normalization.
    Teacher weights/inputs come from BF16 capture and are cast to FP32 for MSE.
    No optimizer step, commit, round transition, or checkpoint export is performed.
    """
    from litebsq.vae_linear import VAELinear

    if not isinstance(module, VAELinear) or module.residual_stages != 2 or module.parallel_parts != 1:
        raise ValueError("A real two-stage, single-part VAELinear is required.")
    decoder = module._modules.get("_parallel_stage_decoder")
    if decoder is None or not module.parallel_stage_decode or decoder.num_models != 2:
        raise ValueError("An existing packed decoder is required.")
    if module.trainable_decode or getattr(module, "_sparse_bit_binding", None) is not None:
        raise ValueError("The source must not have an active bit or decoder training runtime.")
    if module.original_weight is not None or module.always_use_original or not module.temporary:
        raise ValueError("The source must be an ordinary compressed module.")
    if module.has_low_rank_residual() or module.has_sparse_residual():
        raise ValueError("This diagnostic excludes LoRA and sparse residual branches.")
    if any(isinstance(child, (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm))
           for child in module.modules()):
        raise ValueError("Mutable/random decoder layers are outside this diagnostic.")
    parameters = dict(decoder.named_parameters())
    device = next(iter(parameters.values())).device
    if device.type != "cuda" or not sparse_bit_triton_available():
        raise RuntimeError("Real packed direction probes require CUDA/Triton, with no CPU fallback.")
    if any(t.device != device for t in module.state_dict().values()) or any(p.grad is not None for p in module.parameters()):
        raise ValueError("Place the complete source module on one device with no existing gradients.")
    if type(chunk_size) is not int or chunk_size < 1 or type(seed) is not int or not module_name:
        raise ValueError("A positive chunk size, integer seed and module name are required.")
    probes = fixed_probe_groups(candidates_per_stage)
    banks = _code_banks(module)
    n_bits = math.prod(banks[0][1])
    ratio = candidates_per_stage / n_bits
    if candidates_per_stage > n_bits or active_count(n_bits, ratio) != candidates_per_stage:
        raise ValueError("Candidate ratio must yield exactly the specified number per bank.")
    if reference_weight.shape != (module.out_features, module.in_features):
        raise ValueError("Reference must be the original full teacher Linear [out,in] weight.")
    for value in (reference_weight, train_inputs, heldout_inputs):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("Finite floating reference/inputs are required.")
    for inputs in (train_inputs, heldout_inputs):
        if inputs.ndim != 2 or inputs.shape[1] != module.in_features or len(inputs) < 1:
            raise ValueError("Inputs must contain nonempty original-coordinate valid tokens.")

    original = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    original_parameters = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    original_digest = _digest(original, set())
    parameter_ids = {name: id(p) for name, p in module.named_parameters()}
    grad_flags = [(p, p.requires_grad) for p in module.parameters()]
    training_flags = [(child, child.training) for child in module.modules()]
    runtime = {key: getattr(module, key) for key in ("trainable_decode", "cache_decoded_weight", "parallel_stage_decode")}
    had_policy, policy = hasattr(module, "_decoder_compute_dtype"), getattr(module, "_decoder_compute_dtype", None)
    root, manager = nn.Module(), None
    root.add_module("target", module)
    reference = reference_weight.detach().to(device=device, dtype=torch.float32)
    inputs = {"train": train_inputs.detach().to(device), "heldout": heldout_inputs.detach().to(device)}
    result = {"module_name": module_name, "source_module_sha256": original_digest,
              "settings": {"candidates_per_stage": candidates_per_stage, "seed": seed,
                  "active_ratio": ratio, "chunk_size": chunk_size, "proxy_coordinates": "decoder_sensitivity",
                  "source_decoder_storage_dtype": str(next(iter(parameters.values())).dtype),
                  "decoder_storage_dtype_during_probe": "torch.float32",
                  "decoder_compute_dtype": "torch.float32", "decoded_full_weight_dtype": "torch.float32",
                  "sensitivity_calibration_compute_dtype": "torch.float32",
                  "objective_compute_and_reduce_dtype": "torch.float32", "shadow_reduce_dtype": "torch.float64",
                  "reference": "original BF16 teacher weight cast to FP32; original-coordinate BF16 teacher inputs",
                  "candidate_rule": "production affine sampler, first round, seed fixed before gradients/losses",
                  "train_tokens": len(train_inputs), "heldout_tokens": len(heldout_inputs)}, "probes": []}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        module.eval().requires_grad_(False)
        for parameter in parameters.values():
            parameter.data = parameter.data.float()
        module.enable_sparse_bit_decode_graph(parallel_stage_decode=True)
        module._decoder_compute_dtype = torch.float32
        module.train()
        decoder.eval()
        manager = SparseBitTuningManager(root_model=root, targets=[(module_name, module)],
            target_devices={module_name: device}, training_seed=seed, streaming=False,
            config=SparseBitTuningConfig(enabled=True, active_ratio=ratio, optimizer="adamw", bit_lr=4e-5,
                round_steps=128, proxy_coordinates="decoder_sensitivity"))
        manager.initialize_scores()
        binding = manager.bindings[module_name]
        specs = {stage: manager.bank_spec(module_name, stage, 0) for stage in range(2)}
        baseline_scores = {stage: manager.score_module.score_view(spec).detach().clone() for stage, spec in specs.items()}
        grouped = module._get_parallel_stage_grouped_vq_packed(device=device)
        baseline_grouped = grouped.clone()
        with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
            weight = module._decode_weight(dtype=torch.float32)
            loss = linear_output_mse(weight - reference, inputs["train"], chunk_size=chunk_size, recompute=True)
            loss.backward()
        if any(p.grad is not None for p, _ in grad_flags):
            raise RuntimeError("A frozen source parameter received a gradient.")
        gradient_loss = float(loss.detach())
        baseline_weight = weight.detach()
        del weight, loss
        result["banks"], candidates = [], {}
        for stage, spec in specs.items():
            state = manager.sampler_states[spec.canonical_key]
            score = manager.score_module.score_view(spec)
            gradient = manager.score_module.score_chunks[spec.chunk_id].grad
            if spec.n_active != candidates_per_stage or gradient is None:
                raise RuntimeError("Candidate inventory or real score gradient is missing.")
            gradient = gradient[spec.score_start:spec.score_end].detach().cpu()
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Nonfinite real packed score gradients.")
            scale = manager.score_module.coordinate_scales[spec.canonical_key]
            coordinates = candidate_coordinates(state.active_indices(), spec.logical_shape)
            storage = original[banks[stage][0]]
            for ordinal, row in enumerate(coordinates):
                hard = (int(storage[row["block_index"], 0, row["byte_index"]]) >> row["byte_bit"]) & 1
                if bool(score[ordinal] >= 0) != bool(hard):
                    raise RuntimeError("Sampler, score and original packed hard bit disagree.")
                candidates[(stage, ordinal)] = {"stage": stage, "ordinal": ordinal, **row,
                    "hard_bit": hard, "score_gradient": float(gradient[ordinal]), "scale": scale,
                    "full_flip_score_delta": scale * (1 - 2 * hard),
                    "ste_train_prediction": ste_flip_prediction(gradient[ordinal], scale, hard)}
            result["banks"].append({"stage": stage, "canonical_key": spec.canonical_key,
                "logical_shape": list(spec.logical_shape), "scale": scale, "score_radius": scale / 2,
                "sampler": state.to_metadata(), "score_gradient_absolute_max": float(gradient.abs().max()),
                "candidates": [candidates[(stage, i)] for i in range(candidates_per_stage)]})
        baseline_outputs, result["baseline"] = {}, {}
        for split, x in inputs.items():
            baseline_outputs[split], result["baseline"][split] = _output_measurement(baseline_weight - reference, x, chunk_size)
        if result["baseline"]["train"]["loss_fp32"] != gradient_loss:
            raise RuntimeError("Measurement and differentiable local objective have different FP32 baselines.")
        for title, bits in probes:
            with torch.no_grad():
                for stage, ordinal in bits:
                    manager.score_module.score_view(specs[stage])[ordinal].mul_(-1)
                flipped = binding.project_nonstreaming_current()
            if flipped != len(bits):
                raise RuntimeError(f"{title}: projected flip count {flipped} differs from {len(bits)}.")
            with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
                trial_weight = module._decode_weight(dtype=torch.float32).detach()
            row = {"name": title, "bits": [candidates[bit] for bit in bits], "projected_flip_count": flipped,
                   "ste_train_prediction": sum(candidates[bit]["ste_train_prediction"] for bit in bits),
                   "decoded_weight_delta_abs_max": float((trial_weight - baseline_weight).abs().max())}
            for split, x in inputs.items():
                outputs, metrics = _output_measurement(trial_weight - reference, x, chunk_size)
                metrics.update(stable_output_delta(baseline_outputs[split], outputs))
                metrics["delta_loss_fp32"] = metrics["loss_fp32"] - result["baseline"][split]["loss_fp32"]
                metrics["baseline_loss_fp32_ulp"] = result["baseline"][split]["loss_fp32_ulp"]
                metrics["above_baseline_fp32_loss_ulp"] = abs(metrics["shadow_delta_mse_fp64"]) > metrics["baseline_loss_fp32_ulp"]
                row[split] = metrics
                del outputs
            del trial_weight
            with torch.no_grad():
                for stage, spec in specs.items():
                    manager.score_module.score_view(spec).copy_(baseline_scores[stage])
                restored = binding.project_nonstreaming_current()
            if restored != len(bits) or not torch.equal(grouped, baseline_grouped):
                raise RuntimeError(f"{title}: grouped packed baseline was not exactly restored.")
            row["restored_flip_count"] = restored
            result["probes"].append(row)
            print(json.dumps({"target": module_name, "probe": title,
                "prediction": row["ste_train_prediction"], "train_delta": row["train"]["shadow_delta_mse_fp64"],
                "heldout_delta": row["heldout"]["shadow_delta_mse_fp64"]}), flush=True)
        singles = {(row["bits"][0]["stage"], row["bits"][0]["ordinal"]): row
                   for row in result["probes"] if len(row["bits"]) == 1}
        for row in result["probes"]:
            if len(row["bits"]) > 1:
                for split in inputs:
                    total = sum(singles[(bit["stage"], bit["ordinal"])][split]["shadow_delta_mse_fp64"] for bit in row["bits"])
                    row[split]["sum_of_individual_shadow_deltas"] = total
                    row[split]["group_minus_sum_shadow_delta"] = row[split]["shadow_delta_mse_fp64"] - total
        result["direction_summary"] = _direction_summary(result["probes"])
        if any(not torch.equal(parameter.detach(), original_parameters[name].to(device=device, dtype=torch.float32))
               for name, parameter in parameters.items()):
            raise RuntimeError("A frozen decoder parameter changed during probing.")
        decoder_keys = {"_parallel_stage_decoder." + name for name in parameters}
        if _digest(module.state_dict(), decoder_keys) != _digest(original, decoder_keys):
            raise RuntimeError("A persistent non-decoder tensor changed during temporary probing.")
        result["elapsed_seconds"] = time.perf_counter() - started
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    finally:
        if manager is not None:
            manager.detach_runtime()
        if hasattr(module, "_sparse_bit_binding"):
            delattr(module, "_sparse_bit_binding")
        for name, parameter in parameters.items():
            parameter.data = original_parameters[name].to(device=device).clone()
        module.load_state_dict(original, strict=True)
        for parameter, requires_grad in grad_flags:
            parameter.grad = None
            parameter.requires_grad_(requires_grad)
        for child, training in training_flags:
            child.training = training
        for key, value in runtime.items():
            setattr(module, key, value)
        if had_policy:
            module._decoder_compute_dtype = policy
        elif hasattr(module, "_decoder_compute_dtype"):
            delattr(module, "_decoder_compute_dtype")
        module._build_parallel_stage_decode_plan()
        module.clear_decoded_weight_cache()
        if _digest(module.state_dict(), set()) != original_digest:
            raise RuntimeError("Diagnostic did not restore every persistent source byte.")
        if {name: id(p) for name, p in module.named_parameters()} != parameter_ids:
            raise RuntimeError("Diagnostic replaced a source Parameter object.")
    result["restoration"] = {"persistent_state_exact": True, "parameter_objects_preserved": True,
                             "frozen_decoder_unchanged_before_restoration": True,
                             "persistent_non_decoder_unchanged_before_restoration": True,
                             "checkpoint_written": False, "optimizer_steps": 0}
    return result


def run_bit_direction_diagnostic(checkpoint_dir, output_dir):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.calibration import build_calibration_batches
    from experiments.qwen3_w25.local_recovery import BASE_MODEL, capture_shard_inputs
    from experiments.qwen3_w25.summarize_results import _validate_checkpoint
    from train_utils.base_reference import load_frozen_base_reference_model
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    source, output = Path(checkpoint_dir).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(source):
        raise ValueError("Diagnostic output must be separate from its source checkpoint.")
    meta, audit = _validate_checkpoint(source)
    if meta["checkpoint_id"] != SOURCE_ID or not audit["budget_passed"] or Path(meta["base_model_path"]).resolve() != Path(BASE_MODEL).resolve():
        raise ValueError("This diagnostic requires the unchanged best 552282 source and its original teacher.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Bind exactly one authorized CUDA GPU for this isolated diagnostic.")
    output.mkdir(parents=True, exist_ok=False)
    path, device = output / "diagnostic.json", torch.device("cuda:0")
    repo = Path(__file__).resolve().parents[2]
    files = ("experiments/qwen3_w25/bit_direction_diagnostic.py", "experiments/qwen3_w25/diagnose_bit_directions.py",
             "experiments/qwen3_w25/calibration.py", "experiments/qwen3_w25/local_recovery.py",
             "experiments/qwen3_w25/local_bits.py", "experiments/linear_output/objectives.py",
             "sparse_bit_tuning/manager.py", "sparse_bit_tuning/packed_ops.py", "sparse_bit_tuning/triton_kernels.py",
             "sparse_bit_tuning/sampler.py", "sparse_bit_tuning/coordinates.py", "litebsq/vae_linear.py")
    report = {"status": "preparing", "source_checkpoint_id": SOURCE_ID, "source_dir": str(source),
        "source_bit_budget": audit, "source_meta_sha256": hashlib.sha256((source / "checkpoint_meta.json").read_bytes()).hexdigest(),
        "purpose": "fixed_decoder_local_teacher_input_direction_diagnostic_not_model_KL_or_downstream_accuracy",
        "configuration": {"targets": list(TARGETS), "train_documents": 8, "heldout_documents": 8,
            "max_length": 256, "seed": 31, "candidates_per_stage": 16, "chunk_size": 128,
            "teacher_microbatch": 1, "allow_tf32": False, "checkpoint_written": False},
        "limitations": ["teacher-input local MSE is not model KL or downstream accuracy",
            "heldout documents are disjoint within this diagnostic; not a claim of never used in earlier experiments",
            "small predetermined candidates do not estimate whole-model bit usefulness",
            "FP64 shadow reduction measures FP32 residual outputs; it does not remove GEMM rounding",
            "above-loss-ULP is a descriptive resolution filter, not an exact-arithmetic error bound"],
        "code_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files}, "targets": {}}

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    started = time.perf_counter()
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train, heldout, data = build_calibration_batches(tokenizer, seed=31, train_documents=8,
            heldout_documents=8, max_length=256)
        report["data"] = data
        save()
        torch.cuda.reset_peak_memory_stats(device)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundles = capture_shard_inputs(teacher, TARGETS, train, heldout)
        report["teacher_capture_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL, map_location="cpu", strict=True)
        if loaded_meta["checkpoint_id"] != SOURCE_ID:
            raise RuntimeError("Loaded source identity differs from validated metadata.")
        model.eval().requires_grad_(False)
        report["source_loaded_state_tensors"] = len(model.state_dict())
        report["status"] = "running"
        save()
        for name in TARGETS:
            layer = model.get_submodule(name).to(device)
            bundle = bundles.pop(name)
            if any(value.dtype != torch.bfloat16 for value in bundle.values()):
                raise ValueError("The real teacher capture must retain BF16 weights and activations.")
            if any(len(bundle[f"{split}_inputs"]) != data[split]["valid_tokens"] for split in ("train", "heldout")):
                raise RuntimeError("Captured token inventory differs from its document manifest.")
            report["targets"][name] = diagnose_bit_directions(layer, bundle["reference_weight"],
                bundle["train_inputs"], bundle["heldout_inputs"], module_name=name)
            layer.to("cpu")
            del bundle
            gc.collect()
            torch.cuda.empty_cache()
            save()
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save()
