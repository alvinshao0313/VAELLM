"""Token-weighted batch-gradient stability of fixed packed-bit candidates.

No model training occurs. The objective is local teacher-input Linear MSE, not
model KL. A single untouched heldout set measures every predetermined bit flip.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from experiments.linear_output.objectives import linear_output_mse
from experiments.qwen3_w25.bit_direction_diagnostic import (
    SOURCE_ID, TARGETS, _output_measurement, candidate_coordinates, stable_output_delta,
)
from experiments.qwen3_w25.local_bits import _code_banks, _digest
from sparse_bit_tuning.config import SparseBitTuningConfig, active_count
from sparse_bit_tuning.manager import SparseBitTuningManager


SEED = 137
CANDIDATES = 128
PREFIXES = (1, 2, 4, 8)


def weighted_score_gradients(gradients, token_counts, prefix):
    """FP64 analysis of token-weighted FP32 per-batch mean-loss gradients."""
    if gradients.ndim != 2 or gradients.shape[0] != len(token_counts):
        raise ValueError("One gradient vector and valid-token count per batch are required.")
    if (type(prefix) is not int or not 1 <= prefix <= len(token_counts)
            or any(type(n) is not int or n < 1 for n in token_counts)
            or not bool(torch.isfinite(gradients).all())):
        raise ValueError("Require finite gradients, positive token counts and a valid prefix.")
    weights = torch.tensor(token_counts[:prefix], device=gradients.device, dtype=torch.float64)
    return (gradients[:prefix].double() * weights[:, None]).sum(0) / weights.sum()


def direction_statistics(prediction, actual_delta, *, resolution):
    """Benefit precision/recall prevents a mostly-harmful population hiding errors."""
    prediction, actual = torch.as_tensor(prediction).double(), torch.as_tensor(actual_delta).double()
    if (prediction.ndim != 1 or prediction.shape != actual.shape or not len(prediction)
            or not bool(torch.isfinite(prediction).all() and torch.isfinite(actual).all())
            or resolution < 0):
        raise ValueError("Matching finite vectors and nonnegative resolution are required.")
    resolved = actual.abs() > resolution
    useful, harmful = actual < -resolution, actual > resolution
    predicted = prediction < 0
    true_positive = int((predicted & useful).sum())
    predicted_resolved = int((predicted & resolved).sum())
    positive = int(useful.sum())
    count = int(resolved.sum())
    correct = int(((torch.sign(prediction) == torch.sign(actual)) & resolved).sum())
    return {"candidate_count": len(prediction), "resolution": resolution,
        "resolved_count": count, "beneficial_count": positive, "harmful_count": int(harmful.sum()),
        "zero_or_below_resolution_count": int((~resolved).sum()),
        "predicted_decrease_total": int(predicted.sum()), "predicted_decrease_resolved": predicted_resolved,
        "true_beneficial_predictions": true_positive,
        "decrease_precision": true_positive / predicted_resolved if predicted_resolved else None,
        "decrease_recall": true_positive / positive if positive else None,
        "same_sign_count": correct, "same_sign_fraction": correct / count if count else None,
        "zero_prediction_resolved": int(((prediction == 0) & resolved).sum()),
        "always_predict_harmful_fraction": int(harmful.sum()) / count if count else None}


def document_batch_lengths(manifest):
    documents = manifest["train"]["documents"]
    if len(documents) != 32 or len(manifest["heldout"]["documents"]) != 32:
        raise ValueError("This diagnostic requires exactly 32 train and 32 heldout documents.")
    all_documents = documents + manifest["heldout"]["documents"]
    for key in ("row_index", "group_sha256", "token_sha256"):
        if len({row[key] for row in all_documents}) != 64:
            raise ValueError(f"Training/heldout document overlap: {key}.")
    lengths = [sum(int(row["valid_tokens"]) for row in documents[i:i + 4]) for i in range(0, 32, 4)]
    if sum(lengths) != manifest["train"]["valid_tokens"] or any(n < 1 for n in lengths):
        raise ValueError("Document batch token counts disagree with the manifest.")
    return lengths


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def prepare_data_manifest(output_path, previous_report):
    """CPU-only data preparation; seed is fixed, never selected by overlap/results."""
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.calibration import build_calibration_batches
    from experiments.qwen3_w25.local_recovery import BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _, _, manifest = build_calibration_batches(tokenizer, seed=SEED, train_documents=32,
                                               heldout_documents=32, max_length=256)
    prior_path = Path(previous_report).resolve()
    previous = json.loads(prior_path.read_text())
    old_documents = previous["data"]["train"]["documents"] + previous["data"]["heldout"]["documents"]
    overlap = {split: {key: sorted({row[key] for row in manifest[split]["documents"]}
                                 & {row[key] for row in old_documents})
                       for key in ("row_index", "group_sha256", "token_sha256")}
               for split in ("train", "heldout")}
    payload = {"seed": SEED, "candidates_per_stage": CANDIDATES, "prefixes": list(PREFIXES),
        "train_documents": 32, "heldout_documents": 32, "documents_per_batch": 4, "max_length": 256,
        "manifest": manifest, "manifest_sha256": _json_hash(manifest),
        "batch_tokens": document_batch_lengths(manifest), "previous_probe": str(prior_path),
        "previous_probe_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "overlap_with_previous_16_documents": overlap, "seed_resampling_performed": False}
    with Path(output_path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


@contextmanager
def _frozen_probe(module, module_name):
    """Own only this diagnostic's temporary decoder precision and SparseBit binding."""
    if module.trainable_decode or getattr(module, "_sparse_bit_binding", None) is not None:
        raise ValueError("An inactive compressed source module is required.")
    if module.residual_stages != 2 or module.parallel_parts != 1 or not module.parallel_stage_decode:
        raise ValueError("The fixed source must have a packed two-stage single-part decoder.")
    banks = _code_banks(module)
    decoder = module._parallel_stage_decoder
    parameters = dict(decoder.named_parameters())
    device = next(iter(parameters.values())).device
    if device.type != "cuda" or any(p.grad is not None for p in module.parameters()):
        raise ValueError("Use one CUDA-resident source module with no preexisting gradients.")
    if any(t.device != device for t in module.state_dict().values()):
        raise ValueError("The complete source module must reside on the execution device.")
    original = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    original_parameters = {k: v.detach().cpu().clone() for k, v in parameters.items()}
    digest = _digest(original, set())
    identities = {k: id(v) for k, v in module.named_parameters()}
    gradient_flags = [(p, p.requires_grad) for p in module.parameters()]
    training_flags = [(child, child.training) for child in module.modules()]
    runtime = {key: getattr(module, key) for key in ("trainable_decode", "cache_decoded_weight", "parallel_stage_decode")}
    had_policy, policy = hasattr(module, "_decoder_compute_dtype"), getattr(module, "_decoder_compute_dtype", None)
    n_bits = banks[0][1][0] * banks[0][1][-1]
    ratio = CANDIDATES / n_bits
    if active_count(n_bits, ratio) != CANDIDATES:
        raise ValueError("The affine sampler must activate exactly 128 candidates in each stage.")
    root, manager = nn.Module(), None
    root.add_module("target", module)
    evidence = {"source_module_sha256": digest, "source_decoder_storage_dtype": str(next(iter(parameters.values())).dtype)}
    try:
        module.eval().requires_grad_(False)
        for p in parameters.values():
            p.data = p.data.float()
        module.enable_sparse_bit_decode_graph(parallel_stage_decode=True)
        module._decoder_compute_dtype = torch.float32
        module.train()
        decoder.eval()
        manager = SparseBitTuningManager(root_model=root, targets=[(module_name, module)],
            target_devices={module_name: device}, training_seed=SEED, streaming=False,
            config=SparseBitTuningConfig(enabled=True, active_ratio=ratio, optimizer="adamw", bit_lr=4e-5,
                                        round_steps=128, proxy_coordinates="decoder_sensitivity"))
        manager.initialize_scores()
        yield manager, banks, original, evidence
        if any(p.grad is not None for p, _ in gradient_flags):
            raise RuntimeError("A frozen source parameter received gradients.")
        if any(not torch.equal(p.detach(), original_parameters[k].to(device=device, dtype=torch.float32))
               for k, p in parameters.items()):
            raise RuntimeError("A frozen decoder parameter changed during probing.")
        excluded = {"_parallel_stage_decoder." + k for k in parameters}
        if _digest(module.state_dict(), excluded) != _digest(original, excluded):
            raise RuntimeError("Persistent non-decoder state changed during temporary probing.")
        evidence["frozen_state_unchanged_before_restoration"] = True
    finally:
        if manager is not None:
            manager.detach_runtime()
        if hasattr(module, "_sparse_bit_binding"):
            delattr(module, "_sparse_bit_binding")
        for k, p in parameters.items():
            p.data = original_parameters[k].to(device).clone()
        module.load_state_dict(original, strict=True)
        for p, flag in gradient_flags:
            p.grad = None
            p.requires_grad_(flag)
        for child, flag in training_flags:
            child.training = flag
        for key, value in runtime.items():
            setattr(module, key, value)
        if had_policy:
            module._decoder_compute_dtype = policy
        elif hasattr(module, "_decoder_compute_dtype"):
            delattr(module, "_decoder_compute_dtype")
        module._build_parallel_stage_decode_plan()
        module.clear_decoded_weight_cache()
        if _digest(module.state_dict(), set()) != digest or {k: id(v) for k, v in module.named_parameters()} != identities:
            raise RuntimeError("Source persistent bytes or Parameter identities were not exactly restored.")
        evidence["persistent_state_and_parameter_identity_restored"] = True


def diagnose_gradient_accumulation(module, bundle, batch_tokens, *, module_name):
    if len(batch_tokens) != 8 or sum(batch_tokens) != len(bundle["train_inputs"]):
        raise ValueError("Eight document-derived training batches must cover every captured token.")
    device = next(module.parameters()).device
    reference = bundle["reference_weight"].to(device=device, dtype=torch.float32)
    train = bundle["train_inputs"].to(device)
    heldout = bundle["heldout_inputs"].to(device)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    result = {"module_name": module_name, "batch_tokens": batch_tokens, "optimizer_steps": 0,
        "decoder_and_full_weight_dtype": "torch.float32", "objective_dtype": "torch.float32",
        "gradient_dtype": "torch.float32", "gradient_weighted_analysis_dtype": "torch.float64",
        "shadow_reduction_dtype": "torch.float64", "candidates": [], "batches": []}
    with _frozen_probe(module, module_name) as (manager, banks, original, evidence):
        result["evidence"] = evidence
        specs = {stage: manager.bank_spec(module_name, stage, 0) for stage in range(2)}
        binding = manager.bindings[module_name]
        baseline_scores = {stage: manager.score_module.score_view(spec).detach().clone() for stage, spec in specs.items()}
        grouped = module._get_parallel_stage_grouped_vq_packed(device=device)
        baseline_grouped = grouped.clone()
        for stage, spec in specs.items():
            state = manager.sampler_states[spec.canonical_key]
            if spec.n_active != CANDIDATES:
                raise RuntimeError("Actual active count differs from the fixed candidate inventory.")
            scale = manager.score_module.coordinate_scales[spec.canonical_key]
            for ordinal, coordinate in enumerate(candidate_coordinates(state.active_indices(), spec.logical_shape)):
                hard = (int(original[banks[stage][0]][coordinate["block_index"], 0, coordinate["byte_index"]])
                        >> coordinate["byte_bit"]) & 1
                if bool(baseline_scores[stage][ordinal] >= 0) != bool(hard):
                    raise RuntimeError("Candidate coordinates and initial hard scores disagree.")
                result["candidates"].append({"stage": stage, "ordinal": ordinal, **coordinate,
                    "hard_bit": hard, "scale": scale, "full_flip_score_delta": scale * (1 - 2 * hard)})
        gradients = []
        baseline_weight = None
        for batch_index, x in enumerate(train.split(batch_tokens)):
            manager.score_module.zero_grad(set_to_none=True)
            with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
                weight = module._decode_weight(dtype=torch.float32)
                loss = linear_output_mse(weight - reference, x, chunk_size=128, recompute=True)
                loss.backward()
            if baseline_weight is None:
                baseline_weight = weight.detach()
            elif not torch.equal(baseline_weight, weight.detach()):
                raise RuntimeError("The unmodified baseline changed between batch gradient measurements.")
            values = []
            for stage, spec in specs.items():
                gradient = manager.score_module.score_chunks[spec.chunk_id].grad
                if gradient is None:
                    raise RuntimeError("Real packed score gradient is missing.")
                values.append(gradient[spec.score_start:spec.score_end].detach().cpu().clone())
            gradient = torch.cat(values)
            if gradient.dtype != torch.float32 or not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Expected finite FP32 production score gradients.")
            gradients.append(gradient)
            result["batches"].append({"index": batch_index, "document_indices": list(range(4 * batch_index, 4 * batch_index + 4)),
                "tokens": len(x), "loss_fp32": float(loss.detach()), "score_gradients": gradient.tolist()})
            del weight, loss
        gradients = torch.stack(gradients)
        flip_steps = torch.tensor([row["full_flip_score_delta"] for row in result["candidates"]], dtype=torch.float64)
        predictions = {f"batch_{i + 1}": gradients[i].double() * flip_steps for i in range(8)}
        predictions.update({f"prefix_{n}": weighted_score_gradients(gradients, batch_tokens, n) * flip_steps for n in PREFIXES})
        before_output, result["heldout_baseline"] = _output_measurement(baseline_weight - reference, heldout, 128)
        for index, row in enumerate(result["candidates"]):
            stage, ordinal = row["stage"], row["ordinal"]
            with torch.no_grad():
                manager.score_module.score_view(specs[stage])[ordinal].mul_(-1)
                flip_count = binding.project_nonstreaming_current()
            if flip_count != 1:
                raise RuntimeError("A finite single-bit probe must change exactly one grouped packed bit.")
            with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
                trial_weight = module._decode_weight(dtype=torch.float32).detach()
            after_output, metrics = _output_measurement(trial_weight - reference, heldout, 128)
            metrics.update(stable_output_delta(before_output, after_output))
            metrics["delta_loss_fp32"] = metrics["loss_fp32"] - result["heldout_baseline"]["loss_fp32"]
            row["heldout"] = metrics
            row["predictions"] = {key: float(value[index]) for key, value in predictions.items()}
            del trial_weight, after_output
            with torch.no_grad():
                manager.score_module.score_view(specs[stage]).copy_(baseline_scores[stage])
                restore_count = binding.project_nonstreaming_current()
            if restore_count != 1 or not torch.equal(grouped, baseline_grouped):
                raise RuntimeError("A probe did not restore exactly the original grouped packed baseline.")
            row["projected_flip_count"] = flip_count
            row["restored_flip_count"] = restore_count
            if (index + 1) % 32 == 0:
                print(json.dumps({"target": module_name, "completed_single_bit_probes": index + 1,
                                  "total": 2 * CANDIDATES}), flush=True)
        actual = torch.tensor([row["heldout"]["shadow_delta_mse_fp64"] for row in result["candidates"]], dtype=torch.float64)
        ulp = result["heldout_baseline"]["loss_fp32_ulp"]
        result["direction_statistics"] = {}
        for key, values in predictions.items():
            result["direction_statistics"][key] = {
                "all_nonzero": direction_statistics(values, actual, resolution=0.0),
                "above_baseline_loss_ulp": direction_statistics(values, actual, resolution=ulp)}
        batch_signs = torch.stack([torch.sign(predictions[f"batch_{i + 1}"]) for i in range(8)])
        result["batch_pair_sign_agreement"] = [{"batches": [i + 1, j + 1],
            "same_sign_count": int((batch_signs[i] == batch_signs[j]).sum()), "candidate_count": 2 * CANDIDATES}
            for i in range(8) for j in range(i + 1, 8)]
        result["prefix_tokens"] = {str(n): sum(batch_tokens[:n]) for n in PREFIXES}
    result.update(elapsed_seconds=time.perf_counter() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def run_diagnostic(checkpoint_dir, output_dir, prepared_json):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.calibration import build_calibration_batches
    from experiments.qwen3_w25.local_recovery import BASE_MODEL, capture_shard_inputs
    from experiments.qwen3_w25.summarize_results import _validate_checkpoint
    from train_utils.base_reference import load_frozen_base_reference_model
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    source, output, prepared_path = [Path(p).resolve() for p in (checkpoint_dir, output_dir, prepared_json)]
    if output.is_relative_to(source):
        raise ValueError("Diagnostic output must be outside its source checkpoint.")
    meta, audit = _validate_checkpoint(source)
    if meta["checkpoint_id"] != SOURCE_ID or not audit["budget_passed"] or Path(meta["base_model_path"]).resolve() != Path(BASE_MODEL).resolve():
        raise ValueError("The unchanged 552282 source and original teacher are required.")
    prepared = json.loads(prepared_path.read_text())
    if (prepared["seed"] != SEED or prepared["candidates_per_stage"] != CANDIDATES
            or prepared["prefixes"] != list(PREFIXES) or prepared["max_length"] != 256
            or prepared["train_documents"] != 32 or prepared["heldout_documents"] != 32):
        raise ValueError("Prepared configuration does not match the fixed experiment.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Bind one authorized CUDA device.")
    output.mkdir(parents=True, exist_ok=False)
    path, device = output / "diagnostic.json", torch.device("cuda:0")
    report = {"status": "preparing", "source_checkpoint_id": SOURCE_ID, "source_bit_budget": audit,
        "source_dir": str(source), "prepared_json": str(prepared_path),
        "prepared_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        "configuration": {"seed": SEED, "candidates_per_stage": CANDIDATES, "prefixes": list(PREFIXES),
            "gradient_batches": 8, "documents_per_batch": 4, "max_length": 256, "chunk_size": 128,
            "train_documents": 32, "heldout_documents": 32, "targets": list(TARGETS)},
        "limitations": ["Fixed-state local teacher-input MSE diagnostic, not training/convergence/model KL/downstream accuracy",
            "Prefix gradients weight batches by all valid tokens; FP64 combination analyzes FP32 per-batch gradients",
            "Heldout set is disjoint from current calibration, not claimed never used in previous experiments",
            "FP64 shadow loss uses the actual FP32 residual outputs; GEMM rounding remains",
            "One-loss-ULP filtering describes measurement scale, not a rigorous error bound",
            "All eight single batches are reported; no batch, candidate or seed is selected by heldout performance"],
        "targets": {}}

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    started = time.perf_counter()
    save()
    try:
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train, heldout, manifest = build_calibration_batches(tokenizer, seed=SEED, train_documents=32,
                                                            heldout_documents=32, max_length=256)
        lengths = document_batch_lengths(manifest)
        if _json_hash(manifest) != prepared["manifest_sha256"] or lengths != prepared["batch_tokens"]:
            raise RuntimeError("Runtime data differs from the already prepared document/token inventory.")
        report["data"] = manifest
        report["overlap_with_previous_16_documents"] = prepared["overlap_with_previous_16_documents"]
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
            raise RuntimeError("Actual source identity differs from prepared metadata.")
        model.eval().requires_grad_(False)
        report["source_loaded_state_tensors"] = len(model.state_dict())
        report["status"] = "running"
        save()
        for name in TARGETS:
            bundle = bundles.pop(name)
            if any(t.dtype != torch.bfloat16 for t in bundle.values()):
                raise ValueError("Teacher reference weights and captured inputs must retain BF16.")
            if any(len(bundle[f"{split}_inputs"]) != manifest[split]["valid_tokens"] for split in ("train", "heldout")):
                raise ValueError("Captured valid-token counts differ from the prepared manifest.")
            layer = model.get_submodule(name).to(device)
            report["targets"][name] = diagnose_gradient_accumulation(layer, bundle, lengths, module_name=name)
            layer.to("cpu")
            del bundle
            gc.collect()
            torch.cuda.empty_cache()
            save()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save()

