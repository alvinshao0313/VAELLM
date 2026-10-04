"""Read-only regional losses for five deployed BF16 checkpoints on one archive.

The launch record binds source code/evidence hashes, HF shard stats, and five
ordered model identities (source552, width_h128, width_h256, decoder, joint).
Each model needs checkpoint_dir, checkpoint_id, metadata_sha256, weight_stat
({path, size, mtime_ns, inode}) and evidence_sha256 before execution. Pending
models may appear only in a prepared record. CUDA remains visible even during
the existing strict v6 CPU-map load. No training or new loss reduction is used.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

from train_utils.distill_loss_core import (
    build_prediction_token_masks, compute_kl_top_mass_token_loss,
    compute_sft_token_loss, reduce_weighted_token_loss,
)

MODEL_NAMES = ("source552", "width_h128", "width_h256", "decoder", "joint")
SOURCE_ID = "552282f2-868a-4105-90df-46f3c6f6a818"
ARCHIVE_SHA256 = "1b481a4748c2735be951b45db9998fe8bec8e43f69410965cf720c5b3a6c2d53"
PROMPT_WEIGHT = 0.3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_stat(item):
    stat = Path(item["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != tuple(item[k] for k in ("size", "mtime_ns", "inode")):
        raise ValueError(f"Pinned input stat changed: {item['path']}")


def check_inputs(record, *, require_bound=True):
    models = record["models"]
    if tuple(item["name"] for item in models) != MODEL_NAMES:
        raise ValueError("Exactly the five ordered, unique model identities are required.")
    if record["archive"]["sha256"] != ARCHIVE_SHA256:
        raise ValueError("Use the original fixed 256+8 archive, not a resampled dataset.")
    if require_bound and any(item.get("binding_status") != "bound" for item in models):
        raise ValueError("All five model identities must be bound before execution.")
    pins = {**record["source_sha256"], **record["input_evidence_sha256"],
            **record["hf_source"]["metadata_sha256"], record["archive"]["path"]: ARCHIVE_SHA256}
    for path, digest in pins.items():
        if sha256(path) != digest:
            raise ValueError(f"Pinned source/evidence/archive changed: {path}")
    for item in record["hf_source"]["weight_files"]:
        _check_stat(item)
    for item in models:
        if item.get("binding_status") != "bound":
            if require_bound:
                raise ValueError(f"Model identity is still pending: {item['name']}")
            continue
        path = Path(item["checkpoint_dir"])
        if sha256(path / "checkpoint_meta.json") != item["metadata_sha256"]:
            raise ValueError(f"Checkpoint metadata changed: {path}")
        meta = json.loads((path / "checkpoint_meta.json").read_text())
        if (meta["checkpoint_id"] != item["checkpoint_id"] or meta["checkpoint_kind"] != "final_model"
                or meta["converted_module_count"] != 252
                or Path(meta["base_model_path"]).resolve() != Path(record["hf_source"]["path"]).resolve()):
            raise ValueError(f"Unexpected complete model identity: {item['name']}")
        if Path(item["weight_stat"]["path"]).resolve() != (path / meta["state_dict_file"]).resolve():
            raise ValueError("Weight stat must bind the actual metadata state_dict_file.")
        _check_stat(item["weight_stat"])
        if not item["evidence_sha256"]:
            raise ValueError("Each model needs its completed assembly/acceptance evidence.")
        for evidence, digest in item["evidence_sha256"].items():
            if sha256(evidence) != digest:
                raise ValueError(f"Model evidence changed: {evidence}")
    if models[0].get("checkpoint_id") != SOURCE_ID:
        raise ValueError("The reference must be the original552282 BF16 checkpoint.")


def region_rows(kl_token, ce_token, teacher_ce_token, batch, manifest):
    """Sum existing causal token losses in FP64 for reporting, without new loss math."""
    response, prompt = build_prediction_token_masks(labels=batch["labels"], attention_mask=batch["attention_mask"])
    if any(value.shape != response.shape for value in (kl_token, ce_token, teacher_ce_token)):
        raise ValueError("Regional losses must already have exactly one causal shift.")
    if any(not torch.isfinite(value).all() for value in (kl_token, ce_token, teacher_ce_token)):
        raise FloatingPointError("Nonfinite per-token diagnostic loss.")
    current = float(reduce_weighted_token_loss(kl_token, response_mask=response, prompt_mask=prompt,
                                              prompt_loss_weight=PROMPT_WEIGHT))
    rows = []
    for index, sample in enumerate(manifest["samples"]):
        nr, np = int(response[index].sum()), int(prompt[index].sum())
        if nr != sample["causal_response_tokens"] or np != sample["causal_prompt_tokens"]:
            raise ValueError("Fixed archive regional masks/counts changed.")
        r, p = response[index].bool(), prompt[index].bool()
        if not torch.equal(batch["labels"][index, 1:][r], batch["input_ids"][index, 1:][r]):
            raise ValueError("Response labels must equal the actual causal target token IDs.")
        sr = float(kl_token[index][r].double().sum())
        sp = float(kl_token[index][p].double().sum())
        cr = float(ce_token[index][r].double().sum())
        tr = float(teacher_ce_token[index][r].double().sum())
        positions = r.nonzero(as_tuple=False).flatten().cpu().tolist()
        token_ids = batch["labels"][index, 1:][r].cpu().tolist()
        token_kl = kl_token[index][r].float().cpu().tolist()
        token_ce = ce_token[index][r].float().cpu().tolist()
        teacher_ce = teacher_ce_token[index][r].float().cpu().tolist()
        response_tokens = [{"prediction_position": pos, "target_position": pos + 1, "label_id": label,
                            "student_mass_kl": kl, "student_label_ce": ce, "teacher_label_ce": tc}
                           for pos, label, kl, ce, tc in zip(positions, token_ids, token_kl, token_ce, teacher_ce)]
        rows.append({**sample, "archive_batch_index": manifest["batch_index"], "index_in_batch": index,
                     "kl_response_sum": sr, "kl_response_mean": sr / nr if nr else None,
                     "kl_prompt_sum": sp, "kl_prompt_mean": sp / np if np else None,
                     "ce_response_sum": cr, "ce_response_mean": cr / nr if nr else None,
                     "teacher_ce_response_sum": tr, "teacher_ce_response_mean": tr / nr if nr else None,
                     "response_tokens": response_tokens,
                     "kl_weighted_sequence_mean": (sr + PROMPT_WEIGHT * sp) / max(1., nr + PROMPT_WEIGHT * np),
                     "length_group": "response_le12" if nr <= 12 else "response_ge100" if nr >= 100 else "response_13_99"})
    reconstructed = (math.fsum(r["kl_response_sum"] + PROMPT_WEIGHT * r["kl_prompt_sum"] for r in rows)
                     / max(1., math.fsum(r["causal_response_tokens"] + PROMPT_WEIGHT * r["causal_prompt_tokens"] for r in rows)))
    # The production FP32 reduction and independent FP64 sums have normal
    # reduction/denominator rounding; require a small relative numerical error.
    error = abs(reconstructed - current)
    if not math.isclose(reconstructed, current, rel_tol=3e-6, abs_tol=2e-7):
        raise RuntimeError("Regional sums do not recompose the existing FP32 KL reduction.")
    return rows, {"archive_batch_index": manifest["batch_index"], "production_fp32_loss": current,
                  "reconstructed_fp64_loss": reconstructed, "absolute_recomposition_error": error,
                  "production_fp32_denominator": float((response + PROMPT_WEIGHT * prompt).sum().clamp_min(1))}


def aggregate_rows(rows):
    if not rows:
        return {"sequences": 0}
    nr = sum(row["causal_response_tokens"] for row in rows)
    np = sum(row["causal_prompt_tokens"] for row in rows)
    sr = math.fsum(row["kl_response_sum"] for row in rows)
    sp = math.fsum(row["kl_prompt_sum"] for row in rows)
    cr = math.fsum(row["ce_response_sum"] for row in rows)
    tr = math.fsum(row["teacher_ce_response_sum"] for row in rows)
    batches = {}
    for row in rows:
        batches.setdefault(row["archive_batch_index"], []).append(row)
    def batch_denominator(values):
        return max(1., sum(r["causal_response_tokens"] + PROMPT_WEIGHT * r["causal_prompt_tokens"] for r in values))
    def batch_mean(values):
        return (math.fsum(r["kl_response_sum"] + PROMPT_WEIGHT * r["kl_prompt_sum"] for r in values)
                / batch_denominator(values))
    def nonempty_mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return statistics.mean(values) if values else None
    denominator = math.fsum(batch_denominator(values) for values in batches.values())
    empty_response = [row["input_and_label_sha256"] for row in rows if not row["causal_response_tokens"]]
    return {"sequences": len(rows), "represented_batches": len(batches), "causal_response_tokens": nr,
            "zero_response_sequences": len(empty_response), "zero_response_hashes": empty_response,
            "nonempty_response_sequences": len(rows) - len(empty_response),
            "causal_prompt_tokens": np, "kl_response_sum": sr, "kl_prompt_sum": sp, "ce_response_sum": cr,
            "teacher_ce_response_sum": tr,
            "kl_original_token_weighted": (sr + PROMPT_WEIGHT * sp) / denominator,
            "sum_clamped_batch_denominators_fp64": denominator,
            "kl_equal_batch": statistics.mean(batch_mean(values) for values in batches.values()),
            "kl_equal_sequence": statistics.mean(row["kl_weighted_sequence_mean"] for row in rows),
            "kl_response_token_mean": sr / nr if nr else None, "kl_prompt_token_mean": sp / np if np else None,
            "ce_response_token_mean": cr / nr if nr else None,
            "teacher_ce_response_token_mean": tr / nr if nr else None,
            "kl_response_sequence_mean": nonempty_mean("kl_response_mean"),
            "kl_prompt_sequence_mean": nonempty_mean("kl_prompt_mean"),
            "ce_response_sequence_mean": nonempty_mean("ce_response_mean"),
            "teacher_ce_response_sequence_mean": nonempty_mean("teacher_ce_response_mean")}


def summarize_regions(rows, batch_checks):
    ids = [row["input_and_label_sha256"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate observed sequence hash.")
    result = {"all": aggregate_rows(rows)}
    for name in ("response_le12", "response_13_99", "response_ge100"):
        result[name] = aggregate_rows([row for row in rows if row["length_group"] == name])
    denominator = math.fsum(row["production_fp32_denominator"] for row in batch_checks)
    result["original_fp32_batch_reductions"] = {
        "equal_batch": statistics.mean(row["production_fp32_loss"] for row in batch_checks),
        "denominator_weighted": math.fsum(row["production_fp32_loss"] * row["production_fp32_denominator"]
                                         for row in batch_checks) / denominator,
        "maximum_absolute_recomposition_error": max(row["absolute_recomposition_error"] for row in batch_checks)}
    return result


def compare_teacher_observations(models):
    """Report repeated teacher numerical differences without a bitwise gate."""
    reference = models[MODEL_NAMES[0]]["sequences"]
    def flatten(rows):
        return [(row["input_and_label_sha256"], token["prediction_position"], token["target_position"],
                 token["label_id"], token["teacher_label_ce"]) for row in rows for token in row["response_tokens"]]
    original = flatten(reference)
    result = {"reference": MODEL_NAMES[0], "scope": "same-hash response token teacher CE; observation only",
              "numerical_acceptance_threshold": None, "comparisons": {}}
    for name in MODEL_NAMES[1:]:
        current = flatten(models[name]["sequences"])
        if len(current) != len(original) or any(left[:-1] != right[:-1] for left, right in zip(original, current)):
            raise ValueError("Teacher observations do not refer to the same response token identities.")
        delta = [right[-1] - left[-1] for left, right in zip(original, current)]
        result["comparisons"][name] = {
            "response_tokens": len(delta), "nonzero_differences": sum(value != 0 for value in delta),
            "max_absolute_difference": max(map(abs, delta), default=0.),
            "rms_difference": math.sqrt(math.fsum(value * value for value in delta) / len(delta)) if delta else None,
            "mean_signed_difference": statistics.mean(delta) if delta else None}
    return result


@torch.no_grad()
def _observe_model(item, teacher, batches, manifests, device, logger):
    from e2e_common.proxy_trainables import iter_named_vae_module_refs
    from tools.cat_eval import _prepare_model_for_eval
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    started = time.monotonic()
    # Group8 deployment prewarm has large temporary decode tensors. Keep the
    # unchanged teacher on CPU until this student's prewarm has completed.
    teacher.to("cpu")
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    teacher_prewarm_devices = sorted({str(parameter.device) for parameter in teacher.parameters()})
    teacher_prewarm_buffer_devices = sorted({str(buffer.device) for buffer in teacher.buffers()})
    if teacher_prewarm_devices != ["cpu"] or any(value != "cpu" for value in teacher_prewarm_buffer_devices):
        raise RuntimeError("Teacher must be wholly on CPU before loading/prewarming each student.")
    before_student_allocated = torch.cuda.memory_allocated(device)
    logger.info("%s teacher on CPU before student load and group8 prewarm", item["name"])
    torch.cuda.reset_peak_memory_stats(device)
    model, meta, loaded = load_v6_model_checkpoint(item["checkpoint_dir"], map_location="cpu", strict=True,
                                                  expected_kind="final_model")
    if meta["checkpoint_id"] != item["checkpoint_id"] or loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Fresh strict load differs from pinned model.")
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    if {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()} != {torch.bfloat16}:
        raise RuntimeError("This read-only comparison requires actual BF16 model parameters.")
    warmup = _prepare_model_for_eval(model, str(device), 8, logger, {})
    modules = list(iter_named_vae_module_refs(model))
    if len(modules) != 252 or warmup["failed"] or warmup["warmed"] != 252:
        raise RuntimeError("Expected all252 deployment caches warmed by the actual cat_eval path.")
    if any(ref.base_layer.trainable_decode or ref.base_layer._cached_weight is None
           or ref.base_layer._cached_weight.dtype != torch.bfloat16 for ref in modules):
        raise RuntimeError("Inference must retain frozen BF16 deployment caches.")
    prewarm_peak = {"allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "reserved_bytes": torch.cuda.max_memory_reserved(device)}
    after_prewarm_allocated = torch.cuda.memory_allocated(device)
    # Return the exact same BF16 teacher to CUDA only for the paired forwards;
    # the next invocation moves it back before another student's prewarm.
    torch.cuda.empty_cache()
    teacher.to(device)
    torch.cuda.synchronize(device)
    after_teacher_allocated = torch.cuda.memory_allocated(device)
    teacher_forward_devices = sorted({str(parameter.device) for parameter in teacher.parameters()})
    if teacher_forward_devices != [str(device)] or {p.dtype for p in teacher.parameters()} != {torch.bfloat16}:
        raise RuntimeError("Teacher forward requires the unchanged BF16 model on the assigned GPU.")
    logger.info("%s teacher restored to %s after group8 prewarm", item["name"], device)
    versions = {name: tensor._version for name, tensor in model.state_dict().items()}
    pointers = {ref.name: ref.base_layer._cached_weight.data_ptr() for ref in modules}
    if len(versions) != 3423:
        raise RuntimeError("Unexpected complete3423 persistent tensor inventory.")
    rows, checks, timing = [], [], []
    output_dtypes = set()
    for batch, manifest in zip(batches, manifests):
        began = time.monotonic()
        inputs = {key: value.to(device) for key, value in batch.items()}
        forward_inputs = {key: inputs[key] for key in ("input_ids", "attention_mask")}
        # Like cat_eval's BF16 model forward: no train-time FP32 promotion or
        # Accelerator wrapper. The loss primitive itself evaluates in FP32.
        teacher_logits = teacher(**forward_inputs, use_cache=False).logits
        student_logits = model(**forward_inputs, use_cache=False).logits
        output_dtypes.add((str(student_logits.dtype), str(teacher_logits.dtype)))
        kl = compute_kl_top_mass_token_loss(student_logits=student_logits[:, :-1, :],
                teacher_logits=teacher_logits[:, :-1, :], temperature=1.0, top_k=100)
        ce = compute_sft_token_loss(student_logits=student_logits, input_ids=inputs["input_ids"])
        teacher_ce = compute_sft_token_loss(student_logits=teacher_logits, input_ids=inputs["input_ids"])
        batch_rows, check = region_rows(kl, ce, teacher_ce, inputs, manifest)
        rows.extend(batch_rows)
        checks.append(check)
        del teacher_logits, student_logits, kl, ce, teacher_ce, inputs, forward_inputs
        torch.cuda.synchronize(device)
        timing.append(time.monotonic() - began)
        logger.info("%s archive_batch=%d KL=%.9f seconds=%.3f", item["name"], manifest["batch_index"],
                    check["production_fp32_loss"], timing[-1])
    if versions != {name: tensor._version for name, tensor in model.state_dict().items()}:
        raise RuntimeError("A persistent tensor changed during read-only evaluation.")
    if pointers != {ref.name: ref.base_layer._cached_weight.data_ptr() for ref in modules}:
        raise RuntimeError("A deployment cache was replaced during observation.")
    return {"checkpoint_id": meta["checkpoint_id"], "checkpoint_dir": item["checkpoint_dir"],
            "strict_load": True, "persistent_tensor_count": len(versions), "persistent_versions_unchanged": True,
            "deployment_cache_pointers_unchanged": True, "warmup": warmup,
            "teacher_residency": {"before_student_load_and_prewarm": teacher_prewarm_devices,
                                  "buffers_before_student_load_and_prewarm": teacher_prewarm_buffer_devices,
                                  "during_forward": teacher_forward_devices,
                                  "policy": "CPU before every student load/prewarm; GPU after prewarm"},
            "student_prewarm_peak": prewarm_peak,
            "residency_allocated_bytes": {"before_student_load": before_student_allocated,
                                          "after_student_prewarm": after_prewarm_allocated,
                                          "after_teacher_return": after_teacher_allocated},
            "student_parameter_dtype": "torch.bfloat16", "attention_implementation": model.config._attn_implementation,
            "logits_dtype_pairs_student_teacher": sorted(output_dtypes), "sequences": rows, "batch_checks": checks,
            "summary": summarize_regions(rows, checks), "elapsed_seconds": time.monotonic() - started,
            "batch_seconds": timing, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def run_loss_region_diagnostic(launch_record):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.kl_round_diagnostic import validate_batch_archive
    from train_utils.base_reference import load_frozen_base_reference_model

    launch_path = Path(launch_record).resolve()
    record = json.loads(launch_path.read_text())
    check_inputs(record)
    output = Path(record["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "launch_record": str(launch_path), "launch_sha256_at_start": sha256(launch_path),
              "worker_pid": os.getpid(), "worker_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
              "source_sha256": record["source_sha256"], "archive": record["archive"], "models": {},
              "protocol": {"loss": "kl_top_mass", "top_k": 100, "temperature": 1.0,
                           "prompt_loss_weight": PROMPT_WEIGHT, "batch_size": 4, "heldout_batches": 8,
                           "prewarm_group_size": 8, "student_and_teacher_parameters": "BF16",
                           "autocast": False, "loss_math": "existing FP32 full-logits primitives; FP64 reporting sums",
                           "teacher_forward_per_student": True, "chunked_loss": False,
                           "teacher_residency": "initial CPU; CPU during each group8 prewarm; GPU during forwards",
                           "optimizer_updates": 0, "hif4_act": False},
              "limitations": ["The32 heldout ChatML sequences are not the full8 accuracy benchmark.",
                              "Response includes existing empty-think/EOS formatting; response CE is not answer-only CE.",
                              "Length groups describe token counts, not certified dataset/task provenance.",
                              "Equal-sequence/batch summaries are reporting only, not new training objectives.",
                              "Deployment BF16 caches differ from native caches used in training diagnostics."]}
    def save():
        (output / "diagnostic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    started = time.monotonic()
    save()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Bind exactly one assigned visible GPU; do not hide CUDA for the FA2 loader.")
        for key, expected in record["environment"].items():
            if expected is not None and os.environ.get(key) != expected:
                raise ValueError(f"Launch environment differs: {key}")
        device = torch.device("cuda:0")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        archive = torch.load(record["archive"]["path"], map_location="cpu", weights_only=False)
        archive_audit = validate_batch_archive(archive)
        batches, manifests = archive["batches"][256:], archive["manifests"][256:]
        report["data"] = {key: value for key, value in archive_audit.items() if key != "manifests"}
        report["data"]["heldout_manifests"] = manifests
        tokenizer = AutoTokenizer.from_pretrained(record["hf_source"]["path"], use_fast=True, trust_remote_code=True)
        if not tokenizer.is_fast:
            raise RuntimeError("Expected the same fast tokenizer as the accepted full8 evaluations.")
        report["tokenizer"] = {"class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
                               "archive_retokenized": False, "padding_side": tokenizer.padding_side}
        teacher = load_frozen_base_reference_model(record["hf_source"]["path"], access_token=None,
                                                   device="cpu", dtype=torch.bfloat16)
        report["teacher"] = {"path": record["hf_source"]["path"], "class": type(teacher).__name__,
                             "attention_implementation": teacher.config._attn_implementation,
                             "parameter_dtypes": sorted({str(p.dtype) for p in teacher.parameters()}),
                             "initial_parameter_devices": sorted({str(p.device) for p in teacher.parameters()}),
                             "training": teacher.training, "requires_grad_parameters": sum(p.requires_grad for p in teacher.parameters())}
        for item in record["models"]:
            report["models"][item["name"]] = _observe_model(item, teacher, batches, manifests, device, logger)
            save()
            gc.collect()
            torch.cuda.empty_cache()
        report["teacher_repeat_observation"] = compare_teacher_observations(report["models"])
        del teacher
        check_inputs(record)
        report.update(status="complete", source_evidence_weights_unchanged=True, elapsed_seconds=time.monotonic()-started)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        if torch.cuda.is_initialized():
            report["failure_cuda_peak"] = {"allocated_bytes": torch.cuda.max_memory_allocated(),
                                           "reserved_bytes": torch.cuda.max_memory_reserved()}
        raise
    finally:
        save()
    return report
