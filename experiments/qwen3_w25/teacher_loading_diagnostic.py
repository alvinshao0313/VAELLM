"""Isolate an explicit whole-model BF16 cast in the Qwen3 teacher loader.

A uses the existing frozen-reference helper with dtype=BF16 on CPU; B uses
get_qwen3 unchanged and makes only device moves. Both are frozen/eval/FA2 and
consume identical archived B4 inputs. This diagnostic never trains or exports
weights, repairs buffers, or changes the production loader.
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

from train_utils.distill_loss_core import build_prediction_token_masks, compute_sft_token_loss

ARCHIVE_SHA = "1b481a4748c2735be951b45db9998fe8bec8e43f69410965cf720c5b3a6c2d53"
VARIANTS = ("A_explicit_bf16", "B_original_loader")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_inputs(record):
    if record["archive"]["sha256"] != ARCHIVE_SHA or tuple(record["variants"]) != VARIANTS:
        raise ValueError("Expected the exact original256+8 archive and the fixed teacher A/B pair.")
    for path, digest in {**record["source_sha256"], **record["input_evidence_sha256"],
                         **record["hf_source"]["metadata_sha256"], record["archive"]["path"]: ARCHIVE_SHA}.items():
        if sha256(path) != digest:
            raise ValueError(f"Pinned source/evidence changed: {path}")
    for item in record["hf_source"]["weight_files"]:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != tuple(item[k] for k in ("size", "mtime_ns", "inode")):
            raise ValueError(f"HF weight shard stat changed: {item['path']}")


def _tensor_record(value):
    cpu = value.detach().contiguous().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
            "numel": value.numel(), "byte_sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "values_fp64": cpu.double().reshape(-1).tolist()}


def model_inventory(model):
    """Record small non-weight floating tensors including unregistered RoPE alias."""
    counts, signature = {}, []
    for name, parameter in model.named_parameters():
        item = counts.setdefault(str(parameter.dtype), {"tensors": 0, "elements": 0})
        item["tensors"] += 1
        item["elements"] += parameter.numel()
        signature.append((name, list(parameter.shape), str(parameter.dtype)))
    buffers = {name: _tensor_record(value) for name, value in model.named_buffers(remove_duplicate=False)
               if value.is_floating_point()}
    rope = {}
    for name, module in model.named_modules():
        inv, original = getattr(module, "inv_freq", None), getattr(module, "original_inv_freq", None)
        if not isinstance(inv, torch.Tensor) or not isinstance(original, torch.Tensor):
            continue
        rope[name] = {"inv_freq": _tensor_record(inv), "original_inv_freq": _tensor_record(original),
                      "same_python_tensor": inv is original,
                      "same_storage": inv.device == original.device and inv.untyped_storage().data_ptr() == original.untyped_storage().data_ptr(),
                      "same_fp64_values": torch.equal(inv.detach().double().cpu(), original.detach().double().cpu()),
                      "rope_type": getattr(module, "rope_type", None),
                      "max_seq_len_cached": getattr(module, "max_seq_len_cached", None)}
    return {"parameter_dtype_counts": counts, "parameter_tensors": len(signature),
            "parameter_elements": sum(item["elements"] for item in counts.values()),
            "parameter_layout_sha256": hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest(),
            "parameter_value_comparison": "not performed; identical pinned HF shards and normal loaders",
            "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
            "requires_grad_parameters": sum(p.requires_grad for p in model.parameters()),
            "training": model.training, "floating_buffers": buffers, "rope_attributes": rope}


def compare_buffer_values(a, b):
    if set(a["floating_buffers"]) != set(b["floating_buffers"]):
        raise ValueError("Teacher loader variants have different floating buffer names.")
    result = {}
    for name, left in a["floating_buffers"].items():
        right = b["floating_buffers"][name]
        if left["shape"] != right["shape"]:
            raise ValueError(f"Teacher loader variants have different buffer shapes: {name}")
        delta = [y-x for x, y in zip(left["values_fp64"], right["values_fp64"])]
        result[name] = {"A_dtype": left["dtype"], "B_dtype": right["dtype"],
                        "different_values": sum(value != 0 for value in delta),
                        "max_abs_B_minus_A": max(map(abs, delta), default=0.),
                        "rms_B_minus_A": math.sqrt(math.fsum(value*value for value in delta)/len(delta)) if delta else None}
    return result


def response_predictions(logits, batch, manifest):
    response, prompt = build_prediction_token_masks(labels=batch["labels"], attention_mask=batch["attention_mask"])
    mask = response.bool()
    ce = compute_sft_token_loss(student_logits=logits, input_ids=batch["input_ids"])
    # Gather only response rows for compact observations; no model/loss chunking.
    selected = logits[:, :-1, :][mask].float()
    targets = batch["labels"][:, 1:][mask]
    if not torch.equal(targets, batch["input_ids"][:, 1:][mask]):
        raise ValueError("Response labels differ from their causal input token targets.")
    if selected.shape[-1] < 100:
        raise ValueError("Expected the real teacher vocabulary with at least100 tokens.")
    top5_values, top5_ids = selected.topk(5, dim=-1)
    top100_ids = selected.topk(100, dim=-1, sorted=False).indices
    probabilities = (top5_values - torch.logsumexp(selected, dim=-1, keepdim=True)).exp()
    membership = (top100_ids == targets[:, None]).any(-1)
    ce_values = ce[mask].float()
    if not torch.isfinite(ce_values).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite teacher CE or observed top5 probability.")
    ce_list, ids_list, probs_list, in100_list = [v.cpu().tolist() for v in (ce_values, top5_ids, probabilities, membership)]
    rows, offset = [], 0
    for index, sample in enumerate(manifest["samples"]):
        nr, np = int(response[index].sum()), int(prompt[index].sum())
        if nr != sample["causal_response_tokens"] or np != sample["causal_prompt_tokens"]:
            raise ValueError("Archive causal counts changed.")
        positions = mask[index].nonzero(as_tuple=False).flatten().cpu().tolist()
        labels = batch["labels"][index, 1:][mask[index]].cpu().tolist()
        tokens = []
        for local, (position, label) in enumerate(zip(positions, labels)):
            cursor = offset + local
            tokens.append({"prediction_position": position, "target_position": position + 1,
                           "label_id": label, "label_ce": ce_list[cursor],
                           "label_probability_from_ce": math.exp(-ce_list[cursor]),
                           "label_in_teacher_top100": in100_list[cursor],
                           "top5_ids": ids_list[cursor], "top5_probabilities": probs_list[cursor]})
        offset += nr
        total = math.fsum(token["label_ce"] for token in tokens)
        rows.append({**sample, "archive_batch_index": manifest["batch_index"], "index_in_batch": index,
                     "response_ce_sum": total, "response_ce_mean": total / nr if nr else None,
                     "response_tokens": tokens})
    if offset != len(ce_list):
        raise RuntimeError("Response token observation count mismatch.")
    return rows


def compare_ce(reference_rows, current_rows, *, reference_field, current_field):
    def flatten(rows, field):
        return [(row["input_and_label_sha256"], token["prediction_position"], token["target_position"],
                 token["label_id"], token[field]) for row in rows for token in row["response_tokens"]]
    reference, current = flatten(reference_rows, reference_field), flatten(current_rows, current_field)
    if len(reference) != len(current) or any(a[:-1] != b[:-1] for a, b in zip(reference, current)):
        raise ValueError("CE observations do not have the same response token identities.")
    delta = [b[-1] - a[-1] for a, b in zip(reference, current)]
    return {"response_tokens": len(delta), "nonzero_differences": sum(d != 0 for d in delta),
            "max_abs_difference": max(map(abs, delta), default=0.),
            "rms_difference": math.sqrt(math.fsum(d*d for d in delta)/len(delta)) if delta else None,
            "mean_signed_difference": statistics.mean(delta) if delta else None,
            "numerical_acceptance_threshold": None, "scope": "observed numerical differences, not a bitwise gate"}


def summarize_predictions(rows):
    result = {}
    for name, chosen in (("all", rows), ("response_le12", [r for r in rows if r["causal_response_tokens"] <= 12]),
                         ("response_ge100", [r for r in rows if r["causal_response_tokens"] >= 100])):
        values = [token["label_ce"] for row in chosen for token in row["response_tokens"]]
        nonempty = [row["response_ce_mean"] for row in chosen if row["response_ce_mean"] is not None]
        result[name] = {"sequences": len(chosen), "response_tokens": len(values),
                        "token_mean_ce": statistics.mean(values) if values else None,
                        "nonempty_sequence_mean_ce": statistics.mean(nonempty) if nonempty else None,
                        "zero_response_hashes": [r["input_and_label_sha256"] for r in chosen if not r["causal_response_tokens"]]}
    return result


@torch.no_grad()
def observe_variant(name, hf_path, batches, manifests, device, logger):
    from rotation.model_utils import get_qwen3
    from train_utils.base_reference import load_frozen_base_reference_model

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    if name == VARIANTS[0]:
        model = load_frozen_base_reference_model(hf_path, access_token=None, device="cpu", dtype=torch.bfloat16)
    elif name == VARIANTS[1]:
        model = get_qwen3(hf_path, None)
        model.requires_grad_(False)
        model.eval()
        model.config.use_cache = False
    else:
        raise ValueError("Only the specified two teacher loader variants are supported.")
    before = model_inventory(model)
    if before["parameter_devices"] != ["cpu"] or before["requires_grad_parameters"] or model.training:
        raise RuntimeError("Both teachers must be frozen and eval on CPU before device-only movement.")
    model.to(device)
    after_move = model_inventory(model)
    versions = {name: tensor._version for name, tensor in model.state_dict().items()}
    rows, timing, dtypes = [], [], set()
    for batch, manifest in zip(batches, manifests):
        began = time.monotonic()
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False).logits
        dtypes.add(str(logits.dtype))
        rows.extend(response_predictions(logits, inputs, manifest))
        del logits, inputs
        torch.cuda.synchronize(device)
        timing.append(time.monotonic() - began)
        logger.info("%s archive_batch=%d seconds=%.3f", name, manifest["batch_index"], timing[-1])
    after_forward = model_inventory(model)
    if versions != {name: tensor._version for name, tensor in model.state_dict().items()}:
        raise RuntimeError("A persistent teacher tensor changed during no-grad observation.")
    return {"inventory_cpu": before, "inventory_cuda_before_forward": after_move,
            "inventory_cuda_after_forward": after_forward, "persistent_versions_unchanged": True,
            "model_class": type(model).__name__, "attention_implementation": model.config._attn_implementation,
            "use_cache": model.config.use_cache, "logits_dtypes": sorted(dtypes),
            "sequences": rows, "summary": summarize_predictions(rows), "batch_seconds": timing,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def run_teacher_loading_diagnostic(launch_record):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.kl_round_diagnostic import validate_batch_archive

    launch_path = Path(launch_record).resolve()
    record = json.loads(launch_path.read_text())
    check_inputs(record)
    output = Path(record["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "launch_record": str(launch_path), "launch_sha256_at_start": sha256(launch_path),
              "worker_pid": os.getpid(), "worker_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
              "source_sha256": record["source_sha256"], "archive": record["archive"], "variants": {},
              "protocol": {"autocast": False, "teacher_forward": "FA2/BF16 parameter storage; no_grad/eval/use_cacheFalse",
                           "logit_observations": "original FP32 CE primitive and FP32 top5/top100; no full logits saved",
                           "optimizer_steps": 0, "buffer_repair": False},
              "limitations": ["Response includes empty-think/EOS; no semantic-answer mask is invented.",
                              "A explicitly casts floating buffers too; B leaves original loader dtypes intact.",
                              "Parameter layout/dtypes/counts are compared; full parameter values are not hashed.",
                              "This isolates diagnostic teacher loading, not the validity of every formal teacher path."]}
    def save():
        (output / "diagnostic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    began = time.monotonic()
    save()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Bind exactly one assigned visible GPU for this teacher-only diagnostic.")
        for key, value in record["environment"].items():
            if value is not None and os.environ.get(key) != value:
                raise ValueError(f"Launch environment differs: {key}")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger, device = logging.getLogger(__name__), torch.device("cuda:0")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        archive = torch.load(record["archive"]["path"], map_location="cpu", weights_only=True)
        audit = validate_batch_archive(archive)
        batches, manifests = archive["batches"][256:], archive["manifests"][256:]
        report["data"] = {key: value for key, value in audit.items() if key != "manifests"}
        report["data"]["heldout_manifests"] = manifests
        tokenizer = AutoTokenizer.from_pretrained(record["hf_source"]["path"], use_fast=True, trust_remote_code=True)
        if not tokenizer.is_fast:
            raise RuntimeError("Expected the reference fast tokenizer.")
        report["tokenizer"] = {"class": type(tokenizer).__name__, "is_fast": True, "archive_retokenized": False}
        for name in VARIANTS:
            report["variants"][name] = observe_variant(name, record["hf_source"]["path"], batches, manifests, device, logger)
            save()
            gc.collect()
            torch.cuda.empty_cache()
        a, b = [report["variants"][name] for name in VARIANTS]
        region = json.loads(Path(record["region_reference"]).read_text())
        if (region["status"] != "complete" or region["archive"]["sha256"] != ARCHIVE_SHA
                or Path(region["teacher"]["path"]).resolve() != Path(record["hf_source"]["path"]).resolve()):
            raise ValueError("Regional teacher reference is not complete.")
        report["A_vs_region_teacher_ce"] = compare_ce(region["models"]["source552"]["sequences"], a["sequences"],
                                                      reference_field="teacher_label_ce", current_field="label_ce")
        report["B_minus_A_ce"] = compare_ce(a["sequences"], b["sequences"], reference_field="label_ce", current_field="label_ce")
        report["parameter_layout_equal"] = a["inventory_cpu"]["parameter_layout_sha256"] == b["inventory_cpu"]["parameter_layout_sha256"]
        report["floating_buffers_B_minus_A"] = compare_buffer_values(a["inventory_cpu"], b["inventory_cpu"])
        check_inputs(record)
        report.update(status="complete", source_inputs_unchanged=True, elapsed_seconds=time.monotonic()-began)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), elapsed_seconds=time.monotonic()-began)
        if torch.cuda.is_initialized():
            report["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        raise
    finally:
        save()
    return report
