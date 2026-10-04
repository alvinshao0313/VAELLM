"""Read-only ChatML versus explicitly space-separated bare teacher probe.

Only content recoverable from fourteen archived rendered inputs is claimed.
bare_space adds exactly one ASCII space and reports its boundary contribution.
The literal arm failed the CPU pair-prefix audit on nine of fourteen cases and
is not forwarded. No training or production data/template changes are made.
"""
from __future__ import annotations

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

from experiments.qwen3_w25.teacher_loading_diagnostic import ARCHIVE_SHA, model_inventory, sha256

VARIANTS = ("chatml", "bare_space")


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_inputs(record):
    if record["archive"]["sha256"] != ARCHIVE_SHA or tuple(record["variants"]) != VARIANTS:
        raise ValueError("Expected the fixed archive and the two approved formats.")
    for path, digest in {**record["source_sha256"], **record["input_evidence_sha256"],
                         **record["hf_source"]["metadata_sha256"], record["archive"]["path"]: ARCHIVE_SHA}.items():
        if sha256(path) != digest:
            raise ValueError(f"Pinned input changed: {path}")
    for item in record["hf_source"]["weight_files"]:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != tuple(item[k] for k in ("size", "mtime_ns", "inode")):
            raise ValueError(f"HF shard stat changed: {item['path']}")


def recover_single_turn(tokenizer, input_ids, labels):
    """Reject ambiguous conversations; verify the real canonical encoder roundtrip."""
    from train_utils.distill_data import encode_canonical_record

    full = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    start = "<|im_start|>user\n"
    middle = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    end = "<|im_end|>\n<|im_end|>"
    if (not full.startswith(start) or not full.endswith(end) or full.count(middle) != 1
            or full.count("<|im_start|>") != 2 or full.count("<|im_end|>") != 3):
        raise ValueError("Expected untruncated single user/assistant without system, tools or extra turns.")
    user, answer = full[len(start):-len(end)].split(middle)
    if not user or not answer or any(marker in user + answer for marker in ("<|", "<think>", "</think>", "<tool_call>", "<tool_response>")):
        raise ValueError("Empty or ambiguous content containing template/tool markers.")
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if rendered + tokenizer.eos_token != full:
        raise ValueError("Original HF template plus canonical EOS does not reconstruct archived text.")
    canonical = encode_canonical_record({"messages": messages}, tokenizer, text_format="edgerazor_messages",
                                        text_field="text", task="sft", model_max_length=1024)
    if canonical is None or canonical["input_ids"].tolist() != input_ids or canonical["labels"].tolist() != labels:
        raise ValueError("Canonical input/labels do not exactly reconstruct the archived sample.")
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != input_ids:
        raise ValueError("Decoded input cannot be retokenized exactly.")
    answer_start = len(start) + len(user) + len(middle)
    positions = [i for i, (s, e) in enumerate(encoded["offset_mapping"])
                 if e > answer_start and s < answer_start + len(answer)]
    if not positions or any(i == 0 or labels[i] != input_ids[i] for i in positions):
        raise ValueError("Answer content positions do not match the archived response labels.")
    for i in positions:
        s, e = encoded["offset_mapping"][i]
        if s < answer_start or e > answer_start + len(answer):
            raise ValueError("ChatML answer has a token crossing an unsupported template boundary.")
    return {"user": user, "answer": answer, "input_ids": input_ids, "answer_positions": positions,
            "answer_tokens": [{"target_position": i, "offset": list(encoded["offset_mapping"][i]),
                               "includes_user_suffix": False, "includes_added_separator": False} for i in positions],
            "user_utf8_sha256": text_sha(user), "answer_utf8_sha256": text_sha(answer),
            "user_utf8_bytes": len(user.encode("utf-8")), "answer_utf8_bytes": len(answer.encode("utf-8")),
            "full_utf8_sha256": text_sha(full), "canonical_input_exact": True, "canonical_labels_exact": True}


def prepare_bare(tokenizer, user, answer, *, separator):
    """CPU tokenizer audit of the pair contract, checked against real HFLM at runtime."""
    if separator not in ("", " "):
        raise ValueError("Only literal continuation or one explicit ASCII separator is approved.")
    trailing = len(user) - len(user.rstrip())
    pair_context = user[:-trailing] if trailing else user
    continuation = (user[-trailing:] if trailing else "") + separator + answer
    if not pair_context:
        raise ValueError("Empty context would invoke an unapproved BOS/EOS prefix.")
    full = user + separator + answer
    if pair_context + continuation != full:
        raise RuntimeError("Trailing-whitespace transfer changed the input bytes.")
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    context_ids = tokenizer.encode(pair_context, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids[:len(context_ids)] != context_ids:
        raise ValueError("Pair context is not a token prefix; no heuristic boundary repair is allowed.")
    answer_start, answer_end = len(user) + len(separator), len(full)
    tokens = []
    for i, (s, e) in enumerate(encoded["offset_mapping"]):
        if e > answer_start and s < answer_end:
            if i < len(context_ids) or i == 0:
                raise ValueError("Answer content overlaps the independently encoded context.")
            tokens.append({"target_position": i, "offset": [s, e],
                           "includes_user_suffix": s < len(user),
                           "includes_added_separator": bool(separator and s < answer_start and e > len(user))})
    if not tokens:
        raise ValueError("No answer content tokens remain.")
    return {"input_ids": ids, "context_ids": context_ids, "continuation_ids": ids[len(context_ids):],
            "answer_positions": [t["target_position"] for t in tokens], "answer_tokens": tokens,
            "input_utf8_sha256": text_sha(full), "separator_ascii": separator,
            "separator_added_bytes": len(separator), "user_trailing_whitespace_moved_chars": trailing,
            "user_trailing_whitespace_moved_utf8_bytes": len(user[len(user)-trailing:].encode("utf-8")) if trailing else 0,
            "moved_whitespace_utf8_sha256": text_sha(user[len(user)-trailing:] if trailing else ""),
            "pair_context_utf8_sha256": text_sha(pair_context), "pair_continuation_utf8_sha256": text_sha(continuation),
            "prefix_exact": True, "answer_token_count": len(tokens),
            "cross_user_boundary_answer_tokens": sum(t["includes_user_suffix"] for t in tokens),
            "separator_fused_answer_tokens": sum(t["includes_added_separator"] for t in tokens),
            "scored_token_scope": "Tokens overlapping original answer characters; mixed whitespace/separator tokens cannot be subdivided."}


def prepare_inputs(record):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.kl_round_diagnostic import validate_batch_archive

    tokenizer = AutoTokenizer.from_pretrained(record["hf_source"]["path"], use_fast=True, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise ValueError("Official fast tokenizer is required for exact offsets.")
    archive = torch.load(record["archive"]["path"], map_location="cpu", weights_only=True)
    validate_batch_archive(archive)
    samples, summaries = [], []
    for batch, manifest in zip(archive["batches"][256:], archive["manifests"][256:]):
        for index, identity in enumerate(manifest["samples"]):
            if identity["causal_response_tokens"] > 12:
                continue
            mask = batch["attention_mask"][index].bool()
            if mask.tolist() != [True] * int(mask.sum()) + [False] * (mask.numel() - int(mask.sum())):
                raise ValueError("Archive padding is not a contiguous right suffix.")
            ids, labels = [batch[key][index][mask].tolist() for key in ("input_ids", "labels")]
            parsed = recover_single_turn(tokenizer, ids, labels)
            variants = {"chatml": {key: parsed[key] for key in ("input_ids", "answer_positions", "answer_tokens")},
                        "bare_space": prepare_bare(tokenizer, parsed["user"], parsed["answer"], separator=" ")}
            summary = {**identity, "archive_batch_index": manifest["batch_index"], "index_in_batch": index,
                       **{key: parsed[key] for key in ("user_utf8_sha256", "answer_utf8_sha256", "user_utf8_bytes", "answer_utf8_bytes",
                                                       "full_utf8_sha256", "canonical_input_exact", "canonical_labels_exact")},
                       "variants": {name: {key: value for key, value in variant.items() if key not in ("input_ids", "context_ids", "continuation_ids")}
                                    for name, variant in variants.items()}}
            samples.append({"user": parsed["user"], "answer": parsed["answer"], "identity": summary, "variants": variants})
            summaries.append(summary)
    if len(samples) != 14 or sum(len(s["variants"]["chatml"]["answer_positions"]) for s in samples) != 26:
        raise ValueError("Expected all fourteen fixed short responses and their twenty-six archived answer tokens.")
    return tokenizer, samples, {"sequences": 14, "batch_sizes": [4, 4, 4, 2], "archive_answer_tokens": 26,
                                "samples": summaries, "no_sample_dropped_or_replaced": True,
                                "scope": "Recoverable rendered content after production content.strip/template processing, not unknown raw records; no task attribution.",
                                "literal_arm": "Not forwarded: CPU audit found nine non-prefix cases; no sample replacement or boundary repair."}


def answer_observations(logits, input_ids, prepared):
    """Single causal shift; only selected answer logits are converted to FP32."""
    positions = torch.tensor(prepared["answer_positions"], device=logits.device, dtype=torch.long)
    chosen = logits[positions - 1].float()
    labels = input_ids[positions]
    normalizer = torch.logsumexp(chosen, dim=-1)
    ce = normalizer - chosen.gather(1, labels[:, None]).squeeze(1)
    topvalues, topids = chosen.topk(5, dim=-1)
    probabilities = (topvalues - normalizer[:, None]).exp()
    if not torch.isfinite(ce).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite answer observation.")
    tokens = [{**meta, "prediction_position": meta["target_position"] - 1, "label_id": int(label),
               "label_ce": float(loss), "top5_ids": ids, "top5_probabilities": probs}
              for meta, label, loss, ids, probs in zip(prepared["answer_tokens"], labels.cpu().tolist(),
                                                      ce.cpu().tolist(), topids.cpu().tolist(), probabilities.cpu().tolist())]
    total = math.fsum(t["label_ce"] for t in tokens)
    return {"answer_tokens": tokens, "answer_token_count": len(tokens), "answer_nll_sum": total,
            "answer_ce_token_mean": total / len(tokens), "first_answer_token_ce": tokens[0]["label_ce"]}


def _parameter_identity(model):
    return {name: (id(value), value.data_ptr(), value._version, tuple(value.shape), str(value.dtype), str(value.device))
            for name, value in model.named_parameters()}


@torch.no_grad()
def observe_formats(record, tokenizer, samples, logger):
    from rotation.model_utils import get_qwen3
    from lm_eval.models.huggingface import HFLM

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    model = get_qwen3(record["hf_source"]["path"], None)
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    model.to(device)
    before, identities = model_inventory(model), _parameter_identity(model)
    tokenizer_before = {"pad_token_id": tokenizer.pad_token_id, "padding_side": tokenizer.padding_side}
    hf = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=4, device="cuda:0",
              add_bos_token=False, parallelize=False, max_length=1024)
    after = model_inventory(model)
    if hf.model is not model or identities != _parameter_identity(model) or before != after:
        raise RuntimeError("Constructing the real HFLM changed teacher parameter identity/layout/buffer values.")
    if tokenizer.pad_token_id is None:
        raise ValueError("The actual HFLM tokenizer has no padding token.")
    for sample in samples:
        context, continuation = hf._encode_pair(sample["user"], " " + sample["answer"])
        prepared = sample["variants"]["bare_space"]
        if context != prepared["context_ids"] or continuation != prepared["continuation_ids"]:
            raise ValueError("Actual HFLM pair encoding differs from the CPU tokenizer boundary audit.")
    variants, dtypes, timing = {}, set(), {}
    for name in VARIANTS:
        rows, elapsed = [], []
        for start in range(0, 14, 4):
            began = time.monotonic()
            chunk = samples[start:start+4]
            lengths = [len(sample["variants"][name]["input_ids"]) for sample in chunk]
            ids = torch.full((len(chunk), max(lengths)), tokenizer.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, sample in enumerate(chunk):
                ids[i, :lengths[i]] = torch.tensor(sample["variants"][name]["input_ids"], device=device)
                mask[i, :lengths[i]] = 1
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            dtypes.add(str(logits.dtype))
            for i, sample in enumerate(chunk):
                row = answer_observations(logits[i], ids[i], sample["variants"][name])
                rows.append({"input_and_label_sha256": sample["identity"]["input_and_label_sha256"],
                             "iterator_sample_ordinal": sample["identity"]["iterator_sample_ordinal"],
                             "answer_utf8_sha256": sample["identity"]["answer_utf8_sha256"],
                             "answer_utf8_bytes": sample["identity"]["answer_utf8_bytes"],
                             "input_tokens": lengths[i], **row})
            del logits, ids, mask
            torch.cuda.synchronize(device)
            elapsed.append(time.monotonic() - began)
            logger.info("%s samples=%d..%d seconds=%.3f", name, start, start+len(chunk)-1, elapsed[-1])
        variants[name], timing[name] = rows, elapsed
    if identities != _parameter_identity(model) or after != model_inventory(model):
        raise RuntimeError("Teacher state changed during no-grad format observations.")
    return {"variants": variants, "batch_seconds": timing, "teacher_inventory_before_hflm": before,
            "teacher_inventory_after_hflm": after, "hflm_parameter_identity_and_buffers_unchanged": True,
            "teacher_state_unchanged_after_forward": True, "actual_hflm_pair_matches_cpu_audit": True,
            "tokenizer_before_hflm": tokenizer_before,
            "tokenizer_after_hflm": {"pad_token_id": tokenizer.pad_token_id, "padding_side": tokenizer.padding_side},
            "actual_forward_padding": "manual right padding with explicit attention mask; tokenizer padding_side is not used",
            "logits_dtypes": sorted(dtypes), "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def summarize(variants):
    result = {}
    for name, rows in variants.items():
        values = [r["answer_nll_sum"] for r in rows]
        result[name] = {"sequences": len(rows), "answer_tokens": sum(r["answer_token_count"] for r in rows),
                        "sequence_mean_answer_nll": statistics.mean(values), "sequence_median_answer_nll": statistics.median(values),
                        "token_mean_answer_ce": math.fsum(values)/sum(r["answer_token_count"] for r in rows),
                        "sequence_mean_first_answer_ce": statistics.mean(r["first_answer_token_ce"] for r in rows)}
        if name == "chatml":
            continue
        delta = []
        for base, row in zip(variants["chatml"], rows):
            if (base["input_and_label_sha256"], base["answer_utf8_sha256"]) != (row["input_and_label_sha256"], row["answer_utf8_sha256"]):
                raise ValueError("Paired formats have different sample/answer identities.")
            delta.append(row["answer_nll_sum"] - base["answer_nll_sum"])
        result[name]["paired_nll_minus_chatml"] = {"values": delta, "mean": statistics.mean(delta), "median": statistics.median(delta),
                                                  "improved_sequences": sum(d < 0 for d in delta), "equal_sequences": sum(d == 0 for d in delta)}
    return result


def run_teacher_format_diagnostic(launch_record):
    launch = Path(launch_record).resolve()
    record = json.loads(launch.read_text())
    check_inputs(record)
    output = Path(record["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = {"status": "running", "launch_record": str(launch), "launch_sha256_at_start": sha256(launch),
              "worker_pid": os.getpid(), "worker_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
              "source_sha256": record["source_sha256"], "archive": record["archive"], "variants": {},
              "limitations": ["Content is recovered from archived rendered tokens; pre-template fields/leading newlines are not inferred.",
                              "No task provenance or exact benchmark prompt alignment is claimed; no wrong choices are scored.",
                              "bare_space jointly removes wrapping and adds one ASCII separator; this is not a pure wrapping single-variable contrast.",
                              "NLL of a fused separator/answer token includes the separator effect. Literal-pair failures are not a benchmark bug claim.",
                              "Different tokenizations prevent tokenwise CE subtraction; answer sums are paired by unchanged answer bytes.",
                              "This is a selected fourteen-example read-only format probe, not training or a downstream accuracy result."],
              "protocol": {"optimizer_steps": 0, "model_loads": 1, "teacher_loader": "B_original_loader",
                           "dtype_cast_after_load": False, "autocast": False, "no_grad": True,
                           "eval": True, "use_cache": False, "batch_sizes": [4,4,4,2], "variants": list(VARIANTS)}}
    def save():
        (output / "diagnostic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    save()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one assigned GPU must be visible.")
        for key, value in record["environment"].items():
            if value is not None and os.environ.get(key) != value:
                raise ValueError(f"Launch environment differs: {key}")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        tokenizer, samples, preparation = prepare_inputs(record)
        expected = json.loads(Path(record["preparation_report"]).read_text())
        if preparation != expected:
            raise ValueError("CPU-prepared content or boundaries changed.")
        report["data"] = preparation
        report.update(observe_formats(record, tokenizer, samples, logging.getLogger(__name__)))
        reference = json.loads(Path(record["teacher_B_reference"]).read_text())
        if reference["status"] != "complete" or reference["archive"]["sha256"] != ARCHIVE_SHA:
            raise ValueError("Original teacher B reference is incomplete or uses a different archive.")
        by_hash = {row["input_and_label_sha256"]: row for row in reference["variants"]["B_original_loader"]["sequences"]}
        deltas = []
        for row in report["variants"]["chatml"]:
            tokens = {t["target_position"]: t for t in by_hash[row["input_and_label_sha256"]]["response_tokens"]}
            for token in row["answer_tokens"]:
                old = tokens[token["target_position"]]
                if token["label_id"] != old["label_id"]:
                    raise ValueError("ChatML reference label mismatch.")
                deltas.append(token["label_ce"] - old["label_ce"])
        report["chatml_vs_previous_teacher_B"] = {"answer_tokens": len(deltas), "max_abs_difference": max(map(abs, deltas)),
                                                   "rms_difference": math.sqrt(statistics.mean(d*d for d in deltas)),
                                                   "mean_signed_difference": statistics.mean(deltas), "numerical_gate": None,
                                                   "note": "Observed regrouping/reduction floating differences; no bitexact gate."}
        report["summary"] = summarize(report["variants"])
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
