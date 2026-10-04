"""Immutable paired task data for the isolated ChatML/bare-space recovery probe.

Selection uses only raw identity, recoverable content and tokenizer boundaries.
No teacher/model is loaded; the production canonical data rules are unchanged.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from e2e_common.data import _normalize_edgerazor_messages
from e2e_common.lazy_datasets import _build_permutation_indices
from experiments.qwen3_w25.kl_gradient_diagnostic import batch_manifest
from experiments.qwen3_w25.teacher_format_diagnostic import prepare_bare, recover_single_turn, text_sha
from experiments.qwen3_w25.teacher_loading_diagnostic import sha256
from train_utils.distill_data import FORMATTING_VERSION, build_distill_data_collator, encode_canonical_record

FORMAT = "qwen3_format_recovery_archive_v1"
VARIANTS = ("chatml", "bare_space")
SOURCE_SHA = "894a04aea907efebd8b0f52f040c5635750bb30dfb64d9c34e30a97421a155a1"
SOURCE_ROWS = 176597
TRAIN_SEQUENCES, HELDOUT_SEQUENCES = 1024, 32
MAX_LENGTH, BATCH_SIZE, SEED = 1024, 4, 0
IDENTITY_KEYS = ("source_sample_index", "raw_line_number", "raw_utf8_sha256", "content_pair_sha256",
                 "user_sha256", "answer_sha256")
SELECTION_RULES = {
    "order": "existing _build_permutation_indices over all raw rows, seed=0; no worker interleaving",
    "eligibility": "normalized user/assistant only; both full inputs <=1024; canonical roundtrip and bare-space token prefix exact",
    "deduplication": "first visited raw UTF-8 line hash and normalized (user,answer) pair; shared across both formats and splits",
    "raw_hash_scope": "exact UTF-8 line bytes including its original line terminator",
    "content_hash_scope": "UTF-8 compact JSON [normalized_user,normalized_answer], ensure_ascii=False",
    "normalization": "existing _normalize_edgerazor_messages: str(content).strip(), normalized role",
    "split": "first 1024 accepted rows train, next 32 heldout; contiguous B4; no reshuffle",
    "selection_independence": "no model, teacher loss, label probability or downstream result is consulted",
}


class RejectedPair(ValueError):
    """A shared, declared tokenizer/content eligibility failure."""


def _json_sha(value):
    return text_sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _normalized_pair(record):
    messages = _normalize_edgerazor_messages(record)
    if messages is None or [m["role"] for m in messages] != ["user", "assistant"]:
        raise RejectedPair("not_two_normalized_user_assistant_turns")
    return messages, messages[0]["content"], messages[1]["content"]


def encode_paired_record(record, tokenizer):
    """Produce both unpadded inputs or reject the row for both arms together."""
    messages, user, answer = _normalized_pair(record)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full = rendered + tokenizer.eos_token
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if len(full_ids) > MAX_LENGTH:
        raise RejectedPair("chatml_exceeds_1024_without_truncation")
    canonical = encode_canonical_record(record, tokenizer, text_format="edgerazor_messages", text_field="text",
                                        task="sft", model_max_length=MAX_LENGTH)
    if canonical is None or canonical["input_ids"].tolist() != full_ids:
        raise RejectedPair("canonical_input_not_complete_template_plus_eos")
    try:
        parsed = recover_single_turn(tokenizer, full_ids, canonical["labels"].tolist())
    except ValueError as exc:
        raise RejectedPair("canonical_content_not_exactly_recoverable") from exc
    if parsed["user"] != user or parsed["answer"] != answer:
        raise RejectedPair("canonical_rendered_content_differs_from_normalized_source")
    try:
        bare = prepare_bare(tokenizer, user, answer, separator=" ")
    except ValueError as exc:
        raise RejectedPair("bare_space_pair_boundary_not_exact") from exc
    if len(bare["input_ids"]) > MAX_LENGTH:
        raise RejectedPair("bare_space_exceeds_1024_without_truncation")
    if tokenizer.decode(bare["input_ids"], clean_up_tokenization_spaces=False) != user + " " + answer:
        raise RejectedPair("bare_space_content_not_exactly_recoverable")
    bare_labels = [-100] * len(bare["input_ids"])
    for position in bare["answer_positions"]:
        bare_labels[position] = bare["input_ids"][position]
    features = {"chatml": {key: canonical[key].tolist() for key in ("input_ids", "labels")},
                "bare_space": {"input_ids": bare["input_ids"], "labels": bare_labels}}
    metadata = {
        "chatml": {"answer_positions": parsed["answer_positions"], "answer_tokens": parsed["answer_tokens"],
                   "input_utf8_sha256": parsed["full_utf8_sha256"], "canonical_input_exact": True,
                   "canonical_labels_exact": True, "separator_fused_answer_tokens": 0},
        "bare_space": {key: value for key, value in bare.items()
                       if key not in ("input_ids", "context_ids", "continuation_ids")},
    }
    return features, metadata


def _index_source(path):
    path = Path(path)
    before = path.stat()
    offsets, digest = [], hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            offsets.append(offset)
            digest.update(raw)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError("Raw source changed during indexing.")
    return offsets, {"path": str(path.resolve()), "sha256": digest.hexdigest(), "rows": len(offsets),
                     "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "inode": after.st_ino}


def _select_pairs(source, offsets, tokenizer, *, count):
    permutation = _build_permutation_indices(len(offsets), seed=SEED)
    seen_raw, seen_content, selected, feature_rows, skipped = set(), set(), [], [], []
    with Path(source).open("rb") as stream:
        for ordinal, row in enumerate(permutation):
            stream.seek(offsets[int(row)])
            raw = stream.readline()
            identity = {"permutation_ordinal": ordinal, "raw_line_number": int(row) + 1,
                        "raw_utf8_sha256": hashlib.sha256(raw).hexdigest()}
            reason = None
            if identity["raw_utf8_sha256"] in seen_raw:
                reason = "duplicate_raw_line"
            seen_raw.add(identity["raw_utf8_sha256"])
            if reason is None:
                # Malformed JSON/UTF-8 is a source error, not a silently skipped sample.
                record = json.loads(raw.decode("utf-8"))
                try:
                    _, user, answer = _normalized_pair(record)
                    identity.update(user_sha256=text_sha(user), answer_sha256=text_sha(answer),
                                    content_pair_sha256=_json_sha([user, answer]))
                    if identity["content_pair_sha256"] in seen_content:
                        raise RejectedPair("duplicate_normalized_content_pair")
                    seen_content.add(identity["content_pair_sha256"])
                    features, metadata = encode_paired_record(record, tokenizer)
                except RejectedPair as exc:
                    reason = str(exc)
            if reason is not None:
                skipped.append({**identity, "reason": reason})
                continue
            identity.update(source_sample_index=len(selected), formats=metadata)
            selected.append(identity)
            feature_rows.append(features)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} shared eligible unique pairs, required {count}.")
    return selected, feature_rows, {"seed": SEED, "rules": SELECTION_RULES,
                                   "permutation_dtype": str(permutation.dtype),
                                   "full_permutation_sha256": hashlib.sha256(permutation.tobytes()).hexdigest(),
                                   "visited_rows": selected[-1]["permutation_ordinal"] + 1,
                                   "skipped": skipped, "skip_counts": dict(Counter(s["reason"] for s in skipped))}


def _make_manifest(batch, index, samples, variant):
    manifest = batch_manifest(batch, index, sample_offset=index * BATCH_SIZE)
    manifest["index_scope"] = "fixed raw seed0 permutation, shared eligible unique pairs; one-based raw_line_number"
    for item, sample in zip(manifest["samples"], samples):
        item.update({key: sample[key] for key in IDENTITY_KEYS})
        item["answer_positions"] = sample["formats"][variant]["answer_positions"]
    return manifest


def _check_batch(batch, samples, variant, *, pad_id):
    if set(batch) != {"input_ids", "attention_mask", "labels"}:
        raise ValueError("Expected only input_ids, attention_mask and labels.")
    shape = batch["input_ids"].shape
    if len(shape) != 2 or shape[0] != BATCH_SIZE or not 2 <= shape[1] <= MAX_LENGTH or shape[1] % 8:
        raise ValueError("Expected B4, max1024, dynamic right padding to a multiple of eight.")
    if any(v.dtype != torch.int64 or v.device.type != "cpu" or v.shape != shape for v in batch.values()):
        raise ValueError("Paired archive tensors must have the same shape and be CPU int64.")
    for row, sample in enumerate(samples):
        ids, mask, labels = [batch[key][row] for key in ("input_ids", "attention_mask", "labels")]
        length = int(mask.sum())
        if mask.tolist() != [1] * length + [0] * (len(mask) - length) or length < 2:
            raise ValueError("Padding must be a contiguous right suffix.")
        if bool((ids[length:] != pad_id).any()) or bool((labels[length:] != -100).any()):
            raise ValueError("Padding IDs/labels differ from the fixed tokenizer convention.")
        if bool(((labels != -100) & (labels != ids)).any()):
            raise ValueError("Response labels must be their actual input token IDs.")
        positions = sample["formats"][variant]["answer_positions"]
        if not positions or positions != sorted(set(positions)) or any(not 0 < p < length for p in positions):
            raise ValueError("Answer positions must be unique, ordered, causally predictable valid targets.")
        if any(int(labels[p]) != int(ids[p]) for p in positions):
            raise ValueError("An answer token is excluded from response labels.")
        if variant == "bare_space" and (labels != -100).nonzero().flatten().tolist() != positions:
            raise ValueError("Bare response labels must contain exactly the answer-overlap tokens.")


def validate_format_training_archive(archive, *, verify_source=False, tokenizer=None):
    """Validate tensors/provenance; preparation additionally replays the real source."""
    total = TRAIN_SEQUENCES + HELDOUT_SEQUENCES
    if (archive.get("format") != FORMAT or tuple(archive.get("formats", {})) != VARIANTS
            or archive.get("train_count") != TRAIN_SEQUENCES or archive.get("heldout_count") != HELDOUT_SEQUENCES
            or archive.get("max_length") != MAX_LENGTH or archive.get("batch_size") != BATCH_SIZE):
        raise ValueError("Expected the fixed 1024+32 paired-format archive.")
    samples, selection = archive["source_samples"], archive["selection"]
    if len(samples) != total or selection["rules"] != SELECTION_RULES or selection["seed"] != SEED:
        raise ValueError("Shared source count or declared selection rule differs.")
    source = archive["data_source"]
    if source["sha256"] != SOURCE_SHA or source["rows"] != SOURCE_ROWS:
        raise ValueError("Unexpected task source identity.")
    permutation = _build_permutation_indices(source["rows"], seed=SEED)
    if hashlib.sha256(permutation.tobytes()).hexdigest() != selection["full_permutation_sha256"]:
        raise ValueError("Full raw permutation differs.")
    if [s["permutation_ordinal"] for s in samples] != sorted(s["permutation_ordinal"] for s in samples):
        raise ValueError("Accepted raw order changed.")
    trace = sorted(samples + selection["skipped"], key=lambda s: s["permutation_ordinal"])
    if [s["permutation_ordinal"] for s in trace] != list(range(selection["visited_rows"])):
        raise ValueError("Selection trace does not account for the complete consumed prefix.")
    if any(s["raw_line_number"] != int(permutation[s["permutation_ordinal"]]) + 1 for s in trace):
        raise ValueError("Raw row numbers do not match the fixed permutation.")
    if dict(Counter(s["reason"] for s in selection["skipped"])) != selection["skip_counts"]:
        raise ValueError("Skipped row counts differ.")
    for key in ("raw_line_number", "raw_utf8_sha256", "content_pair_sha256"):
        if len({s[key] for s in samples}) != total:
            raise ValueError(f"Shared pairs are not unique by {key}.")
    if [s["source_sample_index"] for s in samples] != list(range(total)):
        raise ValueError("Shared sample order differs.")
    audit = {"training_sequences": TRAIN_SEQUENCES, "heldout_sequences": HELDOUT_SEQUENCES,
             "train_heldout_raw_and_content_intersection": 0, "source_sha256": source["sha256"],
             "visited_rows": selection["visited_rows"], "skip_counts": selection["skip_counts"],
             "source_replay_exact": False, "formats": {}}
    for variant in VARIANTS:
        values = archive["formats"][variant]
        if len(values["batches"]) != total // BATCH_SIZE or len(values["manifests"]) != total // BATCH_SIZE:
            raise ValueError("Each format must contain 256 train and eight heldout batches.")
        for i, (batch, manifest) in enumerate(zip(values["batches"], values["manifests"])):
            group = samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            _check_batch(batch, group, variant, pad_id=archive["tokenizer"]["pad_token_id"])
            if manifest != _make_manifest(batch, i, group, variant):
                raise ValueError(f"{variant} tensor hash, source identity or masks differ at batch {i}.")
        identities = [s["input_and_label_sha256"] for m in values["manifests"] for s in m["samples"]]
        if set(identities[:TRAIN_SEQUENCES]) & set(identities[TRAIN_SEQUENCES:]):
            raise ValueError("Training and heldout token/label hashes overlap.")
        split_stats = {}
        for split, start, end in (("train", 0, TRAIN_SEQUENCES), ("heldout", TRAIN_SEQUENCES, total)):
            ms = values["manifests"][start // BATCH_SIZE:end // BATCH_SIZE]
            split_stats[split] = {"sequences": end - start,
                                 "input_tokens": sum(s["input_tokens"] for m in ms for s in m["samples"]),
                                 "causal_response_tokens": sum(m["causal_response_tokens"] for m in ms),
                                 "causal_prompt_tokens": sum(m["causal_prompt_tokens"] for m in ms),
                                 "semantic_answer_tokens": sum(len(s["formats"][variant]["answer_positions"]) for s in samples[start:end]),
                                 "separator_fused_answer_tokens": sum(s["formats"][variant]["separator_fused_answer_tokens"] for s in samples[start:end]),
                                 "max_padded_length": max(m["shape"][1] for m in ms)}
        audit["formats"][variant] = split_stats
    if verify_source:
        if tokenizer is None:
            raise ValueError("Actual source replay requires the original fast tokenizer.")
        offsets, actual = _index_source(source["path"])
        if any(actual[k] != source[k] for k in actual):
            raise ValueError("Raw source file identity/stat changed.")
        replay_samples, replay_features, replay_selection = _select_pairs(source["path"], offsets, tokenizer, count=total)
        if replay_samples != samples or replay_selection != selection:
            raise ValueError("Shared source selection/recovery replay differs.")
        for i, pair in enumerate(replay_features):
            for variant in VARIANTS:
                batch = archive["formats"][variant]["batches"][i // BATCH_SIZE]
                length = len(pair[variant]["input_ids"])
                if any(batch[key][i % BATCH_SIZE, :length].tolist() != pair[variant][key] for key in ("input_ids", "labels")):
                    raise ValueError("Archived tokens/labels differ from their actual raw source row.")
        audit["source_replay_exact"] = True
    return audit


def prepare_archive(*, source, source_manifest, hf_source, output):
    from transformers import AutoTokenizer

    start = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path, report_path = output / "batches.pt", output / "preparation.json"
    if archive_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite an existing paired archive or preparation record.")
    offsets, source_record = _index_source(source)
    manifest_path = Path(source_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if (source_record["sha256"] != SOURCE_SHA or source_record["rows"] != SOURCE_ROWS
            or manifest["sha256"] != SOURCE_SHA or manifest["samples"] != SOURCE_ROWS
            or manifest["size_bytes"] != source_record["size_bytes"]
            or Path(manifest["path"]).resolve() != Path(source).resolve()):
        raise ValueError("Expected the actual fixed, repaired task176597 source and its manifest.")
    source_record["manifest"] = {"path": str(manifest_path), "sha256": sha256(manifest_path), "record": manifest}
    tokenizer = AutoTokenizer.from_pretrained(hf_source, use_fast=True, trust_remote_code=True, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
        raise ValueError("The official fast tokenizer with existing right-padding convention is required.")
    tokenizer_files = [Path(hf_source) / name for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")]
    tokenizer_record = {"path": str(Path(hf_source).resolve()), "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id, "padding_side": tokenizer.padding_side,
                        "files_sha256": {str(p.resolve()): sha256(p) for p in tokenizer_files}}
    samples, feature_rows, selection = _select_pairs(source, offsets, tokenizer, count=TRAIN_SEQUENCES + HELDOUT_SEQUENCES)
    collator = build_distill_data_collator(tokenizer, model_max_length=MAX_LENGTH, dynamic_padding=True)
    formats = {}
    for variant in VARIANTS:
        batches, manifests = [], []
        for offset in range(0, len(samples), BATCH_SIZE):
            batch = dict(collator([row[variant] for row in feature_rows[offset:offset + BATCH_SIZE]]))
            batches.append(batch)
            manifests.append(_make_manifest(batch, offset // BATCH_SIZE, samples[offset:offset + BATCH_SIZE], variant))
        formats[variant] = {"batches": batches, "manifests": manifests}
    archive = {"format": FORMAT, "data_source": source_record, "selection": selection,
               "tokenizer": tokenizer_record, "source_samples": samples, "formats": formats,
               "train_count": TRAIN_SEQUENCES, "heldout_count": HELDOUT_SEQUENCES,
               "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH, "canonical_formatting_version": FORMATTING_VERSION,
               "data_scope": "paired task-only raw rows; distinct from the historical task50 mixture archive",
               "mask_scope": {"chatml": "unchanged canonical SFT response including format/EOS",
                              "bare_space": "only tokens overlapping normalized answer characters; fused added space stays response, standalone space stays prompt; no EOS appended"}}
    audit = validate_format_training_archive(archive, verify_source=True, tokenizer=tokenizer)
    temporary = output / "batches.pt.tmp"
    torch.save(archive, temporary)
    reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
    if validate_format_training_archive(reloaded) != {**audit, "source_replay_exact": False}:
        raise RuntimeError("Saved/reloaded archive audit differs.")
    temporary.rename(archive_path)
    dependencies = [Path(__file__), Path(__file__).with_name("prepare_format_training_data.py"),
                    Path("e2e_common/data.py"), Path("e2e_common/lazy_datasets.py"), Path("train_utils/distill_data.py"),
                    Path("train_utils/distill_loss_core.py"), Path("experiments/qwen3_w25/teacher_format_diagnostic.py"),
                    Path("experiments/qwen3_w25/kl_gradient_diagnostic.py"),
                    Path("experiments/qwen3_w25/teacher_loading_diagnostic.py")]
    report = {"status": "completed_verified", "archive": {"path": str(archive_path), "sha256": sha256(archive_path),
                                                               "size_bytes": archive_path.stat().st_size},
              "format": FORMAT, "audit": audit, "source": source_record, "tokenizer": tokenizer_record,
              "selection_rules": SELECTION_RULES, "selection_sha256": _json_sha(selection),
              "source_samples_sha256": _json_sha(samples), "source_sha256": {str(p.resolve()): sha256(p) for p in dependencies},
              "elapsed_seconds": time.monotonic() - start, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "boundary": "Data/CPU identity validation only; no model, GPU, training or quality measurement."}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
