"""Derive answer-only training labels without changing paired archive inputs."""
from __future__ import annotations

import hashlib
import json

import torch


def answer_labels(batch, manifest):
    """Keep exactly the archive's unshifted answer targets; retain every input."""
    ids, labels, mask = (batch[key] for key in ("input_ids", "labels", "attention_mask"))
    if (ids.ndim != 2 or labels.shape != ids.shape or mask.shape != ids.shape
            or any(value.device.type != "cpu" or value.dtype != torch.int64
                   for value in (ids, labels, mask)) or len(manifest["samples"]) != len(ids)):
        raise ValueError("Expected CPU int64 archive batches and matching samples.")
    selected = torch.full_like(labels, -100)
    for row, sample in enumerate(manifest["samples"]):
        positions = sample["answer_positions"]
        if (not positions or positions != sorted(set(positions))
                or any(type(position) is not int for position in positions)
                or positions[0] < 1 or positions[-1] >= ids.shape[1]):
            raise ValueError("Expected nonempty unique causal answer target positions.")
        index = torch.tensor(positions, dtype=torch.long)
        if (not bool(mask[row, index].bool().all())
                or not torch.equal(labels[row, index], ids[row, index])):
            raise ValueError("Answer targets must already be fully included in archive labels.")
        selected[row, index] = labels[row, index]
    return {**batch, "labels": selected}


def _update_hash(digest, tensor):
    digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
    digest.update(tensor.contiguous().numpy().tobytes())


def derive_training_labels(data, *, training_label_region, training_format, loss_recipe):
    """Return a training-only view and small provenance; heldout remains original."""
    if training_label_region not in ("response", "answer"):
        raise ValueError("training_label_region must be response or answer.")
    if training_label_region == "answer" and not (
            training_format == "chatml" and loss_recipe["loss_type"] == "kd_top_mass"
            and loss_recipe["alpha"] == .5 and loss_recipe["prompt_loss_weight"] == 0):
        raise ValueError("Answer-only labels are restricted to ChatML CE50/prompt0.")
    if len(data["batches"]) != 264 or len(data["manifests"]) != 264:
        raise ValueError("Expected the fixed 256 train plus 8 heldout B4 archive.")
    batches = list(data["batches"]) if training_label_region == "answer" else data["batches"]
    original_hash, selected_hash, inputs_hash = (hashlib.sha256() for _ in range(3))
    original_tokens = selected_tokens = answer_tokens = sequences = 0
    for index, (batch, manifest) in enumerate(zip(data["batches"][:256], data["manifests"][:256])):
        if len(batch["input_ids"]) != 4:
            raise ValueError("Every training batch must retain the original four sequences.")
        if training_label_region == "answer":
            batches[index] = answer_labels(batch, manifest)
        derived = batches[index]
        _update_hash(original_hash, batch["labels"])
        _update_hash(selected_hash, derived["labels"])
        for key in ("input_ids", "attention_mask"):
            _update_hash(inputs_hash, batch[key])
        original_tokens += int(((batch["labels"][:, 1:] != -100) & batch["attention_mask"][:, 1:].bool()).sum())
        selected_tokens += int(((derived["labels"][:, 1:] != -100) & derived["attention_mask"][:, 1:].bool()).sum())
        answer_tokens += sum(len(sample["answer_positions"]) for sample in manifest["samples"])
        sequences += len(batch["input_ids"])
    if training_label_region == "answer" and selected_tokens != answer_tokens:
        raise ValueError("Derived causal label count differs from declared answer positions.")
    audit = {"training_label_region": training_label_region, "training_batches": 256,
             "training_sequences": sequences, "original_response_tokens": original_tokens,
             "selected_training_tokens": selected_tokens, "answer_tokens": answer_tokens,
             "removed_response_tokens": original_tokens-selected_tokens,
             "original_training_labels_sha256": original_hash.hexdigest(),
             "derived_training_labels_sha256": selected_hash.hexdigest(),
             "unchanged_training_inputs_and_attention_sha256": inputs_hash.hexdigest(),
             "heldout_labels": "unchanged original archive response labels",
             "position_convention": "unshifted target indices; production loss applies its one causal shift"}
    return ({**data, "batches": batches} if training_label_region == "answer" else data), audit
