"""Document-disjoint calibration and full-model NLL for the bounded bit ablation."""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.liftquant_recovery.recovery_data import DATASET
from experiments.liftquant_recovery.recovery_runtime import (
    block_output, clear_caches, first_inputs, prime_packed_cache, tree_to,
)


def _token_hash(ids):
    return hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def document_split(path, tokenizer, train_samples, holdout_samples, seqlen, seed):
    """Split unique documents before sampling windows; never sample with replacement."""
    if min(train_samples, holdout_samples) <= 0 or seqlen < 2:
        raise ValueError("Both document splits must be positive and seqlen must be >= 2.")
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("dataset") != DATASET:
        raise ValueError("Expected the configured RedPajama smoke-row dataset.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Smoke-row JSON must contain a rows list.")
    eligible, seen = [], set()
    duplicates, short = 0, 0
    for position, entry in enumerate(rows):
        text = entry.get("row", {}).get("text")
        if not isinstance(text, str):
            raise ValueError(f"Smoke row {position} has no string text.")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        tokens = tokenizer(text, return_tensors="pt").input_ids.detach().cpu()
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(f"Tokenizer returned an invalid shape for row {position}.")
        if tokens.shape[1] < seqlen + 1:
            short += 1
            continue
        eligible.append((tokens, dict(
            source_row_index=entry.get("row_idx", position), payload_row_position=position,
            text_sha256=digest, document_tokens=int(tokens.shape[1]),
        )))
    required = train_samples + holdout_samples
    if len(eligible) < required:
        raise ValueError(
            f"Insufficient unique eligible documents: eligible={len(eligible)}, "
            f"required={required}, total_rows={len(rows)}, duplicate_rows={duplicates}, "
            f"short_unique_documents={short}, required_tokens={seqlen + 1}."
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    windows, documents = [], []
    for index, (tokens, info) in enumerate(eligible[:required]):
        start = rng.randint(0, tokens.shape[1] - seqlen - 1)
        window = tokens[:, start:start + seqlen].to(dtype=torch.long).contiguous()
        documents.append(dict(
            info, split="train" if index < train_samples else "holdout",
            sample_index=index, window_start=start, window_tokens=seqlen,
            token_sha256=_token_hash(window),
        ))
        windows.append(window)
    train_hashes = {item["token_sha256"] for item in documents[:train_samples]}
    heldout_hashes = {item["token_sha256"] for item in documents[train_samples:]}
    if train_hashes.intersection(heldout_hashes):
        raise ValueError("Identical token window across document splits; choose a prespecified new split before training.")
    ids = torch.cat(windows, dim=0)
    meta = dict(
        dataset=payload["dataset"], scope=payload.get("scope"),
        revision=payload.get("revision"), source=payload.get("source"),
        file_sha256=hashlib.sha256(raw).hexdigest(), seed=seed,
        train_samples=train_samples, holdout_samples=holdout_samples, seqlen=seqlen,
        total_source_rows=len(rows), unique_documents=len(seen),
        eligible_documents=len(eligible), duplicate_rows=duplicates,
        short_unique_documents=short, documents=documents,
        input_sha256=_token_hash(ids), document_disjoint=True,
        sampling="Exact-text deduplication; fixed shuffle; one window per unique document.",
        interpretation=(
            "Small first-shard sample, not representative full training or a downstream benchmark. "
            "Holdout documents must not select hyperparameters or the reported endpoint."
        ),
    )
    return ids, meta


@torch.no_grad()
def model_nll(model, ids, batch_size, device):
    """Score next-token NLL with one decoder block on GPU and layer outputs on CPU.

    Every candidate uses native packed-u8 BF16 arithmetic. Only derived caches and
    residency change; model tensors are not trained or replaced. The caller supplies
    the same held-out ids in the same order for all comparisons.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Full-model NLL requires the confirmed CUDA BF16 runtime.")
    if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] < 2 or batch_size <= 0:
        raise ValueError("Expected nonempty equal-length token sequences and positive batch size.")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token IDs must have an integer dtype.")
    ids = ids.detach().cpu().to(dtype=torch.long)
    training_flags = [(module, module.training) for module in model.modules()]
    model.eval()
    model.cpu()
    backbone = model.model
    per_document = []
    try:
        hidden, kwargs = first_inputs(model, ids, batch_size, device)
        gpu_kwargs = tree_to(kwargs, device)
        for index, block in enumerate(backbone.layers):
            clear_caches(block)
            block.to(device)
            try:
                prime_packed_cache(block)
                outputs = torch.empty_like(hidden, device="cpu")
                for start in range(0, len(hidden), batch_size):
                    prediction = block_output(
                        block, hidden[start:start + batch_size].to(device), gpu_kwargs,
                    )
                    outputs[start:start + batch_size].copy_(prediction.cpu())
                    del prediction
            finally:
                clear_caches(block)
                block.cpu()
            hidden = outputs
            del outputs
            torch.cuda.empty_cache()
            if (index + 1) % 8 == 0 or index + 1 == len(backbone.layers):
                print(f"NLL streamed block {index + 1}/{len(backbone.layers)}", flush=True)
        backbone.norm.to(device)
        model.lm_head.to(device)
        for start in range(0, len(hidden), batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                normalized = backbone.norm(hidden[start:start + batch_size].to(device))
                logits = model.lm_head(normalized)
            labels = ids[start:start + batch_size, 1:].to(device)
            losses = F.cross_entropy(
                logits[:, :-1, :].float().transpose(1, 2), labels, reduction="none",
            )
            values = losses.mean(dim=1)
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite full-model document NLL.")
            per_document.extend(values.cpu().tolist())
            del normalized, logits, labels, losses, values
        mean_nll = float(np.mean(per_document))
        return dict(
            nll_per_document=per_document, mean_nll=mean_nll, ppl=math.exp(mean_nll),
            documents=len(per_document), tokens_per_document=int(ids.shape[1]),
            scored_tokens_per_document=int(ids.shape[1] - 1),
            input_sha256=_token_hash(ids),
            evaluation="Full compressed student; unpadded next-token NLL; packed-u8 BF16.",
        )
    finally:
        for block in backbone.layers:
            clear_caches(block)
        model.cpu()
        for module, training in training_flags:
            module.training = training
        torch.cuda.empty_cache()


def paired_summary(reference, candidate, seed=42):
    """Paired descriptive uncertainty across documents, never across tokens or features."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0 or candidate.shape != reference.shape:
        raise ValueError("Paired metrics must be nonempty one-dimensional arrays of equal length.")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Paired metrics must be finite.")
    difference = candidate - reference
    reference_mean, candidate_mean = float(reference.mean()), float(candidate.mean())
    mean_difference = float(difference.mean())
    relative = mean_difference / reference_mean if reference_mean != 0 else None
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(difference), size=(10000, len(difference)))
    boot_means = difference[sampled].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975]).tolist()
    return dict(
        documents=int(reference.size), reference_mean=reference_mean,
        candidate_mean=candidate_mean, mean_difference=mean_difference,
        relative_mean_difference=relative,
        relative_percent=None if relative is None else 100 * relative,
        win_count=int(np.sum(difference < 0)), tie_count=int(np.sum(difference == 0)),
        loss_count=int(np.sum(difference > 0)), median_difference=float(np.median(difference)),
        differences_per_document=difference.tolist(), bootstrap_95ci=[low, high],
        bootstrap_replicates=10000, bootstrap_seed=seed,
        direction="candidate minus reference; negative is improvement",
        interpretation=(
            "Paired document-bootstrap interval is descriptive for this small fixed sample; "
            "it does not include training-seed uncertainty or establish downstream gains."
        ),
    )
