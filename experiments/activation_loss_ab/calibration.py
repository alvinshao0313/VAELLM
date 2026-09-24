"""Collect fixed, disjoint Alpaca calibration activations from the FP model."""
from __future__ import annotations

import hashlib
import random

import torch
from datasets import load_dataset

from experiments.activation_loss_ab.objectives import block_gram_sum


def collect(lm, names: list[str], *, nsamples: int, seqlen: int, seed: int):
    data = load_dataset("vicgalle/alpaca-gpt4", split="train")
    selected = random.Random(seed).sample(range(len(data)), nsamples)
    grams, counts, handles = {}, {}, []
    model = lm.model
    device = next(model.parameters()).device
    for name in names:
        layer = model.get_submodule(name)
        if layer.in_features % 32:
            raise ValueError(f"Input width is not divisible by 32: {name}")
        grams[name] = torch.zeros(layer.in_features // 32, 32, 32, device=device, dtype=torch.float64)
        counts[name] = 0

    def make_hook(name):
        def hook(_layer, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            grams[name].add_(block_gram_sum(x).double())
            counts[name] += len(x)
        return hook

    digest = hashlib.sha256()
    try:
        handles = [model.get_submodule(name).register_forward_pre_hook(make_hook(name)) for name in names]
        with torch.no_grad():
            for i, row_id in enumerate(selected):
                encoded = lm.tokenizer(data[row_id]["text"], return_tensors="pt", truncation=True, max_length=seqlen)
                digest.update(encoded["input_ids"].numpy().tobytes())
                # Each forward is unpadded, so every captured position is valid.
                model.model(input_ids=encoded["input_ids"].to(device), use_cache=False)
                if (i + 1) % 8 == 0 or i + 1 == len(selected):
                    print(f"CALIBRATION {i+1}/{len(selected)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    for name in names:
        if counts[name] < 32:
            raise RuntimeError("Insufficient calibration positions.")
        grams[name] = (grams[name] / counts[name]).float().cpu()
        if not bool(torch.isfinite(grams[name]).all()):
            raise RuntimeError("Nonfinite calibration second moment.")
    return grams, {
        "dataset": "vicgalle/alpaca-gpt4", "split": "train", "row_ids": selected,
        "max_length": seqlen, "seed": seed, "num_positions_by_module": counts,
        "encoded_input_sha256": digest.hexdigest(), "dataset_fingerprint": data._fingerprint,
        "evaluation_used_for_calibration": False, "second_moment_centered": False,
        "source_model": "Unmodified BF16 model; fixed statistics for every objective and training seed.",
    }
