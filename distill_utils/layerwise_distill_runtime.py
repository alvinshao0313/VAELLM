import os
from typing import List, Optional

import torch

from train_utils.data_utils import get_wikitext2
from train_utils.model_checkpoint_io import META_FILENAME


def resolve_checkpoint_dir(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        if os.path.basename(abs_path) == META_FILENAME:
            return os.path.dirname(abs_path)
        raise FileNotFoundError(f"Expected {META_FILENAME} file, got: {abs_path}")

    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Path does not exist: {abs_path}")

    direct_meta = os.path.join(abs_path, META_FILENAME)
    if os.path.exists(direct_meta):
        return abs_path

    final_model_meta = os.path.join(abs_path, "final_model", META_FILENAME)
    if os.path.exists(final_model_meta):
        return os.path.join(abs_path, "final_model")

    candidates: List[str] = []
    for child in os.listdir(abs_path):
        child_dir = os.path.join(abs_path, child)
        if not os.path.isdir(child_dir):
            continue
        if os.path.exists(os.path.join(child_dir, META_FILENAME)):
            candidates.append(child_dir)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot find checkpoint metadata under: {abs_path}. "
        f"Please pass a directory containing {META_FILENAME}."
    )


def resolve_device(device: str) -> str:
    d = str(device).strip()
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return d


def collect_calib_inputs(
    *,
    model_path: str,
    nsamples: int,
    seed: int,
    seqlen: int,
    access_token: Optional[str] = None,
) -> torch.Tensor:
    tokenizer = None
    if access_token:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            token=access_token,
        )
    samples = get_wikitext2(
        nsamples=int(nsamples),
        seed=int(seed),
        seqlen=int(seqlen),
        model=str(model_path),
        tokenizer=tokenizer,
    )
    if not samples:
        raise RuntimeError("Empty calibration set.")
    return torch.cat([item[0] for item in samples], dim=0).contiguous()
