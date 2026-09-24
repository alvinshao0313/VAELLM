"""Online, uncompressed teacher inputs using the production SFT/LM data path."""
from __future__ import annotations

import hashlib

import torch
from torch.utils.data import DataLoader

from train_utils.config.configs import DistillDataConfig
from train_utils.distill_data import build_distill_data_collator, build_distill_dataset


class _InputCaptured(Exception):
    """Stop exactly before the target Linear, not after a whole teacher forward."""


@torch.no_grad()
def capture_inputs(model, module_name: str, batch: dict, *, microbatch_size: int) -> torch.Tensor:
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Calibration requires an eval-mode, frozen teacher.")
    layer = model.get_submodule(module_name)
    device = next(model.parameters()).device
    captured = []
    current_mask = None

    def hook(_module, inputs):
        x = inputs[0]
        if x.ndim != 3 or x.shape[:2] != current_mask.shape or x.shape[-1] != layer.in_features:
            raise ValueError("Expected dense Linear inputs [sequences, tokens, input_channels].")
        captured.append(x.detach()[current_mask].contiguous())
        raise _InputCaptured()

    handle = layer.register_forward_pre_hook(hook)
    try:
        for start in range(0, len(batch["input_ids"]), microbatch_size):
            stop = start + microbatch_size
            ids = batch["input_ids"][start:stop].to(device)
            mask = batch["attention_mask"][start:stop].to(device)
            current_mask = mask.bool()
            try:
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            except _InputCaptured:
                pass
            else:
                raise RuntimeError(f"Target Linear was not executed: {module_name}")
    finally:
        handle.remove()
    x = torch.cat(captured, dim=0)
    if not len(x) or not torch.isfinite(x).all():
        raise ValueError("No valid calibration tokens, or nonfinite inputs.")
    return x


def build_bundle(args, tokenizer):
    cfg = DistillDataConfig(
        dataset_mix=args.dataset_mix, dataset_task=args.dataset_task,
        model_max_length=args.model_max_length, dynamic_padding=args.dynamic_padding,
        seed=args.seed, data_seed=args.data_seed, group_by_length=False,
    )
    return build_distill_dataset(cfg, tokenizer)


class CalibrationStream:
    """A fresh, reproducible stream per Linear; no hidden fixed calibration subset."""

    def __init__(self, model, tokenizer, bundle, args):
        self.model, self.args = model, args
        self.loader = DataLoader(
            bundle.train_dataset, batch_size=args.batch_size,
            shuffle=not bundle.is_iterable, drop_last=True, num_workers=0,
            generator=torch.Generator().manual_seed(args.data_seed),
            collate_fn=build_distill_data_collator(
                tokenizer, model_max_length=args.model_max_length, dynamic_padding=args.dynamic_padding,
            ),
        )
        self.iterator = iter(self.loader)
        self.digest = hashlib.sha256()
        self.sequences = self.valid_tokens = self.batches = 0

    def next_inputs(self, module_name: str) -> torch.Tensor:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            try:
                batch = next(self.iterator)
            except StopIteration as exc:
                raise ValueError("Dataset cannot produce one complete calibration batch.") from exc
        for key in ("input_ids", "attention_mask"):
            tensor = batch[key].contiguous()
            self.digest.update(str(tuple(tensor.shape)).encode())
            self.digest.update(tensor.numpy().tobytes())
        inputs = capture_inputs(
            self.model, module_name, batch, microbatch_size=self.args.calibration_microbatch_size,
        )
        self.sequences += len(batch["input_ids"])
        self.valid_tokens += len(inputs)
        self.batches += 1
        return inputs

    def metadata(self) -> dict:
        return dict(sequences=self.sequences, valid_tokens=self.valid_tokens, batches=self.batches,
                    input_sha256=self.digest.hexdigest(), mask="all attention_mask-valid tokens, including prompt")
