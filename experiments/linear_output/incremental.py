"""Incremental single-stage trainer for one non-transposed Linear.

The trainer keeps the source matrix and VAE block tensor on CPU between updates.
Each call to ``step`` transfers the complete normalized block set for exactly one
optimizer update, so a caller can sweep teacher activations once and update all
Linear trainers from the same batch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from transformers import get_scheduler

from litebsq.llm_vae import MultiLayerVAE
from train_utils.cat_data_prep import (
    LinearPrepRef,
    materialize_prepared_group_data,
    prepare_group_linear_entries,
)
from train_utils.cat_train_data import (
    apply_stage_norm,
    compute_stage_norm_stats,
    restore_stage_norm,
)
from train_utils.cat_train_pipeline import _fuse_norm_into_decoder, _fuse_q_scale_into_decoder
from train_utils.train_args import create_optimizer

from .artifacts import dump_json, export_linear, state_digest, tensor_digest
from .config import validate, vae_arguments
from .objectives import full_vae_forward
from .output_kernel import output_mse


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _move_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    return value


class LinearTrainer:
    """Train one independent, residual-free W2 VAE incrementally.

    ``step`` consumes one already-captured teacher-input activation tensor. It
    performs exactly one optimizer update over all weight blocks of this
    Linear. The same trainer can therefore be called once per shared teacher
    batch by a sweep coordinator.
    """

    def __init__(self, linear: nn.Linear, name: str, args, output: str | Path):
        validate(args)
        if not isinstance(linear, nn.Linear) or linear.in_features % 32:
            raise ValueError("Only nn.Linear matrices with input width divisible by 32 are supported.")
        if int(getattr(args, "residual_stages", 1)) != 1:
            raise ValueError("LinearTrainer is single-stage and residual-free.")
        self.name = str(name)
        self.args = args
        self.device = torch.device(args.device)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.original = linear.weight.detach().cpu().float().contiguous()
        if not torch.isfinite(self.original).all():
            raise ValueError(f"Nonfinite source weights: {self.name}")
        self.bias = None if linear.bias is None else linear.bias.detach().cpu().float().contiguous()
        if self.bias is not None and not torch.isfinite(self.bias).all():
            raise ValueError(f"Nonfinite source bias: {self.name}")
        self.source_weight_sha256 = tensor_digest(self.original)
        prep = prepare_group_linear_entries(
            group_refs=[LinearPrepRef(self.name, self.original, linear.in_features, linear.out_features, False)],
            activation_weight_by_linear=None,
            channel_protect_count=0,
            channel_axis="input",
            recon_loss_type="mse",
            apply_outlier_channel_removal=False,
        )
        data = materialize_prepared_group_data(
            prepared_entries=prep,
            intra_parallel=(1, 1),
            codebook_dim=32,
            batch_size=args.vae_chunk_vectors,
            normalize_weight=False,
            recon_loss_type="mse",
            train_device="cpu",
            split_weights_by_linear=[self.original],
            shuffle_seed=args.seed,
        )
        self.split_metas = data.split_metas
        self.blocks = data.stacked_data.detach().cpu().contiguous()
        if not torch.equal(self.blocks.reshape_as(self.original), self.original):
            raise RuntimeError("Preparation changed the row-major input-axis layout.")
        self.mean, self.scale = compute_stage_norm_stats(self.blocks)
        self.normalized_blocks = apply_stage_norm(self.blocks, mean=self.mean, scale=self.scale).cpu()
        self.vae = MultiLayerVAE(vae_arguments(args)).to(self.device).train()
        self.optimizer = create_optimizer(self.vae.parameters(), self.vae.args, args.vae_learning_rate)
        self.scheduler = get_scheduler("linear", self.optimizer, num_warmup_steps=0, num_training_steps=args.steps)
        # Keep immutable target blocks on the accelerator for the complete run.
        # Copying each full matrix from CPU every update dominated the sweep.
        self._device_dtype = (
            torch.bfloat16
            if args.vae_autocast_dtype == "bf16" and self.device.type == "cuda"
            else torch.float32
        )
        self.normalized_device = self.normalized_blocks.to(
            device=self.device, dtype=self._device_dtype
        )
        self.original_device = self.original.to(device=self.device)
        self.mean_device = self.mean.to(device=self.device)
        self.scale_device = self.scale.to(device=self.device)
        self.initial_state_sha256 = state_digest(self.vae)
        self.update_count = 0

    def _main_loss(self, decoded: torch.Tensor, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decoded_f = decoded.float()
        blocks_f = self.normalized_device.to(dtype=decoded_f.dtype)
        weight_mse = (decoded_f - blocks_f).square().mean()
        restored = restore_stage_norm(decoded_f, mean=self.mean_device, scale=self.scale_device)
        target = self.original_device
        if self.args.objective == "linear_output_mse":
            main = output_mse(restored, target, inputs, use_triton=True)
        elif self.args.objective == "weight_mse":
            main = weight_mse
        else:
            raise ValueError(f"Unsupported objective: {self.args.objective}")
        return main, weight_mse

    def step(self, inputs: torch.Tensor, step_index: int | None = None) -> dict:
        """Perform one update; ``step_index`` is 1-based when supplied."""
        if inputs.ndim != 2 or inputs.shape[1] != self.original.shape[1]:
            raise ValueError(f"inputs must be [valid_tokens,{self.original.shape[1]}], got {tuple(inputs.shape)}")
        if inputs.shape[0] < 1:
            raise ValueError("At least one valid token is required.")
        expected = self.update_count + 1
        if step_index is not None and int(step_index) != expected:
            raise ValueError(f"step_index must be 1-based and equal next update {expected}, got {step_index}")
        self.vae.train()
        device_inputs = inputs.detach().to(self.device)
        blocks = self.normalized_device
        self.optimizer.zero_grad(set_to_none=True)
        # Keep activation checkpointing enabled for large down_proj matrices.
        # The decoded graph otherwise retains every VAE chunk until backward,
        # which can exceed 80 GB even when static weights are cached.
        decoded, auxiliary, _ = full_vae_forward(
            self.vae, blocks, chunk_vectors=self.args.vae_chunk_vectors, recompute=False
        )
        main, weight_mse = self._main_loss(decoded, device_inputs)
        loss = main * self.vae.model.l1_weight * self.vae.model.num_models + auxiliary
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite loss: {self.name}, update {expected}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), float("inf"), error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        self.update_count = expected
        result = {
            "module": self.name,
            "step": self.update_count,
            "objective": self.args.objective,
            "loss": float(loss.detach().cpu()),
            "main_loss": float(main.detach().cpu()),
            "auxiliary_loss": float(auxiliary.detach().cpu()),
            "weight_mse": float(weight_mse.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "valid_tokens": int(inputs.shape[0]),
            "sequences": int(getattr(self.args, "batch_size", 0)),
            "next_learning_rate": float(self.scheduler.get_last_lr()[0]),
        }
        with (self.output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        del decoded, auxiliary, main, weight_mse, loss, blocks, device_inputs
        return result

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "name": self.name,
            "source_weight_sha256": self.source_weight_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "update_count": int(self.update_count),
            "vae": _cpu_tree(self.vae.state_dict()),
            "optimizer": _cpu_tree(self.optimizer.state_dict()),
            "scheduler": _cpu_tree(self.scheduler.state_dict()),
        }

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        if state.get("name") != self.name:
            raise ValueError(f"Checkpoint module mismatch: {state.get('name')!r} != {self.name!r}")
        if state.get("source_weight_sha256") != self.source_weight_sha256:
            raise ValueError("Checkpoint source weight hash does not match this Linear")
        self.vae.load_state_dict(state["vae"], strict=strict)
        self.optimizer.load_state_dict(_move_tree(state["optimizer"], self.device))
        self.scheduler.load_state_dict(state["scheduler"])
        self.update_count = int(state["update_count"])
        if self.update_count < 0 or self.update_count > int(self.args.steps):
            raise ValueError(f"Invalid checkpoint update_count={self.update_count}")

    def _decode_cpu(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retain the training-precision encoder's bits and decoder output."""
        was_training = self.vae.training
        try:
            self.vae.eval()
            with torch.no_grad():
                decoded, _, bits = full_vae_forward(
                    self.vae, self.normalized_device,
                    chunk_vectors=self.args.vae_chunk_vectors, recompute=False,
                )
                restored = restore_stage_norm(
                    decoded.float(), mean=self.mean_device, scale=self.scale_device
                ).cpu()
            return restored.reshape_as(self.original), bits.detach().cpu()
        finally:
            self.vae.train(was_training)

    @torch.no_grad()
    def _decode_fp32_from_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """Independent export reference: fixed bits, unfused FP32 decoder.

        Re-encoding in FP32 would change signs near zero. Keep the actual bits
        selected by the training encoder and isolate decoder precision changes
        from q-scale/norm fusion and packed serialization correctness.
        """
        # CPU is the IEEE FP32 reference: the deployment CUDA decoder may use
        # Triton TF32 even when its tensor dtype is torch.float32.
        decoder = copy.deepcopy(self.vae.model.decoder).cpu().float().eval()
        q_scale = 1 / math.sqrt(self.args.codebook_bits) if self.args.new_quant else 1.0
        restored = []
        with torch.autocast(device_type="cpu", enabled=False):
            for chunk in bits.split(self.args.vae_chunk_vectors):
                quantized = (chunk.cpu().float() * 2 - 1) * q_scale
                decoded = decoder(quantized)
                restored.append(restore_stage_norm(
                    decoded, mean=self.mean.cpu().float(), scale=self.scale.cpu().float()
                ).cpu())
        return torch.cat(restored, dim=0).reshape_as(self.original)

    def export(self) -> dict:
        """Export native single-stage packed/v6 checkpoint and return its record."""
        if self.update_count != int(self.args.steps):
            raise ValueError(f"Cannot export {self.name} before {self.args.steps} updates; got {self.update_count}")
        training_weight, bits = self._decode_cpu()
        expected_weight = self._decode_fp32_from_bits(bits)
        decoder = copy.deepcopy(self.vae.model.decoder).get_sub_decoder(0).cpu().float().eval()
        q_scale = 1 / math.sqrt(self.args.codebook_bits) if self.args.new_quant else 1.0
        _fuse_q_scale_into_decoder(decoder, q_scale=q_scale)
        _fuse_norm_into_decoder(decoder, mean=float(self.mean.item()), std=float(self.scale.item()))
        payload = {
            "format": "vaellm_group_vae_payload",
            "version": 1,
            "target_common_split_metas": self.split_metas,
            "parts_per_linear": 1,
            "row_parts": 1,
            "col_parts": 1,
            "residual_stages": 1,
            "all_stage_bits": [bits],
            "all_stage_decoders": [[decoder]],
            "all_stage_codebook_dims": [32],
            "all_stage_split_metas": [self.split_metas],
            "protected_channel_quant_format": "none",
            "weight_rotation_specs": [None],
        }
        checkpoint = self.output / "packed"
        parity = export_linear(
            self.name, self._linear_host_source, payload, expected_weight, checkpoint,
            self.args, training_weight=training_weight,
        )
        record = {
            "module": self.name,
            "shape": list(self.original.shape),
            "objective": self.args.objective,
            "transpose": False,
            "residual_stages": 1,
            "steps": self.update_count,
            "batch_size": self.args.batch_size,
            "source_weight_sha256": self.source_weight_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "elapsed_seconds": time.perf_counter() - self.started,
            **parity,
        }
        dump_json(self.output / "record.json", record)
        return record

    @property
    def _linear_host_source(self) -> nn.Linear:
        if not hasattr(self, "_source_linear"):
            source = nn.Linear(self.original.shape[1], self.original.shape[0], bias=self.bias is not None)
            source.weight.data.copy_(self.original)
            if self.bias is not None:
                source.bias.data.copy_(self.bias)
            self._source_linear = source
        return self._source_linear


__all__ = ["LinearTrainer"]
