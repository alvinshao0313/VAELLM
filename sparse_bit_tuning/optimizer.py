from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .config import SparseBitTuningConfig, normalize_bit_optimizer, resolve_bit_lr
from .module import BankSpec, SparseBitTuningModule
from .triton_kernels import launch_adam_update, launch_rms_sgd_update


@dataclass
class OptimizerChunkMeta:
    bank_specs: Tuple[BankSpec, ...]
    n_active: torch.Tensor
    score_offset: torch.Tensor

    @property
    def num_banks(self) -> int:
        return len(self.bank_specs)

    @property
    def max_active(self) -> int:
        return max(int(spec.n_active) for spec in self.bank_specs)


@dataclass
class _BitState:
    exp_avg: Optional[torch.Tensor] = None
    exp_avg_sq: Optional[torch.Tensor] = None


class BitOptimizerManager:
    def __init__(self, module: SparseBitTuningModule, config: SparseBitTuningConfig) -> None:
        self.module = module
        self.config = config.normalized()
        self.optimizer_name = normalize_bit_optimizer(self.config.optimizer)
        self.lr = resolve_bit_lr(self.config.bit_lr, optimizer=self.optimizer_name)
        self.weight_decay = float(self.config.weight_decay)
        self._chunk_meta: Dict[int, OptimizerChunkMeta] = {}
        self._state: Dict[int, _BitState] = {}
        specs_by_chunk: Dict[int, List[BankSpec]] = {}
        for spec in module.bank_specs:
            specs_by_chunk.setdefault(int(spec.chunk_id), []).append(spec)
        for chunk_id, specs in specs_by_chunk.items():
            ordered = tuple(sorted(specs, key=lambda s: int(s.score_start)))
            device = module.score_chunks[int(chunk_id)].device
            self._chunk_meta[int(chunk_id)] = OptimizerChunkMeta(
                bank_specs=ordered,
                n_active=torch.tensor([int(s.n_active) for s in ordered], dtype=torch.int64, device=device),
                score_offset=torch.tensor([int(s.score_start) for s in ordered], dtype=torch.int64, device=device),
            )
            self._state[int(chunk_id)] = _BitState()

    def bit_parameters(self) -> Iterable[nn.Parameter]:
        return self.module.bit_parameters()

    def _ensure_adam_state(self, chunk_id: int) -> _BitState:
        state = self._state[int(chunk_id)]
        score = self.module.score_chunks[int(chunk_id)]
        if state.exp_avg is None:
            state.exp_avg = torch.zeros_like(score, dtype=torch.float32, device=score.device)
            state.exp_avg_sq = torch.zeros_like(score, dtype=torch.float32, device=score.device)
        return state

    def validate_gradients(self) -> None:
        for chunk_id, score in enumerate(self.module.score_chunks):
            if score.grad is None:
                raise RuntimeError(
                    f"Sparse Bit score chunk {chunk_id} has grad=None after a valid backward; "
                    "the bit-aware autograd path is disconnected or the target did not execute."
                )
            if score.grad.dtype != torch.float16:
                raise RuntimeError(
                    f"Sparse Bit score chunk {chunk_id} grad must be FP16, got {score.grad.dtype}."
                )
            if score.grad.device != score.device:
                raise RuntimeError(
                    f"Sparse Bit score chunk {chunk_id} grad device {score.grad.device} != score device {score.device}."
                )

    @torch.no_grad()
    def step_scores(self, *, optimizer_step_in_round: int) -> Tuple[torch.Tensor, ...]:
        """Update score Parameters and return per-chunk old-score/new-score sign flips.

        These counters are authoritative only for streaming mode, where active hard
        state is score>=0 between optimizer steps.  Non-streaming must use packed
        SET counters instead because the packed forward buffer is its source of truth.
        """
        self.validate_gradients()
        t = int(optimizer_step_in_round)
        if t < 1:
            raise ValueError(f"optimizer_step_in_round must be >=1, got {t}.")
        counters: List[torch.Tensor] = []
        for chunk_id, score in enumerate(self.module.score_chunks):
            grad = score.grad
            assert grad is not None
            meta = self._chunk_meta[int(chunk_id)]
            counter = torch.zeros((), device=score.device, dtype=torch.int32)
            if self.optimizer_name == "rms_sgd":
                counter = launch_rms_sgd_update(
                    score,
                    grad,
                    meta,
                    lr=self.lr,
                    eps=1e-8,
                    flip_counter=counter,
                )
            else:
                state = self._ensure_adam_state(int(chunk_id))
                assert state.exp_avg is not None and state.exp_avg_sq is not None
                counter = launch_adam_update(
                    score,
                    grad,
                    state.exp_avg,
                    state.exp_avg_sq,
                    meta,
                    lr=self.lr,
                    step=t,
                    weight_decay=self.weight_decay if self.optimizer_name == "adamw" else 0.0,
                    adamw=self.optimizer_name == "adamw",
                    flip_counter=counter,
                )
            counters.append(counter)
        return tuple(counters)

    @torch.no_grad()
    def reset_round_state(self) -> None:
        for state in self._state.values():
            if state.exp_avg is not None:
                state.exp_avg.zero_()
            if state.exp_avg_sq is not None:
                state.exp_avg_sq.zero_()

    @torch.no_grad()
    def reset_bank_state(self, bank_specs: Sequence[BankSpec]) -> None:
        """Reset Adam/AdamW state only for the supplied logical bank slices."""
        for spec in bank_specs:
            state = self._state[int(spec.chunk_id)]
            start, end = int(spec.score_start), int(spec.score_end)
            if state.exp_avg is not None:
                state.exp_avg[start:end].zero_()
            if state.exp_avg_sq is not None:
                state.exp_avg_sq[start:end].zero_()

    def state_tensors(self) -> Iterable[torch.Tensor]:
        for state in self._state.values():
            if state.exp_avg is not None:
                yield state.exp_avg
            if state.exp_avg_sq is not None:
                yield state.exp_avg_sq

    def clear_state(self) -> None:
        for key in list(self._state):
            self._state[key] = _BitState()

    def state_dict(self) -> dict:
        """Serialize the optimizer-owned live bit state without touching private fields externally."""
        chunks = {}
        for chunk_id in sorted(self._state):
            state = self._state[int(chunk_id)]
            chunks[str(int(chunk_id))] = {
                "exp_avg": None
                if state.exp_avg is None
                else state.exp_avg.detach().to(device="cpu", dtype=torch.float32).contiguous(),
                "exp_avg_sq": None
                if state.exp_avg_sq is None
                else state.exp_avg_sq.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            }
        return {
            "format": "sparse_bit_optimizer_state",
            "version": 1,
            "optimizer_name": str(self.optimizer_name),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "chunks": chunks,
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError(f"BitOptimizerManager state must be dict, got {type(state_dict)}.")
        if str(state_dict.get("format")) != "sparse_bit_optimizer_state" or int(
            state_dict.get("version", -1)
        ) != 1:
            raise ValueError(
                "unsupported BitOptimizerManager state format/version: "
                f"{state_dict.get('format')!r}/{state_dict.get('version')!r}."
            )
        if str(state_dict.get("optimizer_name")) != str(self.optimizer_name):
            raise ValueError(
                f"bit optimizer mismatch: checkpoint={state_dict.get('optimizer_name')!r} "
                f"current={self.optimizer_name!r}."
            )
        if float(state_dict.get("lr")) != float(self.lr):
            raise ValueError(f"bit lr mismatch: checkpoint={state_dict.get('lr')} current={self.lr}.")
        if float(state_dict.get("weight_decay")) != float(self.weight_decay):
            raise ValueError(
                f"bit weight_decay mismatch: checkpoint={state_dict.get('weight_decay')} "
                f"current={self.weight_decay}."
            )
        chunks = state_dict.get("chunks")
        if not isinstance(chunks, dict):
            raise TypeError("BitOptimizerManager state 'chunks' must be a dict.")
        expected = {str(int(chunk_id)) for chunk_id in self._state}
        provided = {str(key) for key in chunks}
        if provided != expected:
            raise ValueError(
                "bit optimizer chunk set mismatch: "
                f"missing={sorted(expected - provided)} extra={sorted(provided - expected)}"
            )
        for chunk_id in sorted(self._state):
            payload = chunks[str(int(chunk_id))]
            if not isinstance(payload, dict):
                raise TypeError(f"bit optimizer chunk {chunk_id} payload must be dict.")
            score = self.module.score_chunks[int(chunk_id)]
            exp_avg = payload.get("exp_avg")
            exp_avg_sq = payload.get("exp_avg_sq")
            if (exp_avg is None) != (exp_avg_sq is None):
                raise ValueError(f"bit optimizer chunk {chunk_id} has partial Adam state.")
            if exp_avg is None:
                self._state[int(chunk_id)] = _BitState()
                continue
            if not torch.is_tensor(exp_avg) or not torch.is_tensor(exp_avg_sq):
                raise TypeError(f"bit optimizer chunk {chunk_id} Adam state must contain tensors.")
            if tuple(exp_avg.shape) != tuple(score.shape) or tuple(exp_avg_sq.shape) != tuple(score.shape):
                raise ValueError(
                    f"bit optimizer chunk {chunk_id} state shape mismatch: "
                    f"exp_avg={tuple(exp_avg.shape)} exp_avg_sq={tuple(exp_avg_sq.shape)} "
                    f"score={tuple(score.shape)}."
                )
            self._state[int(chunk_id)] = _BitState(
                exp_avg=exp_avg.detach().to(device=score.device, dtype=torch.float32).contiguous(),
                exp_avg_sq=exp_avg_sq.detach().to(device=score.device, dtype=torch.float32).contiguous(),
            )

    def offload_state_for_eval(self) -> dict:
        token: dict[int, Tuple[Optional[torch.device], Optional[torch.device]]] = {}
        for chunk_id, state in self._state.items():
            dev_m = state.exp_avg.device if state.exp_avg is not None else None
            dev_v = state.exp_avg_sq.device if state.exp_avg_sq is not None else None
            token[int(chunk_id)] = (dev_m, dev_v)
            if state.exp_avg is not None and state.exp_avg.device.type == "cuda":
                state.exp_avg = state.exp_avg.to("cpu")
            if state.exp_avg_sq is not None and state.exp_avg_sq.device.type == "cuda":
                state.exp_avg_sq = state.exp_avg_sq.to("cpu")
        return token

    def restore_state_after_eval(self, token: dict) -> None:
        for chunk_id, devices in token.items():
            state = self._state[int(chunk_id)]
            dev_m, dev_v = devices
            if state.exp_avg is not None and dev_m is not None and state.exp_avg.device != dev_m:
                state.exp_avg = state.exp_avg.to(dev_m)
            if state.exp_avg_sq is not None and dev_v is not None and state.exp_avg_sq.device != dev_v:
                state.exp_avg_sq = state.exp_avg_sq.to(dev_v)


class SparseBitCompositeOptimizer(torch.optim.Optimizer):
    """A real Optimizer shell so Accelerate/GradScaler see main + bit parameters.

    Main optimizer state is delegated to the existing optimizer.  Bit optimizer state
    stays in BitOptimizerManager and is intentionally absent from ``Optimizer.state``.
    """

    def __init__(
        self,
        *,
        main_optimizer: Optional[torch.optim.Optimizer],
        bit_manager: BitOptimizerManager,
        step_callback,
    ) -> None:
        groups = []
        if main_optimizer is not None:
            for group in main_optimizer.param_groups:
                cloned = {key: value for key, value in group.items() if key != "params"}
                cloned["params"] = list(group["params"])
                groups.append(cloned)
        for param in bit_manager.module.score_chunks:
            groups.append(
                {
                    "params": [param],
                    "lr": 0.0,
                    "weight_decay": 0.0,
                    "_sparse_bit_score_group": True,
                }
            )
        super().__init__(groups, defaults={})
        self.main_optimizer = main_optimizer
        self.bit_manager = bit_manager
        self._step_callback = step_callback
        self._sparse_bit_composite = True

    @property
    def bit_param_ids(self) -> set[int]:
        return self.bit_manager.module.bit_parameter_ids()

    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("SparseBitCompositeOptimizer does not support closures.")
        loss = None
        if self.main_optimizer is not None:
            loss = self.main_optimizer.step()
        self._step_callback()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.main_optimizer is not None:
            self.main_optimizer.zero_grad(set_to_none=set_to_none)
        for param in self.bit_manager.module.score_chunks:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def state_dict(self):
        shell = super().state_dict()
        shell["_sparse_bit_main_optimizer"] = (
            None if self.main_optimizer is None else self.main_optimizer.state_dict()
        )
        return shell

    def load_state_dict(self, state_dict):
        payload = dict(state_dict)
        main_state = payload.pop("_sparse_bit_main_optimizer", None)
        result = super().load_state_dict(payload)
        if self.main_optimizer is not None and main_state is not None:
            self.main_optimizer.load_state_dict(main_state)
        return result

    def offload_training_state_for_eval(self) -> dict:
        token = {"main": [], "bit": self.bit_manager.offload_state_for_eval()}
        if self.main_optimizer is not None:
            for param_state in self.main_optimizer.state.values():
                if not isinstance(param_state, dict):
                    continue
                for key, value in list(param_state.items()):
                    if torch.is_tensor(value) and value.device.type == "cuda":
                        token["main"].append((param_state, key, value.device))
                        param_state[key] = value.detach().to("cpu")
        return token

    def restore_training_state_after_eval(self, token: dict) -> None:
        for param_state, key, device in token.get("main", []):
            value = param_state.get(key)
            if torch.is_tensor(value) and value.device != device:
                param_state[key] = value.to(device)
        self.bit_manager.restore_state_after_eval(token.get("bit", {}))


class SparseBitNoOpLRScheduler:
    """Trainer lifecycle shim for pure-Bit mode; owns no optimizer and no LR state."""

    def __init__(self) -> None:
        self._step_count = 0
        self._is_sparse_bit_noop_scheduler = True

    def step(self, *args, **kwargs) -> None:
        del args, kwargs
        self._step_count += 1

    def get_last_lr(self):
        return [0.0]

    def state_dict(self) -> dict:
        return {"step_count": int(self._step_count)}

    def load_state_dict(self, state_dict: dict) -> None:
        self._step_count = int(state_dict.get("step_count", 0))


def make_noop_scheduler():
    return SparseBitNoOpLRScheduler()
