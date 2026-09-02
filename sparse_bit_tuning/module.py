from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn


_DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class BankSpec:
    canonical_key: str
    module_path: str
    stage_idx: int
    part_idx: int
    logical_shape: Tuple[int, int, int]
    n_bits: int
    n_active: int
    device: torch.device
    chunk_id: int = -1
    score_start: int = -1
    score_end: int = -1

    @property
    def block_count(self) -> int:
        return int(self.logical_shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.logical_shape[2])

    def validate(self) -> None:
        if not self.canonical_key:
            raise ValueError("BankSpec canonical_key cannot be empty.")
        if not self.module_path:
            raise ValueError("BankSpec module_path cannot be empty.")
        if len(self.logical_shape) != 3 or int(self.logical_shape[1]) != 1:
            raise ValueError(
                f"{self.canonical_key}: logical_shape must be [B,1,IN], got {self.logical_shape}."
            )
        expected = int(self.logical_shape[0]) * int(self.logical_shape[2])
        if int(self.n_bits) != expected:
            raise ValueError(f"{self.canonical_key}: n_bits={self.n_bits} != B*IN={expected}.")
        if int(self.n_active) < 1 or int(self.n_active) > int(self.n_bits):
            raise ValueError(
                f"{self.canonical_key}: n_active must satisfy 1<=n_active<=n_bits, "
                f"got {self.n_active}/{self.n_bits}."
            )


@dataclass(frozen=True)
class ScoreChunkSpec:
    chunk_id: int
    device: torch.device
    numel: int
    bank_keys: Tuple[str, ...]


def _module_groups(specs: Sequence[BankSpec]) -> List[List[BankSpec]]:
    ordered = sorted(
        specs,
        key=lambda s: (str(torch.device(s.device)), str(s.module_path), int(s.stage_idx), int(s.part_idx)),
    )
    groups: List[List[BankSpec]] = []
    current: List[BankSpec] = []
    current_key = None
    for spec in ordered:
        spec.validate()
        key = (str(torch.device(spec.device)), str(spec.module_path))
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(spec)
        current_key = key
    if current:
        groups.append(current)
    return groups


def layout_bank_specs(
    specs: Sequence[BankSpec],
    *,
    target_chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> Tuple[List[BankSpec], List[ScoreChunkSpec]]:
    if not specs:
        return [], []
    target_elems = max(1, int(target_chunk_bytes) // torch.tensor([], dtype=torch.float16).element_size())
    laid_out: List[BankSpec] = []
    chunk_specs: List[ScoreChunkSpec] = []
    chunk_id = -1
    chunk_device: torch.device | None = None
    chunk_numel = 0
    chunk_bank_keys: List[str] = []

    def flush() -> None:
        nonlocal chunk_device, chunk_numel, chunk_bank_keys
        if chunk_device is None:
            return
        chunk_specs.append(
            ScoreChunkSpec(
                chunk_id=int(chunk_id),
                device=torch.device(chunk_device),
                numel=int(chunk_numel),
                bank_keys=tuple(chunk_bank_keys),
            )
        )
        chunk_device = None
        chunk_numel = 0
        chunk_bank_keys = []

    for module_specs in _module_groups(specs):
        device = torch.device(module_specs[0].device)
        module_numel = sum(int(spec.n_active) for spec in module_specs)
        needs_new = (
            chunk_device is None
            or device != chunk_device
            or (chunk_numel > 0 and chunk_numel + module_numel > target_elems)
        )
        if needs_new:
            flush()
            chunk_id += 1
            chunk_device = device
        start = int(chunk_numel)
        for spec in module_specs:
            end = start + int(spec.n_active)
            laid_out.append(
                replace(
                    spec,
                    device=device,
                    chunk_id=int(chunk_id),
                    score_start=int(start),
                    score_end=int(end),
                )
            )
            chunk_bank_keys.append(spec.canonical_key)
            start = end
        chunk_numel += module_numel
    flush()

    by_key = {spec.canonical_key: spec for spec in laid_out}
    if len(by_key) != len(laid_out):
        raise ValueError("duplicate sparse-bit canonical bank key.")
    return laid_out, chunk_specs


class SparseBitTuningModule(nn.Module):
    """Owns fixed-shape FP16 score Parameters; optimizer/sampler state lives elsewhere."""

    def __init__(
        self,
        bank_specs: Sequence[BankSpec],
        *,
        target_chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    ) -> None:
        super().__init__()
        laid_out, chunks = layout_bank_specs(bank_specs, target_chunk_bytes=target_chunk_bytes)
        if not laid_out:
            raise ValueError("SparseBitTuningModule requires at least one bank.")
        self._bank_specs: Tuple[BankSpec, ...] = tuple(laid_out)
        self._bank_by_key: Dict[str, BankSpec] = {spec.canonical_key: spec for spec in laid_out}
        self._chunk_specs: Tuple[ScoreChunkSpec, ...] = tuple(chunks)
        params: List[nn.Parameter] = []
        for chunk in chunks:
            params.append(
                nn.Parameter(
                    torch.empty(
                        (int(chunk.numel),),
                        dtype=torch.float16,
                        device=torch.device(chunk.device),
                    ),
                    requires_grad=True,
                )
            )
        self.score_chunks = nn.ParameterList(params)
        self._initialized = False

    @property
    def bank_specs(self) -> Tuple[BankSpec, ...]:
        return self._bank_specs

    @property
    def chunk_specs(self) -> Tuple[ScoreChunkSpec, ...]:
        return self._chunk_specs

    @property
    def initialized(self) -> bool:
        return bool(self._initialized)

    def mark_initialized(self) -> None:
        self._initialized = True

    def bank_spec(self, canonical_key: str) -> BankSpec:
        try:
            return self._bank_by_key[str(canonical_key)]
        except KeyError as exc:
            raise KeyError(f"unknown sparse-bit bank: {canonical_key}") from exc

    def score_view(self, bank: BankSpec | str) -> torch.Tensor:
        spec = self.bank_spec(bank) if isinstance(bank, str) else bank
        chunk = self.score_chunks[int(spec.chunk_id)]
        return chunk[int(spec.score_start) : int(spec.score_end)]

    def module_score_span(self, module_path: str) -> Tuple[torch.Tensor, Tuple[BankSpec, ...]]:
        banks = tuple(spec for spec in self._bank_specs if spec.module_path == str(module_path))
        if not banks:
            raise KeyError(f"no sparse-bit banks for module {module_path!r}.")
        chunk_ids = {int(spec.chunk_id) for spec in banks}
        if len(chunk_ids) != 1:
            raise RuntimeError(f"module {module_path!r} sparse-bit banks span multiple score chunks.")
        ordered = tuple(sorted(banks, key=lambda s: (int(s.stage_idx), int(s.part_idx))))
        start = min(int(spec.score_start) for spec in ordered)
        end = max(int(spec.score_end) for spec in ordered)
        expected = sum(int(spec.n_active) for spec in ordered)
        if end - start != expected:
            raise RuntimeError(f"module {module_path!r} sparse-bit banks are not contiguous.")
        return self.score_chunks[next(iter(chunk_ids))][start:end], ordered

    def bit_parameters(self) -> Iterable[nn.Parameter]:
        return iter(self.score_chunks)

    def bit_parameter_ids(self) -> set[int]:
        return {id(param) for param in self.score_chunks}

    def clear_grads(self) -> None:
        for param in self.score_chunks:
            param.grad = None
