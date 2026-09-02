from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd
from typing import List, Tuple

_MASK64 = 0xFFFFFFFFFFFFFFFF
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_PRIMARY_OFFSET_DOMAIN = 0x243F6A8885A308D3
_PRIMARY_STRIDE_DOMAIN = 0x13198A2E03707344
_SECONDARY_OFFSET_DOMAIN = 0xA4093822299F31D0
_SECONDARY_STRIDE_DOMAIN = 0x082EFA98EC4E6C89


def fnv1a64(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h ^= int(byte)
        h = (h * _FNV_PRIME) & _MASK64
    return h


def splitmix64(value: int) -> int:
    z = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def bank_coverage_seed(training_seed: int, canonical_key: str, coverage_id: int) -> int:
    payload = f"seed={int(training_seed)}|key={str(canonical_key)}|coverage={int(coverage_id)}".encode("utf-8")
    return fnv1a64(payload)


def _resolve_coprime_stride(raw: int, modulus: int) -> int:
    n = int(modulus)
    if n < 1:
        raise ValueError(f"modulus must be >=1, got {n}.")
    if n == 1:
        return 0
    candidate = int(raw) % n
    if candidate == 0:
        candidate = 1
    start = candidate
    while gcd(candidate, n) != 1:
        candidate += 1
        if candidate >= n:
            candidate = 1
        if candidate == start:
            raise RuntimeError(f"failed to find coprime stride for modulus={n}.")
    return candidate


def _mod_inverse(value: int, modulus: int) -> int:
    if int(modulus) == 1:
        return 0
    return pow(int(value), -1, int(modulus))


def _primary_params(training_seed: int, canonical_key: str, coverage_id: int, n_bits: int) -> Tuple[int, int, int]:
    n = int(n_bits)
    base = bank_coverage_seed(training_seed, canonical_key, coverage_id)
    if n == 1:
        return 0, 0, 0
    stride = _resolve_coprime_stride(splitmix64(base ^ _PRIMARY_STRIDE_DOMAIN), n)
    offset = splitmix64(base ^ _PRIMARY_OFFSET_DOMAIN) % n
    return stride, int(offset), _mod_inverse(stride, n)


def _secondary_params(training_seed: int, canonical_key: str, coverage_id: int, cursor: int) -> Tuple[int, int, int]:
    prefix = int(cursor)
    if prefix < 1:
        raise ValueError(f"tail filler requires cursor>=1, got {prefix}.")
    base = bank_coverage_seed(training_seed, canonical_key, coverage_id)
    if prefix == 1:
        return 0, 0, 0
    stride = _resolve_coprime_stride(splitmix64(base ^ _SECONDARY_STRIDE_DOMAIN), prefix)
    offset = splitmix64(base ^ _SECONDARY_OFFSET_DOMAIN) % prefix
    return stride, int(offset), _mod_inverse(stride, prefix)


@dataclass(frozen=True)
class ActiveSubsetMeta:
    remaining: int
    tail: bool
    secondary_stride: int = 0
    secondary_offset: int = 0
    secondary_inverse: int = 0


@dataclass(frozen=True)
class AffineSamplerState:
    canonical_key: str
    training_seed: int
    n_bits: int
    n_active: int
    coverage_id: int = 0
    cursor: int = 0
    stride: int = 0
    offset: int = 0
    inverse: int = 0

    @classmethod
    def create(
        cls,
        *,
        canonical_key: str,
        training_seed: int,
        n_bits: int,
        n_active: int,
        coverage_id: int = 0,
        cursor: int = 0,
    ) -> "AffineSamplerState":
        n = int(n_bits)
        active = int(n_active)
        cov = int(coverage_id)
        cur = int(cursor)
        if n < 1:
            raise ValueError(f"n_bits must be >=1, got {n}.")
        if active < 1 or active > n:
            raise ValueError(f"n_active must satisfy 1<=n_active<=n_bits, got {active}/{n}.")
        if cur < 0 or cur >= n:
            raise ValueError(f"cursor must satisfy 0<=cursor<n_bits, got {cur}/{n}.")
        stride, offset, inverse = _primary_params(training_seed, canonical_key, cov, n)
        return cls(
            canonical_key=str(canonical_key),
            training_seed=int(training_seed),
            n_bits=n,
            n_active=active,
            coverage_id=cov,
            cursor=cur,
            stride=stride,
            offset=offset,
            inverse=inverse,
        )

    def subset_meta(self) -> ActiveSubsetMeta:
        remaining = int(self.n_bits - self.cursor)
        tail = remaining < int(self.n_active)
        if not tail:
            return ActiveSubsetMeta(remaining=remaining, tail=False)
        stride2, offset2, inv2 = _secondary_params(
            self.training_seed,
            self.canonical_key,
            self.coverage_id,
            self.cursor,
        )
        return ActiveSubsetMeta(
            remaining=remaining,
            tail=True,
            secondary_stride=stride2,
            secondary_offset=offset2,
            secondary_inverse=inv2,
        )

    def logical_index(self, active_ordinal: int) -> int:
        q = int(active_ordinal)
        if q < 0 or q >= int(self.n_active):
            raise IndexError(f"active ordinal {q} out of range [0,{self.n_active}).")
        meta = self.subset_meta()
        if (not meta.tail) or q < meta.remaining:
            position = int(self.cursor + q)
        else:
            filler = q - int(meta.remaining)
            prefix_position = (
                int(meta.secondary_stride) * filler + int(meta.secondary_offset)
            ) % int(self.cursor)
            position = prefix_position
        if int(self.n_bits) == 1:
            return 0
        return (int(self.stride) * position + int(self.offset)) % int(self.n_bits)

    def active_indices(self) -> List[int]:
        return [self.logical_index(q) for q in range(int(self.n_active))]

    def advance(self) -> "AffineSamplerState":
        meta = self.subset_meta()
        if meta.tail:
            return AffineSamplerState.create(
                canonical_key=self.canonical_key,
                training_seed=self.training_seed,
                n_bits=self.n_bits,
                n_active=self.n_active,
                coverage_id=self.coverage_id + 1,
                cursor=0,
            )
        next_cursor = int(self.cursor + self.n_active)
        if next_cursor > int(self.n_bits):
            raise RuntimeError(
                f"sampler cursor overflow for {self.canonical_key}: {next_cursor}>{self.n_bits}."
            )
        if next_cursor == int(self.n_bits):
            return AffineSamplerState.create(
                canonical_key=self.canonical_key,
                training_seed=self.training_seed,
                n_bits=self.n_bits,
                n_active=self.n_active,
                coverage_id=self.coverage_id + 1,
                cursor=0,
            )
        return replace(self, cursor=next_cursor)

    def to_metadata(self) -> dict:
        return {
            "canonical_key": self.canonical_key,
            "training_seed": int(self.training_seed),
            "N_bits": int(self.n_bits),
            "N_active": int(self.n_active),
            "coverage_id": int(self.coverage_id),
            "cursor": int(self.cursor),
            "stride": int(self.stride),
            "offset": int(self.offset),
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "AffineSamplerState":
        state = cls.create(
            canonical_key=str(metadata["canonical_key"]),
            training_seed=int(metadata["training_seed"]),
            n_bits=int(metadata["N_bits"]),
            n_active=int(metadata["N_active"]),
            coverage_id=int(metadata["coverage_id"]),
            cursor=int(metadata["cursor"]),
        )
        if int(metadata.get("stride", state.stride)) != int(state.stride):
            raise ValueError(f"sampler stride mismatch for {state.canonical_key}.")
        if int(metadata.get("offset", state.offset)) != int(state.offset):
            raise ValueError(f"sampler offset mismatch for {state.canonical_key}.")
        return state
