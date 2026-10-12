"""Isolated Hadamard factors; this module does not change production dispatch."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch


SLOANE_188_URL = "https://neilsloane.com/hadamard/had.188.tur.txt"
SLOANE_188_SHA256 = "cfa9e9b8f4a61d8275e7bfa8d25dbbfd21cebdb218b5c715c49fc40270cdd107"
PALEY_REFERENCE = (
    "https://doc.sagemath.org/html/en/reference/combinat/"
    "sage/combinat/matrices/hadamard_matrix.html"
)


def _character_matrix(prime: int) -> torch.Tensor:
    if prime < 3 or any(prime % d == 0 for d in range(2, math.isqrt(prime) + 1)):
        raise ValueError("This prototype only constructs prime-field Paley matrices.")
    residues = {x * x % prime for x in range(1, prime)}
    character = torch.tensor(
        [0] + [1 if x in residues else -1 for x in range(1, prime)], dtype=torch.int64
    )
    indices = torch.arange(prime)
    return character[(indices[None, :] - indices[:, None]) % prime]


def paley_252() -> torch.Tensor:
    """Paley I over F_251: [1, 1^T; 1, Q-I], Q_ij=chi(j-i)."""
    q = 251
    h = torch.ones((q + 1, q + 1), dtype=torch.int64)
    h[1:, 1:] = _character_matrix(q) - torch.eye(q, dtype=torch.int64)
    return h.to(torch.int8)


def paley_220() -> torch.Tensor:
    """Paley II: symmetric conference C over F_109, then double its order."""
    q = 109
    c = torch.ones((q + 1, q + 1), dtype=torch.int64)
    c[0, 0] = 0
    c[1:, 1:] = _character_matrix(q)
    eye = torch.eye(q + 1, dtype=torch.int64)
    # C^2=qI; the two block products commute, giving HH^T=2(q+1)I.
    upper = torch.cat((c + eye, c - eye), dim=1)
    lower = torch.cat((c - eye, -c - eye), dim=1)
    return torch.cat((upper, lower), dim=0).to(torch.int8)


def sloane_188(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SLOANE_188_SHA256:
        raise ValueError("The Sloane 188 source does not match the pinned original.")
    rows = raw.decode("ascii").splitlines()
    if len(rows) != 188 or any(len(row) != 188 or set(row) - {"+", "-"} for row in rows):
        raise ValueError("Expected exactly 188 rows of 188 +/- characters.")
    return torch.tensor([[1 if x == "+" else -1 for x in row] for row in rows], dtype=torch.int8)


def validate_factor(h: torch.Tensor) -> dict:
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not bool(((h == 1) | (h == -1)).all()):
        raise ValueError("Expected a square +/-1 matrix.")
    n = h.shape[0]
    integer = h.to(dtype=torch.int64, device="cpu")
    error = integer @ integer.T - n * torch.eye(n, dtype=torch.int64)
    if bool(error.any()):
        raise ValueError(f"Order {n} failed exact integer orthogonality.")
    return {
        "order": n,
        "shape": list(h.shape),
        "entries": [-1, 1],
        "integer_gram_max_abs_error": int(error.abs().max()),
        "symmetric": bool(torch.equal(integer, integer.T)),
        "row_major_int8_sha256": hashlib.sha256(integer.to(torch.int8).numpy().tobytes()).hexdigest(),
    }


def load_factors(sloane_source: Path) -> dict[int, torch.Tensor]:
    result = {188: sloane_188(sloane_source), 220: paley_220(), 252: paley_252()}
    for h in result.values():
        validate_factor(h)
    return result


def probe_factor(n: int, factors: dict[int, torch.Tensor]) -> tuple[torch.Tensor | None, int]:
    """Choose only exact K*2^m factorizations, otherwise use unchanged native code."""
    for k in (252, 220, 188):
        quotient, remainder = divmod(n, k)
        if remainder == 0 and quotient > 0 and quotient & (quotient - 1) == 0:
            return factors[k], k
    from rotation.hadamard_utils import get_hadK

    return get_hadK(n)
