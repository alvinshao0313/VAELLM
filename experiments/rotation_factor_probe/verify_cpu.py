"""CPU-only mathematical and real-axis checks; no model/checkpoint is changed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.rotation_factor_probe.factors import (
    PALEY_REFERENCE, SLOANE_188_SHA256, SLOANE_188_URL, load_factors,
    probe_factor, validate_factor,
)
from rotation.weight_preprocess import _transform_last


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sylvester(n: int, dtype: torch.dtype) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    if h.shape != (n, n):
        raise ValueError("Sylvester reference requires a power of two.")
    return h


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = actual.double() - expected.double()
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / expected.double().norm()),
    }


def check_error(stats: dict, dtype: torch.dtype) -> None:
    # These bound roundoff in normalized orthogonal transforms, not quantization loss.
    atol, relative = (2e-12, 2e-13) if dtype == torch.float64 else (2e-5, 2e-6)
    if stats["max_abs"] > atol or stats["relative_l2"] > relative:
        raise ValueError(f"Transform error exceeds {dtype} tolerances: {stats}")


def axis_check(n: int, factors: dict, dtype: torch.dtype) -> dict:
    h, k = probe_factor(n, factors)
    p = n // k
    generator = torch.Generator(device="cpu").manual_seed(31 + n)
    x = torch.randn((4, n), generator=generator, dtype=dtype)
    actual = _transform_last(x, block_size=n, factor=k, had=h, inverse=False)
    # Explicit small matrices implement H_K X H_p^T / sqrt(n), without an n*n allocation.
    hp = sylvester(p, dtype)
    reference = ((h.to(dtype) @ x.reshape(4, k, p)) @ hp.T / math.sqrt(n)).reshape_as(x)
    restored = _transform_last(actual, block_size=n, factor=k, had=h, inverse=True)
    reference_error, roundtrip_error = errors(actual, reference), errors(restored, x)
    check_error(reference_error, dtype)
    check_error(roundtrip_error, dtype)
    return {
        "dimension": n, "factor": k, "sylvester": p, "rows": 4, "dtype": str(dtype),
        "against_explicit_kronecker_action": reference_error, "roundtrip": roundtrip_error,
        "relative_norm_drift": abs(float(actual.double().norm() / x.double().norm()) - 1),
    }


def two_sided_check(factors: dict) -> dict:
    # Qwen/Llama k/v shape after the original 64 input channels are protected.
    rows, cols = 1024, 4032
    generator = torch.Generator(device="cpu").manual_seed(31)
    x = torch.randn((rows, cols), generator=generator, dtype=torch.float32)
    sout = torch.randint(0, 2, (rows,), generator=generator, dtype=torch.int8) * 2 - 1
    sin = torch.randint(0, 2, (cols,), generator=generator, dtype=torch.int8) * 2 - 1
    hi, ki = probe_factor(cols, factors)
    ho, ko = probe_factor(rows, factors)
    z = _transform_last(x * sin, block_size=cols, factor=ki, had=hi, inverse=False)
    z = _transform_last((z * sout[:, None]).T, block_size=rows, factor=ko, had=ho, inverse=False).T
    restored = _transform_last(z, block_size=cols, factor=ki, had=hi, inverse=True) * sin
    restored = _transform_last(restored.T, block_size=rows, factor=ko, had=ho, inverse=True).T * sout[:, None]
    error = errors(restored, x)
    check_error(error, x.dtype)
    return {"shape": [rows, cols], "dtype": str(x.dtype), "seed": 31,
            "input_factor": ki, "output_factor": ko, "roundtrip": error,
            "relative_norm_drift": abs(float(z.double().norm() / x.double().norm()) - 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "cpu_validation.json").exists():
        raise FileExistsError("CPU validation output already exists.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=-1 for this CPU-only check.")
    torch.set_num_threads(4)
    started = time.time()
    paths = [Path(__file__).resolve(), Path(__file__).with_name("factors.py").resolve(),
             Path("rotation/weight_preprocess.py").resolve(), Path("rotation/hadamard_utils.py").resolve()]
    source_before = {str(p): sha(p) for p in paths}
    factors = load_factors(output / "had.188.tur.txt")
    report = {
        "scope": "isolated_cpu_factor_and_transform_validation_only",
        "started_unix": started, "argv": sys.argv, "python": sys.executable,
        "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(),
        "source_sha256": source_before,
        "external_data": {"url": SLOANE_188_URL, "sha256": SLOANE_188_SHA256,
                          "path": str(output / "had.188.tur.txt"), "bytes": 35532},
        "paley_reference": PALEY_REFERENCE,
        "tolerances": {"float64": {"max_abs": 2e-12, "relative_l2": 2e-13},
                       "float32": {"max_abs": 2e-5, "relative_l2": 2e-6}},
        "factors": {}, "axes": [],
        "boundaries": ["No production dispatch or metadata changed", "No GPU or VAE/model training",
                       "Random inputs, not model weights; no downstream quality claim"],
    }
    for k, h in factors.items():
        path = output / f"h{k}.npy"
        with path.open("xb") as f:
            np.save(f, h.numpy(), allow_pickle=False)
        report["factors"][str(k)] = {**validate_factor(h), "artifact": str(path), "file_sha256": sha(path)}
        print(json.dumps(report["factors"][str(k)]), flush=True)
    from rotation.hadamard_utils import get_hadK
    rejected = []
    for n in (4032, 12032, 14080):
        try:
            get_hadK(n)
        except AssertionError:
            rejected.append(n)
        else:
            raise ValueError(f"Native dispatch unexpectedly accepts {n}; review this new condition.")
        for dtype in (torch.float64, torch.float32):
            item = axis_check(n, factors, dtype)
            report["axes"].append(item)
            print(json.dumps(item), flush=True)
    report["native_rejected_dimensions"] = rejected
    report["two_sided"] = two_sided_check(factors)
    if source_before != {str(p): sha(p) for p in paths}:
        raise ValueError("A source changed during CPU validation.")
    report["source_sha_unchanged"] = True
    report["finished_unix"] = time.time()
    report["elapsed_seconds"] = report["finished_unix"] - started
    report["status"] = "verified"
    (output / "cpu_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "two_sided": report["two_sided"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
