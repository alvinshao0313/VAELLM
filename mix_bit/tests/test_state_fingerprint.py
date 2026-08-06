from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mix_bit.state_fingerprint import (
    STATE_FINGERPRINT_CHUNK_BYTES,
    STATE_FINGERPRINT_KIND,
    compare_state_fingerprints,
    fingerprint_model_state,
    fingerprint_tensor,
    write_state_fingerprint_manifest,
)


def _entry(tensor: torch.Tensor) -> dict:
    return fingerprint_tensor(tensor)


def test_fingerprint_float32_tensor_is_stable_and_matches_dtype_shape_numel():
    torch.manual_seed(0)
    t = torch.randn(3, 5, dtype=torch.float32)
    entry = fingerprint_tensor(t)
    assert entry["dtype"] == "float32"
    assert entry["shape"] == [3, 5]
    assert entry["numel"] == 15
    assert isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64
    # Same input -> same hash.
    assert fingerprint_tensor(t.clone())["sha256"] == entry["sha256"]


def test_fingerprint_bfloat16_tensor_matches_uint8_byte_view():
    t = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    entry = fingerprint_tensor(t)
    assert entry["dtype"] == "bfloat16"
    assert entry["shape"] == [2, 4]
    assert entry["numel"] == 8
    # Independent byte-level hash: header + raw bytes via uint8 view.
    import hashlib

    header = {"dtype": "bfloat16", "shape": [2, 4], "numel": 8}
    header_bytes = json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = hashlib.sha256()
    expected.update(header_bytes)
    expected.update(t.view(torch.uint8).numpy().tobytes(order="C"))
    assert entry["sha256"] == expected.hexdigest()


def test_fingerprint_uint8_and_bool_and_int64_tensors():
    for dtype in (torch.uint8, torch.bool, torch.int64):
        t = torch.zeros(4, dtype=dtype)
        entry = fingerprint_tensor(t)
        assert entry["dtype"] == str(dtype).replace("torch.", "")
        assert entry["numel"] == 4
        assert len(entry["sha256"]) == 64


def test_fingerprint_zero_length_tensor_has_valid_hash():
    t = torch.zeros(0, dtype=torch.float32)
    entry = fingerprint_tensor(t)
    assert entry["numel"] == 0
    assert entry["shape"] == [0]
    assert len(entry["sha256"]) == 64
    # Zero-length tensor is stable.
    assert fingerprint_tensor(torch.zeros(0, dtype=torch.float32))["sha256"] == entry["sha256"]


def test_fingerprint_rejects_non_contiguous_tensor():
    t = torch.randn(4, 4, dtype=torch.float32)
    non_contig = t.t()  # transpose -> non-contiguous
    assert not non_contig.is_contiguous()
    with pytest.raises(ValueError, match="non-contiguous"):
        fingerprint_tensor(non_contig)


def test_fingerprint_rejects_sparse_layout_tensor():
    t = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([1.0, 2.0]),
        size=(2, 2),
    )
    with pytest.raises(ValueError, match="layout"):
        fingerprint_tensor(t)


def test_fingerprint_uses_bounded_chunk_bytes():
    # chunk_bytes smaller than the tensor's byte footprint should still hash correctly.
    t = torch.arange(1024, dtype=torch.float32)
    entry_small = fingerprint_tensor(t, chunk_bytes=64)
    entry_full = fingerprint_tensor(t, chunk_bytes=STATE_FINGERPRINT_CHUNK_BYTES)
    assert entry_small["sha256"] == entry_full["sha256"]


def test_mutation_detection_single_value_change():
    torch.manual_seed(1)
    t = torch.randn(8, dtype=torch.float32)
    base = fingerprint_tensor(t)
    mutated = t.clone()
    mutated[3] = mutated[3] + 1.0
    entry = fingerprint_tensor(mutated)
    with pytest.raises(ValueError, match="sha256"):
        compare_state_fingerprints(
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 1, "entries": {"x": base}},
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 1, "entries": {"x": entry}},
        )


def test_mutation_detection_dtype_change():
    t = torch.zeros(4, dtype=torch.float32)
    base = fingerprint_tensor(t)
    other = fingerprint_tensor(t.to(torch.bfloat16))
    with pytest.raises(ValueError, match="dtype"):
        compare_state_fingerprints(
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 1, "entries": {"x": base}},
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 1, "entries": {"x": other}},
        )


def test_mutation_detection_shape_change():
    t = torch.zeros(4, dtype=torch.float32)
    base = fingerprint_tensor(t)
    other = fingerprint_tensor(t.view(2, 2))
    with pytest.raises(ValueError, match="shape"):
        compare_state_fingerprints(
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 1, "entries": {"x": base}},
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 1, "entries": {"x": other}},
        )


def test_mutation_detection_key_missing_and_extra_key():
    base = fingerprint_tensor(torch.zeros(2))
    other = fingerprint_tensor(torch.zeros(2))
    with pytest.raises(ValueError, match="key set mismatch"):
        compare_state_fingerprints(
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 2, "entries": {"a": base, "b": other}},
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 1, "entries": {"a": base}},
        )
    with pytest.raises(ValueError, match="key set mismatch"):
        compare_state_fingerprints(
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 1, "entries": {"a": base}},
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 2, "entries": {"a": base, "b": other}},
        )


def test_mutation_detection_kind_mismatch():
    base = fingerprint_tensor(torch.zeros(2))
    with pytest.raises(ValueError, match="kind"):
        compare_state_fingerprints(
            {"kind": "other", "key_count": 1, "entries": {"a": base}},
            {"kind": STATE_FINGERPRINT_KIND, "key_count": 1, "entries": {"a": base}},
        )


def test_fingerprint_model_state_round_trip_matches_identical_model():
    torch.manual_seed(2)
    a = nn.Linear(4, 4, bias=False)
    b = nn.Linear(4, 4, bias=False)
    b.load_state_dict(a.state_dict())
    fa = fingerprint_model_state(a)
    fb = fingerprint_model_state(b)
    compare_state_fingerprints(fa, fb)  # must not raise
    assert fa["key_count"] == len(fa["entries"])
    assert fa["kind"] == STATE_FINGERPRINT_KIND


def test_fingerprint_model_state_detects_mutation_in_one_tensor():
    a = nn.Linear(4, 4, bias=False)
    b = nn.Linear(4, 4, bias=False)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        b.weight[0, 0] += 1.0
    fa = fingerprint_model_state(a)
    fb = fingerprint_model_state(b)
    with pytest.raises(ValueError, match="sha256"):
        compare_state_fingerprints(fa, fb)


def test_fingerprint_model_state_does_not_clone_tensors(monkeypatch: pytest.MonkeyPatch):
    """fingerprint_model_state must succeed even if torch.Tensor.clone raises.

    This guards against any hidden full-tensor clone inside the streaming path.
    """
    a = nn.Linear(8, 8, bias=False)

    original_clone = torch.Tensor.clone

    def _exploding_clone(self, *args, **kwargs):
        raise RuntimeError("clone must not be called inside fingerprint_model_state")

    monkeypatch.setattr(torch.Tensor, "clone", _exploding_clone)
    try:
        fingerprint_model_state(a)
    finally:
        monkeypatch.setattr(torch.Tensor, "clone", original_clone)


def test_write_state_fingerprint_manifest_is_atomic_and_canonical(tmp_path: Path):
    a = nn.Linear(4, 4, bias=False)
    payload = fingerprint_model_state(a)
    out_path = tmp_path / "subdir" / "state_fingerprint.json"
    abs_path = write_state_fingerprint_manifest(out_path, payload)
    assert Path(abs_path).resolve() == out_path.resolve()
    assert out_path.is_file()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["kind"] == STATE_FINGERPRINT_KIND
    assert loaded["key_count"] == len(loaded["entries"])
    # No temp file left behind.
    assert not (out_path.parent / "state_fingerprint.json.tmp").exists()


def test_compare_state_fingerprints_key_count_mismatch_rejected():
    base = fingerprint_tensor(torch.zeros(2))
    payload = {
        "kind": STATE_FINGERPRINT_KIND,
        "key_count": 5,
        "entries": {"a": base},
    }
    actual = {
        "kind": STATE_FINGERPRINT_KIND,
        "key_count": 1,
        "entries": {"a": base},
    }
    with pytest.raises(ValueError, match="key_count"):
        compare_state_fingerprints(payload, actual)
