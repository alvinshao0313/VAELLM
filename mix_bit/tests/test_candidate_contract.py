from __future__ import annotations

import math

import pytest

from mix_bit.candidate_contract import (
    candidate_mode_from_payload,
    validate_mode_payload,
    validate_module_spec_mode_contract,
)
from mix_bit.schema import CandidateMode


def valid_mode_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "s1",
        "nominal_bit": 0.5,
        "codebook_bits": 4,
        "codebook_dim": 8,
        "residual_stages": 1,
    }
    base.update(overrides)
    return base


def packed_vq_spec(bits: int) -> dict[str, object]:
    return {
        "storage_format": "bitpack_u8",
        "dtype": "uint8",
        "logical_dtype": "bool",
        "pack_bits": 8,
        "logical_shape": [8, 1, bits],
        "shape": [8, 1, (bits + 7) // 8],
    }


def decoder_spec(bits: int, dim: int) -> dict[str, object]:
    return {
        "in_dim": bits,
        "out_dim": dim,
        "hidden_dim": 8,
        "num_res_blocks": 0,
        "norm_type": "layer",
        "activation_type": "swish",
        "decoder_type": "linear",
        "use_checkpoint": False,
        "param_dtype": "float32",
    }


S1_MODE = CandidateMode(
    name="s1",
    nominal_bit=0.5,
    codebook_bits=4,
    codebook_dim=8,
    residual_stages=1,
)

S2_MODE = CandidateMode(
    name="s2",
    nominal_bit=1.0,
    codebook_bits=4,
    codebook_dim=8,
    residual_stages=2,
)


def s1_single_part_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "residual_stages": 1,
        "codebook_dim": 8,
        "stage_codebook_dims": [8],
        "parallel_parts": 1,
        "vq_weights": [packed_vq_spec(4)],
        "decoders": [decoder_spec(4, 8)],
        "stage_vq_weights": None,
        "stage_decoders": None,
    }
    base.update(overrides)
    return base


def s2_single_part_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "residual_stages": 2,
        "codebook_dim": 8,
        "stage_codebook_dims": [8, 8],
        "parallel_parts": 1,
        "stage_vq_weights": [packed_vq_spec(4), packed_vq_spec(4)],
        "stage_decoders": [decoder_spec(4, 8), decoder_spec(4, 8)],
        "vq_weights": [packed_vq_spec(4)],
        "decoders": [decoder_spec(4, 8)],
    }
    base.update(overrides)
    return base


def s2_parallel_parts_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "residual_stages": 2,
        "codebook_dim": 8,
        "stage_codebook_dims": [8, 8],
        "parallel_parts": 2,
        "stage_vq_weights": [
            [packed_vq_spec(4), packed_vq_spec(4)],
            [packed_vq_spec(4), packed_vq_spec(4)],
        ],
        "stage_decoders": [
            [decoder_spec(4, 8), decoder_spec(4, 8)],
            [decoder_spec(4, 8), decoder_spec(4, 8)],
        ],
        "vq_weights": [packed_vq_spec(4), packed_vq_spec(4)],
        "decoders": [decoder_spec(4, 8), decoder_spec(4, 8)],
    }
    base.update(overrides)
    return base


def test_mode_payload_requires_all_five_fields() -> None:
    for missing in (
        "name",
        "nominal_bit",
        "codebook_bits",
        "codebook_dim",
        "residual_stages",
    ):
        payload = valid_mode_payload()
        del payload[missing]
        with pytest.raises(ValueError, match=missing):
            candidate_mode_from_payload(payload, label="mode")


def test_mode_payload_rejects_non_finite_nominal_bit() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="nominal_bit"):
            candidate_mode_from_payload(
                valid_mode_payload(nominal_bit=bad),
                label="mode",
            )


def test_mode_payload_rejects_nominal_bit_inconsistent_with_structure() -> None:
    with pytest.raises(ValueError, match="nominal_bit"):
        candidate_mode_from_payload(
            valid_mode_payload(nominal_bit=0.75),
            label="mode",
        )


def test_validate_mode_payload_rejects_same_name_wrong_nominal_bit() -> None:
    payload = valid_mode_payload(
        name="s1",
        nominal_bit=1.0,
        codebook_bits=8,
        codebook_dim=8,
        residual_stages=1,
    )
    with pytest.raises(ValueError, match="nominal_bit"):
        validate_mode_payload(payload, S1_MODE, label="completed")


def test_validate_mode_payload_rejects_same_name_wrong_codebook_bits() -> None:
    payload = valid_mode_payload(
        name="s1",
        nominal_bit=0.5,
        codebook_bits=8,
        codebook_dim=16,
        residual_stages=1,
    )
    with pytest.raises(ValueError, match="codebook_bits"):
        validate_mode_payload(payload, S1_MODE, label="completed")


def test_validate_mode_payload_rejects_same_name_wrong_codebook_dim() -> None:
    payload = valid_mode_payload(
        name="s1",
        nominal_bit=0.5,
        codebook_bits=4,
        codebook_dim=16,
        residual_stages=2,
    )
    with pytest.raises(ValueError, match="codebook_dim"):
        validate_mode_payload(payload, S1_MODE, label="completed")


def test_validate_mode_payload_rejects_same_name_wrong_residual_stages() -> None:
    payload = valid_mode_payload(
        name="s1",
        nominal_bit=0.5,
        codebook_bits=4,
        codebook_dim=8,
        residual_stages=2,
    )
    with pytest.raises(ValueError, match="residual_stages"):
        validate_mode_payload(payload, S1_MODE, label="completed")


def test_s2_single_part_contract_accepts_exact_structure() -> None:
    validate_module_spec_mode_contract(
        s2_single_part_spec(),
        S2_MODE,
        label="module[0]",
    )


def test_s2_parallel_parts_contract_accepts_exact_structure() -> None:
    validate_module_spec_mode_contract(
        s2_parallel_parts_spec(),
        S2_MODE,
        label="module[0]",
    )


def test_contract_rejects_wrong_residual_stages() -> None:
    with pytest.raises(ValueError, match="residual_stages"):
        validate_module_spec_mode_contract(
            s2_single_part_spec(residual_stages=3, stage_codebook_dims=[8, 8, 8]),
            S2_MODE,
            label="module[0]",
        )


def test_contract_rejects_wrong_codebook_dim() -> None:
    with pytest.raises(ValueError, match="codebook_dim"):
        validate_module_spec_mode_contract(
            s2_single_part_spec(codebook_dim=16, stage_codebook_dims=[16, 16]),
            S2_MODE,
            label="module[0]",
        )


def test_contract_rejects_short_stage_codebook_dims() -> None:
    with pytest.raises(ValueError, match="stage_codebook_dims"):
        validate_module_spec_mode_contract(
            s2_single_part_spec(stage_codebook_dims=[8]),
            S2_MODE,
            label="module[0]",
        )


def test_contract_rejects_wrong_stage_codebook_dim() -> None:
    with pytest.raises(ValueError, match="stage_codebook_dims"):
        validate_module_spec_mode_contract(
            s2_single_part_spec(stage_codebook_dims=[8, 16]),
            S2_MODE,
            label="module[0]",
        )


def test_contract_rejects_wrong_stage_count() -> None:
    with pytest.raises(ValueError, match="stage_vq_weights"):
        validate_module_spec_mode_contract(
            s2_single_part_spec(
                stage_vq_weights=[packed_vq_spec(4)],
                stage_decoders=[decoder_spec(4, 8)],
            ),
            S2_MODE,
            label="module[0]",
        )


def test_contract_rejects_wrong_parallel_part_count() -> None:
    with pytest.raises(ValueError, match="stage_vq_weights"):
        validate_module_spec_mode_contract(
            s2_parallel_parts_spec(
                stage_vq_weights=[
                    [packed_vq_spec(4)],
                    [packed_vq_spec(4), packed_vq_spec(4)],
                ],
            ),
            S2_MODE,
            label="module[0]",
        )


def test_contract_rejects_vq_logical_bits_mismatch() -> None:
    with pytest.raises(ValueError, match="logical_shape"):
        validate_module_spec_mode_contract(
            s2_single_part_spec(
                stage_vq_weights=[packed_vq_spec(8), packed_vq_spec(4)],
            ),
            S2_MODE,
            label="module[0]",
        )


def test_contract_rejects_non_bitpacked_storage() -> None:
    bad_vq = dict(packed_vq_spec(4))
    bad_vq["storage_format"] = "raw_bool"
    with pytest.raises(ValueError, match="storage_format"):
        validate_module_spec_mode_contract(
            s2_single_part_spec(stage_vq_weights=[bad_vq, packed_vq_spec(4)]),
            S2_MODE,
            label="module[0]",
        )


def test_contract_rejects_decoder_in_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="in_dim"):
        validate_module_spec_mode_contract(
            s2_single_part_spec(
                stage_decoders=[decoder_spec(8, 8), decoder_spec(4, 8)],
            ),
            S2_MODE,
            label="module[0]",
        )


def test_contract_rejects_decoder_out_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="out_dim"):
        validate_module_spec_mode_contract(
            s2_single_part_spec(
                stage_decoders=[decoder_spec(4, 16), decoder_spec(4, 8)],
            ),
            S2_MODE,
            label="module[0]",
        )


def test_s1_legacy_fields_are_validated_without_stage_fields() -> None:
    validate_module_spec_mode_contract(
        s1_single_part_spec(),
        S1_MODE,
        label="module[0]",
    )
