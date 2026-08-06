from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from litebsq.bitpack import validate_bitpack_u8_spec
from mix_bit.schema import CandidateMode

NOMINAL_BIT_TOLERANCE = 1e-12


def _contract_error(label: str, field: str, actual: object, expected: object) -> ValueError:
    return ValueError(
        f"{label}: {field} mismatch: actual={actual!r} expected={expected!r}"
    )


def candidate_mode_from_payload(payload: Mapping[str, object], *, label: str) -> CandidateMode:
    """Parse exactly the five required mode fields and reject missing/invalid values."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label}: payload must be a mapping, got {type(payload)}")

    required_keys = ("name", "nominal_bit", "codebook_bits", "codebook_dim", "residual_stages")
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"{label}: missing required field {key!r}")

    name = str(payload["name"])
    if not name:
        raise ValueError(f"{label}: name cannot be empty")

    nominal_bit = float(payload["nominal_bit"])
    if not math.isfinite(nominal_bit):
        raise ValueError(f"{label}: nominal_bit must be finite, got {nominal_bit!r}")

    codebook_bits = int(payload["codebook_bits"])
    codebook_dim = int(payload["codebook_dim"])
    residual_stages = int(payload["residual_stages"])

    if codebook_bits < 1:
        raise _contract_error(label, "codebook_bits", codebook_bits, ">= 1")
    if codebook_dim < 1:
        raise _contract_error(label, "codebook_dim", codebook_dim, ">= 1")
    if residual_stages < 1:
        raise _contract_error(label, "residual_stages", residual_stages, ">= 1")

    derived_nominal_bit = residual_stages * codebook_bits / codebook_dim
    if abs(nominal_bit - derived_nominal_bit) > NOMINAL_BIT_TOLERANCE:
        raise _contract_error(label, "nominal_bit", nominal_bit, derived_nominal_bit)

    return CandidateMode(
        name=name,
        nominal_bit=nominal_bit,
        codebook_bits=codebook_bits,
        codebook_dim=codebook_dim,
        residual_stages=residual_stages,
    )


def validate_mode_payload(
    payload: Mapping[str, object],
    expected: CandidateMode,
    *,
    label: str,
) -> None:
    """Require all five mode fields to equal expected; nominal_bit tolerance is 1e-12."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label}: payload must be a mapping, got {type(payload)}")

    required_keys = ("name", "nominal_bit", "codebook_bits", "codebook_dim", "residual_stages")
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"{label}: missing required field {key!r}")

    actual_name = str(payload["name"])
    actual_nominal_bit = float(payload["nominal_bit"])
    actual_codebook_bits = int(payload["codebook_bits"])
    actual_codebook_dim = int(payload["codebook_dim"])
    actual_residual_stages = int(payload["residual_stages"])

    if actual_name != str(expected.name):
        raise _contract_error(label, "name", actual_name, expected.name)
    if abs(actual_nominal_bit - float(expected.nominal_bit)) > NOMINAL_BIT_TOLERANCE:
        raise _contract_error(label, "nominal_bit", actual_nominal_bit, expected.nominal_bit)
    if actual_codebook_bits != int(expected.codebook_bits):
        raise _contract_error(label, "codebook_bits", actual_codebook_bits, expected.codebook_bits)
    if actual_codebook_dim != int(expected.codebook_dim):
        raise _contract_error(label, "codebook_dim", actual_codebook_dim, expected.codebook_dim)
    if actual_residual_stages != int(expected.residual_stages):
        raise _contract_error(label, "residual_stages", actual_residual_stages, expected.residual_stages)


def _validate_vq_spec(
    vq_spec: object,
    mode: CandidateMode,
    *,
    label: str,
    field: str,
) -> None:
    if not isinstance(vq_spec, dict):
        raise TypeError(f"{label}: {field} must be a dict, got {type(vq_spec)}")
    normalized = validate_bitpack_u8_spec(vq_spec, arg_name=f"{label}.{field}")
    logical_shape = normalized.get("logical_shape")
    if not isinstance(logical_shape, (list, tuple)) or len(logical_shape) == 0:
        raise ValueError(f"{label}: {field}.logical_shape must be a non-empty list/tuple")
    if int(normalized["logical_shape"][-1]) != mode.codebook_bits:
        raise _contract_error(
            label,
            f"{field}.logical_shape[-1]",
            int(normalized["logical_shape"][-1]),
            mode.codebook_bits,
        )


def _validate_decoder_spec(
    decoder: object,
    mode: CandidateMode,
    *,
    label: str,
    field: str,
) -> None:
    if not isinstance(decoder, dict):
        raise TypeError(f"{label}: {field} must be a dict, got {type(decoder)}")
    in_dim = int(decoder["in_dim"])
    out_dim = int(decoder["out_dim"])
    if in_dim != mode.codebook_bits:
        raise _contract_error(label, f"{field}.in_dim", in_dim, mode.codebook_bits)
    if out_dim != mode.codebook_dim:
        raise _contract_error(label, f"{field}.out_dim", out_dim, mode.codebook_dim)


def _validate_stage_part_group(
    group: object,
    mode: CandidateMode,
    *,
    label: str,
    field: str,
    parallel_parts: int,
    validate_item: Any,
) -> None:
    if parallel_parts == 1:
        validate_item(group, mode, label=label, field=field)
        return
    if not isinstance(group, (list, tuple)):
        raise TypeError(f"{label}: {field} must be a list/tuple, got {type(group)}")
    if len(group) != parallel_parts:
        raise _contract_error(label, field, len(group), parallel_parts)
    for part_idx, item in enumerate(group):
        validate_item(item, mode, label=label, field=f"{field}[{part_idx}]")


def _validate_stage_collection(
    collection: object,
    mode: CandidateMode,
    *,
    label: str,
    field: str,
    stages: int,
    parallel_parts: int,
    validate_item: Any,
) -> None:
    if collection is None:
        raise ValueError(f"{label}: missing required field {field!r}")
    if not isinstance(collection, (list, tuple)):
        raise TypeError(f"{label}: {field} must be a list/tuple, got {type(collection)}")
    if len(collection) != stages:
        raise _contract_error(label, field, len(collection), stages)
    for stage_idx, stage_group in enumerate(collection):
        _validate_stage_part_group(
            stage_group,
            mode,
            label=label,
            field=f"{field}[{stage_idx}]",
            parallel_parts=parallel_parts,
            validate_item=validate_item,
        )


def _validate_legacy_part_collection(
    collection: object,
    mode: CandidateMode,
    *,
    label: str,
    field: str,
    parallel_parts: int,
    validate_item: Any,
) -> None:
    if collection is None:
        raise ValueError(f"{label}: missing required field {field!r}")
    if not isinstance(collection, (list, tuple)):
        raise TypeError(f"{label}: {field} must be a list/tuple, got {type(collection)}")
    if len(collection) != parallel_parts:
        raise _contract_error(label, field, len(collection), parallel_parts)
    for part_idx, item in enumerate(collection):
        validate_item(item, mode, label=label, field=f"{field}[{part_idx}]")


def _stage_group_matches_legacy(
    stage_group: object,
    legacy_group: object,
    *,
    parallel_parts: int,
) -> bool:
    if parallel_parts == 1:
        return stage_group == legacy_group
    if not isinstance(stage_group, (list, tuple)) or not isinstance(legacy_group, (list, tuple)):
        return False
    if len(stage_group) != len(legacy_group):
        return False
    return all(left == right for left, right in zip(stage_group, legacy_group))


def _validate_single_stage_spec(
    spec: Mapping[str, object],
    mode: CandidateMode,
    *,
    label: str,
    parallel_parts: int,
) -> None:
    _validate_legacy_part_collection(
        spec.get("vq_weights"),
        mode,
        label=label,
        field="vq_weights",
        parallel_parts=parallel_parts,
        validate_item=_validate_vq_spec,
    )
    _validate_legacy_part_collection(
        spec.get("decoders"),
        mode,
        label=label,
        field="decoders",
        parallel_parts=parallel_parts,
        validate_item=_validate_decoder_spec,
    )

    stage_vq_weights = spec.get("stage_vq_weights")
    stage_decoders = spec.get("stage_decoders")
    if stage_vq_weights is not None:
        _validate_stage_collection(
            stage_vq_weights,
            mode,
            label=label,
            field="stage_vq_weights",
            stages=1,
            parallel_parts=parallel_parts,
            validate_item=_validate_vq_spec,
        )
        legacy_vq = spec["vq_weights"]
        if parallel_parts == 1:
            legacy_group = legacy_vq[0]  # type: ignore[index]
        else:
            legacy_group = legacy_vq
        if not _stage_group_matches_legacy(
            stage_vq_weights[0],  # type: ignore[index]
            legacy_group,
            parallel_parts=parallel_parts,
        ):
            raise _contract_error(
                label,
                "stage_vq_weights[0]",
                stage_vq_weights[0],
                legacy_group,
            )
    if stage_decoders is not None:
        _validate_stage_collection(
            stage_decoders,
            mode,
            label=label,
            field="stage_decoders",
            stages=1,
            parallel_parts=parallel_parts,
            validate_item=_validate_decoder_spec,
        )
        legacy_decoders = spec["decoders"]
        if parallel_parts == 1:
            legacy_group = legacy_decoders[0]  # type: ignore[index]
        else:
            legacy_group = legacy_decoders
        if not _stage_group_matches_legacy(
            stage_decoders[0],  # type: ignore[index]
            legacy_group,
            parallel_parts=parallel_parts,
        ):
            raise _contract_error(
                label,
                "stage_decoders[0]",
                stage_decoders[0],
                legacy_group,
            )


def _validate_multi_stage_spec(
    spec: Mapping[str, object],
    mode: CandidateMode,
    *,
    label: str,
    parallel_parts: int,
    stages: int,
) -> None:
    _validate_stage_collection(
        spec.get("stage_vq_weights"),
        mode,
        label=label,
        field="stage_vq_weights",
        stages=stages,
        parallel_parts=parallel_parts,
        validate_item=_validate_vq_spec,
    )
    _validate_stage_collection(
        spec.get("stage_decoders"),
        mode,
        label=label,
        field="stage_decoders",
        stages=stages,
        parallel_parts=parallel_parts,
        validate_item=_validate_decoder_spec,
    )


def validate_module_spec_mode_contract(
    spec: Mapping[str, object],
    mode: CandidateMode,
    *,
    label: str,
) -> None:
    """Require actual VAE structure, VQ storage and decoder dimensions to match mode."""
    if not isinstance(spec, Mapping):
        raise TypeError(f"{label}: spec must be a mapping, got {type(spec)}")

    parallel_parts = int(spec.get("parallel_parts", 1))
    if parallel_parts < 1:
        raise _contract_error(label, "parallel_parts", parallel_parts, ">= 1")

    if "residual_stages" not in spec:
        raise ValueError(f"{label}: missing required field 'residual_stages'")
    residual_stages = int(spec["residual_stages"])
    if residual_stages != mode.residual_stages:
        raise _contract_error(label, "residual_stages", residual_stages, mode.residual_stages)

    if "codebook_dim" not in spec:
        raise ValueError(f"{label}: missing required field 'codebook_dim'")
    codebook_dim = int(spec["codebook_dim"])
    if codebook_dim != mode.codebook_dim:
        raise _contract_error(label, "codebook_dim", codebook_dim, mode.codebook_dim)

    stage_codebook_dims = spec.get("stage_codebook_dims")
    if not isinstance(stage_codebook_dims, (list, tuple)):
        raise ValueError(f"{label}: stage_codebook_dims must be a list/tuple")
    if len(stage_codebook_dims) != residual_stages:
        raise _contract_error(
            label,
            "stage_codebook_dims",
            len(stage_codebook_dims),
            residual_stages,
        )
    for idx, dim in enumerate(stage_codebook_dims):
        if int(dim) != mode.codebook_dim:
            raise _contract_error(
                label,
                f"stage_codebook_dims[{idx}]",
                int(dim),
                mode.codebook_dim,
            )

    if residual_stages > 1:
        _validate_multi_stage_spec(
            spec,
            mode,
            label=label,
            parallel_parts=parallel_parts,
            stages=residual_stages,
        )
    else:
        _validate_single_stage_spec(
            spec,
            mode,
            label=label,
            parallel_parts=parallel_parts,
        )
