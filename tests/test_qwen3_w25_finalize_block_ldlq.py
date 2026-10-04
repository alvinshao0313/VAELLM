"""CPU packed-code and report contracts for full BlockLDLQ assembly."""

from copy import deepcopy
import hashlib

import pytest
import torch

from experiments.qwen3_w25.block_ldlq_model import CODE_KEYS, SOURCE_ID
from experiments.qwen3_w25.finalize_block_ldlq import (
    ARMS, PILOT_TARGET, SHARDS, SHARED_CODE, _load_patch, audit_code_patch, validate_worker_reports,
)
from litebsq.bitpack import pack_bool_tensor_to_uint8


def _code_pair():
    generator = torch.Generator().manual_seed(31)
    codes = torch.randint(0, 2, (2, 5, 1, 24), generator=generator).bool()
    changed = codes.clone()
    for stage, row, bit in ((0, 0, 0), (0, 2, 23), (1, 1, 11), (1, 3, 17)):
        changed[stage, row, 0, bit] = ~changed[stage, row, 0, bit]

    def pack(values):
        return {key: pack_bool_tensor_to_uint8(values[stage], logical_shape=(5, 1, 24))
                for stage, key in enumerate(CODE_KEYS)}

    return codes, changed, pack(codes), pack(changed), pack


def test_real_packed_xor_counts_both_stages_and_last_bit_without_mutation():
    _, _, original, patch, _ = _code_pair()
    before = {key: value.clone() for key, value in original.items()}
    report = audit_code_patch(original, patch, [2, 2])
    assert report == {"stage_flips": [2, 2], "changed_vectors": 4, "maximum_flips_per_vector": 1,
                      "vectors": 5, "packed_padding_bits": 0, "packed_bytes": 30}
    assert all(torch.equal(value, before[key]) for key, value in original.items())
    assert audit_code_patch(original, original, [0, 0])["maximum_flips_per_vector"] == 0


@pytest.mark.parametrize("second_stage", [0, 1])
def test_two_flips_in_one_vector_are_rejected_within_or_across_stages(second_stage):
    _, changed, original, _, pack = _code_pair()
    changed[second_stage, 0, 0, 1] = ~changed[second_stage, 0, 0, 1]
    counts = [3, 2] if second_stage == 0 else [2, 3]
    with pytest.raises(ValueError, match="More than one bit"):
        audit_code_patch(original, pack(changed), counts)


def test_packed_schema_and_claimed_counts_are_checked():
    _, _, original, patch, _ = _code_pair()
    with pytest.raises(ValueError, match="XOR counts"):
        audit_code_patch(original, patch, [3, 1])
    with pytest.raises(ValueError, match="exactly both"):
        audit_code_patch(original, {CODE_KEYS[0]: patch[CODE_KEYS[0]]}, [2, 2])
    for invalid in (patch[CODE_KEYS[0]].long(), torch.zeros(5, 1, 4, dtype=torch.uint8),
                    torch.zeros(5, 1, 6, dtype=torch.uint8)[:, :, ::2]):
        with pytest.raises(ValueError, match="dtype, shape, layout"):
            audit_code_patch(original, {**patch, CODE_KEYS[0]: invalid}, [2, 2])
    backing = torch.zeros(6, 1, 3, dtype=torch.uint8)
    with pytest.raises(ValueError, match="backing storage"):
        audit_code_patch(original, {**patch, CODE_KEYS[0]: backing[:5]}, [2, 2])


def _statistics(feedback):
    return {"feedback": feedback, "blocks": 752, "block_size": 16, "bits_per_stage": 24,
            "candidates_per_vector": 49, "candidate_vector_scores": 150929408, "stage_flips": [0, 0],
            "block_reports": [{"block": 751 - index, "visited_blocks": index + 1, "total_blocks": 752,
                               "stage_flips": [0, 0]} for index in range(752)]}


def _reports():
    pins = {name: "a" * 64 for name in SHARED_CODE}
    pilot = {"status": "verified", "feedback_dev_improved": True, "source_checkpoint_id": SOURCE_ID,
             "source_meta_sha256": "b" * 64, "target": PILOT_TARGET, "code_sha256": pins,
             "code_sha256_after": dict(pins), "data": {"train": {"valid_tokens": 74773}},
             "configuration": {"passes": 1, "damping_ratio": 0.01},
             "arms": {arm: {"statistics": _statistics(arm == "feedback_on")} for arm in ARMS}}
    workers = []
    for targets in SHARDS:
        workers.append({"status": "verified", "source_checkpoint_id": SOURCE_ID,
            "source_meta_sha256": "b" * 64, "pilot_report_sha256": "c" * 64,
            "data": deepcopy(pilot["data"]), "configuration": dict(pilot["configuration"]),
            "targets": list(targets), "code_sha256": dict(pins), "code_sha256_after": dict(pins),
            "modules": {name: {arm: {"patch_file": f"patches/{arm}.{name}.pt", "patch_sha256": "d" * 64,
                "statistics": pilot["arms"][arm]["statistics"], "metrics": {}, "proxy": 0.0} for arm in ARMS}
                for name in targets}})
    return workers, pilot


def _validate(workers, pilot):
    validate_worker_reports(workers, pilot, source_meta_sha256="b" * 64, pilot_report_sha256="c" * 64)


def test_complete_report_partition_is_read_only_and_excludes_reused_pilot():
    workers, pilot = _reports()
    before = deepcopy((workers, pilot))
    _validate(workers, pilot)
    assert (workers, pilot) == before
    assert sum(len(worker["targets"]) for worker in workers) == 35
    assert all(PILOT_TARGET not in worker["targets"] for worker in workers)


@pytest.mark.parametrize("fault", ["duplicate", "missing_target", "missing_module", "missing_arm", "missing_patch",
                                  "data", "configuration", "source", "pilot_hash", "pin", "incomplete_blocks"])
def test_incomplete_or_mismatched_worker_reports_are_rejected(fault):
    workers, pilot = deepcopy(_reports())
    first = workers[0]
    name = first["targets"][0]
    if fault == "duplicate":
        first["targets"][1] = name
    elif fault == "missing_target":
        first["targets"].pop()
    elif fault == "missing_module":
        del first["modules"][name]
    elif fault == "missing_arm":
        del first["modules"][name]["feedback_on"]
    elif fault == "missing_patch":
        del first["modules"][name]["feedback_on"]["patch_file"]
    elif fault == "data":
        first["data"]["train"]["valid_tokens"] -= 1
    elif fault == "configuration":
        first["configuration"]["passes"] = 2
    elif fault == "source":
        first["source_checkpoint_id"] = "another-source"
    elif fault == "pilot_hash":
        first["pilot_report_sha256"] = "e" * 64
    elif fault == "pin":
        first["code_sha256"][SHARED_CODE[0]] = "e" * 64
    else:
        first["modules"][name]["feedback_on"]["statistics"]["block_reports"].pop()
    with pytest.raises(ValueError):
        _validate(workers, pilot)


def test_actual_patch_serialization_hash_and_path_containment(tmp_path):
    _, _, _, patch, _ = _code_pair()
    worker = tmp_path / "worker"
    (worker / "patches").mkdir(parents=True)
    path = worker / "patches/arm.pt"
    torch.save(patch, path)
    item = {"patch_file": "patches/arm.pt", "patch_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    restored = _load_patch(worker, item)
    assert all(torch.equal(value, patch[key]) for key, value in restored.items())
    with pytest.raises(ValueError, match="SHA256"):
        _load_patch(worker, {**item, "patch_sha256": "0" * 64})
    outside = tmp_path / "outside.pt"
    torch.save(patch, outside)
    with pytest.raises(ValueError, match="inside its worker"):
        _load_patch(worker, {"patch_file": "../outside.pt", "patch_sha256": hashlib.sha256(outside.read_bytes()).hexdigest()})
