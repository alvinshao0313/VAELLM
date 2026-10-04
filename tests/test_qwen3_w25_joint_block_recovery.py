from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.joint_block_recovery import (
    load_training_document_selection, recovery_configuration, selected_plan_from_manifest, shard_layers,
)
from experiments.qwen3_w25.joint_block_recovery_finalize import apply_joint_patch, validate_shard_coverage
from experiments.qwen3_w25.local_decoder import _storage_bits
from experiments.qwen3_w25.protected_bit_recovery import _file_sha256
from test_qwen3_w25_joint_block_validation import _state_fixture


def test_fixed_teacher_input_candidate_and_36_unique_blocks():
    cfg = recovery_configuration()
    assert cfg["num_shards"] == 4 and cfg["layer_assignment"] == "layer_index % 4"
    assert cfg["input_source"] == "original_teacher_block_inputs" and not cfg["cross_block_gradients"]
    assert cfg["train_documents"] == 512 and cfg["heldout_documents"] == 64
    assert cfg["maximum_train_documents"] == 512 and cfg["steps"] == 128 and cfg["decoder_lr"] == 1.25e-5
    assert cfg["attention"] == {"lr_divisor": 128, "round_steps": 128}
    assert cfg["mlp"] == {"lr_divisor": 256, "round_steps": 256}
    assert cfg["heldout_role"] == "calibration_dev" and cfg["heldout_used_for_selection"]
    assert not any(cfg[key] for key in ("heldout_used_for_gradients", "heldout_used_for_early_stopping",
                                        "heldout_used_for_per_block_selection"))
    assert set(cfg["selection_evidence"]) == {"64", "512"} and "superseded" in cfg["recipe_status"]
    groups = [shard_layers(i) for i in range(4)]
    assert [len(g) for g in groups] == [9] * 4
    assert sorted(v for group in groups for v in group) == list(range(36))
    for invalid in (-1, 4, True, 1.5):
        with pytest.raises(ValueError, match="shard_index"):
            shard_layers(invalid)


def test_shard_inventory_rejects_missing_duplicate_and_incomplete():
    # These are manifest inventories only, not synthetic optimization evidence.
    workers = [{"status": "complete", "protocol_version": 1, "shard_index": i,
                "layers": shard_layers(i), "blocks": dict.fromkeys(map(str, shard_layers(i)))} for i in range(4)]
    validate_shard_coverage(workers)
    with pytest.raises(ValueError, match="four"):
        validate_shard_coverage(workers[:3])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_shard_coverage([workers[0], workers[0], workers[2], workers[3]])
    invalid = deepcopy(workers)
    invalid[2]["blocks"].pop("34")
    with pytest.raises(ValueError, match="nine assigned"):
        validate_shard_coverage(invalid)
    invalid = deepcopy(workers)
    invalid[1]["status"] = "training"
    with pytest.raises(ValueError, match="Incomplete"):
        validate_shard_coverage(invalid)


@pytest.mark.parametrize("change_codes", [False, True])
def test_actual_protected_joint_patch_preserves_all_other_bytes(change_codes):
    # Real registered tensors test the new assembly helper. No CPU SparseBit
    # surrogate or optimizer trajectory is presented as training evidence.
    model, block, source, inventory = _state_fixture()
    block.eval().requires_grad_(False)
    bits = _storage_bits(model)
    parameter_ids = {k: id(v) for k, v in block.named_parameters()}
    patch = {key: source[key].clone() for key in inventory["allowed_state_keys"]}
    for entry in inventory["modules"].values():
        for key in entry["decoder_keys"]:
            patch[key].add_(0.03125)
    if change_codes:
        key = next(iter(inventory["modules"].values()))["code_banks"][0]["key"]
        patch[key].reshape(-1)[0] ^= 1
    actual_inventory, audit = apply_joint_patch(block, source, patch, prefix="block")
    assert actual_inventory == inventory and audit["total_changed_code_bits"] == int(change_codes)
    assert _storage_bits(model) == bits and {k: id(v) for k, v in block.named_parameters()} == parameter_ids
    _assert_state_equal(model.state_dict(), source | patch, "Actual joint patch assembly")
    fresh = deepcopy(model)
    fresh.load_state_dict(source | patch, strict=True)
    _assert_state_equal(fresh.state_dict(), source | patch, "Fresh CPU block instance load")


def test_joint_patch_rejects_extra_state_missing_code_and_wrong_dtype():
    _model, block, source, inventory = _state_fixture()
    block.eval().requires_grad_(False)
    patch = {key: source[key].clone() for key in inventory["allowed_state_keys"]}
    frozen = next(key for key in source if key.endswith("protected_input_qvalues"))
    invalid = patch | {frozen: source[frozen].clone()}
    with pytest.raises(ValueError, match="inventory"):
        apply_joint_patch(block, source, invalid, prefix="block")
    invalid = dict(patch)
    invalid.pop(next(iter(inventory["modules"].values()))["code_banks"][0]["key"])
    with pytest.raises(ValueError, match="inventory"):
        apply_joint_patch(block, source, invalid, prefix="block")
    invalid = dict(patch)
    key = next(iter(inventory["modules"].values()))["decoder_keys"][0]
    invalid[key] = invalid[key].float()
    with pytest.raises(ValueError, match="shape/dtype"):
        apply_joint_patch(block, source, invalid, prefix="block")
    _assert_state_equal({"block." + key: value for key, value in block.state_dict().items()}, source,
                        "Rejected malformed patches preserve the source")


def test_real_completed_pilot_retains_fixed_prefix_and_public_heldout():
    path = Path("result/qwen3_8b_w25_20260930/local_bits_recipe_protected/pilot.json")
    if not path.is_file():
        pytest.skip("Local completed source pilot is not present")
    pilot = json.loads(path.read_text())
    assert pilot["status"] == "complete"
    plan = selected_plan_from_manifest(pilot["data"], pilot["document_plan"])
    assert plan["train_document_indices"] == list(range(512)) and plan["train_tokens"] == 74773
    assert plan["heldout_document_indices"] == list(range(512, 576)) and plan["heldout_tokens"] == 9183
    invalid = deepcopy(pilot["data"])
    invalid["heldout"]["documents"][0] = deepcopy(invalid["train"]["documents"][0])
    with pytest.raises(ValueError, match="Repeated"):
        selected_plan_from_manifest(invalid, pilot["document_plan"])


def test_completed_real_paired_reports_bind_the_512_calibration_dev_choice():
    pilot_path = Path("result/qwen3_8b_w25_20260930/local_bits_recipe_protected/pilot.json")
    if not pilot_path.is_file():
        pytest.skip("Local completed source pilot is not present")
    pilot = json.loads(pilot_path.read_text())
    if not all(Path(path).is_file() for path in recovery_configuration()["selection_evidence"].values()):
        pytest.skip("Local completed L16 paired reports are not present")
    source = Path(pilot["source_checkpoint_dir"])
    selected = load_training_document_selection(source, pilot, _file_sha256(pilot_path))
    assert selected["selected_train_documents"] == 512 and selected["selection_block_index"] == 16
    assert selected["reports"]["512"]["after_native"]["heldout_mse"] < selected["reports"]["64"]["after_native"]["heldout_mse"]
    assert selected["calibration_dev_documents_sha256"] == selected_plan_from_manifest(
        pilot["data"], pilot["document_plan"])["heldout_documents_sha256"]
    with pytest.raises(ValueError, match="source/data/configuration"):
        load_training_document_selection(source / "wrong_source", pilot, _file_sha256(pilot_path))
