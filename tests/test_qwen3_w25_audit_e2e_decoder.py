"""Real CPU v6 export contracts for one projection; no 252-module/GPU claim."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.audit_e2e_decoder import (
    _audit_decoder_export_state, _inference_spec, _validate_export_storage,
)
from experiments.qwen3_w25.local_bits import _code_banks
from litebsq.autoencoder import Decoder
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model, save_v6_full_checkpoint


TARGET = "model.q_proj"


def _model(dtype, *, bits=8):
    torch.manual_seed(97)
    root = nn.Module()
    root.model = nn.Module()
    root.register_buffer("outside_linear", torch.arange(7, dtype=torch.bfloat16))
    protected = torch.arange(32, dtype=torch.int64) * 7
    qvalues, scales = encode_protected_channel_weight(torch.randn(32, 256), quant_format="int8")
    root.model.q_proj = VAELinear(
        in_features=256, out_features=256, bias=None, codebook_dim=32, transpose=False,
        compressed_in_features=224, protected_input_indices=protected,
        protected_input_qvalues=qvalues, protected_input_scales=scales,
        protected_channel_quant_format="int8",
        stage_vq_weights=[torch.randint(0, 2, (1792, 1, bits), dtype=torch.bool) for _ in range(2)],
        stage_decoders=[Decoder(in_dim=bits, out_dim=32, hidden_dim=8, num_res_blocks=0,
            norm_type="layer", decoder_type="symmetric", use_checkpoint=True).to(dtype)
            for _ in range(2)],
    )
    root.model.q_proj.pack_parallel_stage_decoder_(trainable=False)
    return root.eval()


def _save_reload(root, path):
    meta = save_v6_full_checkpoint(root, str(path), checkpoint_kind="final_model",
        compressed_targets=(TARGET,), base_model_path="cpu-real-audit-fixture", save_config=False)
    serialized = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    skeleton = nn.Module()
    skeleton.model = nn.Module()
    skeleton.model.q_proj = nn.Linear(256, 256, bias=False)
    skeleton.register_buffer("outside_linear", torch.zeros(7, dtype=torch.bfloat16))
    loaded, loaded_meta, _ = load_v6_full_checkpoint_into_model(skeleton, str(path),
        expected_kind="final_model", map_location="cpu", strict=True)
    assert loaded_meta["checkpoint_id"] == meta["checkpoint_id"]
    assert _assert_state_equal(loaded.state_dict(), serialized, "Real CPU strict reload") == len(serialized)
    budget = audit_checkpoint(path, max_bpw=2.5, max_lora_rank=0)
    assert budget["budget_passed"], budget["contract_violations"]
    assert budget["compressed_target_count"] == 1
    return loaded, loaded_meta, serialized, budget


def _export(tmp_path, dtype=torch.float32, *, bits=8, update=True, flip_codes=False):
    root, source_meta, original, source_budget = _save_reload(_model(dtype, bits=bits), tmp_path / "source")
    packed = root.model.q_proj._parallel_stage_decoder
    inventory = {TARGET: sorted(TARGET + "._parallel_stage_decoder." + key for key in packed.state_dict())}
    banks = {TARGET: [(TARGET + "." + key, shape) for key, shape in _code_banks(root.model.q_proj)]}
    packed.bfloat16()
    packed.use_checkpoint = False
    with torch.no_grad():
        if update:
            next(packed.parameters()).reshape(-1)[0].add_(1)
        if flip_codes:
            state = root.state_dict()
            for index, (key, _shape) in enumerate(banks[TARGET]):
                state[key].reshape(-1)[0].bitwise_xor_(1 << index)
    _loaded, final_meta, actual, final_budget = _save_reload(root, tmp_path / "final")
    assert source_meta["checkpoint_id"] != final_meta["checkpoint_id"]
    return original, actual, inventory, banks, source_meta, final_meta, source_budget, final_budget


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_real_decoder_export_strict_reload_and_precision_storage(tmp_path, dtype):
    original, actual, inventory, banks, source_meta, final_meta, before, after = _export(tmp_path, dtype)
    source_spec, final_spec = source_meta["converted_modules"][0], final_meta["converted_modules"][0]
    assert {row["param_dtype"] for row in source_spec["stage_decoders"]} == {str(dtype).removeprefix("torch.")}
    assert {row["param_dtype"] for row in final_spec["stage_decoders"]} == {"bfloat16"}
    assert _inference_spec(source_spec, export_decoder_dtype=True) == _inference_spec(final_spec)
    report = _audit_decoder_export_state(original, actual, inventory, banks, train_mode="decoder")
    assert report["decoder_source_dtypes"] == [str(dtype)]
    assert report["decoder_target_dtype"] == "torch.bfloat16"
    assert report["decoder_dtype_converted_tensors"] == (len(inventory[TARGET]) if dtype == torch.float32 else 0)
    assert report["changed_decoder_elements"] == 1
    assert report["changed_code_bits"] == 0
    assert report["packed_padding_unchanged"]
    assert report["frozen_bytes_unchanged"]
    expected_saved = before["components"]["decoder"]["storage_bits"] - after["components"]["decoder"]["storage_bits"]
    assert report["decoder_storage_source_bits"] == before["components"]["decoder"]["storage_bits"]
    assert report["decoder_storage_target_bits"] == after["components"]["decoder"]["storage_bits"]
    assert report["decoder_storage_saved_bits"] == expected_saved
    assert report["decoder_storage_saved_bytes"] == expected_saved // 8
    assert (expected_saved > 0) == (dtype == torch.float32)
    assert before["user_linear_bits"] - after["user_linear_bits"] == expected_saved
    assert before["packed_linear_bits"] - after["packed_linear_bits"] == expected_saved
    assert before["full_model_tensor_bytes"] - after["full_model_tensor_bytes"] == expected_saved // 8
    assert before["outside_linear_bits"] == after["outside_linear_bits"]
    _validate_export_storage(before, after, report)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_conversion_without_a_numerical_decoder_update_is_rejected(tmp_path, dtype):
    original, actual, inventory, banks, *_ = _export(tmp_path, dtype, update=False)
    for key in inventory[TARGET]:
        assert torch.equal(actual[key], original[key].bfloat16())
    with pytest.raises(ValueError, match="No actual numerical decoder update"):
        _audit_decoder_export_state(original, actual, inventory, banks, train_mode="decoder")


def test_decoder_audit_rejects_frozen_dtype_bytes_and_decoder_schema_changes(tmp_path):
    original, actual, inventory, banks, *_ = _export(tmp_path)
    decoder_key = inventory[TARGET][0]
    extra_storage = torch.empty(actual[decoder_key].numel() + 4, dtype=torch.bfloat16)
    enlarged_view = extra_storage[:actual[decoder_key].numel()].reshape_as(actual[decoder_key])
    enlarged_view.copy_(actual[decoder_key])
    for key, replacement in (
        (TARGET + ".protected_input_scales", actual[TARGET + ".protected_input_scales"].float()),
        (TARGET + ".protected_input_qvalues", actual[TARGET + ".protected_input_qvalues"].bitwise_xor(1)),
        ("outside_linear", actual["outside_linear"] + 1),
        (decoder_key, actual[decoder_key].float()),
        (decoder_key, actual[decoder_key].unsqueeze(0)),
        (decoder_key, torch.full_like(actual[decoder_key], float("inf"))),
        (decoder_key, enlarged_view),
    ):
        corrupted = {**actual, key: replacement}
        with pytest.raises(ValueError):
            _audit_decoder_export_state(original, corrupted, inventory, banks, train_mode="decoder")
    with pytest.raises(ValueError):
        _audit_decoder_export_state({**original, decoder_key: original[decoder_key].half()},
            actual, inventory, banks, train_mode="decoder")


def test_metadata_normalization_keeps_all_other_inference_fields(tmp_path):
    _original, _actual, _inventory, _banks, source_meta, final_meta, *_ = _export(tmp_path)
    source, final = source_meta["converted_modules"][0], final_meta["converted_modules"][0]
    snapshot = deepcopy(source)
    normalized = _inference_spec(source, export_decoder_dtype=True)
    assert source == snapshot
    assert normalized == _inference_spec(final)
    assert _inference_spec(source) != _inference_spec(final)
    for field, value in (("transpose", not final["transpose"]), ("codebook_dim", 16)):
        assert normalized != _inference_spec({**final, field: value})
    corrupted = deepcopy(final)
    corrupted["protected_input_scales"]["dtype"] = "float32"
    assert normalized != _inference_spec(corrupted)
    corrupted = deepcopy(final)
    corrupted["stage_decoders"][1]["hidden_dim"] += 1
    assert normalized != _inference_spec(corrupted)
    corrupted = deepcopy(source)
    corrupted["stage_decoders"][1]["param_dtype"] = "float16"
    assert _inference_spec(corrupted, export_decoder_dtype=True) != _inference_spec(final)
    # A same-named dtype field outside the decoder trees must remain meaningful.
    external = {**source, "other_spec": {"param_dtype": "float32"}}
    assert _inference_spec(external, export_decoder_dtype=True)["other_spec"]["param_dtype"] == "float32"


def test_storage_audit_rejects_inconsistent_real_export_ledgers(tmp_path):
    original, actual, inventory, banks, _source_meta, _final_meta, before, after = _export(tmp_path)
    report = _audit_decoder_export_state(original, actual, inventory, banks, train_mode="decoder")
    _validate_export_storage(before, after, report)
    for field in ("original_linear_weights", "user_linear_bits", "packed_linear_bits", "code_padding_bits", "full_model_tensor_bytes"):
        corrupted = deepcopy(after)
        corrupted[field] += 8
        with pytest.raises(ValueError):
            _validate_export_storage(before, corrupted, report)
    for component, field in (("decoder", "storage_bits"), ("decoder", "unique_storages"),
                             ("protected_weights", "storage_bits")):
        corrupted = deepcopy(after)
        corrupted["components"][component][field] += 1
        with pytest.raises(ValueError):
            _validate_export_storage(before, corrupted, report)
    corrupted = deepcopy(after)
    corrupted["outside_linear_bits"]["other_model"] += 8
    with pytest.raises(ValueError):
        _validate_export_storage(before, corrupted, report)
    with pytest.raises(ValueError):
        _validate_export_storage(before, {**after, "budget_passed": False}, report)


def test_sparse_bit_real_packed_xor_and_padding_contracts_survive_export(tmp_path):
    original, actual, inventory, banks, _source_meta, _final_meta, before, after = _export(
        tmp_path, bits=7, flip_codes=True)
    assert before["code_padding_bits"] == after["code_padding_bits"] > 0
    report = _audit_decoder_export_state(original, actual, inventory, banks, train_mode="decoder_sparse_bit")
    assert report["stage_changed_bits"] == {TARGET: [1, 1]}
    assert report["changed_code_bits"] == 2
    assert report["has_committed_code_change"]
    assert report["packed_padding_unchanged"]
    assert report["non_decoder_non_code_bytes_unchanged"]
    _validate_export_storage(before, after, report)
    with pytest.raises(ValueError):
        _audit_decoder_export_state(original, actual, inventory, banks, train_mode="decoder")
    first_code = banks[TARGET][0][0]
    corrupted = deepcopy(actual)
    corrupted[first_code].reshape(-1)[0].bitwise_xor_(128)
    with pytest.raises(RuntimeError, match="padding bits changed"):
        _audit_decoder_export_state(original, corrupted, inventory, banks, train_mode="decoder_sparse_bit")
    with pytest.raises(ValueError):
        _audit_decoder_export_state(original, {**actual, first_code: actual[first_code].float()},
            inventory, banks, train_mode="decoder_sparse_bit")
