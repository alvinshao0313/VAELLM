from copy import deepcopy
import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.qwen3_w25.assemble_categories import _load_state
from experiments.qwen3_w25.local_bit_recovery import save_code_patch
from experiments.qwen3_w25.local_bit_recovery_finalize import _apply_code_patch, _save_final
from experiments.qwen3_w25.local_bits import _code_banks, _digest
from experiments.qwen3_w25.local_decoder import _storage_bits, _validate_payload_profile
from experiments.qwen3_w25.local_recovery import TARGETS, shard_targets
from experiments.qwen3_w25.protected_bit_recovery import (
    SOURCE_CHECKPOINT_ID, load_recipe_selection, recovery_configuration, validate_pilot_source_model,
)
from experiments.qwen3_w25.protected_bit_recovery_finalize import read_shards, validate_data_plan
from litebsq.autoencoder import Decoder
from litebsq.misc import set_module_by_name
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def test_fixed_configuration_and_complete_partition():
    for divisor in (64, 128):
        cfg = recovery_configuration(lr_divisor=divisor, num_shards=4)
        assert cfg["train_documents"] == 512 and cfg["heldout_documents"] == 64
        assert cfg["recipe_mode"] == "uniform" and cfg["focused_selection"] is None
        assert cfg["recipes"] == {group: {"steps": 128, "round_steps": 128, "lr_divisor": divisor}
                                  for group in ("attention", "mlp")}
        assert not cfg["repeated_diagnostic_used_for_selection"]
    groups = [shard_targets(index, 4) for index in range(4)]
    assert sum(map(len, groups)) == len(set().union(*map(set, groups))) == 252
    with pytest.raises(ValueError, match="divisor"):
        recovery_configuration(lr_divisor=32, num_shards=4)


@pytest.mark.parametrize("status", ["running", "failed"])
def test_incomplete_pilot_never_selects_a_recipe(tmp_path, status):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"status": status}), encoding="utf-8")
    with pytest.raises(ValueError, match="completed combined pilot"):
        load_recipe_selection(path, tmp_path / "source", {"checkpoint_id": SOURCE_CHECKPOINT_ID})


def test_wrong_source_and_missing_shards_are_rejected(tmp_path):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"status": "complete", "source_checkpoint_id": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed protected source"):
        load_recipe_selection(path, tmp_path / "source", {"checkpoint_id": SOURCE_CHECKPOINT_ID})
    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "worker.json").write_text(json.dumps({"configuration": {"num_shards": 3}}))
    with pytest.raises(ValueError, match="Missing/invalid shard"):
        read_shards(tmp_path, {}, [shard])


def _source(tmp_path):
    torch.manual_seed(381)
    cfg = Qwen3Config(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=36,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=64)
    original = Qwen3ForCausalLM(cfg).bfloat16().eval()
    original_dir = tmp_path / "original"
    original.save_pretrained(original_dir)
    model = deepcopy(original)
    for target in TARGETS:
        linear = model.get_submodule(target)
        indices = torch.arange(0, 16, dtype=torch.long)
        qvalues, scales = encode_protected_channel_weight(linear.weight[:, indices].T, quant_format="int8")
        compressed = linear.in_features - len(indices)
        bits = 7 if target.endswith("down_proj") else 5
        module = VAELinear(in_features=linear.in_features, out_features=linear.out_features, bias=None,
            codebook_dim=16, transpose=target.endswith(("q_proj", "o_proj", "down_proj")),
            compressed_in_features=compressed, protected_input_indices=indices,
            protected_input_qvalues=qvalues, protected_input_scales=scales,
            protected_channel_quant_format="int8",
            stage_vq_weights=[torch.randint(0, 2, (compressed * linear.out_features // 16, 1, bits), dtype=torch.bool)
                              for _ in range(2)],
            stage_decoders=[Decoder(in_dim=bits, out_dim=16, hidden_dim=8, num_res_blocks=0,
                                    norm_type="layer", decoder_type="linear").bfloat16() for _ in range(2)])
        module.pack_parallel_stage_decoder_(trainable=False)
        assert _validate_payload_profile(module) == "no_rotation_int8_input"
        set_module_by_name(model, target, module)
    source = tmp_path / "source"
    save_v6_full_checkpoint(model, str(source), checkpoint_kind="final_model", compressed_targets=TARGETS,
                            base_model_path=str(original_dir))
    loaded, meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True)
    return loaded, meta, source


def test_real_252_protected_code_patches_freeze_payload_and_canonical_reload(tmp_path):
    # Real small modules and checkpoint I/O test storage contracts only. These
    # patches do not claim to come from a GPU optimization or production shard.
    model, meta, source = _source(tmp_path)
    target = TARGETS[0]
    observed = {"targets": {target: {"source_module_sha256": _digest(model.get_submodule(target).state_dict(), set())}}}
    validate_pilot_source_model(model, observed)
    protected = model.get_submodule(target).protected_input_qvalues
    protected.reshape(-1)[0].bitwise_xor_(1)
    with pytest.raises(ValueError, match="Source module bytes differ"):
        validate_pilot_source_model(model, observed)
    protected.reshape(-1)[0].bitwise_xor_(1)
    original = _load_state(source)
    expected = dict(original)
    audit = audit_checkpoint(source, max_bpw=2.5, max_lora_rank=0)
    assert audit["budget_passed"]
    first = None
    for index, target in enumerate(TARGETS):
        module = model.get_submodule(target)
        state = module.state_dict()
        banks = _code_banks(module)
        keys = {key for key, _ in banks}
        codes = {key: state[key].clone() for key in keys}
        changed = deepcopy(module)
        changed.get_stage_part_vq_storage(0, 0).reshape(-1)[0].bitwise_xor_(1)
        patch_file = tmp_path / f"patch_{index}.pt"
        report = save_code_patch(changed, codes, patch_file)
        patch = torch.load(patch_file, map_location="cpu", weights_only=True)
        frozen = _digest(state, keys)
        report.update(source_module_sha256=_digest(state, set()), evidence={
            "code_state_keys": [key for key, _ in banks], "frozen_sha256_before": frozen,
            "frozen_sha256_after": frozen, "persistent_storage_bits_before": _storage_bits(module),
            "persistent_storage_bits_after": _storage_bits(module), "stage_changed_bits": [1, 0],
            "final_changed_bits": 1})
        if first is None:
            first = deepcopy(module), deepcopy(patch), deepcopy(report), target
        assert _apply_code_patch(module, patch, report, target=target) == [1, 0]
        expected.update({target + "." + key: value for key, value in patch.items()})
    output = tmp_path / "recovered"
    finalized = _save_final(model, output, meta, expected, audit, {"cpu_test": "protected code storage only"})
    assert finalized["status"] == "verified" and finalized["checkpoint_id"] != meta["checkpoint_id"]
    for key in ("user_linear_bits", "packed_linear_bits", "code_padding_bits", "full_model_tensor_bytes"):
        assert finalized["bit_budget"][key] == audit[key]
    assert finalized["serialized_tensors_compared"] == finalized["reloaded_tensors_compared"] == len(expected)
    reloaded, new_meta, _ = load_v6_model_checkpoint(str(output), map_location="cpu", strict=True)
    assert new_meta.get("immutable_resume_contract") is None
    actual = reloaded.state_dict()
    for key, tensor in original.items():
        if not key.endswith((".vq_weight", ".vq_weight_s1")):
            assert torch.equal(actual[key].view(torch.uint8), tensor.view(torch.uint8)), key
    for target in (TARGETS[0], TARGETS[3], TARGETS[-1]):
        module = model.get_submodule(target)
        x = torch.randn(3, module.in_features, dtype=torch.bfloat16)
        with torch.no_grad():
            torch.testing.assert_close(reloaded.get_submodule(target)(x), module(x), rtol=0, atol=0)

    module, patch, report, target = first
    for field in ("protected_input_qvalues", "protected_input_scales", "protected_input_indices"):
        bad = deepcopy(module)
        value = getattr(bad, field)
        with torch.no_grad():
            value.reshape(-1)[0].add_(1)
        with pytest.raises(ValueError, match="Source module digest"):
            _apply_code_patch(bad, patch, report, target=target)
    with pytest.raises(ValueError, match="exactly the two packed"):
        _apply_code_patch(module, {**patch, "protected_input_qvalues": module.protected_input_qvalues}, report, target=target)
    padded = deepcopy(patch)
    padded["vq_weight"].reshape(-1)[0].bitwise_xor_(128)
    with pytest.raises(RuntimeError, match="padding bits"):
        _apply_code_patch(module, padded, report, target=target)
    with pytest.raises(ValueError, match="no-op"):
        _apply_code_patch(module, {key: module.state_dict()[key].clone() for key in patch}, report, target=target)
