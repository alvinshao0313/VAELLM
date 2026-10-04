from copy import deepcopy
import hashlib
import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from experiments.qwen3_w25.local_bits import _changed_bits, _code_banks, _digest
from experiments.qwen3_w25.local_decoder import _storage_bits
from experiments.qwen3_w25.local_bit_recovery_finalize import (
    _apply_code_patch, _code_sha, _inside, _read_shards, _save_final,
)
from experiments.qwen3_w25.local_recovery import TARGETS
from litebsq.autoencoder import Decoder
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def _source(tmp_path):
    torch.manual_seed(57)
    config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, max_position_embeddings=64)
    original = Qwen3ForCausalLM(config).bfloat16().eval()
    original_dir = tmp_path / "original"
    original.save_pretrained(original_dir)
    model = deepcopy(original)
    for name in TARGETS:
        linear = model.get_submodule(name)
        bits = 7 if name.endswith("down_proj") else 5
        module = VAELinear(
            in_features=linear.in_features, out_features=linear.out_features, bias=None,
            codebook_dim=16, transpose=name.endswith(("o_proj", "down_proj")),
            stage_vq_weights=[torch.randint(0, 2, (linear.weight.numel() // 16, 1, bits), dtype=torch.bool)
                              for _ in range(2)],
            stage_decoders=[Decoder(in_dim=bits, out_dim=16, hidden_dim=8, num_res_blocks=0,
                                    norm_type="layer", decoder_type="linear").bfloat16() for _ in range(2)],
            weight_rotation_spec=rotation_spec("two_sided", 0, 31, name),
        )
        module.pack_parallel_stage_decoder_(trainable=False)
        set_module_by_name(model, name, module)
    source_dir = tmp_path / "source"
    save_v6_full_checkpoint(model, str(source_dir), checkpoint_kind="final_model", compressed_targets=TARGETS,
                            base_model_path=str(original_dir))
    loaded, meta, _ = load_v6_model_checkpoint(str(source_dir), map_location="cpu", strict=True)
    return loaded, meta, source_dir


def _real_patch(module, index):
    # These are real serialization/bit-contract fixtures, not simulated GPU
    # optimization. No gradient, training loss or completed-shard claim is made.
    state = module.state_dict()
    banks = _code_banks(module)
    keys = [key for key, _ in banks]
    patch = {key: state[key].clone() for key in keys}
    patch[keys[0]].reshape(-1)[0].bitwise_xor_(1)
    if index % 2:
        patch[keys[1]].reshape(-1)[-1].bitwise_xor_(4)
    changes = [_changed_bits(state[key], patch[key], logical) for key, logical in banks]
    frozen = _digest(state, set(keys))
    report = {
        "source_module_sha256": _digest(state, set()),
        "stage_code_sha256_before": [_code_sha(state[key]) for key in keys],
        "stage_code_sha256_after": [_code_sha(patch[key]) for key in keys],
        "stage_changed_bits": changes,
        "code_schema": {key: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for key, tensor in patch.items()},
        "evidence": {
            "code_state_keys": keys, "frozen_sha256_before": frozen, "frozen_sha256_after": frozen,
            "persistent_storage_bits_before": _storage_bits(module),
            "persistent_storage_bits_after": _storage_bits(module),
            "stage_changed_bits": changes, "final_changed_bits": sum(changes),
        },
    }
    return patch, report


def test_real_252_packed_patches_freeze_all_other_bytes_and_survive_canonical_reload(tmp_path):
    model, meta, source_dir = _source(tmp_path)
    original = _load_state(source_dir)
    expected = dict(original)
    source_hash = hashlib.sha256((source_dir / "pytorch_model.bin").read_bytes()).hexdigest()
    audit = audit_checkpoint(source_dir, max_bpw=2.5, max_lora_rank=0)
    assert audit["budget_passed"]
    first = None
    total = 0
    for index, target in enumerate(TARGETS):
        module = model.get_submodule(target)
        patch, report = _real_patch(module, index)
        if first is None:
            first = deepcopy(module), deepcopy(patch), deepcopy(report), target
        changes = _apply_code_patch(module, patch, report, target=target)
        assert changes == ([1, 1] if index % 2 else [1, 0])
        total += sum(changes)
        expected.update({target + "." + key: tensor for key, tensor in patch.items()})
    assert total == 378
    changed = [key for key in expected if not torch.equal(expected[key], original[key])]
    assert len(changed) == 378
    assert all(key.endswith((".vq_weight", ".vq_weight_s1")) for key in changed)
    output = tmp_path / "recovered"
    finalized = _save_final(model, output, meta, expected, audit,
                            {"source_checkpoint_id": meta["checkpoint_id"], "cpu_test": "packed storage only"})
    assert finalized["status"] == "verified"
    assert finalized["checkpoint_id"] != meta["checkpoint_id"]
    assert finalized["serialized_tensors_compared"] == finalized["reloaded_tensors_compared"] == len(expected)
    assert finalized["bit_budget"]["compressed_target_count"] == 252
    assert finalized["bit_budget"]["code_padding_bits"] > 0
    assert finalized["bit_budget"]["user_linear_bits"] == audit["user_linear_bits"]
    assert finalized["bit_budget"]["packed_linear_bits"] == audit["packed_linear_bits"]
    assert source_hash == hashlib.sha256((source_dir / "pytorch_model.bin").read_bytes()).hexdigest()
    _assert_state_equal(_load_state(source_dir), original, "Source untouched")
    reloaded, new_meta, _ = load_v6_model_checkpoint(str(output), map_location="cpu", strict=True)
    assert new_meta.get("immutable_resume_contract") is None
    assert new_meta["converted_modules"] == meta["converted_modules"]
    assert not (output / "cat_runtime_state.pt").exists()
    for target in (TARGETS[0], TARGETS[3], TARGETS[-1]):
        module = model.get_submodule(target)
        x = torch.randn(3, module.in_features, dtype=torch.bfloat16)
        with torch.no_grad():
            actual, reference = reloaded.get_submodule(target)(x), module(x)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert json.loads((output / "finalize.json").read_text())["status"] == "verified"

    module, patch, report, target = first
    with pytest.raises(ValueError, match="exactly the two packed"):
        _apply_code_patch(module, {**patch, "weight_rotation.input_signs": module.weight_rotation.input_signs.clone()},
                          report, target=target)
    with pytest.raises(ValueError, match="exactly the two packed"):
        _apply_code_patch(module, {"vq_weight": patch["vq_weight"]}, report, target=target)
    bad = deepcopy(report)
    bad["evidence"]["stage_changed_bits"][0] += 1
    with pytest.raises(ValueError, match="XOR changes"):
        _apply_code_patch(module, patch, bad, target=target)
    bad = deepcopy(report)
    bad["stage_code_sha256_after"][0] = "0" * 64
    with pytest.raises(ValueError, match="bank fingerprint"):
        _apply_code_patch(module, patch, bad, target=target)
    padded = deepcopy(patch)
    padded["vq_weight"].reshape(-1)[0].bitwise_xor_(128)
    with pytest.raises(RuntimeError, match="padding bits changed"):
        _apply_code_patch(module, padded, report, target=target)
    with pytest.raises(ValueError, match="schema/storage"):
        _apply_code_patch(module, {**patch, "vq_weight": patch["vq_weight"].float()}, report, target=target)
    frozen_tampered = deepcopy(module)
    with torch.no_grad():
        next(frozen_tampered._parallel_stage_decoder.parameters()).reshape(-1)[0].add_(1)
    with pytest.raises(ValueError, match="Source module digest"):
        _apply_code_patch(frozen_tampered, patch, report, target=target)
    unchanged = {key: module.state_dict()[key].clone() for key in patch}
    with pytest.raises(ValueError, match="XOR changes"):
        _apply_code_patch(module, unchanged, report, target=target)


def test_missing_worker_and_shard_artifact_escape_are_rejected(tmp_path):
    directory = tmp_path / "shard"
    directory.mkdir()
    (directory / "worker.json").write_text(json.dumps({"configuration": {"num_shards": 3}}))
    with pytest.raises(ValueError, match="Missing/invalid shard count"):
        _read_shards(tmp_path / "source", {"checkpoint_id": "not-a-production-source"}, [directory])
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a tensor patch")
    with pytest.raises(ValueError, match="outside shard artifact"):
        _inside(directory, "../outside.pt")
