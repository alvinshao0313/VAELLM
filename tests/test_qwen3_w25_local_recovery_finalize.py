from copy import deepcopy
import hashlib
import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from experiments.qwen3_w25.local_decoder import optimize_local_decoder
from experiments.qwen3_w25.local_recovery import TARGETS, recovery_config
from experiments.qwen3_w25.local_recovery_finalize import (
    _apply_patch, _inside, _read_shards, _save_final, _validate_source,
)
from litebsq.autoencoder import Decoder
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def _make_source(tmp_path):
    torch.manual_seed(53)
    config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=36, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=64)
    original = Qwen3ForCausalLM(config).to(torch.bfloat16)
    original_dir = tmp_path / "qwen3_original"
    original.save_pretrained(original_dir)
    model = deepcopy(original)
    for name in TARGETS:
        dense = model.get_submodule(name)
        layer = VAELinear(
            in_features=dense.in_features, out_features=dense.out_features, bias=None,
            codebook_dim=16, transpose=name.endswith(("o_proj", "down_proj")),
            stage_vq_weights=[torch.randint(0, 2, (dense.weight.numel() // 16, 1, 8), dtype=torch.bool)
                              for _ in range(2)],
            stage_decoders=[Decoder(in_dim=8, out_dim=16, hidden_dim=8, num_res_blocks=0,
                                    norm_type="layer", decoder_type="linear").to(torch.bfloat16)
                            for _ in range(2)],
            weight_rotation_spec=rotation_spec("two_sided", 0, 31, name),
        )
        layer.pack_parallel_stage_decoder_(trainable=False)
        set_module_by_name(model, name, layer)
    source_dir = tmp_path / "source"
    save_v6_full_checkpoint(model, str(source_dir), checkpoint_kind="final_model",
                            compressed_targets=TARGETS, base_model_path=str(original_dir))
    source, meta, _ = load_v6_model_checkpoint(str(source_dir), strict=True, map_location="cpu")
    return original, source, meta, source_dir


def test_real_seven_projection_patches_and_252_tensor_roundtrip(tmp_path):
    # Real local optimization and canonical save/reload on all 252 small modules.
    # This tests the private patch/save path, not production 8B/full-shard acceptance.
    original, source, meta, source_dir = _make_source(tmp_path)
    _validate_source(meta)
    source_hash = hashlib.sha256((source_dir / "pytorch_model.bin").read_bytes()).hexdigest()
    expected = dict(_load_state(source_dir))
    config = recovery_config(steps=2, lr=0.03, num_shards=36)
    token_counts = {"train": {"valid_tokens": 11}, "heldout": {"valid_tokens": 7}}
    first_valid = None
    for target in TARGETS[:7]:
        module = source.get_submodule(target)
        updated = deepcopy(module)
        train_x = torch.randn(11, module.in_features, dtype=torch.bfloat16)
        heldout_x = torch.randn(7, module.in_features, dtype=torch.bfloat16)
        report = optimize_local_decoder(updated, original.get_submodule(target).weight.float(), train_x, heldout_x,
                                        steps=config["steps"], lr=config["lr"], chunk_size=256)
        report.update(target=target, source_checkpoint_id=meta["checkpoint_id"],
                      source_checkpoint_dir=str(source_dir), protocol_version=1)
        patch = {key: value.clone() for key, value in updated.state_dict().items()
                 if key.startswith("_parallel_stage_decoder.")}
        if first_valid is None:
            first_valid = (deepcopy(module), deepcopy(patch), deepcopy(report), target)
        deltas = _apply_patch(module, patch, report, target=target, source_dir=source_dir,
                              source_id=meta["checkpoint_id"], config=config, data=token_counts)
        assert all(value > 0 for value in deltas)
        _assert_state_equal(module.state_dict(), updated.state_dict(), target)
        expected.update({target + "." + key: value for key, value in patch.items()})
    output = tmp_path / "recovered"
    report = _save_final(source, output, meta, expected, {"source_checkpoint_id": meta["checkpoint_id"]})
    assert report["status"] == "verified"
    assert report["serialized_tensors_compared"] == report["reloaded_tensors_compared"] == len(expected)
    assert report["bit_budget"]["compressed_target_count"] == 252
    assert report["bit_budget"]["budget_passed"] and report["bit_budget"]["user_linear_bpw"] <= 2.5
    assert json.loads((output / "finalize.json").read_text())["status"] == "verified"
    assert source_hash == hashlib.sha256((source_dir / "pytorch_model.bin").read_bytes()).hexdigest()
    module, patch, evidence, target = first_valid
    kwargs = dict(target=target, source_dir=source_dir, source_id=meta["checkpoint_id"], config=config, data=token_counts)
    frozen_patch = dict(patch, vq_weight=module.vq_weight.clone())
    with pytest.raises(ValueError, match="exactly every packed decoder"):
        _apply_patch(module, frozen_patch, evidence, **kwargs)
    bad = deepcopy(evidence)
    bad["evidence"]["frozen_sha256_before"] = "0" * 64
    with pytest.raises(ValueError, match="Frozen payload digest"):
        _apply_patch(module, patch, bad, **kwargs)
    unchanged = {key: value.clone() for key, value in module.state_dict().items() if key in patch}
    with pytest.raises(ValueError, match="Saved decoder update"):
        _apply_patch(module, unchanged, evidence, **kwargs)


def test_missing_shards_and_artifact_escape_rejected(tmp_path):
    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "worker.json").write_text(json.dumps({"status": "preparing", "config": recovery_config(steps=2, lr=0.03, num_shards=36)}))
    with pytest.raises(ValueError, match="Missing/invalid shard count"):
        _read_shards(tmp_path / "source", {"checkpoint_id": "test-source"}, [shard])
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a patch")
    with pytest.raises(ValueError, match="outside shard artifact"):
        _inside(shard, "../outside.pt")
