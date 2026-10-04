from copy import deepcopy
import hashlib
import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.qwen3_w25.assemble_categories import (
    CATEGORIES, _assert_state_equal, _expected_state, _load_state,
    _parse_categories, _replace_modules, _save_and_verify, _validate_sources,
)
from litebsq.autoencoder import Decoder
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def _compress(model, name, *, donor=False):
    linear = model.get_submodule(name)
    bits, stages, dtype = (9, 1, torch.float32) if donor else (8, 2, torch.bfloat16)
    module = VAELinear(
        in_features=linear.in_features, out_features=linear.out_features,
        bias=None, codebook_dim=16, transpose=donor,
        stage_vq_weights=[torch.randint(0, 2, (linear.weight.numel() // 16, 1, bits), dtype=torch.bool)
                          for _ in range(stages)],
        stage_decoders=[Decoder(in_dim=bits, out_dim=16, hidden_dim=8, num_res_blocks=0,
                                norm_type="layer", decoder_type="linear").to(dtype)
                        for _ in range(stages)],
        weight_rotation_spec=rotation_spec("two_sided", 0, 31 if not donor else 53, name),
    )
    module.pack_parallel_stage_decoder_(trainable=False)
    set_module_by_name(model, name, module)


def test_real_252_module_canonical_roundtrip_and_source_immutability(tmp_path):
    # Keep the actual 36x7 module topology; the 8B shape gate is tested separately
    # below. This exercises real v6 reconstruction, not a patched model loader.
    torch.manual_seed(31)
    config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=36, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=64)
    original = Qwen3ForCausalLM(config).to(torch.bfloat16)
    original_dir = tmp_path / "qwen3_tiny_original"
    original.save_pretrained(original_dir)
    base, donor = deepcopy(original), deepcopy(original)
    names = [name for name, module in original.named_modules()
             if name.rsplit(".", 1)[-1] in CATEGORIES]
    selected = [name for name in names if name.endswith(".down_proj")]
    assert len(names) == 252 and len(selected) == 36
    for name in names:
        _compress(base, name)
    for name in selected:
        _compress(donor, name, donor=True)
    base_dir, donor_dir = tmp_path / "base", tmp_path / "donor"
    base_saved = save_v6_full_checkpoint(base, str(base_dir), checkpoint_kind="final_model",
                                         compressed_targets=names, base_model_path=str(original_dir))
    donor_saved = save_v6_full_checkpoint(donor, str(donor_dir), checkpoint_kind="category_boundary",
                                          compressed_targets=selected,
                                          pending_dense_targets=sorted(set(names) - set(selected)),
                                          base_model_path=str(original_dir))
    sources = [path for directory in (base_dir, donor_dir) for path in directory.iterdir() if path.is_file()]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    base_state, donor_state = _load_state(base_dir), _load_state(donor_dir)
    expected = _expected_state(base_state, donor_state, selected, names)
    base, base_meta, _ = load_v6_model_checkpoint(str(base_dir), strict=True, map_location="cpu")
    donor, donor_meta, _ = load_v6_model_checkpoint(str(donor_dir), strict=True, map_location="cpu")
    _replace_modules(base, donor, selected)
    _assert_state_equal(base.state_dict(), expected, "assembled")
    provenance = {"base_checkpoint_id": base_saved["checkpoint_id"],
                  "donor_checkpoint_id": donor_saved["checkpoint_id"], "replaced_targets": selected}
    output = tmp_path / "assembled"
    assembled_meta = dict(base_meta)
    donor_specs = {spec["name"]: spec for spec in donor_meta["converted_modules"]}
    assembled_meta["converted_modules"] = [donor_specs.get(spec["name"], spec)
                                           for spec in base_meta["converted_modules"]]
    report = _save_and_verify(base, output, assembled_meta, expected, provenance)
    assert report["status"] == "verified"
    assert report["serialized_tensors_compared"] == report["reloaded_tensors_compared"] == len(expected)
    assert report["bit_budget"]["compressed_target_count"] == 252
    assert report["bit_budget"]["budget_passed"] and report["bit_budget"]["user_linear_bpw"] <= 2.5
    assert report["bit_budget"]["code_padding_bits"] > 0
    assert json.loads((output / "assembly_report.json").read_text())["status"] == "verified"
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    reloaded, _, _ = load_v6_model_checkpoint(str(output), strict=True, map_location="cpu")
    # A real packed decoder/rotation forward must also survive the dtype/topology change.
    x = torch.randn(3, 128, dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(reloaded.get_submodule(selected[0])(x), donor.get_submodule(selected[0])(x),
                                   rtol=0, atol=0)
    with pytest.raises(ValueError, match="wrong Qwen3-8B projection"):
        _validate_sources(base_dir, donor_dir, ["down_proj"])


def test_backbone_mismatch_and_exact_source_bytes_are_rejected():
    base = {"proj.code": torch.ones(3, dtype=torch.uint8), "embed": torch.ones(2)}
    donor = {"proj.code": torch.zeros(4, dtype=torch.uint8), "embed": torch.zeros(2)}
    with pytest.raises(ValueError, match="Uncompressed backbone.*tensor bytes differ"):
        _expected_state(base, donor, ["proj"], ["proj"])
    donor["embed"] = base["embed"].clone()
    expected = _expected_state(base, donor, ["proj"], ["proj"])
    assert torch.equal(expected["proj.code"], donor["proj.code"])
    with pytest.raises(ValueError, match="shape/dtype"):
        _assert_state_equal({"x": torch.ones(2).bfloat16()}, {"x": torch.ones(2)}, "dtype")
    with pytest.raises(ValueError, match="tensor bytes differ"):
        _assert_state_equal({"x": torch.tensor([-0.0])}, {"x": torch.tensor([0.0])}, "signed zero")


def test_lora_tensor_and_ambiguous_category_list_rejected(tmp_path):
    torch.save({"model.layers.0.mlp.down_proj.low_rank_a": torch.zeros(2, 1)}, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="LoRA tensor"):
        _load_state(tmp_path)
    for text in ("", "down_proj,down_proj", "down_proj,", "all"):
        with pytest.raises(ValueError, match="comma-separated"):
            _parse_categories(text)
    assert _parse_categories("o_proj, down_proj") == ["o_proj", "down_proj"]
