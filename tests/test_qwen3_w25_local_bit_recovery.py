import json

import pytest
import torch

from experiments.qwen3_w25.local_bit_recovery import (
    SOURCE_CHECKPOINT_ID, bit_recovery_configuration, load_pilot_selection, save_code_patch,
)
from experiments.qwen3_w25.local_recovery import shard_targets
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec


def _layer_and_codes():
    layer = VAELinear(
        in_features=4, out_features=4, bias=None, codebook_dim=4, transpose=False,
        stage_vq_weights=[torch.zeros(4, 1, 9, dtype=torch.bool) for _ in range(2)],
        stage_decoders=[Decoder(in_dim=9, out_dim=4, decoder_type="linear").bfloat16() for _ in range(2)],
        weight_rotation_spec=rotation_spec("two_sided", 0, 31, "patch_test"),
    ).eval()
    layer.pack_parallel_stage_decoder_(trainable=False)
    source = {key: tensor.clone() for key, tensor in layer.state_dict().items() if key in {"vq_weight", "vq_weight_s1"}}
    return layer, source


def test_configuration_preserves_full_manifest_and_complete_shard_partition():
    for selected in (32, 128, 512):
        config = bit_recovery_configuration(train_documents=selected, num_shards=3)
        assert config["maximum_train_documents"] == 512 and config["heldout_documents"] == 64
        assert config["train_documents"] == selected and config["steps"] == config["round_steps"] == 128
    groups = [shard_targets(index, 3) for index in range(3)]
    assert all(len(group) == 84 for group in groups)
    assert len(set().union(*map(set, groups))) == 252
    with pytest.raises(ValueError, match="pilot-selected"):
        bit_recovery_configuration(train_documents=64, num_shards=3)
    with pytest.raises(ValueError, match="num_shards"):
        bit_recovery_configuration(train_documents=32, num_shards=0)


@pytest.mark.parametrize("status", ["running", "failed"])
def test_unfinished_pilot_rejected_before_source_or_runtime_access(tmp_path, status):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"status": status}), encoding="utf-8")
    with pytest.raises(ValueError, match="completed pilot"):
        load_pilot_selection(path, tmp_path / "source", {"checkpoint_id": SOURCE_CHECKPOINT_ID})


def test_wrong_source_pilot_rejected(tmp_path):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"status": "complete", "source_checkpoint_id": "wrong-source"}), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed original down3"):
        load_pilot_selection(path, tmp_path / "source", {"checkpoint_id": SOURCE_CHECKPOINT_ID})


def test_real_code_patch_contains_both_uint8_banks_and_only_actual_code_changes(tmp_path):
    layer, source = _layer_and_codes()
    frozen = {key: value.clone() for key, value in layer.state_dict().items() if key not in source}
    layer.get_stage_part_vq_storage(0, 0)[..., 0].bitwise_xor_(1)
    path = tmp_path / "codes.pt"
    report = save_code_patch(layer, source, path)
    patch = torch.load(path, map_location="cpu", weights_only=True)
    assert set(patch) == {"vq_weight", "vq_weight_s1"}
    assert all(value.dtype == torch.uint8 and value.device.type == "cpu" for value in patch.values())
    assert report["stage_changed_bits"] == [4, 0]
    assert report["stage_code_sha256_before"][0] != report["stage_code_sha256_after"][0]
    assert report["stage_code_sha256_before"][1] == report["stage_code_sha256_after"][1]
    for key, value in patch.items():
        assert torch.equal(value, layer.state_dict()[key])
    for key, value in frozen.items():
        assert torch.equal(value, layer.state_dict()[key])
    with pytest.raises(FileExistsError):
        save_code_patch(layer, source, path)


def test_unchanged_or_padding_only_patch_rejected_without_writing(tmp_path):
    layer, source = _layer_and_codes()
    path = tmp_path / "codes.pt"
    with pytest.raises(ValueError, match="actual logical bit change"):
        save_code_patch(layer, source, path)
    assert not path.exists()
    layer.get_stage_part_vq_storage(0, 0)[..., -1].bitwise_or_(128)
    with pytest.raises(RuntimeError, match="padding bits"):
        save_code_patch(layer, source, path)
    assert not path.exists()
