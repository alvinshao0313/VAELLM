from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.qwen3_w25.joint_block_validation import (
    audit_joint_state, bit_configuration, collect_state_inventory, configuration, require_native_runtime,
    validate_core_evidence,
)
from experiments.qwen3_w25.local_bits import calibrate_local_bits_cpu
from experiments.qwen3_w25.local_block import PROJECTIONS
from litebsq.autoencoder import Decoder
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear


def _state_fixture():
    # Real protected VAELinears exercise only state accounting and CPU scale
    # calibration here; no optimizer, CUDA proxy or training evidence is fabricated.
    torch.manual_seed(31)
    model, block = nn.Module(), nn.Module()
    block.add_module("self_attn", nn.Module())
    block.add_module("mlp", nn.Module())
    block.register_buffer("frozen_norm", torch.ones(16, dtype=torch.bfloat16))
    for name in PROJECTIONS:
        indices = torch.tensor([0, 3, 7, 11])
        qvalues, scales = encode_protected_channel_weight(torch.randn(4, 16).bfloat16(), quant_format="int8")
        layer = VAELinear(in_features=16, out_features=16, bias=None,
            codebook_dim=4, compressed_in_features=12, transpose=name == "mlp.down_proj",
            stage_vq_weights=[torch.randint(0, 2, (48, 1, 8), dtype=torch.bool) for _ in range(2)],
            stage_decoders=[Decoder(in_dim=8, out_dim=4, hidden_dim=8, num_res_blocks=0,
                norm_type="layer", decoder_type="symmetric").bfloat16() for _ in range(2)],
            protected_input_indices=indices, protected_input_qvalues=qvalues,
            protected_input_scales=scales, protected_channel_quant_format="int8")
        layer.pack_parallel_stage_decoder_(trainable=False)
        block.set_submodule(name, layer)
    model.add_module("block", block)
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    inventory = collect_state_inventory(block, prefix="block")
    return model, block, source, inventory


def test_actual_protected_state_audit_limits_changes_and_checks_both_stages():
    model, block, source, inventory = _state_fixture()
    block.requires_grad_(False)
    require_native_runtime(block)
    block.get_submodule(PROJECTIONS[0]).trainable_decode = True
    with pytest.raises(RuntimeError, match="native BF16"):
        require_native_runtime(block)
    block.get_submodule(PROJECTIONS[0]).trainable_decode = False
    assert len([row for entry in inventory["modules"].values() for row in entry["code_banks"]]) == 14
    changed = deepcopy(source)
    for entry in inventory["modules"].values():
        for key in entry["decoder_keys"]:
            changed[key].add_(0.03125)
    decoder_only = audit_joint_state(source, changed, inventory)
    assert decoder_only["total_changed_code_bits"] == 0
    code_key = inventory["modules"][PROJECTIONS[0]]["code_banks"][0]["key"]
    changed[code_key].view(-1)[0] ^= 1
    audit = audit_joint_state(source, changed, inventory)
    assert audit["total_changed_code_bits"] == 1
    assert audit["complete_source_tensors_compared"] == len(source)
    assert all(all(value > 0 for value in row["stage_parameter_delta_l2_saved"])
               for row in audit["modules"].values())
    forbidden = next(name for name in changed if name.endswith("protected_input_qvalues"))
    changed[forbidden].view(-1)[0] ^= 1
    with pytest.raises(RuntimeError, match="frozen source tensor"):
        audit_joint_state(source, changed, inventory)


def test_saved_no_update_and_schema_changes_are_rejected():
    _model, _block, source, inventory = _state_fixture()
    with pytest.raises(RuntimeError, match="Both decoder stages"):
        audit_joint_state(source, deepcopy(source), inventory)
    changed = deepcopy(source)
    key = inventory["modules"][PROJECTIONS[0]]["decoder_keys"][0]
    changed[key] = changed[key].float()
    with pytest.raises(RuntimeError, match="shape/dtype"):
        audit_joint_state(source, changed, inventory)


def test_real_cpu_scale_mapping_keeps_teacher_input_protocol_explicit():
    _model, block, _source, _inventory = _state_fixture()
    calibrations = {name: calibrate_local_bits_cpu(block.get_submodule(name)) for name in PROJECTIONS}
    rates, rounds = bit_configuration(calibrations)
    for name in PROJECTIONS:
        divisor = 128 if name.startswith("self_attn.") else 256
        assert rates[name] == max(row["scale"] for row in calibrations[name]["stages"]) / divisor
        assert rounds[name] == divisor
    cfg = configuration()
    assert cfg["block_index"] == 16 and cfg["maximum_train_documents"] == 512
    assert cfg["train_documents"] == cfg["heldout_documents"] == 64
    assert cfg["input_source"] == "original_teacher_block_inputs"
    assert cfg["steps"] == 128 and cfg["decoder_lr"] == 1.25e-5
    with pytest.raises(ValueError, match="exactly seven"):
        bit_configuration({})


def test_actual_inventory_prefix_is_independent_of_block16():
    _model, block, _source, _inventory = _state_fixture()
    # A deliberately incomplete trajectory must get past the actual tensor-scope
    # check for each prefix. No optimization metrics or GPU success are invented.
    for prefix in ("model.layers.0", "model.layers.16", "model.layers.35"):
        inventory = collect_state_inventory(block, prefix=prefix)
        relative = {key.removeprefix(prefix + ".") for key in inventory["allowed_state_keys"]}
        incomplete = {"settings": {"steps": 128, "decoder_lr": 1.25e-5, "bit_lrs": {}, "round_steps": {},
                                   "train_tokens": 1, "heldout_tokens": 1},
                      "evidence": {"allowed_state_keys": sorted(relative)},
                      "modules": dict.fromkeys(PROJECTIONS), "training_loss": []}
        plan = {"train_tokens": 1, "heldout_tokens": 1}
        with pytest.raises(RuntimeError, match="training-loss trajectory"):
            validate_core_evidence(incomplete, inventory, {}, {}, {}, plan, prefix=prefix)
        if prefix != "model.layers.16":
            with pytest.raises(RuntimeError, match="parameter/bank scope"):
                validate_core_evidence(incomplete, inventory, {}, {}, {}, plan)


def test_train_document_choice_changes_only_the_explicit_prefix_size():
    default = configuration()
    assert default == configuration(train_documents=64)
    assert configuration(train_documents=512) == {**default, "train_documents": 512}
    for invalid in (0, 32, 128, 513, True, 64.0):
        with pytest.raises(ValueError, match="exactly 64 or 512"):
            configuration(train_documents=invalid)
