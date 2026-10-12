"""Replay the saved single-pass choices without another neighbor search."""
from __future__ import annotations

import torch

from experiments.joint_reconstruction_probe.core import rebuild, require, schema, storage_bits
from experiments.rotation_factor_probe.capture import tensor_record


def replay(diagnostic, choices, control, neighbor):
    layer = rebuild(diagnostic, "gate_proj")
    initial = {name: tensor_record(value) for name, value in layer.state_dict().items()}
    require(initial == control["initial_payload"] == neighbor["local"]["initial_payload"],
            "Original decoder/control/neighbor initial payloads must be identical.")
    blocks = diagnostic["rotated_weight"].numel() // 32
    require(choices.device.type == "cpu" and choices.dtype == torch.uint8
            and choices.shape == (blocks,) and int(choices.max()) <= 64,
            "Require the saved complete CPU uint8 choice vector, 0=noop and 1..64=one bit.")
    require(tensor_record(choices) == neighbor["selection_tensor"], "Saved choices changed.")
    histogram = torch.bincount(choices.long(), minlength=65).tolist()
    require(histogram == neighbor["local"]["selection"]["choice_histogram"], "Choice inventory changed.")
    before_schema, before_bits = schema(layer), storage_bits(layer)
    counts = []
    for stage in range(2):
        bank = diagnostic["stage_codes"][stage].clone()
        selected = (choices > stage * 32) & (choices <= (stage + 1) * 32)
        rows = torch.arange(blocks)[selected]
        bits = choices[selected].long() - 1 - stage * 32
        bank[rows, 0, bits // 8] ^= (1 << (bits % 8)).to(torch.uint8)
        layer.get_stage_part_vq_storage(stage, 0).copy_(bank)
        counts.append(int(selected.sum()))
    layer._build_parallel_stage_decode_plan()
    final = {name: tensor_record(value) for name, value in layer.state_dict().items()}
    require(final == neighbor["local"]["final_payload"], "Replayed banks differ from the actual exact64 endpoint.")
    require(all(final[name] == value for name, value in initial.items()
                if name not in {"vq_weight", "vq_weight_s1"}), "Replay changed non-code state.")
    require(counts == neighbor["local"]["selection"]["stage_xor"]
            and sum(counts) == neighbor["local"]["selection"]["totals"]["beneficial_blocks"],
            "Replay no longer changes exactly one bit per selected block.")
    require(schema(layer) == before_schema and storage_bits(layer) == before_bits,
            "Replay changed persistent geometry or storage.")
    return layer, {"initial_decoder_and_original_payload_exact": True,
        "replayed_payload_exact": True, "noncodes_unchanged": True,
        "stage_initial_xor_from_original": counts, "persistent_bits": before_bits,
        "target": tensor_record(diagnostic["rotated_weight"])}
