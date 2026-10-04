"""Export a measured L0-down decoder/joint endpoint onto the original BF16 source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.kl_round_diagnostic import TARGET

DECODER_NAMES = tuple("_parallel_stage_decoder." + suffix for suffix in (
    "linear_in.conv.weight", "linear_in.conv.bias", "norm_out.weight", "norm_out.bias",
    "linear_out.conv.weight", "linear_out.conv.bias"))
CODE_NAMES = ("vq_weight", "vq_weight_s1")
SELECTED_KEYS = {TARGET + "." + name for name in (*DECODER_NAMES, *CODE_NAMES)}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_payload(payload, *, expected_source_id):
    """Validate the saved FP32 endpoint and project only its decoder leaves to BF16."""
    if (not isinstance(expected_source_id, str) or not expected_source_id.strip()
            or payload.get("format") != "qwen3_kl_joint_bridge_v1"
            or payload.get("source_checkpoint_id") != expected_source_id
            or payload.get("target") != TARGET or payload.get("steps") != 256
            or type(payload.get("joint")) is not bool):
        raise ValueError("Expected a completed same-source 256-step L0-down bridge payload.")
    if set(payload["decoder_fp32"]) != set(DECODER_NAMES) or set(payload["codes"]) != set(CODE_NAMES):
        raise ValueError("Payload must contain exactly six decoder leaves and two persistent code banks.")
    projected = {}
    for part, dtype in (("decoder_fp32", torch.float32), ("codes", torch.uint8)):
        for name, value in payload[part].items():
            if (not isinstance(value, torch.Tensor) or value.dtype != dtype or value.device.type != "cpu"
                    or value.layout != torch.strided or not value.is_contiguous() or not torch.isfinite(value).all()):
                raise ValueError(f"Expected a finite contiguous CPU {dtype} tensor: {name}")
            projected[TARGET + "." + name] = value.bfloat16() if part == "decoder_fp32" else value
    if any(not torch.isfinite(value).all() for value in projected.values()):
        raise ValueError("FP32-to-BF16 endpoint projection overflowed.")
    return projected


@torch.no_grad()
def apply_payload_(model, payload, *, expected_source_id):
    from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load

    projected = project_payload(payload, expected_source_id=expected_source_id)
    layer = model.get_submodule(TARGET)
    state = model.state_dict()
    for name, value in projected.items():
        if name not in state or state[name].shape != value.shape or state[name].dtype != value.dtype:
            raise ValueError(f"Payload shape/dtype disagrees with original BF16 source geometry: {name}")
    for name, value in projected.items():
        state[name].copy_(value)
    model.requires_grad_(False)
    layer.disable_trainable_decode()
    refresh_vae_linear_runtime_after_state_load(layer)
    grouped = torch.cat([layer.get_stage_part_vq_storage(stage, part)
                         for stage, part in layer._parallel_stage_layout], dim=1).contiguous()
    if not torch.equal(layer._parallel_stage_grouped_vq_packed, grouped) or layer._cached_weight is not None:
        raise RuntimeError("Exported target grouped codes/cache did not refresh from persistent payloads.")
    model.eval()
    return projected


def export_joint_endpoint(diagnostic, output):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.assemble_categories import _load_state, _save_and_verify, _validate_sources
    from experiments.qwen3_w25.kl_round_export import validate_codes, LOGICAL_SHAPE
    from experiments.qwen3_w25.local_bits import _changed_bits
    from train_utils.checkpoint_bpw import audit_state_dict
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    diagnostic, output = [Path(path).resolve() for path in (diagnostic, output)]
    report = json.loads(diagnostic.read_text())
    source_id = report["source_checkpoint_id"]
    if (report["status"] != "complete" or report["steps"] != 256 or report["target"] != TARGET
            or not isinstance(source_id, str) or not source_id.strip() or type(report["joint"]) is not bool
            or not report["frozen_state"]["unchanged"] or not report["frozen_cache_pointers_unchanged"]
            or not report["source_pins_data_unchanged"]
            or report["runtime"]["frozen_tensor_count"] != (3415 if report["joint"] else 3417)):
        raise ValueError("Expected the completed frozen-state-verified L0-down bridge diagnostic.")
    source = Path(report["source_dir"]).resolve()
    if json.loads((source / "checkpoint_meta.json").read_text())["checkpoint_id"] != source_id:
        raise ValueError("Source checkpoint identity differs from the endpoint report.")
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(diagnostic.parent):
        raise ValueError("Use a fresh export directory outside protected source and diagnostic inputs.")
    payload_path = Path(report["selected_payload"]["path"]).resolve()
    if payload_path.parent != diagnostic.parent or _sha(payload_path) != report["selected_payload"]["sha256"]:
        raise ValueError("Saved endpoint payload identity differs from its completed diagnostic.")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    projected = project_payload(payload, expected_source_id=source_id)
    if payload["joint"] != report["joint"]:
        raise ValueError("Payload and measured training arm differ.")
    validate_codes({TARGET + "." + name: payload["codes"][name] for name in CODE_NAMES})
    repo = Path(__file__).resolve().parents[2]
    pins = dict(report["source_sha256"])
    for name in ("kl_joint_export.py", "export_kl_joint.py", "assemble_categories.py", "kl_round_export.py"):
        path = Path(__file__).parent / name
        pins[str(path)] = _sha(path)
    evidence = {str(diagnostic): _sha(diagnostic), str(payload_path): _sha(payload_path),
                report["data_archive"]["path"]: report["data_archive"]["sha256"]}
    def check_inputs():
        for path, digest in {**pins, **report["input_metadata_sha256"], **evidence}.items():
            if _sha(repo / path) != digest:
                raise ValueError(f"Pinned source or endpoint evidence changed: {path}")
        for item in report["input_stats"]:
            stat = Path(item["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != tuple(item[key] for key in ("size", "mtime_ns", "inode")):
                raise ValueError(f"Protected source input stat changed: {item['path']}")
    check_inputs()
    meta, _, _, original_model = _validate_sources(source, source, ["down_proj"])
    if meta["checkpoint_id"] != source_id:
        raise ValueError("Wrong original complete BF16 source.")
    original = _load_state(source)
    if len(original) != 3423 or not SELECTED_KEYS <= original.keys():
        raise ValueError("Expected original3423 tensor inventory and selected6+2 tensors.")
    for name, value in projected.items():
        if value.shape != original[name].shape or value.dtype != original[name].dtype:
            raise ValueError(f"Projected endpoint changed source geometry or dtype: {name}")
    xor = [_changed_bits(original[TARGET + "." + name], payload["codes"][name], LOGICAL_SHAPE) for name in CODE_NAMES]
    if xor != report["net_xor_by_stage"] or (not report["joint"] and any(xor)):
        raise ValueError("Actual endpoint codes disagree with reported changes or decoder-only freeze.")
    changes = {name: int((payload["decoder_fp32"][name] != original[TARGET + "." + name].float()).sum())
               for name in DECODER_NAMES}
    if changes != report["decoder_changed_elements"]:
        raise ValueError("Saved FP32 decoder differs from the reported endpoint update.")
    source_budget = audit_state_dict(meta, original, max_bpw=2.5, max_lora_rank=0)
    expected = {**original, **projected}
    budget = audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0)
    budget_fields = ("user_linear_bits", "packed_linear_bits", "original_linear_weights", "components",
                     "outside_linear_bits", "code_padding_bits")
    if not source_budget["budget_passed"] or not budget["budget_passed"]:
        raise ValueError("Source or endpoint exceeds the2.5bpw/rank0 contract.")
    if any(source_budget[field] != budget[field] for field in budget_fields):
        raise ValueError("Same-geometry decoder/code substitution changed actual storage accounting.")
    # The existing FA2 loader requires CUDA visible even with map_location CPU.
    # Keep the assigned GPU visible; this assembly performs no model forward.
    model, loaded_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True,
                                                   expected_kind="final_model")
    if loaded_meta["checkpoint_id"] != source_id:
        raise ValueError("Loaded source identity changed.")
    _assert_state_equal(model.state_dict(), original, "Original BF16 source before endpoint replacement")
    apply_payload_(model, payload, expected_source_id=source_id)
    _assert_state_equal(model.state_dict(), expected, "Eight-tensor endpoint source union")
    frozen = {name: value for name, value in original.items() if name not in SELECTED_KEYS}
    compared = _assert_state_equal({name: model.state_dict()[name] for name in frozen}, frozen,
                                  "3415 frozen original BF16 source tensors")
    if compared != 3415:
        raise ValueError("Incomplete frozen tensor inventory.")
    provenance = {"base_checkpoint_dir": str(source), "base_checkpoint_id": source_id,
                  "donor_checkpoint_id": source_id, "donor_kind": "saved_local_decoder_joint_endpoint",
                  "joint": report["joint"], "replaced_targets": [TARGET],
                  "replaced_tensor_keys": sorted(SELECTED_KEYS), "frozen_tensors_compared": compared,
                  "source_tensor_count": len(original), "original_model_path": str(original_model),
                  "input_evidence_sha256": evidence, "source_sha256": pins,
                  "base_meta_sha256": _sha(source / "checkpoint_meta.json"),
                  "bf16_projection": "exact torch.bfloat16 cast of the six saved FP32 leaves; persistent uint8 codes unchanged by export",
                  "decoder_changed_elements_after_bf16_cast": {name: int((projected[TARGET + "." + name] != original[TARGET + "." + name]).sum()) for name in DECODER_NAMES},
                  "code_net_xor_by_stage": xor, "source_bit_budget": source_budget,
                  "purpose": "inference_only endpoint assembly for full downstream evaluation; no retraining/exact-resume"}
    tokenizer = AutoTokenizer.from_pretrained(str(original_model), use_fast=True, local_files_only=True)
    result = _save_and_verify(model, output, meta, expected, provenance, tokenizer)
    if any(result["bit_budget"][field] != budget[field] for field in budget_fields):
        raise ValueError("Serialized checkpoint changed actual storage accounting.")
    if result["serialized_tensors_compared"] != 3423 or result["reloaded_tensors_compared"] != 3423:
        raise ValueError("Incomplete3423 tensor save/fresh verification.")
    check_inputs()
    return result
