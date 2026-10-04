"""Assemble two saved width-probe decoders into their original BF16 v6 source."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.expand_decoder import validate_packed_r0_decoder
from experiments.qwen3_w25.kl_gradient_diagnostic import SOURCE_ID, TARGETS
from experiments.qwen3_w25.widen_decoder import widen_packed_decoder_h128_to_h256_

PARAMETER_SUFFIXES = ("linear_in.conv.weight", "linear_in.conv.bias", "norm_out.weight", "norm_out.bias",
                      "linear_out.conv.weight", "linear_out.conv.bias")
SELECTED_KEYS = {name + "._parallel_stage_decoder." + suffix for name in TARGETS for suffix in PARAMETER_SUFFIXES}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_payload(payload):
    if (payload.get("format") != "qwen3_decoder_width_probe_v1" or payload.get("source_checkpoint_id") != SOURCE_ID
            or payload.get("targets") != list(TARGETS) or payload.get("hidden_dim") not in (128, 256)
            or payload.get("steps") != 256):
        raise ValueError("Expected a completed same-source H128/H256 256-step decoder-width payload.")
    for precision, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        values = payload[precision]
        if set(values) != SELECTED_KEYS:
            raise ValueError("Payload must contain exactly the twelve selected decoder leaves.")
        for name, value in values.items():
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != dtype
                    or value.layout != torch.strided or not value.is_contiguous() or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid finite contiguous CPU {precision} decoder tensor: {name}")
    _assert_state_equal(payload["bf16"], {name: value.bfloat16() for name, value in payload["fp32"].items()},
                        "Exact FP32-to-BF16 payload projection")


def width_specs(source_meta, hidden_dim):
    if hidden_dim not in (128, 256):
        raise ValueError("Only the measured H128/H256 endpoints are supported.")
    specs = deepcopy(source_meta["converted_modules"])
    found = set()
    for spec in specs:
        if spec["name"] not in TARGETS:
            continue
        found.add(spec["name"])
        for field in ("decoders", "stage_decoders"):
            for decoder in spec[field]:
                if (decoder["hidden_dim"] != 128 or decoder["decoder_type"] != "symmetric"
                        or decoder["num_res_blocks"] != 0 or decoder["param_dtype"] != "bfloat16"):
                    raise ValueError("Expected original BF16 H128/R0 symmetric source specs.")
                if hidden_dim == 256:
                    decoder.update(hidden_dim=256, decoder_type="asymmetric")
    if found != set(TARGETS):
        raise ValueError("Source metadata omits a selected target.")
    return specs


@torch.no_grad()
def apply_payload_(model, payload):
    """Validate shapes before replacing either module, then copy saved BF16 leaves."""
    from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load

    validate_payload(payload)
    hidden_dim = payload["hidden_dim"]
    modules = [(name, model.get_submodule(name)) for name in TARGETS]
    for name, module in modules:
        decoder = validate_packed_r0_decoder(module)
        if any(value.dtype != torch.bfloat16 for value in decoder.parameters()):
            raise ValueError("The assembled base must retain original BF16 decoder storage.")
        for suffix in PARAMETER_SUFFIXES:
            key = name + "._parallel_stage_decoder." + suffix
            shape = list(model.state_dict()[key].shape)
            if hidden_dim == 256:
                if suffix.startswith("linear_in."):
                    shape[0] *= 2
                elif suffix.startswith("norm_out."):
                    shape[-1] *= 2
                elif suffix == "linear_out.conv.weight":
                    shape[1] *= 2
            if list(payload["bf16"][key].shape) != shape:
                raise ValueError(f"Payload shape disagrees with real source geometry: {key}")
    transformations = {}
    for name, module in modules:
        if hidden_dim == 256:
            transformations[name] = widen_packed_decoder_h128_to_h256_(module)
        for suffix in PARAMETER_SUFFIXES:
            relative = "_parallel_stage_decoder." + suffix
            module.state_dict()[relative].copy_(payload["bf16"][name + "." + relative])
        module.requires_grad_(False)
        module.disable_trainable_decode()
        refresh_vae_linear_runtime_after_state_load(module)
    model.eval()
    return transformations


def export_width_endpoint(diagnostic, output):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.assemble_categories import _load_state, _save_and_verify, _validate_sources
    from train_utils.checkpoint_bpw import audit_state_dict
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    diagnostic, output = map(lambda path: Path(path).resolve(), (diagnostic, output))
    report = json.loads(diagnostic.read_text())
    if (report["status"] != "complete" or report["mode"] != "width_arm" or report["steps"] != 256
            or report["source_checkpoint_id"] != SOURCE_ID or report["targets"] != list(TARGETS)
            or not report["frozen_state"]["unchanged"] or not report["source_pins_data_unchanged"]
            or report["runtime"]["frozen_tensor_count"] != 3411 or report["runtime"]["code_bank_count"] != 504):
        raise ValueError("Expected the verified complete two-decoder 256-step width diagnostic.")
    source = Path(report["source_dir"]).resolve()
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(diagnostic.parent):
        raise ValueError("Use a fresh export directory outside source and diagnostic inputs.")
    payload_path = Path(report["selected_payload"]["path"]).resolve()
    if payload_path.parent != diagnostic.parent or _sha(payload_path) != report["selected_payload"]["sha256"]:
        raise ValueError("Selected payload identity differs from its completed diagnostic.")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    validate_payload(payload)
    if payload["hidden_dim"] != report["hidden_dim"]:
        raise ValueError("Payload and measured width differ.")
    repo = Path(__file__).resolve().parents[2]
    pins = dict(report["source_sha256"])
    for name in ("decoder_width_export.py", "export_decoder_width.py", "assemble_categories.py", "widen_decoder.py", "expand_decoder.py"):
        path = Path(__file__).parent / name
        pins[str(path)] = _sha(path)
    evidence = {str(diagnostic): _sha(diagnostic), str(payload_path): _sha(payload_path)}
    def check_inputs():
        for path, digest in {**pins, **report["input_metadata_sha256"], **evidence}.items():
            if _sha(repo / path) != digest:
                raise ValueError(f"Pinned source or diagnostic changed: {path}")
        for item in report["input_stats"]:
            stat = Path(item["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != tuple(item[key] for key in ("size", "mtime_ns", "inode")):
                raise ValueError(f"Protected source input stat changed: {item['path']}")
    check_inputs()
    meta, _, _, original_model = _validate_sources(source, source, ["gate_proj", "down_proj"])
    if meta["checkpoint_id"] != SOURCE_ID:
        raise ValueError("Wrong original complete BF16 source.")
    original = _load_state(source)
    if len(original) != 3423 or not SELECTED_KEYS <= original.keys():
        raise ValueError("Expected the original3423 tensor inventory and twelve decoder leaves.")
    source_budget = audit_state_dict(meta, original, max_bpw=2.5, max_lora_rank=0)
    expected = {**original, **payload["bf16"]}
    assembled_meta = dict(meta, converted_modules=width_specs(meta, payload["hidden_dim"]))
    budget = audit_state_dict(assembled_meta, expected, max_bpw=2.5, max_lora_rank=0)
    added_bits = sum((payload["bf16"][name].numel() - original[name].numel()) * 16 for name in SELECTED_KEYS)
    if (not source_budget["budget_passed"] or not budget["budget_passed"]
            or budget["user_linear_bits"] - source_budget["user_linear_bits"] != added_bits
            or budget["outside_linear_bits"] != source_budget["outside_linear_bits"]
            or budget["code_padding_bits"] != source_budget["code_padding_bits"]):
        raise ValueError("Width-only substitution changed another storage component or exceeded2.5bpw.")
    # Keep assigned CUDA visible: the existing FA2 loader checks CUDA even when
    # all checkpoint tensors are requested on CPU. No forward occurs here.
    model, loaded_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True,
                                                   expected_kind="final_model")
    if loaded_meta["checkpoint_id"] != SOURCE_ID:
        raise ValueError("Loaded source identity changed.")
    _assert_state_equal(model.state_dict(), original, "Original BF16 source before replacement")
    transformations = apply_payload_(model, payload)
    _assert_state_equal(model.state_dict(), expected, "Twelve-decoder-leaf source union")
    frozen = {name: value for name, value in original.items() if name not in SELECTED_KEYS}
    compared = _assert_state_equal({name: model.state_dict()[name] for name in frozen}, frozen,
                                  "3411 frozen original BF16 source tensors including all packed codes")
    code_keys = [name for name, value in original.items() if value.dtype == torch.uint8
                 and (name.endswith("vq_weight") or name.endswith("vq_weight_s1"))]
    if compared != 3411 or len(code_keys) != 504:
        raise ValueError("Wrong frozen tensor or code-bank inventory.")
    provenance = {"base_checkpoint_dir": str(source), "base_checkpoint_id": SOURCE_ID,
                  "donor_checkpoint_id": SOURCE_ID, "donor_kind": "two_saved_decoder_payloads_from_same_source",
                  "hidden_dim": payload["hidden_dim"], "replaced_targets": list(TARGETS),
                  "replaced_tensor_keys": sorted(SELECTED_KEYS), "frozen_tensors_compared": compared,
                  "frozen_code_banks": len(code_keys), "source_tensor_count": len(original),
                  "original_model_path": str(original_model), "input_evidence_sha256": evidence,
                  "source_sha256": pins, "base_meta_sha256": _sha(source / "checkpoint_meta.json"),
                  "decoder_transformations": transformations, "bf16_projection_exact": True,
                  "added_decoder_storage_bits": added_bits, "source_bit_budget": source_budget,
                  "purpose": "inference_only width endpoint assembly; no retraining or exact-resume contract"}
    tokenizer = AutoTokenizer.from_pretrained(str(original_model), use_fast=True, local_files_only=True)
    result = _save_and_verify(model, output, assembled_meta, expected, provenance, tokenizer)
    for field in ("user_linear_bits", "packed_linear_bits", "original_linear_weights", "components", "outside_linear_bits"):
        if result["bit_budget"][field] != budget[field]:
            raise ValueError(f"Serialized checkpoint changed actual storage accounting: {field}")
    if result["serialized_tensors_compared"] != 3423 or result["reloaded_tensors_compared"] != 3423:
        raise ValueError("Incomplete3423 tensor save/fresh verification.")
    check_inputs()
    return result
