"""Save isolated round-replay hard codes and assemble them into the BF16 source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from experiments.qwen3_w25.kl_gradient_diagnostic import SOURCE_ID

TARGET = "model.layers.0.mlp.down_proj"
CODE_KEYS = (TARGET + ".vq_weight", TARGET + ".vq_weight_s1")
CODE_SHAPE = (1540096, 1, 6)
LOGICAL_SHAPE = (1540096, 1, 48)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_codes(codes):
    if set(codes) != set(CODE_KEYS):
        raise ValueError("Expected exactly the two L0 down persistent code banks.")
    for name, value in codes.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or tuple(value.shape) != CODE_SHAPE:
            raise ValueError(f"Wrong packed code dtype/shape: {name}")
        if value.device.type != "cpu" or value.layout != torch.strided or not value.is_contiguous():
            raise ValueError(f"Expected contiguous CPU packed codes: {name}")


def save_code_payload(path, codes, report):
    path = Path(path)
    if path.exists():
        raise ValueError("Do not overwrite a final-code payload.")
    codes = {name: value.detach().cpu().contiguous().clone() for name, value in codes.items()}
    validate_codes(codes)
    if report["source_checkpoint_id"] != SOURCE_ID or report["steps"] != 256:
        raise ValueError("Final-code payload must come from the defined source and 256-step replay.")
    payload = {"format": "qwen3_l0_down_round_codes_v1", "source_checkpoint_id": SOURCE_ID,
               "source_fingerprint": report["source_fingerprint"], "target": TARGET,
               "round_steps": report["round_steps"], "steps": 256,
               "data_archive_sha256": report["data_archive"]["sha256"],
               "logical_shape": list(LOGICAL_SHAPE), "final_bits": report["final_bits"], "codes": codes}
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    validate_codes(loaded["codes"])
    if any(not torch.equal(loaded["codes"][name], value) for name, value in codes.items()):
        raise RuntimeError("Final codes did not survive their independent CPU read.")
    return {"path": str(path.resolve()), "sha256": _sha(path), "tensor_count": 2,
            "logical_shape": list(LOGICAL_SHAPE), "packed_shape": list(CODE_SHAPE), "dtype": "uint8",
            "tensor_sha256": {name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                              for name, value in codes.items()}}


def compare_replay(replay, reference):
    """Compare recorded trajectory values; timings, worker IDs and source-code pins may differ."""
    if replay.get("status") != "complete" or reference.get("status") != "complete":
        raise ValueError("Both replay and reference must be complete.")
    fields = ("source_checkpoint_id", "source_fingerprint", "steps", "round_steps", "target",
              "configuration", "registration", "data_archive", "final_bits", "heldout",
              "round_reset_events", "frozen_state", "cumulative_flip_events_all_steps")
    different = [field for field in fields if replay[field] != reference[field]]
    if len(replay["steps_observed"]) != 256 or len(reference["steps_observed"]) != 256:
        raise ValueError("Expected both complete 256-step trajectories.")
    mismatches = []
    for step, (actual, expected) in enumerate(zip(replay["steps_observed"], reference["steps_observed"]), 1):
        for field in ("step", "batch_manifest_index", "loss", "telemetry", "before", "gradient", "after"):
            if actual.get(field) != expected.get(field):
                mismatches.append({"step": step, "field": field})
    return {"recorded_trajectory_exact": not different and not mismatches,
            "different_summary_fields": different, "different_step_fields": mismatches,
            "scope": "all recorded scalar losses/flip telemetry and recorded score/gradient/moment samples; not all score bytes"}


def export_replay(diagnostic, reference_diagnostic, output):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.assemble_categories import (
        _assert_state_equal, _load_state, _save_and_verify, _validate_sources)
    from experiments.qwen3_w25.local_bits import _changed_bits
    from train_utils.checkpoint_bpw import audit_state_dict
    from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    diagnostic, reference_diagnostic, output = map(lambda p: Path(p).resolve(),
                                                   (diagnostic, reference_diagnostic, output))
    report = json.loads(diagnostic.read_text())
    reference = json.loads(reference_diagnostic.read_text())
    comparison = compare_replay(report, reference)
    # A new trajectory is still real evidence, but cannot silently stand in for the prior paired result.
    if not comparison["recorded_trajectory_exact"]:
        comparison_path = diagnostic.parent / "replay_comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
        raise ValueError(f"Replay differs from its deterministic reference; retain evidence for review: {comparison_path}")
    source = Path(report["source_dir"]).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Use a fresh export directory outside the source.")
    payload_path = Path(report["final_codes"]["path"]).resolve()
    if payload_path.parent != diagnostic.parent or _sha(payload_path) != report["final_codes"]["sha256"]:
        raise ValueError("Final-code payload path or hash differs from the completed replay.")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if (payload["format"] != "qwen3_l0_down_round_codes_v1" or payload["target"] != TARGET
            or payload["source_checkpoint_id"] != SOURCE_ID or payload["logical_shape"] != list(LOGICAL_SHAPE)):
        raise ValueError("Unsupported code payload identity or geometry.")
    for field in ("source_fingerprint", "steps", "round_steps", "final_bits"):
        if payload[field] != report[field]:
            raise ValueError(f"Code payload disagrees with completed replay: {field}")
    if payload["data_archive_sha256"] != report["data_archive"]["sha256"]:
        raise ValueError("Code payload came from a different data archive.")
    validate_codes(payload["codes"])
    meta, _, _, original = _validate_sources(source, source, ["down_proj"])
    if meta["checkpoint_id"] != SOURCE_ID:
        raise ValueError("Wrong complete BF16 source checkpoint.")
    stat = (source / "pytorch_model.bin").stat()
    fingerprint = {"weight_stat": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino},
                   "metadata_sha256": {name: _sha(source / name) for name in ("checkpoint_meta.json", "config.json")}}
    if fingerprint != report["source_fingerprint"]:
        raise ValueError("The protected source changed since replay.")
    state = _load_state(source)
    if len(state) != 3423 or {k for k in state if k.startswith(TARGET + ".vq_weight")} != set(CODE_KEYS):
        raise ValueError("Source inventory differs from 3423 tensors and exactly two target code banks.")
    validate_codes({name: state[name] for name in CODE_KEYS})
    expected = dict(state)
    for stage, name in enumerate(CODE_KEYS):
        codes = payload["codes"][name]
        changed = _changed_bits(state[name], codes, LOGICAL_SHAPE)
        if changed != report["final_bits"][f"{TARGET}|stage={stage}|part=0"]["xor_vs_source"]:
            raise ValueError("Saved-code XOR differs from the measured final hard state.")
        expected[name] = codes
    source_budget = audit_state_dict(meta, state, max_bpw=2.5, max_lora_rank=0)
    budget = audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0)
    for field in ("user_linear_bits", "packed_linear_bits", "original_linear_weights", "components"):
        if budget[field] != source_budget[field]:
            raise ValueError(f"Code-only substitution changed storage accounting: {field}")
    if not source_budget["budget_passed"] or not budget["budget_passed"]:
        raise ValueError("The source or code-only model exceeds the bit budget.")
    model, _, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True, expected_kind="final_model")
    target = model.get_submodule(TARGET)
    with torch.no_grad():
        for name in CODE_KEYS:
            target.state_dict()[name.removeprefix(TARGET + ".")].copy_(expected[name])
    refresh_vae_linear_runtime_after_state_load(model)
    model.eval()
    _assert_state_equal(model.state_dict(), expected, "Two-codebank source union")
    frozen = {name: value for name, value in state.items() if name not in CODE_KEYS}
    compared = _assert_state_equal({name: model.state_dict()[name] for name in frozen}, frozen,
                                  "3421 frozen original BF16 source tensors")
    provenance = {"base_checkpoint_dir": str(source), "base_checkpoint_id": SOURCE_ID,
                  "donor_checkpoint_id": SOURCE_ID, "donor_kind": "two_codebank_payload_from_same_source",
                  "donor_payload": str(payload_path), "donor_payload_sha256": _sha(payload_path),
                  "diagnostic": str(diagnostic), "diagnostic_sha256": _sha(diagnostic),
                  "reference_diagnostic": str(reference_diagnostic), "reference_sha256": _sha(reference_diagnostic),
                  "replay_comparison": comparison, "replaced_tensor_keys": list(CODE_KEYS),
                  "frozen_tensors_compared": compared, "source_tensor_count": len(state),
                  "base_meta_sha256": _sha(source / "checkpoint_meta.json"),
                  "purpose": "inference_only code-bank assembly; original BF16 decoder/backbone; no exact-resume contract"}
    tokenizer = AutoTokenizer.from_pretrained(str(original), use_fast=True, local_files_only=True)
    result = _save_and_verify(model, output, meta, expected, provenance, tokenizer)
    for field in ("user_linear_bits", "packed_linear_bits", "original_linear_weights", "components"):
        if result["bit_budget"][field] != source_budget[field]:
            raise ValueError(f"Serialized checkpoint changed actual storage accounting: {field}")
    return result
