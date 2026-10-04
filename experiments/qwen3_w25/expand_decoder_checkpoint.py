"""Create a separate R1 Qwen3 v6 base without changing any existing tensor."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import EXPECTED, _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.expand_decoder import expand_packed_decoder_r0_to_r1_, validate_packed_r0_decoder
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.checkpoint_v6 import _atomic_write_json, save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


REPO = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "experiments/qwen3_w25/expand_decoder_checkpoint.py",
    "experiments/qwen3_w25/expand_decoder.py",
    "experiments/qwen3_w25/assemble_categories.py",
    "experiments/qwen3_w25/summarize_results.py",
    "litebsq/autoencoder.py", "litebsq/parallel_layers.py", "litebsq/vae_linear.py",
    "train_utils/v6_model_loader.py", "train_utils/checkpoint_v6.py",
    "train_utils/checkpoint_bpw.py", "rotation/model_utils.py",
)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def expanded_specs(meta):
    expected = deepcopy(meta["converted_modules"])
    for spec in expected:
        for field in ("decoders", "stage_decoders"):
            for decoder in spec[field]:
                _require(decoder["num_res_blocks"] == 0, "Expected only R0 source decoders.")
                decoder["num_res_blocks"] = 1
    return expected


def expand_checkpoint(source_dir, output_dir, report_path):
    source, output, report_path = (Path(path).resolve() for path in (source_dir, output_dir, report_path))
    _require(not output.exists() and not report_path.exists(), "Expansion output/report already exists.")
    _require(not output.is_relative_to(source) and not report_path.is_relative_to(source),
             "Expansion must not write inside the protected source checkpoint.")
    started = time.monotonic()
    source_meta, source_budget = _validate_checkpoint(source)
    _require(source_meta["checkpoint_id"] == "552282f2-868a-4105-90df-46f3c6f6a818",
             "This experiment is bound to the approved 552282 source.")
    code_pins = {name: sha256(REPO / name) for name in CODE_FILES}
    source_pins = {name: sha256(source / name) for name in ("checkpoint_meta.json", "pytorch_model.bin", "config.json")}
    report = {
        "status": "running", "source_checkpoint": str(source), "source_checkpoint_id": source_meta["checkpoint_id"],
        "checkpoint_dir": str(output), "source_sha256": source_pins, "code_sha256": code_pins,
        "started_at": datetime.now().astimezone().isoformat(), "worker_pid": os.getpid(),
        "worker_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()[19]),
        "method": "one zero-output existing residual block per packed decoder; existing tensors unchanged",
        "purpose": "new independent inference/round base; not exact resume or a quality result",
    }
    try:
        original = _load_state(source)
        _require(source_budget["budget_passed"], "Source bit budget is invalid.")
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True,
                                                       expected_kind="final_model")
        _require(loaded_meta["checkpoint_id"] == source_meta["checkpoint_id"], "Loaded source identity changed.")
        _assert_state_equal(model.state_dict(), original, "Loaded source before expansion")
        model.eval().requires_grad_(False)
        modules = [(name, model.get_submodule(name)) for name in sorted(EXPECTED)]
        for _name, module in modules:
            validate_packed_r0_decoder(module)
        expected_specs = expanded_specs(source_meta)
        report["modules"] = {name: expand_packed_decoder_r0_to_r1_(module) for name, module in modules}
        expected_state = model.state_dict()
        _require(set(original) < set(expected_state), "Expansion did not add an independent parameter inventory.")
        report["original_tensors_compared"] = _assert_state_equal(
            {key: expected_state[key] for key in original}, original, "All original tensors after expansion")
        added = sorted(set(expected_state) - set(original))
        expected_added = {name + "._parallel_stage_decoder.blocks.0." + suffix for name in EXPECTED
                          for suffix in ("norm1.weight", "norm1.bias", "linear1.conv.weight", "linear1.conv.bias")}
        _require(set(added) == expected_added, "Unexpected new tensor names outside the R1 residual blocks.")
        for key in added:
            tensor = expected_state[key]
            _require(tensor.dtype == torch.bfloat16 and tensor.device.type == "cpu", f"Invalid new storage: {key}.")
            initial = 1 if key.endswith("norm1.weight") else 0
            _require(bool(torch.all(tensor == initial)), f"New residual initialization differs: {key}.")
        extra_params = sum(expected_state[key].numel() for key in added)
        _require(extra_params == 8451072, "R1 new parameter count disagrees with the approved H128 geometry.")
        tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True, local_files_only=True)
        provenance = {"source_checkpoint": str(source), "source_checkpoint_id": source_meta["checkpoint_id"],
                      "source_sha256": source_pins, "code_sha256": code_pins,
                      "initialization": "residual Linear weight/bias zero; LayerNorm weight one/bias zero",
                      "added_decoder_blocks": 1, "existing_tensors_preserved": True}
        saved = save_v6_full_checkpoint(
            model, str(output), checkpoint_kind="final_model", compressed_targets=source_meta["compressed_targets"],
            pending_dense_targets=(), skip_targets=(), train_mode="none", after_category_mode="none",
            lora_config=None, base_model_path=source_meta["base_model_path"], tokenizer=tokenizer,
            extra_meta={"decoder_expansion": provenance},
        )
        report["checkpoint_id"] = saved["checkpoint_id"]
        meta, budget = _validate_checkpoint(output)
        _require(meta["checkpoint_id"] != source_meta["checkpoint_id"], "Expansion reused source UUID.")
        _require(meta["converted_modules"] == expected_specs, "Saved topology changed beyond residual block depth.")
        _require(meta.get("immutable_resume_contract") is None and not meta.get("post_norm_head_linear"),
                 "Expanded inference base inherited resume/head training state.")
        serialized = _load_state(output)
        report["serialized_tensors_compared"] = _assert_state_equal(serialized, expected_state, "Expanded serialized state")
        del model, modules, expected_state
        gc.collect()
        reloaded, reloaded_meta, _ = load_v6_model_checkpoint(str(output), map_location="cpu", strict=True,
                                                            expected_kind="final_model")
        report["reloaded_tensors_compared"] = _assert_state_equal(reloaded.state_dict(), serialized,
                                                                 "Fresh strict R1 state")
        _require(reloaded_meta["checkpoint_id"] == meta["checkpoint_id"], "Reload identity differs.")
        del reloaded
        gc.collect()
        _require(budget["budget_passed"] and budget["code_padding_bits"] == source_budget["code_padding_bits"],
                 "Expanded bit budget or code padding is invalid.")
        _require(budget["user_linear_bits"] - source_budget["user_linear_bits"] == extra_params * 16,
                 "Physical budget increase does not equal new BF16 residual parameters.")
        _require(budget["outside_linear_bits"] == source_budget["outside_linear_bits"],
                 "Expansion changed storage outside the target projections.")
        for name, digest in code_pins.items():
            _require(sha256(REPO / name) == digest, f"Migration code changed: {name}.")
        for name, digest in source_pins.items():
            _require(sha256(source / name) == digest, f"Protected source changed: {name}.")
        report.update(status="verified", added_tensor_count=len(added), added_parameters=extra_params,
                      bit_budget=budget, source_bit_budget=source_budget,
                      existing_tensor_bytes_unchanged=True, numerical_forward_checked=False,
                      cuda_initialized=torch.cuda.is_initialized())
        return report
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      completed_at=datetime.now().astimezone().isoformat())
        _atomic_write_json(report, str(report_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = expand_checkpoint(args.source_dir, args.output_dir, args.report)
    print(json.dumps({key: result[key] for key in ("status", "checkpoint_id", "added_parameters",
                                                  "reloaded_tensors_compared", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
