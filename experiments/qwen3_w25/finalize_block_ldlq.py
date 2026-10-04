"""Assemble both complete down BlockLDLQ arms from two workers and the L27 pilot.

Only the 72 packed code tensors may change. This consumes real hard-code
patches, not decoder optimizer state, and preserves every other source byte.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.block_ldlq_model import CODE_KEYS, SOURCE_ID


REPO = Path(__file__).resolve().parents[2]
ARMS = ("feedback_off", "feedback_on")
TARGETS = tuple(f"model.layers.{index}.mlp.down_proj" for index in range(36))
PILOT_TARGET = TARGETS[27]
SHARDS = (TARGETS[:18], tuple(name for name in TARGETS[18:] if name != PILOT_TARGET))
SHARED_CODE = (
    "experiments/qwen3_w25/block_ldlq.py", "experiments/qwen3_w25/block_ldlq_model.py",
    "experiments/qwen3_w25/calibration.py", "experiments/qwen3_w25/local_recovery.py",
    "litebsq/vae_linear.py", "litebsq/fused_multistage_decoder.py", "litebsq/bitpack.py",
    "rotation/weight_preprocess.py",
)


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}.")
    return value


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_worker_reports(workers, pilot, *, source_meta_sha256, pilot_report_sha256):
    """Validate the fixed partition and provenance before reading any patch."""
    _require(len(workers) == 2, "Exactly two worker reports are required.")
    _require(pilot.get("status") == "verified" and pilot.get("feedback_dev_improved") is True,
             "The completed pilot must permit extension through improved native dev error.")
    _require(pilot.get("source_checkpoint_id") == SOURCE_ID
             and pilot.get("source_meta_sha256") == source_meta_sha256
             and pilot.get("target") == PILOT_TARGET, "Pilot source/target identity disagrees.")
    _require(set(pilot.get("arms", {})) == set(ARMS), "Pilot must contain both completed arms.")
    pilot_pins = pilot.get("code_sha256")
    _require(isinstance(pilot_pins, dict) and set(SHARED_CODE) <= pilot_pins.keys()
             and pilot.get("code_sha256_after") == pilot_pins, "Pilot code pins are missing or changed.")
    seen = set()
    for index, worker in enumerate(workers):
        _require(worker.get("status") == "verified", f"Worker {index} is not verified.")
        _require(worker.get("source_checkpoint_id") == SOURCE_ID
                 and worker.get("source_meta_sha256") == source_meta_sha256
                 and worker.get("pilot_report_sha256") == pilot_report_sha256,
                 f"Worker {index} source/pilot identity disagrees.")
        _require(worker.get("data") == pilot["data"]
                 and worker.get("configuration") == pilot["configuration"],
                 f"Worker {index} data or mathematical configuration differs from the pilot.")
        targets = worker.get("targets")
        _require(isinstance(targets, list) and len(targets) == len(set(targets))
                 and not seen.intersection(targets), "Duplicate or invalid worker targets.")
        _require(tuple(targets) == SHARDS[index], f"Worker {index} does not cover its complete ordered shard.")
        seen.update(targets)
        modules = worker.get("modules")
        _require(isinstance(modules, dict) and set(modules) == set(targets),
                 f"Worker {index} module reports are missing or extra.")
        pins = worker.get("code_sha256")
        _require(isinstance(pins, dict) and set(SHARED_CODE) <= pins.keys(), f"Worker {index} code pins are missing.")
        _require(all(pins[name] == pilot_pins[name] for name in pins.keys() & pilot_pins.keys()),
                 f"Worker {index} changed a shared pilot implementation.")
        if "code_sha256_after" in worker:
            _require(worker["code_sha256_after"] == pins, f"Worker {index} source changed during execution.")
        for name, module in modules.items():
            _require(isinstance(module, dict) and set(module) == set(ARMS), f"Missing arm report: {name}.")
            for arm, item in module.items():
                _require(isinstance(item, dict) and {"patch_file", "patch_sha256", "statistics", "metrics", "proxy"}
                         <= item.keys(), f"Missing patch/report fields: {name}/{arm}.")
                _validate_statistics(item["statistics"], arm)
    _require(seen == set(TARGETS) - {PILOT_TARGET}, "Workers must cover exactly the 35 non-pilot down projections.")
    for arm in ARMS:
        _validate_statistics(pilot["arms"][arm]["statistics"], arm)


def _validate_statistics(statistics, arm):
    expected = {"feedback": arm == "feedback_on", "blocks": 752, "block_size": 16,
                "bits_per_stage": 24, "candidates_per_vector": 49, "candidate_vector_scores": 150929408}
    _require(isinstance(statistics, dict) and all(statistics.get(key) == value for key, value in expected.items()),
             f"Wrong complete one-pass candidate geometry for {arm}.")
    flips = statistics.get("stage_flips")
    _require(isinstance(flips, list) and len(flips) == 2 and all(type(n) is int and n >= 0 for n in flips),
             "Expected two nonnegative integer stage flip counts.")
    blocks = statistics.get("block_reports")
    _require(isinstance(blocks, list) and len(blocks) == 752
             and [item.get("block") for item in blocks] == list(range(751, -1, -1)),
             "Missing or reordered reverse block reports.")
    _require(all(item.get("visited_blocks") == index + 1 and item.get("total_blocks") == 752
                 for index, item in enumerate(blocks)), "Incomplete block traversal counters.")
    per_block = [item.get("stage_flips") for item in blocks]
    _require(all(isinstance(row, list) and len(row) == 2 and all(type(n) is int and n >= 0 for n in row)
                 and sum(row) <= 4096 for row in per_block), "Invalid per-block single-flip counts.")
    _require([sum(row[stage] for row in per_block) for stage in range(2)] == flips,
             "Stage totals disagree with per-block flip counts.")


def audit_code_patch(original_pair, patch, reported_stage_flips):
    """Count actual packed XOR bits and enforce at most one flip per D16 vector.

    Banks are [vectors, 1, 3] uint8 for K24; every stored bit is logical, so
    there is no packed padding. Both stage counts share the same vector axis.
    """
    _require(isinstance(original_pair, dict) and isinstance(patch, dict)
             and set(original_pair) == set(patch) == set(CODE_KEYS), "A patch must contain exactly both packed banks.")
    _require(isinstance(reported_stage_flips, list) and len(reported_stage_flips) == 2
             and all(type(n) is int and n >= 0 for n in reported_stage_flips), "Invalid reported stage flip counts.")
    _require(isinstance(original_pair[CODE_KEYS[0]], torch.Tensor), "Original code bank must be a tensor.")
    shape = original_pair[CODE_KEYS[0]].shape
    _require(len(shape) == 3 and shape[0] > 0 and tuple(shape[1:]) == (1, 3),
             "K24 code banks must have shape [vectors, 1, 3].")
    for state in (original_pair, patch):
        for key, value in state.items():
            _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                     and value.dtype == torch.uint8 and value.layout == torch.strided
                     and value.shape == shape and value.is_contiguous()
                     and value.storage_offset() == 0 and value.untyped_storage().nbytes() == value.numel(),
                     f"Packed bank dtype, shape, layout or backing storage is invalid: {key}.")
    lookup = torch.tensor([value.bit_count() for value in range(256)], dtype=torch.int16)
    stage_flips, changed_vectors, maximum = [0, 0], 0, 0
    for start in range(0, shape[0], 65536):
        counts = []
        for stage, key in enumerate(CODE_KEYS):
            xor = original_pair[key][start:start + 65536] ^ patch[key][start:start + 65536]
            per_vector = lookup[xor.long()].sum(dim=(1, 2), dtype=torch.int16)
            stage_flips[stage] += int(per_vector.sum())
            counts.append(per_vector)
        total = counts[0] + counts[1]
        maximum = max(maximum, int(total.max()))
        _require(maximum <= 1, "More than one bit changed across the two stages of a vector.")
        changed_vectors += int(torch.count_nonzero(total))
    _require(stage_flips == reported_stage_flips, "Actual packed XOR counts disagree with reported stage flips.")
    return {"stage_flips": stage_flips, "changed_vectors": changed_vectors,
            "maximum_flips_per_vector": maximum, "vectors": shape[0], "packed_padding_bits": 0,
            "packed_bytes": sum(value.numel() for value in patch.values())}


def _verify_pins(pins):
    _require(isinstance(pins, dict) and bool(pins), "Missing code pins.")
    for name, expected in pins.items():
        path = (REPO / name).resolve()
        _require(not Path(name).is_absolute() and path.is_relative_to(REPO)
                 and path.is_file() and _sha(path) == expected, f"Code pin changed or is outside the repository: {name}.")


def _load_patch(directory, item):
    relative = Path(item["patch_file"])
    path = (directory / relative).resolve()
    _require(not relative.is_absolute() and path.is_relative_to(directory) and path.is_file(),
             "Patch path must resolve to a file inside its worker directory.")
    _require(_sha(path) == item["patch_sha256"], f"Patch SHA256 changed: {path}.")
    return torch.load(path, map_location="cpu", weights_only=True)


def finalize(source, worker_dirs, pilot_dir, output):
    from transformers import AutoTokenizer

    from experiments.qwen3_w25.block_ldlq_model import save_and_verify, validate_layer
    from experiments.qwen3_w25.summarize_results import _validate_checkpoint
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    source, pilot_dir, output = (Path(path).resolve() for path in (source, pilot_dir, output))
    workers = tuple(Path(path).resolve() for path in worker_dirs)
    _require(len(workers) == len(set(workers)) == 2, "Provide two distinct worker directories.")
    _require(not output.exists() and all(not output.is_relative_to(path) for path in (source, pilot_dir, *workers)),
             "Use a new output directory outside the immutable inputs.")
    output.mkdir(parents=True)
    started = time.perf_counter()
    report_path = output / "finalize.json"
    report = {"status": "running", "source": str(source), "pilot_dir": str(pilot_dir),
              "worker_dirs": [str(path) for path in workers], "arms": {},
              "code_sha256": {str(Path(__file__).resolve().relative_to(REPO)): _sha(__file__)}}
    _write(report_path, report)
    try:
        pilot_path = pilot_dir / "pilot.json"
        worker_paths = [directory / "worker.json" for directory in workers]
        pilot, worker_reports = _read(pilot_path), [_read(path) for path in worker_paths]
        source_meta_sha = _sha(source / "checkpoint_meta.json")
        pilot_sha = _sha(pilot_path)
        validate_worker_reports(worker_reports, pilot, source_meta_sha256=source_meta_sha, pilot_report_sha256=pilot_sha)
        _require(Path(pilot["source"]).resolve() == source, "Pilot source directory differs.")
        pin_sets = [pilot["code_sha256"], *(worker["code_sha256"] for worker in worker_reports), report["code_sha256"]]
        for pins in pin_sets:
            _verify_pins(pins)
        source_meta, source_budget = _validate_checkpoint(source)
        _require(source_meta["checkpoint_id"] == SOURCE_ID and source_budget["budget_passed"], "Invalid approved source.")
        report.update(source_checkpoint_id=SOURCE_ID, source_meta_sha256=source_meta_sha,
                      pilot_report_sha256=pilot_sha, worker_report_sha256=[_sha(path) for path in worker_paths],
                      source_budget=source_budget, configuration=pilot["configuration"], data=pilot["data"])
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True, expected_kind="final_model")
        original = _load_state(source)
        _require(loaded_meta["checkpoint_id"] == SOURCE_ID and len(original) == 3423, "Unexpected source inventory/identity.")
        _assert_state_equal(model.state_dict(), original, "BlockLDLQ original strict source")
        model.eval().requires_grad_(False)
        for name in TARGETS:
            validate_layer(model.get_submodule(name))
        code_names = {name + "." + key for name in TARGETS for key in CODE_KEYS}
        frozen = {key: value for key, value in original.items() if key not in code_names}
        _require(len(code_names) == 72 and len(frozen) == 3351, "Unexpected full-model frozen/code inventory.")
        tokenizer = AutoTokenizer.from_pretrained(source_meta["base_model_path"], use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        for arm in ARMS:
            expected, audits = dict(original), {}
            for directory, worker in zip(workers, worker_reports):
                for name in worker["targets"]:
                    item = worker["modules"][name][arm]
                    patch = _load_patch(directory, item)
                    pair = {key: original[name + "." + key] for key in CODE_KEYS}
                    audits[name] = audit_code_patch(pair, patch, item["statistics"]["stage_flips"])
                    expected.update({name + "." + key: value for key, value in patch.items()})
            pilot_final = pilot_dir / arm / "final_model"
            pilot_meta, _pilot_budget = _validate_checkpoint(pilot_final)
            _require(pilot_meta["checkpoint_id"] == pilot["arms"][arm]["export"]["checkpoint_id"], "Pilot final UUID changed.")
            _require({item["name"]: item for item in pilot_meta["converted_modules"]}
                     == {item["name"]: item for item in source_meta["converted_modules"]}, "Pilot topology changed.")
            pilot_state = _load_state(pilot_final)
            pilot_codes = {PILOT_TARGET + "." + key for key in CODE_KEYS}
            _assert_state_equal({key: value for key, value in pilot_state.items() if key not in pilot_codes},
                                {key: value for key, value in original.items() if key not in pilot_codes},
                                "Current pilot non-code state versus original source")
            patch = {key: pilot_state[PILOT_TARGET + "." + key] for key in CODE_KEYS}
            pair = {key: original[PILOT_TARGET + "." + key] for key in CODE_KEYS}
            audits[PILOT_TARGET] = audit_code_patch(pair, patch, pilot["arms"][arm]["statistics"]["stage_flips"])
            expected.update({PILOT_TARGET + "." + key: value for key, value in patch.items()})
            _require(set(audits) == set(TARGETS), "Complete assembly is missing a down projection.")
            model.load_state_dict(expected, strict=True)
            actual = model.state_dict()
            frozen_count = _assert_state_equal({key: actual[key] for key in frozen}, frozen,
                                               "All non-code source tensors remain exact")
            provenance = {"source_checkpoint_id": SOURCE_ID, "source_meta_sha256": source_meta_sha,
                          "feedback": arm == "feedback_on", "configuration": pilot["configuration"],
                          "pilot_report_sha256": pilot_sha, "worker_report_sha256": report["worker_report_sha256"],
                          "data_manifest_sha256": hashlib.sha256(json.dumps(pilot["data"], sort_keys=True).encode()).hexdigest(),
                          "code_sha256": {name: sha for pins in pin_sets for name, sha in pins.items()},
                          "reused_pilot_target": PILOT_TARGET, "updated_targets": list(TARGETS)}
            export = save_and_verify(model, output / arm / "final_model", source_meta, expected, provenance, tokenizer)
            _require(export["serialized_tensors_compared"] == export["fresh_tensors_compared"] == 3423,
                     "Full saved/fresh state verification is incomplete.")
            for key in ("user_linear_bits", "packed_linear_bits", "code_padding_bits", "outside_linear_bits",
                        "full_model_tensor_bytes", "components"):
                _require(export["bit_budget"][key] == source_budget[key], f"Code-only export changed storage: {key}.")
            report["arms"][arm] = {"export": export, "frozen_tensors_compared": frozen_count,
                                    "code_bank_count": 72, "updated_target_count": 36,
                                    "code_audits": audits, "full_downstream_evaluation_pending": True}
            _write(report_path, report)
            del pilot_state, patch, pair, expected, actual
            gc.collect()
        _require(_sha(pilot_path) == pilot_sha and [_sha(path) for path in worker_paths] == report["worker_report_sha256"]
                 and _sha(source / "checkpoint_meta.json") == source_meta_sha, "Input provenance changed during assembly.")
        for pins in pin_sets:
            _verify_pins(pins)
        report.update(status="verified", elapsed_seconds=time.perf_counter() - started)
        _write(report_path, report)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        _write(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--worker-dirs", type=Path, nargs=2, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.source, args.worker_dirs, args.pilot_dir, args.output)


if __name__ == "__main__":
    main()
