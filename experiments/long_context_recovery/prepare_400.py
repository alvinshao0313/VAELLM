"""Prepare an unbound 400-step full-data run from verified 8K resource evidence."""
import argparse
import json
from pathlib import Path
import sys

from experiments.longbench_v1.common import check_hashes, file_stat, sha256, source_pins, write_json
from experiments.long_context_recovery.contract import SOURCE, SOURCE_ID, require
from experiments.long_context_recovery.data import read_manifest
from experiments.long_context_recovery.prepare import training_arguments


def prepare(manifest_path, output, resource_run):
    from experiments.long_context_recovery.train_400 import parse_configuration
    from experiments.qwen3_w25.subset_step_export_callback import stage_commands
    from experiments.w25_eval.subset_export import CODE_FILES
    import torch
    root = Path(__file__).resolve().parents[2]
    manifest_path, output, resource_run = map(lambda p: Path(p).resolve(), (manifest_path, output, resource_run))
    require(not output.exists(), "Use a new independent full-data output directory.")
    manifest = read_manifest(manifest_path)
    resource = json.loads((resource_run / "launch.json").read_text())
    proof = json.loads((resource_run / "verification.json").read_text())
    require(resource["status"] == proof["status"] == "verified"
            and resource["dispatcher_exit_code"] == 0
            and proof["native_vs_independent_tensors_compared"] == 3423,
            "Reuse only completed native 8K training/export evidence.")
    check_hashes(resource["source_sha256"])
    require(all(x["shape"] == [1, 8192] and x["response_non_special_tokens_shifted"] > 0
                for x in proof["batches"]) and len(proof["batches"]) == 4,
            "Short validation must contain four actual supervised 8K inputs.")
    argv = training_arguments(manifest_path, output, manifest["alias"])
    argv[argv.index("--save_steps") + 1] = "400"
    argv[argv.index("--save_total_limit") + 1] = "1"
    cfg, _hf, args, names, architecture, inventory = parse_configuration(argv)
    require(not torch.cuda.is_initialized(), "Preparation must remain CPU-only.")
    sources = source_pins()
    sources.update({str(root / name): sha256(root / name) for name in CODE_FILES})
    for directory in (root / "compressed_e2e_fintuning", Path(__file__).parent):
        sources.update({str(path): sha256(path) for path in directory.rglob("*.py")})
    for name in ("experiments/qwen3_w25/train_subset_400.py",
                 "experiments/qwen3_w25/subset_step_export_callback.py",
                 "experiments/qwen3_w25/aux_step_export_callback.py",
                 "experiments/qwen3_w25/step_export_callback.py",
                 "experiments/qwen3_w25/summarize_results.py",
                 "experiments/w25_eval/validate_decoder_subset_training.py"):
        sources[str(root / name)] = sha256(root / name)
    meta = json.loads((SOURCE / "checkpoint_meta.json").read_text())
    model_path = Path(meta["base_model_path"])
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    evidence_paths = [manifest_path, Path(manifest["source"]["path"]),
        Path(manifest["canonical_cpu_report"]["path"]), resource_run / "launch.json",
        resource_run / "verification.json", resource_run / "observations.json",
        resource_run / "step_0004/audit.json", SOURCE / "checkpoint_meta.json",
        SOURCE / meta["state_dict_file"], model_path / "config.json", model_path / "tokenizer_config.json",
        model_path / "tokenizer.json", model_path / "generation_config.json"]
    old_options, new_options = dict(zip(resource["training_argv"][::2], resource["training_argv"][1::2])), dict(zip(argv[::2], argv[1::2]))
    differences = {key: {"resource": old_options.get(key), "full_data": value}
                   for key, value in new_options.items() if old_options.get(key) != value}
    require(set(differences) == {"--long_context_manifest", "--run_root_dir", "--save_steps", "--save_total_limit"},
            "Unexpected science/config difference from the real 8K resource probe.")
    commands = stage_commands(checkpoint=output / "NATIVE_RUN/trainer_state/checkpoint-400",
                              source_dir=SOURCE, output_dir=output / "step_0400")
    for name, command, _allowed in commands:
        if name in ("export", "audit"):
            command[2] = "experiments.w25_eval.subset_" + name
        require("--lm_limit" not in command, "Formal evaluation must not inherit a diagnostic limit.")
    record = {"schema": "long_context_ov72_full_data_400_v1", "status": "prepared_not_scheduled",
        "gpu": None, "queued": False, "automatic_retry": False, "output_dir": str(output),
        "cwd": str(root), "python_executable": sys.executable,
        "source_checkpoint": str(SOURCE), "source_checkpoint_id": SOURCE_ID,
        "training_argv": argv, "argv": [sys.executable, "-u", "-m", "experiments.long_context_recovery.train_400", *argv],
        "source_sha256": sources, "input_evidence_sha256": {str(path): sha256(path) for path in evidence_paths},
        "hf_weight_stats": [file_stat(model_path / name) for name in sorted(set(index["weight_map"].values()))],
        "selected_targets": list(names), "expected_inventory": inventory, "architecture": architecture,
        "data_registration": cfg.data.data_registration,
        "environment": {"CUDA_VISIBLE_DEVICES": None, "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                        "PYTHONPATH": str(root), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "schedule": {"max_steps": args.max_steps, "warmup_steps": 20, "save_steps": 400,
                     "save_total_limit": 1, "stop_step": 400},
        "resource_validation": {"path": str(resource_run), "status": "verified", "gpu_short_repeated": False,
            "peak_allocated_bytes": max(x["peak_allocated_bytes"] for x in proof["steps"]),
            "stable_step_seconds": [x["elapsed_seconds"] for x in proof["steps"][1:]]},
        "argv_differences_from_resource": differences,
        "callback_difference": "Budget400Callback replaces the four-step observer; no data/resource selection carried over.",
        "full8_stage_templates": commands, "cuda_initialized": False,
        "current_goal_evaluation": "Eight tasks screen regression only. Complete candidate LongBench21 is required for nine-term acceptance.",
        "comparison_boundary": "LongAlpaca original SFT stream, max8192, B1 and OV72 decoder-only form one recovery recipe; not a single-variable length claim.",
        "experience_review": ["kd_and_data.md: native prefix truncation/skip retained; actual exposure recorded.",
            "experiment_design.md: OV restoration attribution is diagnostic and does not predict training gains."]}
    output.mkdir(parents=True)
    write_json(output / "launch.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource_run", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.manifest, args.output, args.resource_run)
    print(json.dumps({key: result[key] for key in ("status", "gpu", "schedule", "argv_differences_from_resource",
                      "expected_inventory", "cuda_initialized")}, ensure_ascii=False, indent=2))
