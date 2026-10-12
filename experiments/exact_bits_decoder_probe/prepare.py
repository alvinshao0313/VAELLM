"""CPU pairing for fixed exact64 assignments followed by the existing decoder500."""
from __future__ import annotations

import argparse
import ast
import gc
import json
from pathlib import Path
import sys

import torch

from experiments.exact_bits_decoder_probe.replay import replay
from experiments.joint_reconstruction_probe.core import require
from experiments.longbench_v1.common import check_hashes, sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-weight-root", type=Path, required=True)
    parser.add_argument("--neighbor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    full, neighbor, output = (p.resolve() for p in
        (args.full_weight_root, args.neighbor_root, args.output_root))
    require(not (output / "preparation.json").exists() and not (output / "run").exists(),
            "Do not overwrite a preparation or run.")
    old = read(full / "preparation.json")
    prior = read(neighbor / "preparation.json")
    control = read(full / "formal/report.json")
    finite = read(neighbor / "run/report.json")
    require(control["status"] == finite["status"] == "completed"
            and (full / "formal.exit").read_text().strip() == "0"
            and (neighbor / "run.exit").read_text().strip() == "0", "Both previous experiments must have completed.")
    for root, report_path in ((full, full / "formal/report.json"), (neighbor, neighbor / "run/report.json")):
        review = read(root / "review.json")
        require(review["status"] == "verified" and review["report_sha256"] == sha256(report_path)
                and review["preparation_sha256"] == sha256(root / "preparation.json"),
                "Prior independent review or its source changed.")
    require(old["data_dir"] == prior["data_dir"], "Activation and original weight archives differ.")
    check_hashes(prior["source_sha256"])
    check_hashes(prior["input_sha256"])
    sources, inputs = dict(prior["source_sha256"]), dict(prior["input_sha256"])
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        ast.parse(path.read_text())
        sources[str(path)] = sha256(path)
    for path in (full / "formal/report.json", full / "formal.exit", full / "review.json",
                 neighbor / "preparation.json", neighbor / "run/report.json", neighbor / "run.exit",
                 neighbor / "review.json"):
        inputs[str(path)] = sha256(path)
    torch.set_num_threads(4)
    cases = {}
    for model in ("qwen", "llama"):
        info, c, n = prior["models"][model], control["cases"][model], finite["cases"][model]
        arm = c["arms"]["decoder"]
        require(arm["status"] == "completed" and arm["steps"] == arm["schedule_steps"] == 500
                and arm["joint"] is False and arm["stage_final_net_xor"] == [0, 0],
                "Reference must be the completed original-code full-matrix decoder500 control.")
        require(info["initial_payload_file"] == c["initial_payload_file"]
                and info["initial_payload_sha256"] == c["initial_payload_file_sha256"], "Initial payload source differs.")
        for path, expected in ((info["initial_payload_file"], info["initial_payload_sha256"]),
                               (n["selection_file"], n["selection_file_sha256"])):
            require(sha256(path) == expected, "Initial payload or selected codes changed.")
            inputs[path] = expected
        diagnostic = torch.load(info["initial_payload_file"], map_location="cpu", weights_only=True, mmap=True)
        choices = torch.load(n["selection_file"], map_location="cpu", weights_only=True)
        layer, evidence = replay(diagnostic, choices, c["arms"]["decoder"], n)
        cases[model] = {"initial_payload_file": info["initial_payload_file"],
            "selection_file": n["selection_file"], "model_type": n["model_type"],
            "source_checkpoint_id": n["checkpoint_id"], "cpu_pairing": evidence,
            "control_elapsed_seconds": c["arms"]["decoder"]["elapsed_seconds"],
            "control_peak_allocated_bytes": c["arms"]["decoder"]["peak_allocated_bytes"],
            "control_peak_reserved_bytes": c["arms"]["decoder"]["peak_reserved_bytes"]}
        del diagnostic, choices, layer
        gc.collect()
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    record = {"status": "cpu_ready", "python": sys.executable, "torch": torch.__version__,
        "source_sha256": sources, "input_sha256": inputs, "cases": cases,
        "full_weight_report": str(full / "formal/report.json"),
        "neighbor_report": str(neighbor / "run/report.json"), "data_dir": prior["data_dir"],
        "output_root": str(output), "cuda_initialized": False, "new_training_arms": 2,
        "steps": 500, "schedule_steps": 500, "seed": 31, "joint": False,
        "recipe": "Unchanged full_weight_bits_probe.core.train decoder branch; only its initial packed banks differ.",
        "comparison_budget": "Same 500 decoder updates; candidate additionally used the already completed exact64 search. Not equal total compute.",
        "cost_estimate": {"training_seconds_from_controls": sum(c["control_elapsed_seconds"] for c in cases.values()),
            "estimated_total_seconds": [60, 180], "suggested_available_window_seconds": 600,
            "peak_reserved_bytes_from_control": max(c["control_peak_reserved_bytes"] for c in cases.values()),
            "wall_budget_seconds": 600, "boundary": "Estimate from completed identical decoder geometry; no automatic wait or retry."},
        "validation_reuse": "Reuse actual fullweight two-step and completed decoder500, plus exact64 compact replay. No additional mechanism run, search, or control retraining.",
        "command": [sys.executable, "-m", "experiments.exact_bits_decoder_probe.run", "--preparation",
                    str(output / "preparation.json")]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preparation.json", record)
    print(json.dumps({"status": record["status"], "sha256": sha256(output / "preparation.json"),
        "source_count": len(sources), "input_count": len(inputs), "cpu_pairing": cases}, ensure_ascii=False))


if __name__ == "__main__":
    main()
