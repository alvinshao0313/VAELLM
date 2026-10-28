"""Evaluate one actual completed C/G/D native endpoint: full8, full21, original nine-term score."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result/qwen3_8b_mmlu65_20261004"
HERE = Path(__file__).resolve().parent
SNAPSHOT = RESULT / "rotation_factor_probe/source_snapshot_20261005_01"
sys.path.insert(0, str(SNAPSHOT))

from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.scoring import score_directory
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.model_inventory import model_inventory
from train_utils.checkpoint_bpw import audit_checkpoint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(native_report, output, gpu):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU driver; only children use the assigned GPU.")
    native = read(native_report)
    require(native["status"] == "completed" and native["exit_code"] == 0
            and native["profile"] in ("C", "G", "D"), "Require an actual completed native C/G/D report.")
    binding = native["saved_boundaries"][-1]
    require(binding["kind"] == "final_model", "Use the actual final save.")
    checkpoint = Path(binding["path"])
    meta = read(checkpoint / "checkpoint_meta.json")
    require(meta["checkpoint_id"] == binding["checkpoint_id"] and meta.get("lora_config") is None
            and not meta["pending_dense_targets"] and meta["converted_module_count"] == 252,
            "Require a complete rank0 model.")
    for name, key in (("checkpoint_meta.json", "metadata_sha256"), ("cat_runtime_state.pt", "runtime_sha256")):
        require(sha256(checkpoint / name) == binding[key], "Native save binding changed: " + name)
    state_path = checkpoint / meta["state_dict_file"]
    actual = state_path.stat()
    require(all(binding["weight_stat"][key] == value for key, value in
        dict(bytes=actual.st_size, inode=actual.st_ino, device=actual.st_dev, mtime_ns=actual.st_mtime_ns).items()),
        "Native weight binding changed.")
    api = module_from(RESULT / "qwen_aux_retention32_batch_20261011/evaluate.py", "_recompression_helpers")
    resources = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    devices = {int(line.split(",")[0]): line.split(",")[1].strip() for line in resources.splitlines()}
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    require(not any(line.split(",")[0].strip() == devices[gpu] for line in active.splitlines()), "Assigned GPU is occupied.")
    require(not output.exists(), "Never overwrite an evaluation.")
    output.mkdir(parents=True)
    report = dict(status="preparing", profile=native["profile"], gpu=gpu, pid=os.getpid(),
        start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        checkpoint_dir=str(checkpoint), checkpoint_id=meta["checkpoint_id"], native_report=str(native_report),
        started_unix=time.time(), stages={}, automatic_retry=False, automatic_resume=False,
        protocol=dict(max_linear_bpw=2.6, max_lora_rank=0, full8_samples=24742, longbench_samples=4750))
    report_path = output / "evaluation.json"
    write_json(report_path, report)

    def stage(name, argv, seconds, visible=""):
        command = ["timeout", "--signal=TERM", "--kill-after=20s", str(seconds) + "s", *argv]
        row = dict(command=command, started_unix=time.time(), gpu=visible)
        report["stages"][name] = row
        write_json(report_path, report)
        try:
            run_logged_stage(command, name=name, output_dir=output, env=api.environment(SNAPSHOT, visible))
        finally:
            row.update(finished_unix=time.time(), exit_code=(output / (name + ".exit")).read_text().strip()
                       if (output / (name + ".exit")).exists() else None)
            write_json(report_path, report)

    try:
        reference = read(api.REFERENCE_CONFIG)
        config = make_config(reference["archive"]["path"], kind="compressed", hf_path=meta["base_model_path"],
                             checkpoint=checkpoint, mode="full")
        require(all(config[key] == reference[key] for key in reference if key != "model"),
                "Keep the original shared archive, model policy and complete generation protocol.")
        budget = audit_checkpoint(checkpoint, max_bpw=2.6, max_lora_rank=0)
        require(budget["budget_passed"] and not budget["contract_violations"]
                and not budget["components"].get("residual_lora", {}).get("storage_bits", 0),
                "Actual logical storage including decoder/protection must pass rank0/2.6.")
        write_json(output / "bit_budget.json", budget)
        pins = {str(path): sha256(path) for path in [native_report, checkpoint / "checkpoint_meta.json",
            checkpoint / "cat_runtime_state.pt", HERE / "evaluate_endpoint.py", HERE / "longbench.py", *api.source_paths()]}
        config["model"]["recompression"] = dict(max_linear_bpw=2.6, max_lora_rank=0, bit_budget=budget, source_sha256=pins)
        config_path = output / "longbench_config.json"
        write_json(config_path, config)
        report.update(status="running", bit_budget=budget, source_sha256=pins)
        write_json(report_path, report)
        for name, tasks, batch in (("mmlu", "mmlu", "64"), ("seven", api.SEVEN, "32")):
            stage(name, [sys.executable, "-B", "-u", str(api.SAMPLE_EVAL), "--checkpoint_dir", str(checkpoint),
                "--strict", "--eval_lm_eval", "--eval_device", "cuda:0", "--tasks", tasks, "--num_fewshot", "0",
                "--lm_batch_size", batch, "--tokenizer_use_fast", "true", "--prewarm_group_size", "8",
                "--eval_log_dir", str(output / (name + "_logs")), "--output_json", str(output / (name + ".json"))], 3600, str(gpu))
        paths = [output / "mmlu.json", output / "seven.json"]
        full8 = collect_results(paths, model_inventory(meta["base_model_path"]), checkpoint=(checkpoint, meta))
        template = read(api.TEMPLATE)["nine_task_argv"]
        index = template.index("--baseline_results")
        baseline_paths = template[index + 1:index + 3]
        baseline = collect_results(baseline_paths, model_inventory(meta["base_model_path"]))
        require(full8["versions"] == baseline["versions"] and full8["sample_count"] == 24742,
                "Require the original complete downstream protocol.")
        for path, base, count in zip(paths, baseline_paths, (14042, 10700), strict=True):
            lm = read(path)["evals"]["lm_eval"]
            require(lm["configs"] == read(base)["configs"] and lm["sample_artifacts"]["total_count"] == count,
                    "Require complete matched configurations and raw per-question artifacts.")
            for row in lm["sample_artifacts"]["tasks"].values():
                artifact = Path(row["path"])
                require(artifact.is_relative_to(output) and sha256(artifact) == row["sha256"], "Raw sample binding changed.")
        write_json(output / "full8_summary.json", {"checkpoint_id": meta["checkpoint_id"], **full8})
        report["full8_mean_percent"] = full8["mean_percent"]
        write_json(report_path, report)
        stage("longbench", [sys.executable, "-B", "-u", str(HERE / "longbench.py"), "--config", str(config_path),
                            "--output", str(output / "longbench")], 86580, str(gpu))
        scores_path = output / "longbench/scores.json"
        scores = score_directory(config["archive"]["path"], output / "longbench", scores_path)
        require(scores["sample_count"] == 4750, "Require all4750 LongBench examples.")
        formula_dir = RESULT / "qwen_b32_mlp_s3_20261023"
        sys.path.insert(0, str(formula_dir))
        formula = module_from(formula_dir / "summarize_nine.py", "_recompression_nine_formula")
        nine = formula.summarize(base_model_path=meta["base_model_path"], checkpoint_dir=str(checkpoint),
            baseline_results=baseline_paths, compressed_results=list(map(str, paths)),
            longbench_manifest=config["archive"]["path"],
            baseline_longbench_scores=template[template.index("--baseline_longbench_scores") + 1],
            compressed_longbench_scores=str(scores_path))
        require(nine["quality_target_evaluated"] and nine["checkpoint_id"] == meta["checkpoint_id"]
                and abs(nine["baseline_nine_task_score"] - 68.04714491790848) < 1e-12, "Original nine-term reference changed.")
        write_json(output / "nine_task_acceptance.json", nine)
        check_hashes(pins); check_config(config); check_stat(config["model"]["weight_stat"])
        report.update(status="complete_full8_full21_nine", nine_task_score=nine["compressed_nine_task_score"], passed=nine["passed"])
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["finished_unix"] = time.time()
        write_json(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-report", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--output", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--gpu", type=int, choices=(5, 6, 7), required=True)
    args = parser.parse_args()
    print(evaluate(args.native_report, args.output, args.gpu), flush=True)
