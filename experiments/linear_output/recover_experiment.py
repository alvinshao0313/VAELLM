"""Recover the failed shard and finish the authorized full W2 A/B experiment.

One process per GPU. Existing successful output shards are immutable inputs.
Every subprocess exit and artifact is checked; no output directory is deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
TASKS = "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"
MIX = "edgerazor_ii_7m=0.614,edgerazor_ii_gen=0.121,edgerazor_tulu=0.050,edgerazor_am=0.115,vaellm_eval_task=0.100"
RANGES = [(0, 9), (9, 18), (18, 27), (27, 36)]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def check_worker(path):
    m = json.loads((Path(path) / "manifest.json").read_text())
    if m["status"] != "COMPLETE" or len(m["records"]) != 63:
        raise RuntimeError(f"Incomplete worker: {path}")
    if {r["module"] for r in m["records"]} != set(m["modules"]):
        raise RuntimeError(f"Inconsistent worker module list: {path}")
    for rec in m["records"]:
        if rec["steps"] != 5000 or rec["code_payload_bpw"] != 2.0:
            raise RuntimeError(f"Wrong training budget/bit rate: {rec['module']}")
        packed = Path(path) / "linears" / rec["module"] / "packed"
        if not (packed / "checkpoint_meta.json").is_file():
            raise RuntimeError(f"Missing checkpoint: {packed}")


def training_command(path, objective, lo, hi):
    return [sys.executable, "-u", "-m", "experiments.linear_output.run_shared",
            "--model_path", "Qwen/Qwen3-8B", "--output_dir", str(path),
            "--objective", objective, "--steps", "5000", "--batch_size", "8",
            "--codebook_bits", "64", "--codebook_dim", "32", "--residual_stages", "1",
            "--seed", "31", "--data_seed", "31", "--vae_chunk_vectors", "262144",
            "--calibration_microbatch_size", "1", "--layer_start", str(lo), "--layer_end", str(hi),
            "--dataset_mix", MIX, "--dataset_task", "sft", "--model_max_length", "1024",
            "--dynamic_padding", "true", "--log_every", "100"]


def evaluation_summary(log_path):
    text = Path(log_path).read_text(errors="replace")
    marker = "Evaluation summary:\n"
    if "All evaluations completed." not in text or marker not in text:
        raise RuntimeError(f"Evaluation did not finish: {log_path}")
    summary, _ = json.JSONDecoder().raw_decode(text.rsplit(marker, 1)[1].lstrip())
    lm = summary["evals"]["lm_eval"]
    if set(lm["tasks"]) != set(TASKS.split(",")) or lm["limit"] is not None:
        raise RuntimeError("Evaluation task coverage/limit differs from the full experiment")
    if lm["num_fewshot"] != 0:
        raise RuntimeError("Evaluation fewshot mismatch")
    metrics = lm["task_metrics"]
    if any(not isinstance(metrics.get(t), (float, int)) or not math.isfinite(metrics[t])
           for t in TASKS.split(",")):
        raise RuntimeError("Missing/nonfinite evaluation metric")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    root = Path(args.root).resolve()
    old = REPO / "result/linear_output/full_output_w2_v5"
    output_workers = [root / "output/worker_0_9"] + [old / f"worker_{a}_{b}" for a, b in RANGES[1:]]
    weight_workers = [root / f"weight/worker_{a}_{b}" for a, b in RANGES]
    output_model = old / "final_model"
    weight_model = root / "weight/final_model"
    for w in output_workers[1:]:
        check_worker(w)
    for path in [root, output_model]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    jobs = {"output_0_9": dict(gpu=4, path=str(output_workers[0]),
                command=training_command(output_workers[0], "linear_output_mse", 0, 9))}
    for gpu, (a, b), w in zip(range(4, 8), RANGES, weight_workers):
        jobs[f"weight_{a}_{b}"] = dict(gpu=gpu, path=str(w), command=training_command(w, "weight_mse", a, b))
    for job in jobs.values():
        job["status"] = "PENDING"
    if args.dry_run:
        print(json.dumps({"jobs": jobs, "output_workers": list(map(str, output_workers)),
                          "tasks": TASKS, "output_model": str(output_model)}, indent=2))
        return
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    source_hashes = {str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in (REPO / "experiments/linear_output").glob("*.py")}
    state = {"status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "jobs": jobs, "source_sha256": source_hashes, "output_models": [str(output_model), str(weight_model)]}
    env = dict(os.environ, PYTHONPATH=str(REPO), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    running, done, failed = {}, set(), set()

    def persist():
        write_json(root / "status.json", state)

    def launch(name):
        job = jobs[name]
        handle = (root / "logs" / f"{name}.log").open("w")
        proc = subprocess.Popen(job["command"], cwd=REPO,
                env=dict(env, CUDA_VISIBLE_DEVICES=str(job["gpu"])),
                stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        running[name] = (proc, handle)
        job.update(status="RUNNING", pid=proc.pid)
        persist()

    def merge_eval(label, workers, model):
        state[label + "_postprocess"] = "RUNNING"
        persist()
        with (root / "logs" / f"{label}_merge.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "experiments.linear_output.merge_workers",
                "--model_path", "Qwen/Qwen3-8B", "--workers", *map(str, workers),
                "--output_dir", str(model), "--expected_target_count", "252"],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log_path = root / "logs" / f"{label}_eval.log"
        with log_path.open("w") as log:
            subprocess.run([sys.executable, "tools/cat_eval.py", "--checkpoint_dir", str(model),
                "--eval_linear_mse", "--eval_lm_eval", "--eval_hif4_act", "false",
                "--tasks", TASKS, "--num_fewshot", "0", "--lm_batch_size", "8",
                "--eval_device", "cuda", "--eval_log_dir", str(root / f"{label}_eval")],
                cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES="4"),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        result = evaluation_summary(log_path)
        write_json(root / f"{label}_evaluation.json", result)
        state[label + "_postprocess"] = "COMPLETE"
        persist()

    try:
        launch("output_0_9")
        for a, b in RANGES[1:]:
            launch(f"weight_{a}_{b}")
        output_processed = False
        while running:
            for name, (proc, handle) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                handle.close()
                del running[name]
                jobs[name]["exit_code"] = code
                try:
                    if code:
                        raise RuntimeError(f"Exit code {code}")
                    check_worker(jobs[name]["path"])
                    jobs[name]["status"] = "COMPLETE"
                    done.add(name)
                except Exception as exc:
                    jobs[name].update(status="FAILED", error=str(exc))
                    failed.add(name)
                persist()
            if "output_0_9" in done and not output_processed:
                try:
                    merge_eval("output", output_workers, output_model)
                except Exception as exc:
                    state["output_postprocess"] = "FAILED"
                    state["output_error"] = repr(exc)
                    failed.add("output_postprocess")
                    persist()
                output_processed = True
                launch("weight_0_9")
            if running:
                time.sleep(30)
        if all(f"weight_{a}_{b}" in done for a, b in RANGES):
            try:
                merge_eval("weight", weight_workers, weight_model)
            except Exception as exc:
                state.update(weight_postprocess="FAILED", weight_error=repr(exc))
                persist()
                raise
        if failed:
            raise RuntimeError(f"Failed stages: {sorted(failed)}; see logs and status.json")
        paired_records = []
        for workers in (output_workers, weight_workers):
            paired_records.append({r["module"]: r for w in workers
                for r in json.loads((w / "manifest.json").read_text())["records"]})
        for name, rec in paired_records[0].items():
            control = paired_records[1][name]
            for key in ("initial_state_sha256", "source_weight_sha256", "steps", "batch_size", "residual_stages"):
                if rec[key] != control[key]:
                    raise RuntimeError(f"A/B pairing mismatch: {name}, {key}")
        state["paired_initialization_and_budget_check"] = "PASSED"
        output = json.loads((root / "output_evaluation.json").read_text())
        weight = json.loads((root / "weight_evaluation.json").read_text())
        rows = ["# W2 output-MSE / weight-MSE comparison", "",
                "All 252 target Linears compressed, 5000 updates, batch 8, one 64/32 stage.", "",
                "| Task | Output objective (%) | Weight objective (%) | Difference (pp) |",
                "|---|---:|---:|---:|"]
        diffs = []
        for task in TASKS.split(","):
            a, b = (r["evals"]["lm_eval"]["task_metrics"][task] * 100 for r in (output, weight))
            ka, kb = (r["evals"]["lm_eval"]["task_metric_keys"][task] for r in (output, weight))
            if ka != kb:
                raise RuntimeError(f"Metric mismatch for {task}: {ka} vs {kb}")
            rows.append(f"| {task} | {a:.3f} | {b:.3f} | {a-b:+.3f} |")
            diffs.append(a-b)
        rows += ["", f"Unweighted mean difference over eight tasks: {sum(diffs)/len(diffs):+.3f} pp.",
                 "", "This A/B does not establish superiority to a CAT model with different training/recovery settings."]
        (root / "COMPARISON.md").write_text("\n".join(rows) + "\n")
        state["status"] = "COMPLETE"
    except Exception as exc:
        state.update(status="FAILED", error=repr(exc))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
