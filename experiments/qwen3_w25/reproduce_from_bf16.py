"""Rebuild the measured Qwen3-8B success chain from dense BF16 weights."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile

from experiments.qwen3_w25.reproduction_inputs import prepare_format_launch, prepare_inputs
from experiments.qwen3_w25.reproduction_recipe import REPO, cat_command, training_command


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment(gpu, *, format_recovery=False):
    env = os.environ.copy()
    for key in ("FLASH_ATTENTION_DETERMINISTIC", "NVIDIA_TF32_OVERRIDE"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(REPO), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               WANDB_MODE="offline", OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES=str(gpu))
    if format_recovery:
        env.update(FLASH_ATTENTION_DETERMINISTIC="1", MKL_NUM_THREADS="4")
    return env


def fast_training_command(recipe, *, source, model, output, gpu):
    command = training_command(recipe, source_dir=source, hf_model=model, output_dir=output, export_gpu=gpu)
    command[2] = "experiments.qwen3_w25.train_reproduction_stage"
    index = command.index("--export_cuda_visible_devices")
    del command[index:index + 2]
    # Stop the last trajectory at its measured best point without shortening its scheduler.
    command.extend(["--finish_step", "1200" if recipe == "last_joint" else "2000"])
    return command


def check_gpus(gpus):
    for gpu in gpus:
        query = subprocess.run(["nvidia-smi", "--id=" + gpu, "--query-compute-apps=pid",
                                "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        if query.stdout.strip():
            raise RuntimeError(f"GPU {gpu} is occupied by PID(s) {query.stdout.strip()}; select two idle GPUs.")


class Pipeline:
    def __init__(self, root, model, gpus, inputs):
        self.root, self.model, self.gpus = root, model, gpus
        self.logs = root / "logs"
        self.logs.mkdir()
        self.record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                       "model_path": str(model), "gpus": gpus, "inputs": inputs, "stages": {},
                       "recipe": "BF16 -> CAT -> CE10/CE50 -> MLP assembly -> KL/ChatML256 -> attention assembly -> joint1200",
                       "last_schedule_steps": 2000, "last_finish_step": 1200}
        self.save()

    def save(self):
        write_json(self.root / "pipeline.json", self.record)

    def run_group(self, jobs):
        """Run only explicitly independent stages together, each in its own CUDA namespace."""
        check_gpus([str(gpu) for _name, _command, gpu, _format, _allowed in jobs])
        running, waiter = [], None
        try:
            for name, command, gpu, format_recovery, allowed in jobs:
                env = environment(gpu, format_recovery=format_recovery)
                log = (self.logs / f"{name}.log").open("x", encoding="utf-8")
                log.write(shlex.join(command) + "\n")
                log.flush()
                row = {"status": "running", "command": command, "gpu": gpu,
                       "log": str(self.logs / f"{name}.log"),
                       "started_at": datetime.now(timezone.utc).isoformat()}
                self.record["stages"][name] = row
                try:
                    process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    row["status"] = "launch_failed"
                    raise
                row["pid"] = process.pid
                running.append((name, process, log, allowed))
                print(f"[{name}] PID {process.pid}; log: {row['log']}", flush=True)
            self.save()
            waiter = ThreadPoolExecutor(max_workers=len(running))
            pending = {waiter.submit(process.wait): (name, process, log, allowed)
                       for name, process, log, allowed in running}
            for future in as_completed(pending):
                name, process, log, allowed = pending[future]
                code = future.result()
                log.close()
                (self.logs / f"{name}.exit").write_text(f"{code}\n", encoding="utf-8")
                self.record["stages"][name].update(status="completed" if code in allowed else "failed",
                    exit_code=code, completed_at=datetime.now(timezone.utc).isoformat())
                self.save()
                if code not in allowed:
                    raise RuntimeError(f"Stage {name} exited {code}; see {self.logs / (name + '.log')}.")
        finally:
            # Only terminate unfinished children started by this pipeline if this group fails.
            for name, process, log, allowed in running:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    self.record["stages"][name].update(status="interrupted", exit_code=process.returncode)
                    (self.logs / f"{name}.exit").write_text(f"{process.returncode}\n", encoding="utf-8")
                elif self.record["stages"][name]["status"] == "running":
                    self.record["stages"][name].update(
                        status="completed" if process.returncode in allowed else "failed", exit_code=process.returncode)
                    (self.logs / f"{name}.exit").write_text(f"{process.returncode}\n", encoding="utf-8")
                log.close()
            if waiter is not None:
                waiter.shutdown(wait=True)
            self.save()

    def run(self, name, command, *, gpu=None, allowed=(0,)):
        self.run_group([(name, command, self.gpus[0] if gpu is None else gpu, False, allowed)])

    def export(self, name, directory, source, *, subset=False):
        result = json.loads((directory / "stage_result.json").read_text())
        checkpoint = Path(result["training_checkpoint_dir"])
        selected = directory / f"step_{result['global_step']:04d}"
        selected.mkdir()
        final = selected / "final_model"
        module = "e2e_subset_export" if subset else "export_e2e_selected"
        self.run(name + "_export", [sys.executable, "-m", "experiments.qwen3_w25." + module,
                                   "--checkpoint_dir", str(checkpoint), "--output_dir", str(final)])
        audit = [sys.executable, "-m", "experiments.qwen3_w25." + ("e2e_subset_audit" if subset else "audit_e2e_decoder"),
                 "--source_dir", str(source), "--checkpoint_dir", str(final), "--output", str(selected / "audit.json")]
        audit.extend(["--training_checkpoint_dir", str(checkpoint)] if subset else ["--train_mode", "decoder"])
        self.run(name + "_audit", audit)
        return final

    def execute(self):
        root, model = self.root, self.model
        gpu0, gpu1 = self.gpus
        archive = root / "02_format_data"
        self.run("format_data", [sys.executable, "-m", "experiments.qwen3_w25.prepare_format_training_data",
                 "--source", self.record["inputs"]["task_source"],
                 "--source_manifest", self.record["inputs"]["task_manifest"],
                 "--hf_source", str(model), "--output", str(archive)])

        cat = root / "01_cat"
        self.run("cat", cat_command(hf_model=model, output_dir=cat))
        candidates = list(cat.glob("*/final_model/checkpoint_meta.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one completed CAT final_model under {cat}; found {len(candidates)}.")
        cat_model = candidates[0].parent
        ce10, ce50 = root / "03_ce10", root / "04_ce50"
        self.run_group([
            ("ce10_train", fast_training_command("ce10", source=cat_model, model=model, output=ce10, gpu=gpu1), gpu0, False, (0,)),
            ("ce50_train", fast_training_command("ce50", source=cat_model, model=model, output=ce50, gpu=gpu0), gpu1, False, (0,)),
        ])
        ce10_model = self.export("ce10", ce10, cat_model)
        ce50_model = self.export("ce50", ce50, cat_model)
        combined = root / "05_ce50_attention_ce10_mlp" / "final_model"
        self.run("assemble_mlp", [sys.executable, "-m", "experiments.qwen3_w25.assemble_categories",
                 "--base", str(ce50_model), "--donor", str(ce10_model),
                 "--categories", "gate_proj,up_proj,down_proj", "--output", str(combined)])

        decoder = root / "06_decoder_kl"
        format_root = root / "07_chatml_joint256"
        format_root.mkdir()
        formal = training_command("formal_reference", source_dir=combined, hf_model=model,
                                  output_dir=format_root / "formal_configuration_only", export_gpu=gpu0)
        format_env = environment(gpu1, format_recovery=True)
        launch = prepare_format_launch(root=format_root, source_dir=combined, hf_model=model,
                    archive_dir=archive, environment=format_env, formal_command=formal)
        self.run_group([
            ("decoder_train", fast_training_command("decoder", source=combined, model=model, output=decoder, gpu=gpu1), gpu0, False, (0,)),
            ("chatml256_train", [sys.executable, "-m", "experiments.qwen3_w25.train_format_recovery",
                                "--launch_record", str(launch)], gpu1, True, (0,)),
        ])
        decoder_model = self.export("decoder", decoder, combined)
        chatml_model = format_root / "final_model"
        self.run("chatml256_export", [sys.executable, "-m", "experiments.qwen3_w25.export_kl_joint",
                 "--diagnostic", str(format_root / "format_observations" / "diagnostic.json"), "--output", str(chatml_model)])
        source = root / "08_joint_mlp_decoder_attention" / "final_model"
        self.run("assemble_attention", [sys.executable, "-m", "experiments.qwen3_w25.assemble_runtime_categories",
                 "--base", str(chatml_model), "--donor", str(decoder_model),
                 "--categories", "q_proj,k_proj,v_proj,o_proj", "--output", str(source)])

        last = root / "09_joint1200"
        self.run("last_train", fast_training_command("last_joint", source=source, model=model, output=last, gpu=gpu1))
        final = self.export("last", last, source, subset=True)
        evaluation = root / "evaluation"
        evaluation.mkdir()
        jobs = []
        for name, tasks, batch, gpu in (("mmlu", "mmlu", "64", gpu0),
                ("seven", "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa", "32", gpu1)):
            command = [sys.executable, "tools/cat_eval.py", "--checkpoint_dir", str(final), "--strict",
                       "--eval_lm_eval", "--eval_device", "cuda:0", "--tasks", tasks,
                       "--num_fewshot", "0", "--lm_batch_size", batch, "--tokenizer_use_fast", "true",
                       "--prewarm_group_size", "8", "--eval_log_dir", str(evaluation / (name + "_logs")),
                       "--output_json", str(evaluation / (name + ".json"))]
            jobs.append((name, command, gpu, False, (0,)))
        self.run_group(jobs)
        self.run("acceptance", [sys.executable, "-m", "experiments.qwen3_w25.summarize_results",
                 "--checkpoint_dir", str(final), "--results", str(evaluation / "mmlu.json"),
                 str(evaluation / "seven.json"), "--output", str(root / "acceptance.json")], allowed=(0, 1))
        acceptance = json.loads((root / "acceptance.json").read_text())
        (root / "final_model").symlink_to(final.relative_to(root), target_is_directory=True)
        self.record.update(status="passed" if acceptance["passed"] else "not_met",
                           final_model=str(final), mean_percent=acceptance["mean_percent"],
                           completed_at=datetime.now(timezone.utc).isoformat())
        self.save()
        print(json.dumps({"status": self.record["status"], "final_model": str(root / "final_model"),
                          "mean_percent": acceptance["mean_percent"],
                          "user_linear_bpw": acceptance["bit_budget"]["user_linear_bpw"]}, ensure_ascii=False), flush=True)
        return 0 if acceptance["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True, help="Original dense BF16 Qwen3-8B directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Fresh root; never overwritten or resumed.")
    parser.add_argument("--gpus", required=True, help="Two distinct physical GPU indices, e.g. 0,1.")
    parser.add_argument("--check_inputs", action="store_true", help="Only check local input files and hashes on CPU.")
    args = parser.parse_args()
    gpus = [part.strip() for part in args.gpus.split(",")]
    if len(gpus) != 2 or any(not part.isdecimal() for part in gpus) or len({int(part) for part in gpus}) != 2:
        parser.error("--gpus requires two different physical GPU indices.")
    if Path.cwd().resolve() != REPO:
        parser.error(f"Run from {REPO}; the canonical dataset aliases are relative to the repository.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ:
        parser.error("Run one ordinary Python process, not torchrun or a distributed launcher.")
    model, root = args.model_path.resolve(), args.output_dir.resolve()
    if args.check_inputs:
        with tempfile.TemporaryDirectory(prefix="qwen3_bf16_inputs_") as temporary:
            result = prepare_inputs(Path(temporary), model)
            print(json.dumps({"status": "inputs_checked", "model_path": str(model),
                              "task_source": result["task_source"], "versions": result["versions"],
                              "scope": "CPU file/config checks only; no GPU training or evaluation"}, ensure_ascii=False, indent=2))
        return 0
    if root.exists():
        parser.error(f"Output already exists: {root}; use a new directory.")
    root.mkdir(parents=True)
    pipeline, code = None, 2
    try:
        inputs = prepare_inputs(root, model)
        pipeline = Pipeline(root, model, gpus, inputs)
        code = pipeline.execute()
        return code
    except BaseException as exc:
        failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if pipeline is not None:
            pipeline.record.update(failure)
            pipeline.save()
        else:
            write_json(root / "pipeline.json", failure)
        raise
    finally:
        (root / "run.exit").write_text(f"{code}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
