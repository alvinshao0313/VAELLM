"""Run isolated native LoRA export and fixed evaluation while training is paused."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import time

from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback, stage_commands as fixed_stage_commands
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.w25_eval.lora_state import inspect_step


LOG = logging.getLogger(__name__)


def stage_commands(*, checkpoint, source_dir, output_dir, diagnostic_eval_limit=None):
    commands = fixed_stage_commands(checkpoint=checkpoint, source_dir=source_dir, output_dir=output_dir,
                                    diagnostic_eval_limit=diagnostic_eval_limit)
    commands[0][1][2] = "experiments.w25_eval.lora_export"
    commands[1][1][2] = "experiments.w25_eval.lora_audit"
    commands[1][1].extend(["--training_checkpoint_dir", str(checkpoint)])
    if diagnostic_eval_limit is None:
        commands[-1][1][2] = "experiments.w25_eval.summarize_lora_results"
        commands[-1][1].extend(["--audit", str(Path(output_dir) / "audit.json"),
                                  "--minimum_mean_percent", "69.5", "--minimum_mmlu_percent", "65"])
    return commands


class LoraStepExportCallback(AuxStepExportCallback):
    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step == self._last_step:
            return control
        checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{step}"
        meta, saved, _snapshot, source, _base, _architecture = inspect_step(checkpoint)
        if saved["global_step"] != step or source.resolve() != self.source_dir:
            raise ValueError("Saved LoRA step/source differs from callback boundary.")
        output = self.run_root_dir / f"step_{step:04d}"
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "running", "source_checkpoint": str(checkpoint), "source_checkpoint_id": meta["checkpoint_id"],
            "global_step": step, "max_steps": int(args.max_steps), "export_cuda_visible_devices": self.env["CUDA_VISIBLE_DEVICES"],
            "diagnostic_eval_limit": self.diagnostic_eval_limit, "stages": {}}
        path = output / ("diagnostic.json" if self.diagnostic_eval_limit is not None else "step_export.json")
        started = time.monotonic()
        try:
            query = subprocess.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
            if query.stdout.strip():
                raise RuntimeError(f"Export GPU is occupied: {query.stdout.strip()}.")
            record["export_gpu_idle_checked"] = True
            for name, command, allowed in stage_commands(checkpoint=checkpoint, source_dir=source,
                    output_dir=output, diagnostic_eval_limit=self.diagnostic_eval_limit):
                record["stages"][name] = {"command": command}
                path.write_text(json.dumps(record, indent=2) + "\n")
                record["stages"][name]["exit_code"] = run_logged_stage(
                    command, name=name, output_dir=output, env=self.env, allowed_exit_codes=allowed)
            record["status"] = "diagnostic_complete" if self.diagnostic_eval_limit is not None else "evaluated"
            self._last_step = step
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            path.write_text(json.dumps(record, indent=2) + "\n")
        LOG.info("LoRA step %d independent export/evaluation finished; resuming training.", step)
        return control
