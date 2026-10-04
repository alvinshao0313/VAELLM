"""Pause final-norm training for isolated export and fixed full eight-task evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import time

from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback, stage_commands as head_stage_commands
from experiments.qwen3_w25.e2e_norm_export import inspect_step
from experiments.qwen3_w25.step_export_callback import run_logged_stage


LOG = logging.getLogger(__name__)


def stage_commands(*, checkpoint, source_dir, output_dir, diagnostic_eval_limit=None):
    """Reuse the fixed full evaluation shards, changing only export and audit."""
    commands = head_stage_commands(checkpoint=checkpoint, source_dir=source_dir, output_dir=output_dir,
                                   diagnostic_eval_limit=diagnostic_eval_limit)
    commands[0][1][2] = "experiments.qwen3_w25.e2e_norm_export"
    commands[1][1][2] = "experiments.qwen3_w25.e2e_norm_audit"
    commands[1][1].extend(["--training_checkpoint_dir", str(checkpoint)])
    return commands


class NormStepExportCallback(AuxStepExportCallback):
    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step == self._last_step:
            return control
        checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{step}"
        meta, saved_state, _snapshot, source, _base = inspect_step(checkpoint)
        if saved_state["global_step"] != step or source.resolve() != self.source_dir:
            raise ValueError("Saved step/source does not match the callback boundary.")
        output = self.run_root_dir / f"step_{step:04d}"
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "running", "source_checkpoint": str(checkpoint), "source_checkpoint_id": meta["checkpoint_id"],
                  "global_step": step, "max_steps": int(args.max_steps),
                  "export_cuda_visible_devices": self.env["CUDA_VISIBLE_DEVICES"],
                  "diagnostic_eval_limit": self.diagnostic_eval_limit, "stages": {}}
        record_path = output / ("diagnostic.json" if self.diagnostic_eval_limit is not None else "step_export.json")
        started = time.monotonic()
        LOG.info("Pausing final-norm training at step %d for saved-step export/evaluation on GPU %s.",
                 step, self.env["CUDA_VISIBLE_DEVICES"])
        try:
            query = subprocess.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
            if query.stdout.strip():
                raise RuntimeError(f"Export GPU is occupied: {query.stdout.strip()}.")
            record["export_gpu_idle_checked"] = True
            for name, command, allowed in stage_commands(checkpoint=checkpoint, source_dir=self.source_dir,
                    output_dir=output, diagnostic_eval_limit=self.diagnostic_eval_limit):
                record["stages"][name] = {"command": command}
                record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                record["stages"][name]["exit_code"] = run_logged_stage(
                    command, name=name, output_dir=output, env=self.env, allowed_exit_codes=allowed)
            record["status"] = "diagnostic_complete" if self.diagnostic_eval_limit is not None else "evaluated"
            self._last_step = step
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        LOG.info("Final-norm step %d export/evaluation finished in %.1fs; resuming training.",
                 step, record["elapsed_seconds"])
        return control
