"""Isolated native subset export; full64/32 evaluation and private four-step diagnostics."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import subprocess
import time

from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback, stage_commands as head_stage_commands
from experiments.qwen3_w25.e2e_subset_export import inspect_step
from experiments.qwen3_w25.audit_e2e_decoder import persistent_inventory
from experiments.qwen3_w25.step_export_callback import run_logged_stage


LOG = logging.getLogger(__name__)


def stage_commands(*, checkpoint, source_dir, output_dir, diagnostic_eval_limit=None):
    """Reuse the fixed full evaluation shards, changing only export and audit."""
    commands = head_stage_commands(checkpoint=checkpoint, source_dir=source_dir, output_dir=output_dir,
                                   diagnostic_eval_limit=diagnostic_eval_limit)
    commands[0][1][2] = "experiments.qwen3_w25.e2e_subset_export"
    commands[1][1][2] = "experiments.qwen3_w25.e2e_subset_audit"
    commands[1][1].extend(["--training_checkpoint_dir", str(checkpoint)])
    return commands


class SubsetStepExportCallback(AuxStepExportCallback):
    def __init__(self, *, selected_targets, diagnostic_stop_step=None, **kwargs):
        super().__init__(**kwargs)
        if diagnostic_stop_step is not None and (self.diagnostic_eval_limit is None
                or diagnostic_stop_step != 4):
            raise ValueError("The private diagnostic stop is only four steps with limited inference.")
        self.selected_targets = tuple(selected_targets)
        self.diagnostic_stop_step = diagnostic_stop_step
        self.diagnostic = {"batches": [], "optimizer_steps": []}
        self._input_hook = None
        self._observed_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if self.diagnostic_stop_step is None:
            return control
        if args.max_steps != 2000 or args.save_steps != 2 or args.get_warmup_steps(args.max_steps) != 20:
            raise ValueError("Diagnostic must preserve the 2000-step schedule and warmup20, saving every2.")
        model = kwargs["model"]
        inventory, banks = persistent_inventory(model)
        expected = {key for name in self.selected_targets for key in inventory[name]}
        active = {name: value for name, value in model.named_parameters() if value.requires_grad}
        decoder = {name for name in active if not name.startswith("sparse_bit_tuning.")}
        if decoder != expected or any(value.dtype.is_floating_point and str(value.dtype) != "torch.float32"
                                      for value in active.values()):
            raise ValueError("Diagnostic trainables escaped selected FP32 decoder/score inventory.")
        score_elements = sum(active[name].numel() for name in set(active) - decoder)
        expected_score_elements = sum(math.prod(shape) for name in self.selected_targets for _key, shape in banks[name])
        if score_elements != expected_score_elements:
            raise ValueError("Diagnostic score parameters do not cover every selected code bit.")
        self.diagnostic["trainable_inventory"] = {
            "decoder_keys": sorted(decoder), "decoder_tensors": len(decoder),
            "decoder_elements": sum(active[name].numel() for name in decoder),
            "score_keys": sorted(set(active) - decoder),
            "score_elements": score_elements, "expected_active_bits": expected_score_elements,
            "selected_code_banks": sum(len(banks[name]) for name in self.selected_targets)}
        def observe_inputs(_module, positional, values):
            inputs = values.get("input_ids")
            if inputs is None and positional:
                inputs = positional[0]
            if inputs is not None:
                mask = values.get("attention_mask")
                self.diagnostic["batches"].append({"step": self._observed_step,
                    "shape": list(inputs.shape), "padded_tokens": inputs.numel(),
                    "attention_tokens": None if mask is None else int(mask.sum().item())})
        self._input_hook = model.register_forward_pre_hook(observe_inputs, with_kwargs=True)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._observed_step = int(state.global_step) + 1
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.diagnostic_stop_step is not None:
            optimizer = kwargs["optimizer"]
            while hasattr(optimizer, "optimizer"):
                optimizer = optimizer.optimizer
            main = getattr(optimizer, "main_optimizer", optimizer)
            moment_dtypes = sorted({str(value.dtype) for item in main.state.values()
                                   for key, value in item.items() if key in ("exp_avg", "exp_avg_sq")})
            self.diagnostic["optimizer_steps"].append({"step": int(state.global_step),
                "main_learning_rates_after_scheduler": [group["lr"] for group in main.param_groups],
                "main_moment_dtypes": moment_dtypes})
            if int(state.global_step) == self.diagnostic_stop_step:
                control.should_training_stop = True
        return control

    def close(self):
        if self._input_hook is not None:
            self._input_hook.remove()
            self._input_hook = None

    def on_train_end(self, args, state, control, **kwargs):
        self.close()
        return control

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step == self._last_step:
            return control
        checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{step}"
        meta, saved_state, _snapshot, source, _base = inspect_step(checkpoint)
        if (saved_state["global_step"] != step or source.resolve() != self.source_dir
                or tuple(f"model.layers.{layer}.mlp.down_proj" for layer in meta["target_layers"])
                != self.selected_targets):
            raise ValueError("Saved step/source does not match the callback boundary.")
        output = self.run_root_dir / f"step_{step:04d}"
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "running", "source_checkpoint": str(checkpoint), "source_checkpoint_id": meta["checkpoint_id"],
                  "global_step": step, "max_steps": int(args.max_steps),
                  "export_cuda_visible_devices": self.env["CUDA_VISIBLE_DEVICES"],
                  "diagnostic_eval_limit": self.diagnostic_eval_limit, "stages": {}}
        record_path = output / ("diagnostic.json" if self.diagnostic_eval_limit is not None else "step_export.json")
        started = time.monotonic()
        LOG.info("Pausing subset training at step %d for saved-step export/evaluation on GPU %s.",
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
        LOG.info("Subset step %d export/evaluation finished in %.1fs; resuming training.",
                 step, record["elapsed_seconds"])
        return control
