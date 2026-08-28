import argparse
import logging
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer, TrainerCallback

try:
    from transformers.trainer_callback import ProgressCallback
except Exception:  # pragma: no cover
    ProgressCallback = None

from compressed_e2e_fintuning.teacher_targets import (
    StudentHiddenCollector,
    TeacherHiddenTargetCollector,
    TeacherTargetBatch,
    copy_detached_tensor_to_cpu,
    resolve_pre_mlp_modules,
)
from e2e_common.dense_loss import (
    compute_dense_loss_from_logits,
    compute_dense_loss_from_offloaded_teacher,
    get_output_logits,
)
from train_utils.distill_losses import (
    DistillTokenRegions,
    build_distill_token_regions,
    compute_teacher_entropy_mean_and_gamma,
    is_eakld_top_loss,
)

# Rank-local EAKLD telemetry keys (no all_reduce; logging-rank microbatch stats).
_EAKLD_WEIGHTED_KEYS = (
    "teacher_entropy_mean",
    "gamma_reverse",
    "lambda_forward",
    "forward_kl",
    "reverse_kl",
    "eakld_total",
)
from train_utils.distill_token_stats import DistillTokenStatsAccumulator
from train_utils.lora_training import (
    _capture_pre_mlp_hiddens_from_modules,
    _compute_named_pre_mlp_hidden_alignment_loss,
    build_distill_hidden_layer_weights,
    compute_distill_hidden_alignment_loss,
    compute_selected_distill_hidden_alignment_loss,
    is_adaptive_hidden_alignment_layer_weighting,
    parse_distill_hidden_alignment_layer_weighting,
)


def _resolve_e2e_pre_mlp_capture_modules(model: nn.Module):
    return tuple(
        (f"model.layers.{layer_id}.post_attention_layernorm", module)
        for layer_id, module in enumerate(resolve_pre_mlp_modules(model))
    )


class E2ETrainerLogCallback(TrainerCallback):
    """Write Trainer metrics into the run FileHandler, matching LoRA distill logging."""

    def __init__(self, *, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not bool(getattr(state, "is_world_process_zero", True)):
            return
        if not logs:
            return
        values = dict(logs)
        values.pop("total_flos", None)
        ordered_keys = (
            "loss",
            "distill_loss",
            "hidden_loss",
            "pre_mlp_hidden_loss",
            "train_loss",
            "eval_loss",
            "learning_rate",
            "grad_norm",
            "epoch",
        )
        parts = []
        for key in ordered_keys:
            if key in values:
                parts.append(f"{key}={values.pop(key)}")
        for key in sorted(values):
            parts.append(f"{key}={values[key]}")
        if not parts:
            return
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            fn="",
            lno=0,
            msg="E2E train: step=%s %s",
            args=(str(getattr(state, "global_step", "unknown")), " ".join(parts)),
            exc_info=None,
        )
        for handler in list(getattr(self.logger, "handlers", [])):
            if not isinstance(handler, logging.FileHandler):
                continue
            if record.levelno < handler.level:
                continue
            handler.handle(record)


class _QuietProgressCallback(ProgressCallback if ProgressCallback is not None else object):
    def on_log(self, args, state, control, logs=None, **kwargs):
        return


class E2EDistillTokenStatsCallback(TrainerCallback):
    """Mirror _LoraDistillTokenStatsCallback for VAEDecoderE2ETrainer.

    Same boundary / window_start_step / reduce-before-rank0 / resume semantics
    as category distillation; only the log prefix differs ("E2E token stats").
    """

    def __init__(self, *, trainer, logger):
        self._trainer = trainer
        self._logger = logger
        self.window_start_step = None

    def on_step_end(self, args, state, control, **kwargs):
        logging_steps = getattr(state, "logging_steps", None)
        if not isinstance(logging_steps, int) or logging_steps <= 0:
            raise ValueError(
                f"state.logging_steps must be a positive integer, got {logging_steps!r}."
            )

        global_step = int(getattr(state, "global_step", 0))
        if self.window_start_step is None:
            self.window_start_step = global_step

        if global_step <= 0 or global_step % logging_steps != 0:
            return

        # Reduce across ranks before checking rank0 so all ranks participate in
        # the collective; only rank0 writes the resulting log line.
        stats = self._trainer.distill_token_stats.consume_global(self._trainer.accelerator)
        window_optimizer_steps = global_step - self.window_start_step + 1
        self.window_start_step = global_step + 1

        if stats is None:
            return

        if not bool(getattr(state, "is_world_process_zero", True)):
            return

        _log_e2e_trainer_message_to_file_handlers(
            self._logger,
            "E2E token stats: step=%s window_optimizer_steps=%d avg_prompt_tokens=%.4f avg_response_tokens=%.4f global_samples=%d",
            str(global_step),
            int(window_optimizer_steps),
            float(stats.avg_prompt_tokens_per_sample),
            float(stats.avg_response_tokens_per_sample),
            int(stats.global_samples),
        )


def _log_e2e_trainer_message_to_file_handlers(logger, message: str, *args) -> None:
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        fn="",
        lno=0,
        msg=message,
        args=args,
        exc_info=None,
    )
    for handler in list(getattr(logger, "handlers", [])):
        if not isinstance(handler, logging.FileHandler):
            continue
        if record.levelno < handler.level:
            continue
        handler.handle(record)


def replace_progress_log_callback(trainer):
    if ProgressCallback is None:
        return trainer
    callback_handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(callback_handler, "callbacks", None)
    if not isinstance(callbacks, list):
        return trainer
    for idx, callback in enumerate(callbacks):
        if isinstance(callback, ProgressCallback) and not isinstance(callback, _QuietProgressCallback):
            callbacks[idx] = _QuietProgressCallback()
    return trainer


def build_vae_hidden_layer_weights(
    *,
    num_layers: int,
    layer_weighting: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return build_distill_hidden_layer_weights(
        num_layers=int(num_layers),
        layer_weighting=str(layer_weighting),
        device=device,
        dtype=dtype,
    )


def compute_vae_hidden_alignment_loss(
    *,
    teacher_hidden_states,
    student_hidden_states,
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    loss_device: torch.device,
    eps: float = 1e-6,
) -> torch.Tensor:
    loss = compute_distill_hidden_alignment_loss(
        teacher_hidden_states=teacher_hidden_states,
        student_hidden_states=student_hidden_states,
        attention_mask=attention_mask,
        layer_weighting=str(layer_weighting),
        eps=float(eps),
    )
    return loss.to(device=torch.device(loss_device))


def _causal_lm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].to(device=shift_logits.device).contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, int(shift_logits.shape[-1])),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compute_choice_scores_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"choice logits must have shape [B, C, L, V], got {tuple(logits.shape)}.")
    if labels.ndim != 3:
        raise ValueError(f"choice labels must have shape [B, C, L], got {tuple(labels.shape)}.")
    if tuple(logits.shape[:3]) != tuple(labels.shape):
        raise ValueError(
            f"choice logits/labels shape mismatch: {tuple(logits.shape[:3])} vs {tuple(labels.shape)}."
        )

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].to(device=shift_logits.device).contiguous()
    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * valid_mask.to(dtype=token_log_probs.dtype)).sum(dim=-1)


def compute_choice_kd_loss_from_scores(
    *,
    student_scores: torch.Tensor,
    teacher_scores: Optional[torch.Tensor],
    answer_index: torch.Tensor,
    loss_type: str,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    norm = str(loss_type or "").strip().lower()
    if norm not in {"choice_kd", "choice_kd_ce"}:
        raise ValueError(f"Unsupported choice KD loss_type={loss_type!r}.")
    if teacher_scores is None:
        raise ValueError(f"loss_type={norm} requires teacher_scores.")
    if student_scores.ndim != 2 or teacher_scores.ndim != 2:
        raise ValueError("student_scores and teacher_scores must have shape [B, C].")
    if tuple(student_scores.shape) != tuple(teacher_scores.shape):
        raise ValueError(
            f"student/teacher choice score shape mismatch: {tuple(student_scores.shape)} vs {tuple(teacher_scores.shape)}."
        )

    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must satisfy 0 <= alpha <= 1, got {alpha}.")

    teacher_scores = teacher_scores.detach().to(device=student_scores.device)
    answer_index = answer_index.to(device=student_scores.device, dtype=torch.long)
    kd_loss = F.kl_div(
        F.log_softmax(student_scores.float() / temperature, dim=-1),
        F.softmax(teacher_scores.float() / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)
    if norm == "choice_kd":
        return kd_loss
    ce_loss = F.cross_entropy(student_scores.float(), answer_index)
    return ce_loss * (1.0 - alpha) + kd_loss * alpha


class VAEDecoderE2ETrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_type: str = "sft",
        teacher_model: Optional[nn.Module] = None,
        distill_temperature: float = 1.0,
        distill_alpha: float = 0.5,
        hidden_loss_weight: float = 0.0,
        pre_mlp_hidden_loss_weight: float = 0.0,
        prompt_kd_weight: float = 0.0,
        eakld_confidence_k: int = 16,
        hidden_layer_weighting: str = "uniform",
        saved_tensor_offload=None,
        streaming_offload_manager=None,
        teacher_output_offload: str = "none",
        teacher_model_offload: str = "none",
        teacher_output_pin_memory: bool = True,
        teacher_output_chunk_tokens: int = 8,
        **kwargs,
    ):
        self.loss_type = str(loss_type).strip().lower()
        self.teacher_model = teacher_model
        self.distill_temperature = float(distill_temperature)
        self.distill_alpha = float(distill_alpha)
        self.hidden_loss_weight = float(hidden_loss_weight)
        if self.hidden_loss_weight < 0.0:
            raise ValueError(f"hidden_loss_weight must be >= 0, got {self.hidden_loss_weight}.")
        self.pre_mlp_hidden_loss_weight = float(pre_mlp_hidden_loss_weight)
        if self.pre_mlp_hidden_loss_weight < 0.0:
            raise ValueError(
                "pre_mlp_hidden_loss_weight must be >= 0, "
                f"got {self.pre_mlp_hidden_loss_weight}."
            )
        self.prompt_kd_weight = float(prompt_kd_weight)
        if self.prompt_kd_weight < 0.0:
            raise ValueError(f"prompt_kd_weight must be >= 0, got {self.prompt_kd_weight}.")
        self.eakld_confidence_k = int(eakld_confidence_k)
        if self.eakld_confidence_k < 2:
            raise ValueError(f"eakld_confidence_k must be >= 2, got {self.eakld_confidence_k}.")
        try:
            self.hidden_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
                str(hidden_layer_weighting)
            )
        except (ValueError, argparse.ArgumentTypeError) as exc:
            raise ValueError(
                str(exc).replace(
                    "--distill_hidden_alignment_layer_weighting",
                    "hidden_layer_weighting",
                )
            ) from exc
        self.teacher_output_offload = str(teacher_output_offload).strip().lower()
        if self.teacher_output_offload not in {"none", "cpu"}:
            raise ValueError("teacher_output_offload must be one of: none | cpu.")
        self.teacher_model_offload = str(teacher_model_offload).strip().lower()
        if self.teacher_model_offload not in {"none", "cpu"}:
            raise ValueError("teacher_model_offload must be one of: none | cpu.")
        if self.teacher_model_offload == "cpu" and self.teacher_output_offload != "cpu":
            raise ValueError(
                "teacher_model_offload=cpu requires teacher_output_offload=cpu."
            )
        self.teacher_output_pin_memory = bool(teacher_output_pin_memory)
        self.teacher_output_chunk_tokens = int(teacher_output_chunk_tokens)
        if self.teacher_output_chunk_tokens < 1:
            raise ValueError("teacher_output_chunk_tokens must be >= 1.")
        self._active_teacher_targets: Optional[TeacherTargetBatch] = None
        self._last_teacher_target_stats: Dict[str, object] = {}
        self._logged_teacher_target_stats = False
        self._teacher_device = None
        self.saved_tensor_offload = saved_tensor_offload
        self.streaming_offload_manager = streaming_offload_manager
        self._last_loss_parts: Dict[str, float] = {}
        # Rank-local EAKLD telemetry accumulator (reset on each training log flush).
        self._eakld_telemetry_weighted_sums: Dict[str, float] = {}
        self._eakld_telemetry_weight = 0.0
        self._eakld_gamma_zero_weight = 0.0
        self._eakld_gamma_one_weight = 0.0
        # Logging-window token telemetry; consumed by E2EDistillTokenStatsCallback.
        self.distill_token_stats = DistillTokenStatsAccumulator()
        super().__init__(*args, **kwargs)
        # Custom compute_loss returns token-mean losses. HF treats models with
        # forward(**kwargs) as accepting num_items_in_batch and then skips
        # dividing by gradient_accumulation_steps, which inflates logged loss
        # and gradients. Disable that path explicitly.
        self.model_accepts_loss_kwargs = False
        if self.model is not None:
            self.model.accepts_loss_kwargs = False

    def _ensure_teacher_device(self, device: torch.device) -> None:
        if self.teacher_model is None:
            return
        if self._teacher_device == device:
            return
        self.teacher_model.to(device)
        self.teacher_model.eval()
        self._teacher_device = device

    def offload_teacher_to_cpu(self) -> Optional[torch.device]:
        """Move teacher off GPU for eval; returns previous device for restore."""
        if self.teacher_model is None:
            return None
        previous = self._teacher_device
        self.teacher_model.to("cpu")
        self._teacher_device = torch.device("cpu")
        return previous

    def restore_teacher_device(self, device: Optional[torch.device]) -> None:
        if device is None or self.teacher_model is None:
            return
        if device.type == "cpu":
            self.offload_teacher_to_cpu()
            return
        self._ensure_teacher_device(device)

    def _compute_teacher_outputs(self, teacher_inputs: Dict[str, torch.Tensor], *, output_hidden_states: bool = False):
        if self.teacher_model is None:
            raise RuntimeError("当前 loss_type 需要 teacher，但 trainer.teacher_model 为空。")
        input_tensor = next(value for value in teacher_inputs.values() if torch.is_tensor(value))
        self._ensure_teacher_device(device=input_tensor.device)
        with torch.no_grad():
            return self.teacher_model(**teacher_inputs, output_hidden_states=bool(output_hidden_states))

    def _compute_hidden_alignment_loss(self, teacher_outputs, student_outputs, inputs, *, loss_device):
        return compute_vae_hidden_alignment_loss(
            teacher_hidden_states=teacher_outputs.hidden_states,
            student_hidden_states=student_outputs.hidden_states,
            attention_mask=inputs.get("attention_mask"),
            layer_weighting=self.hidden_layer_weighting,
            loss_device=loss_device,
        )

    def _store_loss_parts(
        self,
        *,
        distill_loss: torch.Tensor,
        hidden_loss: Optional[torch.Tensor] = None,
        pre_mlp_hidden_loss: Optional[torch.Tensor] = None,
    ) -> None:
        parts = {"distill_loss": float(distill_loss.detach().float().item())}
        if hidden_loss is not None:
            parts["hidden_loss"] = float(hidden_loss.detach().float().item())
        if pre_mlp_hidden_loss is not None:
            parts["pre_mlp_hidden_loss"] = float(
                pre_mlp_hidden_loss.detach().float().item()
            )
        self._last_loss_parts = parts

    def _record_eakld_telemetry(self, telemetry: Dict[str, torch.Tensor]) -> None:
        if not telemetry:
            return
        valid_tokens = float(telemetry["valid_tokens"].detach().float().item())
        weight = max(valid_tokens, 1.0)
        for key in _EAKLD_WEIGHTED_KEYS:
            value = float(telemetry[key].detach().float().item())
            self._eakld_telemetry_weighted_sums[key] = (
                self._eakld_telemetry_weighted_sums.get(key, 0.0) + value * weight
            )
        self._eakld_telemetry_weight += weight
        gamma_reverse = float(telemetry["gamma_reverse"].detach().float().item())
        if gamma_reverse <= 1e-6:
            self._eakld_gamma_zero_weight += weight
        if gamma_reverse >= 1.0 - 1e-6:
            self._eakld_gamma_one_weight += weight

    def _consume_eakld_telemetry_logs(self) -> Dict[str, float]:
        # Rank-local telemetry: no all_reduce (avoids deadlock if only some ranks log).
        total_weight = float(self._eakld_telemetry_weight)
        if total_weight <= 0.0:
            return {}
        sums = self._eakld_telemetry_weighted_sums
        logs = {
            "eakld/teacher_entropy_mean": sums["teacher_entropy_mean"] / total_weight,
            "eakld/gamma_reverse_mean": sums["gamma_reverse"] / total_weight,
            "eakld/lambda_forward_mean": sums["lambda_forward"] / total_weight,
            "eakld/gamma_reverse_zero_fraction": (
                self._eakld_gamma_zero_weight / total_weight
            ),
            "eakld/gamma_reverse_one_fraction": (
                self._eakld_gamma_one_weight / total_weight
            ),
            "eakld/forward_kl_mean": sums["forward_kl"] / total_weight,
            "eakld/reverse_kl_mean": sums["reverse_kl"] / total_weight,
            "eakld/total_mean": sums["eakld_total"] / total_weight,
        }
        self._eakld_telemetry_weighted_sums = {}
        self._eakld_telemetry_weight = 0.0
        self._eakld_gamma_zero_weight = 0.0
        self._eakld_gamma_one_weight = 0.0
        return logs

    def log(self, logs, start_time=None):
        for key, value in getattr(self, "_last_loss_parts", {}).items():
            logs.setdefault(key, value)
        # Rank-local telemetry merge into the existing training log event.
        for key, value in self._consume_eakld_telemetry_logs().items():
            logs.setdefault(key, value)
        return super().log(logs, start_time=start_time)

    def _release_active_teacher_targets(self) -> None:
        targets = self._active_teacher_targets
        self._active_teacher_targets = None
        if targets is not None:
            targets.clear()

    def _build_distill_token_regions(
        self,
        inputs: Dict[str, torch.Tensor],
        reference_logits: torch.Tensor,
    ) -> DistillTokenRegions:
        return build_distill_token_regions(
            labels=inputs.get("labels"),
            attention_mask=inputs.get("attention_mask"),
            reference_logits=reference_logits,
        )

    def _build_cpu_teacher_targets(
        self,
        *,
        inputs: Dict[str, torch.Tensor],
        logits_required: bool,
        hidden_required: bool,
        pre_mlp_hidden_required: bool,
        eakld_metadata_required: bool,
    ) -> TeacherTargetBatch:
        if self._active_teacher_targets is not None:
            raise RuntimeError(
                "Active teacher targets already exist; release them before building a new batch."
            )

        targets = TeacherTargetBatch()
        teacher_outputs = None
        teacher_logits = None
        regions: Optional[DistillTokenRegions] = None
        try:
            teacher_inputs = dict(inputs)
            teacher_inputs.pop("labels", None)
            teacher_inputs.pop("num_items_in_batch", None)

            collector_ctx = (
                TeacherHiddenTargetCollector(
                    model=self.teacher_model,
                    attention_mask=inputs.get("attention_mask"),
                    layer_weighting=self.hidden_layer_weighting,
                    pin_memory=self.teacher_output_pin_memory,
                    score_chunk_tokens=64,
                )
                if hidden_required
                else nullcontext(None)
            )
            pre_mlp_reference_hidden_required = bool(
                pre_mlp_hidden_required
                and is_adaptive_hidden_alignment_layer_weighting(self.hidden_layer_weighting)
            )
            pre_mlp_capture_modules = (
                _resolve_e2e_pre_mlp_capture_modules(self.teacher_model)
                if pre_mlp_hidden_required
                else ()
            )
            pre_mlp_capture_ctx = (
                _capture_pre_mlp_hiddens_from_modules(pre_mlp_capture_modules)
                if pre_mlp_hidden_required
                else nullcontext(None)
            )

            with collector_ctx as collector, pre_mlp_capture_ctx as captured_pre_mlp:
                teacher_outputs = self._compute_teacher_outputs(
                    teacher_inputs,
                    output_hidden_states=pre_mlp_reference_hidden_required,
                )

            logits_cpu = None
            gamma_cpu = None
            entropy_mean_cpu = None
            valid_count_cpu = None
            prompt_gamma_cpu = None
            prompt_entropy_mean_cpu = None
            prompt_valid_count_cpu = None
            if logits_required:
                teacher_logits = get_output_logits(teacher_outputs)
                # Single CPU copy of full teacher logits, reused for EAKLD metadata.
                logits_cpu = copy_detached_tensor_to_cpu(
                    teacher_logits,
                    pin_memory=self.teacher_output_pin_memory,
                )
                if eakld_metadata_required:
                    regions = self._build_distill_token_regions(inputs, teacher_logits)
                    entropy_mean, gamma, valid_count = compute_teacher_entropy_mean_and_gamma(
                        logits_cpu,
                        regions.response_mask,
                        confidence_k=self.eakld_confidence_k,
                    )
                    gamma_cpu = gamma.detach().reshape(()).to(device="cpu", dtype=torch.float32)
                    entropy_mean_cpu = entropy_mean.detach().reshape(()).to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    valid_count_cpu = valid_count.detach().reshape(()).to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    if self.prompt_kd_weight > 0.0:
                        (
                            prompt_entropy_mean,
                            prompt_gamma,
                            prompt_valid_count,
                        ) = compute_teacher_entropy_mean_and_gamma(
                            logits_cpu,
                            regions.prompt_mask,
                            confidence_k=self.eakld_confidence_k,
                        )
                        prompt_gamma_cpu = prompt_gamma.detach().reshape(()).to(
                            device="cpu",
                            dtype=torch.float32,
                        )
                        prompt_entropy_mean_cpu = prompt_entropy_mean.detach().reshape(()).to(
                            device="cpu",
                            dtype=torch.float32,
                        )
                        prompt_valid_count_cpu = prompt_valid_count.detach().reshape(()).to(
                            device="cpu",
                            dtype=torch.float32,
                        )

            hidden_layer_indices: tuple = ()
            hidden_cpu_by_layer: Dict[int, torch.Tensor] = {}
            num_hidden_layers = 0
            if hidden_required:
                if collector is None:
                    raise RuntimeError("hidden_required=True but teacher hidden collector is missing.")
                hidden_layer_indices, hidden_cpu_by_layer, num_hidden_layers = collector.finalize()

            pre_mlp_hidden_cpu_by_name: Dict[str, torch.Tensor] = {}
            pre_mlp_reference_hidden_cpu = None
            if pre_mlp_hidden_required:
                if captured_pre_mlp is None:
                    raise RuntimeError(
                        "pre_mlp_hidden_required=True but teacher pre-MLP capture is missing."
                    )
                pre_mlp_hidden_cpu_by_name = {
                    logical_name: copy_detached_tensor_to_cpu(
                        captured_pre_mlp[logical_name],
                        pin_memory=self.teacher_output_pin_memory,
                    )
                    for logical_name, _module in pre_mlp_capture_modules
                }
                if pre_mlp_reference_hidden_required:
                    if teacher_outputs.hidden_states is None:
                        raise RuntimeError(
                            "adaptive pre-MLP hidden alignment requires teacher hidden_states."
                        )
                    pre_mlp_reference_hidden_cpu = copy_detached_tensor_to_cpu(
                        teacher_outputs.hidden_states[0],
                        pin_memory=self.teacher_output_pin_memory,
                    )

            del teacher_outputs, teacher_logits, regions
            teacher_outputs = None
            teacher_logits = None
            regions = None

            targets = TeacherTargetBatch(
                logits_cpu=logits_cpu,
                eakld_gamma_cpu=gamma_cpu,
                teacher_entropy_mean_cpu=entropy_mean_cpu,
                teacher_valid_token_count_cpu=valid_count_cpu,
                eakld_prompt_gamma_cpu=prompt_gamma_cpu,
                teacher_prompt_entropy_mean_cpu=prompt_entropy_mean_cpu,
                teacher_prompt_valid_token_count_cpu=prompt_valid_count_cpu,
                hidden_cpu_by_layer=dict(hidden_cpu_by_layer),
                hidden_layer_indices=tuple(hidden_layer_indices),
                num_hidden_layers=int(num_hidden_layers),
                pre_mlp_hidden_cpu_by_name=dict(pre_mlp_hidden_cpu_by_name),
                pre_mlp_reference_hidden_cpu=pre_mlp_reference_hidden_cpu,
            )
            self._last_teacher_target_stats = {
                "logits_device": "cpu" if logits_required else "none",
                "hidden_layer_indices": tuple(hidden_layer_indices),
                "hidden_layer_count": len(hidden_layer_indices),
                "num_hidden_layers": int(num_hidden_layers),
                "pre_mlp_hidden_layer_count": len(pre_mlp_hidden_cpu_by_name),
            }
            return targets
        except Exception:
            targets.clear()
            del teacher_outputs, teacher_logits, regions
            raise

        finally:
            if self.teacher_model_offload == "cpu":
                self.offload_teacher_to_cpu()

    def _compute_choice_kd_loss(self, model, inputs, return_outputs: bool):
        if float(self.hidden_loss_weight) > 0.0:
            raise ValueError("dataset_task=mcqa does not support hidden_loss_weight > 0.")
        if float(self.pre_mlp_hidden_loss_weight) > 0.0:
            raise ValueError(
                "dataset_task=mcqa does not support pre_mlp_hidden_loss_weight > 0."
            )
        if self.teacher_model is None:
            raise ValueError(f"loss_type={self.loss_type} requires teacher_model for MCQA choice KD.")

        choice_input_ids = inputs["choice_input_ids"]
        choice_attention_mask = inputs.get("choice_attention_mask")
        choice_labels = inputs["choice_labels"]
        answer_index = inputs["answer_index"]
        if choice_input_ids.ndim != 3:
            raise ValueError(f"choice_input_ids must have shape [B, C, L], got {tuple(choice_input_ids.shape)}.")
        batch_size, choice_count, seq_len = choice_input_ids.shape
        flat_student_inputs = {
            "input_ids": choice_input_ids.reshape(batch_size * choice_count, seq_len),
        }
        if choice_attention_mask is not None:
            flat_student_inputs["attention_mask"] = choice_attention_mask.reshape(batch_size * choice_count, seq_len)

        offload_context = (
            self.saved_tensor_offload.context()
            if self.saved_tensor_offload is not None
            else nullcontext()
        )
        with offload_context:
            outputs = model(**flat_student_inputs, output_hidden_states=False)
        student_logits = get_output_logits(outputs).reshape(batch_size, choice_count, seq_len, -1)

        flat_teacher_inputs = dict(flat_student_inputs)
        try:
            teacher_outputs = self._compute_teacher_outputs(flat_teacher_inputs, output_hidden_states=False)
            teacher_logits = get_output_logits(teacher_outputs).to(device=student_logits.device)
        finally:
            if self.teacher_model_offload == "cpu":
                self.offload_teacher_to_cpu()
        teacher_logits = teacher_logits.reshape(batch_size, choice_count, seq_len, -1)

        student_scores = compute_choice_scores_from_logits(student_logits, choice_labels)
        teacher_scores = compute_choice_scores_from_logits(teacher_logits, choice_labels.to(device=teacher_logits.device))
        loss = compute_choice_kd_loss_from_scores(
            student_scores=student_scores,
            teacher_scores=teacher_scores,
            answer_index=answer_index,
            loss_type=self.loss_type,
            temperature=self.distill_temperature,
            alpha=self.distill_alpha,
        )
        self._store_loss_parts(distill_loss=loss)
        return (loss, outputs) if return_outputs else loss

    def _compute_legacy_dense_loss(self, model, inputs, return_outputs: bool):
        labels = inputs.get("labels")
        student_inputs = dict(inputs)
        student_inputs.pop("labels", None)
        # Custom distill losses ignore HF num_items_in_batch scaling.
        student_inputs.pop("num_items_in_batch", None)
        hidden_loss_enabled = float(self.hidden_loss_weight) > 0.0
        pre_mlp_hidden_loss_enabled = float(self.pre_mlp_hidden_loss_weight) > 0.0
        pre_mlp_reference_hidden_required = bool(
            pre_mlp_hidden_loss_enabled
            and is_adaptive_hidden_alignment_layer_weighting(self.hidden_layer_weighting)
        )
        student_pre_mlp_capture_modules = (
            _resolve_e2e_pre_mlp_capture_modules(model)
            if pre_mlp_hidden_loss_enabled
            else ()
        )
        student_pre_mlp_capture_ctx = (
            _capture_pre_mlp_hiddens_from_modules(student_pre_mlp_capture_modules)
            if pre_mlp_hidden_loss_enabled
            else nullcontext(None)
        )
        offload_context = (
            self.saved_tensor_offload.context()
            if self.saved_tensor_offload is not None
            else nullcontext()
        )
        with offload_context, student_pre_mlp_capture_ctx as captured_student_pre_mlp:
            outputs = model(**student_inputs, output_hidden_states=hidden_loss_enabled)
        logits = get_output_logits(outputs)

        ce_loss = None
        if labels is not None:
            ce_loss = _causal_lm_cross_entropy(logits, labels)

        loss_type = self.loss_type
        if loss_type in {"sft", "origin"}:
            if ce_loss is None:
                raise ValueError(f"loss_type={loss_type} requires labels.")
            hidden_loss = None
            pre_mlp_hidden_loss = None
            loss = ce_loss
            if hidden_loss_enabled or pre_mlp_hidden_loss_enabled:
                teacher_inputs = dict(inputs)
                teacher_inputs.pop("labels", None)
                teacher_inputs.pop("num_items_in_batch", None)
                teacher_pre_mlp_capture_modules = (
                    _resolve_e2e_pre_mlp_capture_modules(self.teacher_model)
                    if pre_mlp_hidden_loss_enabled
                    else ()
                )
                teacher_pre_mlp_capture_ctx = (
                    _capture_pre_mlp_hiddens_from_modules(teacher_pre_mlp_capture_modules)
                    if pre_mlp_hidden_loss_enabled
                    else nullcontext(None)
                )
                with teacher_pre_mlp_capture_ctx as captured_teacher_pre_mlp:
                    teacher_outputs = self._compute_teacher_outputs(
                        teacher_inputs,
                        output_hidden_states=(hidden_loss_enabled or pre_mlp_reference_hidden_required),
                    )
                if hidden_loss_enabled:
                    hidden_loss = self._compute_hidden_alignment_loss(
                        teacher_outputs,
                        outputs,
                        inputs,
                        loss_device=loss.device,
                    )
                    loss = loss + float(self.hidden_loss_weight) * hidden_loss
                if pre_mlp_hidden_loss_enabled:
                    teacher_reference_hidden = (
                        teacher_outputs.hidden_states[0]
                        if pre_mlp_reference_hidden_required
                        else None
                    )
                    pre_mlp_hidden_loss = _compute_named_pre_mlp_hidden_alignment_loss(
                        teacher_by_name=dict(captured_teacher_pre_mlp),
                        student_by_name=dict(captured_student_pre_mlp),
                        attention_mask=inputs.get("attention_mask"),
                        layer_weighting=self.hidden_layer_weighting,
                        teacher_reference_hidden=teacher_reference_hidden,
                        teacher_targets_on_cpu=False,
                    )
                    loss = loss + float(self.pre_mlp_hidden_loss_weight) * pre_mlp_hidden_loss
            self._store_loss_parts(
                distill_loss=ce_loss,
                hidden_loss=hidden_loss,
                pre_mlp_hidden_loss=pre_mlp_hidden_loss,
            )
            return (loss, outputs) if return_outputs else loss

        teacher_inputs = dict(inputs)
        teacher_inputs.pop("labels", None)
        teacher_inputs.pop("num_items_in_batch", None)
        teacher_pre_mlp_capture_modules = (
            _resolve_e2e_pre_mlp_capture_modules(self.teacher_model)
            if pre_mlp_hidden_loss_enabled
            else ()
        )
        teacher_pre_mlp_capture_ctx = (
            _capture_pre_mlp_hiddens_from_modules(teacher_pre_mlp_capture_modules)
            if pre_mlp_hidden_loss_enabled
            else nullcontext(None)
        )
        with teacher_pre_mlp_capture_ctx as captured_teacher_pre_mlp:
            teacher_outputs = self._compute_teacher_outputs(
                teacher_inputs,
                output_hidden_states=(hidden_loss_enabled or pre_mlp_reference_hidden_required),
            )
        teacher_logits = get_output_logits(teacher_outputs).to(device=logits.device)
        regions = self._build_distill_token_regions(inputs, logits)
        telemetry: Dict[str, torch.Tensor] = {}
        distill_loss = compute_dense_loss_from_logits(
            loss_type=loss_type,
            student_logits=logits,
            teacher_logits=teacher_logits,
            ce_loss=ce_loss,
            mask=regions.response_mask,
            temperature=self.distill_temperature,
            alpha=self.distill_alpha,
            eakld_confidence_k=int(self.eakld_confidence_k),
            telemetry_out=telemetry,
            prompt_mask=regions.prompt_mask,
            prompt_kd_weight=self.prompt_kd_weight,
        )
        self._record_eakld_telemetry(telemetry)
        hidden_loss = None
        pre_mlp_hidden_loss = None
        loss = distill_loss
        if hidden_loss_enabled:
            hidden_loss = self._compute_hidden_alignment_loss(
                teacher_outputs,
                outputs,
                inputs,
                loss_device=distill_loss.device,
            )
            loss = loss + float(self.hidden_loss_weight) * hidden_loss
        if pre_mlp_hidden_loss_enabled:
            teacher_reference_hidden = (
                teacher_outputs.hidden_states[0]
                if pre_mlp_reference_hidden_required
                else None
            )
            pre_mlp_hidden_loss = _compute_named_pre_mlp_hidden_alignment_loss(
                teacher_by_name=dict(captured_teacher_pre_mlp),
                student_by_name=dict(captured_student_pre_mlp),
                attention_mask=inputs.get("attention_mask"),
                layer_weighting=self.hidden_layer_weighting,
                teacher_reference_hidden=teacher_reference_hidden,
                teacher_targets_on_cpu=False,
            )
            loss = loss + float(self.pre_mlp_hidden_loss_weight) * pre_mlp_hidden_loss
        self._store_loss_parts(
            distill_loss=distill_loss,
            hidden_loss=hidden_loss,
            pre_mlp_hidden_loss=pre_mlp_hidden_loss,
        )
        return (loss, outputs) if return_outputs else loss

    def _compute_teacher_first_cpu_loss(self, model, inputs, return_outputs: bool):
        loss_type = self.loss_type
        hidden_required = float(self.hidden_loss_weight) > 0.0
        pre_mlp_hidden_required = float(self.pre_mlp_hidden_loss_weight) > 0.0
        logits_required = loss_type not in {"sft", "origin"}
        eakld_metadata_required = (
            loss_type in {"eakld", "eakld_kd"}
            or is_eakld_top_loss(loss_type)
        )
        needs_teacher = hidden_required or pre_mlp_hidden_required or logits_required

        labels = inputs.get("labels")
        student_inputs = dict(inputs)
        student_inputs.pop("labels", None)
        student_inputs.pop("num_items_in_batch", None)

        targets: Optional[TeacherTargetBatch] = None
        try:
            if needs_teacher:
                targets = self._build_cpu_teacher_targets(
                    inputs=inputs,
                    logits_required=logits_required,
                    hidden_required=hidden_required,
                    pre_mlp_hidden_required=pre_mlp_hidden_required,
                    eakld_metadata_required=eakld_metadata_required,
                )

            student_collector_ctx = (
                StudentHiddenCollector(
                    model=model,
                    layer_indices=targets.hidden_layer_indices,
                )
                if hidden_required
                else nullcontext(None)
            )
            student_pre_mlp_capture_modules = (
                _resolve_e2e_pre_mlp_capture_modules(model)
                if pre_mlp_hidden_required
                else ()
            )
            student_pre_mlp_capture_ctx = (
                _capture_pre_mlp_hiddens_from_modules(student_pre_mlp_capture_modules)
                if pre_mlp_hidden_required
                else nullcontext(None)
            )
            offload_context = (
                self.saved_tensor_offload.context()
                if self.saved_tensor_offload is not None
                else nullcontext()
            )
            with (
                offload_context,
                student_collector_ctx as student_collector,
                student_pre_mlp_capture_ctx as captured_student_pre_mlp,
            ):
                outputs = model(**student_inputs, output_hidden_states=False)
            logits = get_output_logits(outputs)

            ce_loss = None
            if labels is not None:
                ce_loss = _causal_lm_cross_entropy(logits, labels)

            if loss_type in {"sft", "origin"}:
                if ce_loss is None:
                    raise ValueError(f"loss_type={loss_type} requires labels.")
                if targets is not None and targets.logits_cpu is not None:
                    raise RuntimeError("sft/origin must not cache teacher logits.")
                distill_loss = ce_loss
            else:
                if targets is None or targets.logits_cpu is None:
                    raise RuntimeError(
                        "Dense distillation with teacher_output_offload=cpu requires teacher logits on CPU."
                    )
                if eakld_metadata_required and (
                    targets.eakld_gamma_cpu is None
                    or targets.teacher_entropy_mean_cpu is None
                    or targets.teacher_valid_token_count_cpu is None
                ):
                    raise RuntimeError(
                        "EAKLD-family loss requires teacher logits, gamma, and "
                        "entropy scalars on CPU."
                    )
                regions = self._build_distill_token_regions(inputs, logits)
                if eakld_metadata_required and self.prompt_kd_weight > 0.0 and (
                    targets.eakld_prompt_gamma_cpu is None
                    or targets.teacher_prompt_entropy_mean_cpu is None
                    or targets.teacher_prompt_valid_token_count_cpu is None
                ):
                    raise RuntimeError(
                        "prompt_kd_weight > 0 requires prompt-region teacher "
                        "scalars on CPU."
                    )
                telemetry: Dict[str, torch.Tensor] = {}
                distill_loss = compute_dense_loss_from_offloaded_teacher(
                    loss_type=loss_type,
                    student_logits=logits,
                    teacher_logits_cpu=targets.logits_cpu,
                    teacher_gamma_cpu=targets.eakld_gamma_cpu,
                    teacher_entropy_mean_cpu=targets.teacher_entropy_mean_cpu,
                    teacher_valid_token_count_cpu=(
                        targets.teacher_valid_token_count_cpu
                    ),
                    ce_loss=ce_loss,
                    mask=regions.response_mask,
                    temperature=self.distill_temperature,
                    alpha=self.distill_alpha,
                    eakld_confidence_k=int(self.eakld_confidence_k),
                    sequence_chunk_size=int(self.teacher_output_chunk_tokens),
                    telemetry_out=telemetry if eakld_metadata_required else None,
                    prompt_mask=regions.prompt_mask,
                    prompt_kd_weight=self.prompt_kd_weight,
                    teacher_prompt_gamma_cpu=targets.eakld_prompt_gamma_cpu,
                    teacher_prompt_entropy_mean_cpu=(
                        targets.teacher_prompt_entropy_mean_cpu
                    ),
                    teacher_prompt_valid_token_count_cpu=(
                        targets.teacher_prompt_valid_token_count_cpu
                    ),
                )
                if eakld_metadata_required:
                    self._record_eakld_telemetry(telemetry)

            hidden_loss = None
            pre_mlp_hidden_loss = None
            loss = distill_loss
            if hidden_required:
                if targets is None or student_collector is None:
                    raise RuntimeError("hidden alignment requires teacher and student collectors.")
                hidden_loss = compute_selected_distill_hidden_alignment_loss(
                    teacher_hidden_by_layer=targets.hidden_cpu_by_layer,
                    student_hidden_by_layer=student_collector.collected(),
                    hidden_layer_indices=targets.hidden_layer_indices,
                    attention_mask=inputs.get("attention_mask"),
                    layer_weighting=self.hidden_layer_weighting,
                    num_layers=int(targets.num_hidden_layers),
                    loss_device=distill_loss.device,
                )
                loss = loss + float(self.hidden_loss_weight) * hidden_loss

            if pre_mlp_hidden_required:
                if targets is None or captured_student_pre_mlp is None:
                    raise RuntimeError(
                        "pre-MLP hidden alignment requires teacher and student captures."
                    )
                pre_mlp_hidden_loss = _compute_named_pre_mlp_hidden_alignment_loss(
                    teacher_by_name=targets.pre_mlp_hidden_cpu_by_name,
                    student_by_name=dict(captured_student_pre_mlp),
                    attention_mask=inputs.get("attention_mask"),
                    layer_weighting=self.hidden_layer_weighting,
                    teacher_reference_hidden=targets.pre_mlp_reference_hidden_cpu,
                    teacher_targets_on_cpu=True,
                )
                loss = loss + float(self.pre_mlp_hidden_loss_weight) * pre_mlp_hidden_loss

            self._store_loss_parts(
                distill_loss=distill_loss,
                hidden_loss=hidden_loss,
                pre_mlp_hidden_loss=pre_mlp_hidden_loss,
            )

            if torch.is_grad_enabled() and torch.is_tensor(loss) and bool(loss.requires_grad):
                self._active_teacher_targets = targets
                targets = None
            else:
                if targets is not None:
                    targets.clear()
                targets = None
                self._active_teacher_targets = None

            return (loss, outputs) if return_outputs else loss
        except Exception:
            if targets is not None:
                targets.clear()
            self._active_teacher_targets = None
            raise

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        del num_items_in_batch, kwargs
        # Record token-window telemetry exactly once from original labels before
        # dispatching to choice/dense/CPU paths. MCQA requests expose no
        # ordinary rank-2 `labels`; skip them explicitly rather than fabricating.
        if bool(getattr(model, "training", False)):
            original_labels = inputs.get("labels")
            if isinstance(original_labels, torch.Tensor):
                self.distill_token_stats.update(
                    original_labels, inputs.get("attention_mask")
                )
        if "choice_input_ids" in inputs:
            return self._compute_choice_kd_loss(model, inputs, return_outputs=bool(return_outputs))
        if self.teacher_output_offload == "none":
            return self._compute_legacy_dense_loss(model, inputs, return_outputs=bool(return_outputs))
        if self.teacher_output_offload == "cpu":
            return self._compute_teacher_first_cpu_loss(model, inputs, return_outputs=bool(return_outputs))
        raise ValueError("teacher_output_offload must be one of: none | cpu.")

    def training_step(self, model, inputs, num_items_in_batch=None):
        track_peak = (
            self.teacher_output_offload == "cpu"
            and not self._logged_teacher_target_stats
            and torch.cuda.is_available()
            and torch.device(self.args.device).type == "cuda"
        )
        if track_peak:
            torch.cuda.reset_peak_memory_stats(torch.device(self.args.device))

        try:
            try:
                loss = super().training_step(
                    model,
                    inputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                self._release_active_teacher_targets()
                loss = super().training_step(model, inputs)
        finally:
            self._release_active_teacher_targets()

        if self.streaming_offload_manager is not None:
            self.streaming_offload_manager.offload_all(synchronize=True)

        target_device = torch.device(self.args.device)
        if torch.is_tensor(loss) and loss.device != target_device:
            loss = loss.to(device=target_device)

        if not self._logged_teacher_target_stats and self.teacher_output_offload == "cpu":
            peak_bytes = (
                int(torch.cuda.max_memory_allocated(target_device))
                if track_peak
                else -1
            )
            logging.getLogger("compressed_e2e_fintuning").info(
                "Teacher target first-step stats: logits_device=%s hidden_layers=%s "
                "hidden_layer_count=%d pre_mlp_hidden_layer_count=%d "
                "peak_allocated_bytes=%d teacher_model_offload=%s",
                self._last_teacher_target_stats.get("logits_device", "none"),
                self._last_teacher_target_stats.get("hidden_layer_indices", ()),
                int(self._last_teacher_target_stats.get("hidden_layer_count", 0)),
                int(self._last_teacher_target_stats.get("pre_mlp_hidden_layer_count", 0)),
                peak_bytes,
                self.teacher_model_offload,
            )
            self._logged_teacher_target_stats = True

        return loss
