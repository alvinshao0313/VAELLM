import argparse
import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Dict, Optional, Sequence

import torch
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
from e2e_common.selective_topk_head import (
    extract_teacher_topk_targets,
    is_selective_student_topk_loss,
    move_teacher_topk_targets_to_device,
    parse_selective_student_topk_k,
    selective_student_lm_head,
)
from train_utils.distill_loss_core import (
    compute_selected_kl_top_model_level_loss,
    normalize_model_level_loss_type,
)
from train_utils.config.configs import DistillLossConfig
from train_utils.distill_losses import (
    DistillTokenRegions,
    build_distill_token_regions,
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


class VAEDecoderE2ETrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_type: str = "sft",
        top_k: int = 100,
        teacher_model: Optional[nn.Module] = None,
        distill_temperature: float = 1.0,
        distill_alpha: float = 0.5,
        hidden_loss_weight: float = 0.0,
        pre_mlp_hidden_loss_weight: float = 0.0,
        prompt_kd_weight: float = 0.0,
        hidden_layer_weighting: str = "uniform",
        saved_tensor_offload=None,
        streaming_offload_manager=None,
        teacher_output_offload: str = "none",
        teacher_model_offload: str = "none",
        teacher_output_pin_memory: bool = True,
        teacher_output_chunk_tokens: int = 8,
        selective_student_topk: bool = False,
        selective_student_topk_chunk_rows: int = 32,
        decoder_param_ids: Optional[Sequence[int]] = None,
        distill_hif4_act_controller=None,
        decoder_lr: Optional[float] = None,
        sparse_bit_manager=None,
        aux_trainable_parameters=None,
        loss_config: Optional[DistillLossConfig] = None,
        **kwargs,
    ):
        if loss_config is not None:
            loss_config.validate()
            self.loss_config = loss_config
        else:
            self.loss_config = DistillLossConfig(
                loss_type=normalize_model_level_loss_type(loss_type),
                top_k=int(top_k),
                temperature=float(distill_temperature),
                alpha=float(distill_alpha),
                prompt_loss_weight=float(prompt_kd_weight),
                hidden_loss_weight=float(hidden_loss_weight),
                pre_mlp_hidden_loss_weight=float(pre_mlp_hidden_loss_weight),
                hidden_layer_weighting=str(hidden_layer_weighting),
                selective_student_topk=bool(selective_student_topk),
                selective_student_topk_chunk_rows=int(selective_student_topk_chunk_rows),
            )
            self.loss_config.validate()
        self.loss_type = str(self.loss_config.loss_type)
        self.top_k = int(self.loss_config.top_k)
        self.teacher_model = teacher_model
        self.distill_temperature = float(self.loss_config.temperature)
        self.distill_alpha = float(self.loss_config.alpha)
        self.hidden_loss_weight = float(self.loss_config.hidden_loss_weight)
        self.pre_mlp_hidden_loss_weight = float(self.loss_config.pre_mlp_hidden_loss_weight)
        self.prompt_kd_weight = float(self.loss_config.prompt_loss_weight)
        try:
            self.hidden_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
                str(self.loss_config.hidden_layer_weighting)
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
        self.selective_student_topk = bool(self.loss_config.selective_student_topk)
        self.selective_student_topk_chunk_rows = int(self.loss_config.selective_student_topk_chunk_rows)
        if self.selective_student_topk and not is_selective_student_topk_loss(self.loss_type):
            raise ValueError("selective_student_topk only supports loss_type=kl_top.")
        self._active_teacher_targets: Optional[TeacherTargetBatch] = None
        self._last_teacher_target_stats: Dict[str, object] = {}
        self._logged_teacher_target_stats = False
        self._teacher_device = None
        self.saved_tensor_offload = saved_tensor_offload
        self.streaming_offload_manager = streaming_offload_manager
        self.decoder_param_ids = frozenset(int(v) for v in (decoder_param_ids or ()))
        self.distill_hif4_act_controller = distill_hif4_act_controller
        self.decoder_lr = None if decoder_lr is None else float(decoder_lr)
        self.sparse_bit_manager = sparse_bit_manager
        self.aux_trainable_parameters = dict(aux_trainable_parameters or {})
        self._sparse_bit_main_optimizer = None
        self._sparse_bit_main_parameters = ()
        self._last_sparse_bit_telemetry: Dict[str, float] = {}
        self._last_loss_parts: Dict[str, float] = {}
        # Task-9 v6 exact-step checkpointing is opt-in. Legacy callers keep the
        # existing checkpoint lifecycle until runtime_v6 explicitly configures it.
        self._v6_step_checkpoint_context = None
        self._v6_selected_vae_modules = ()
        self._v6_exact_resume_loaded = False
        # Logging-window token telemetry; consumed by E2EDistillTokenStatsCallback.
        self.distill_token_stats = DistillTokenStatsAccumulator()
        super().__init__(*args, **kwargs)
        if self.sparse_bit_manager is not None:
            if int(getattr(self.args, 'n_gpu', 0)) > 1 and int(getattr(self.accelerator, 'num_processes', 1)) == 1 and not bool(getattr(self, 'is_model_parallel', False)):
                raise RuntimeError(
                    'Sparse Bit Tuning does not support single-process torch.nn.DataParallel. '
                    'For DP, launch with torchrun/DDP so each process owns one GPU.'
                )
            from sparse_bit_tuning.amp import install_main_grad_clip_filter, install_sparse_bit_grad_scaler

            bit_param_ids = self.sparse_bit_manager.score_module.bit_parameter_ids()
            self._sparse_bit_main_parameters = tuple(
                param
                for param in self.model.parameters()
                if bool(param.requires_grad) and id(param) not in bit_param_ids
            )
            install_sparse_bit_grad_scaler(self.accelerator)
            install_main_grad_clip_filter(self.accelerator, self._sparse_bit_main_parameters)
        # Custom compute_loss returns token-mean losses. HF treats models with
        # forward(**kwargs) as accepting num_items_in_batch and then skips
        # dividing by gradient_accumulation_steps, which inflates logged loss
        # and gradients. Disable that path explicitly.
        self.model_accepts_loss_kwargs = False
        if self.model is not None:
            self.model.accepts_loss_kwargs = False

    def configure_v6_step_checkpoint(self, *, context: dict, selected_vae_modules) -> None:
        """Enable lightweight v6 optimizer-step checkpoints for the active E2E run."""
        if not isinstance(context, dict):
            raise TypeError(f"v6 step checkpoint context must be dict, got {type(context)}.")
        required = {
            "round_base_dir",
            "round_base_checkpoint_id",
            "train_mode",
            "compressed_targets",
            "pending_dense_targets",
            "skip_targets",
            "norm_train_mode",
            "lm_head_train_mode",
            "lora_config",
            "resolved_learning_rates",
            "target_layers",
            "target_modules",
            "immutable_resume_contract",
            "base_model_path",
        }
        missing = sorted(required - set(context))
        if missing:
            raise ValueError(f"v6 step checkpoint context missing required fields: {missing}")
        round_base_dir = os.path.abspath(str(context["round_base_dir"]))
        if not os.path.isdir(round_base_dir):
            raise FileNotFoundError(f"v6 round_base_dir does not exist: {round_base_dir}")
        if not str(context["round_base_checkpoint_id"]).strip():
            raise ValueError("v6 round_base_checkpoint_id must be non-empty.")
        if bool(getattr(self.args, "save_only_model", False)):
            raise ValueError(
                "v6 exact-step resume requires save_only_model=false so optimizer/RNG state is retained."
            )
        if bool(getattr(self.args, "ignore_data_skip", False)):
            raise ValueError("v6 exact-step resume requires ignore_data_skip=false.")
        normalized = dict(context)
        normalized["round_base_dir"] = round_base_dir
        self._v6_step_checkpoint_context = normalized
        self._v6_selected_vae_modules = tuple(selected_vae_modules or ())
        self._v6_exact_resume_loaded = False

    def _v6_step_checkpoint_enabled(self) -> bool:
        return isinstance(self._v6_step_checkpoint_context, dict)

    def save_model(self, output_dir=None, _internal_call: bool = False):
        # Trainer._save_checkpoint always calls save_model first. In v6 exact-step
        # mode round_base owns the frozen/full model, so checkpoint-N must not
        # duplicate full model or PEFT adapter weights.
        if self._v6_step_checkpoint_enabled() and bool(_internal_call):
            target = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(target, exist_ok=True)
            return None
        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _sparse_bit_optimizer_step(self) -> None:
        if self.sparse_bit_manager is None:
            raise RuntimeError("Sparse Bit optimizer callback executed without a manager.")
        telemetry = self.sparse_bit_manager.optimizer_step()
        self._last_sparse_bit_telemetry = {
            "bit/round": float(telemetry.global_bit_round),
            "bit/round_step": float(telemetry.bit_round_step),
            "bit/step_flip_count": float(telemetry.step_flip_count),
            "bit/cumulative_flip_count": float(telemetry.cumulative_flip_count),
            "bit/stable_counter": float(telemetry.stable_counter),
            "bit/stable_steps": float(telemetry.stable_steps),
            "bit/had_flip": float(bool(telemetry.had_flip)),
            "bit/round_ended": float(bool(telemetry.round_ended)),
        }

    def create_optimizer(self):
        if self.sparse_bit_manager is None:
            if getattr(self, "model_level_trainable_selection", None) is not None or self.decoder_param_ids:
                from compressed_e2e_fintuning.optimizer_grouping import create_decoder_grouped_optimizer

                return create_decoder_grouped_optimizer(self)
            return super().create_optimizer()
        from sparse_bit_tuning.optimizer import SparseBitCompositeOptimizer

        if isinstance(self.optimizer, SparseBitCompositeOptimizer):
            return self.optimizer
        if self.optimizer is not None:
            raise RuntimeError(
                "Sparse Bit E2E does not accept a pre-built external optimizer; "
                "let Trainer create the continuous optimizer and Composite wrapper."
            )

        bit_params = tuple(self.sparse_bit_manager.score_module.bit_parameters())
        bit_param_ids = {id(param) for param in bit_params}
        main_params = tuple(self._sparse_bit_main_parameters)
        if any(id(param) in bit_param_ids for param in main_params):
            raise RuntimeError("Sparse Bit score/main optimizer parameter sets overlap.")

        main_optimizer = None
        if main_params:
            old_requires_grad = [bool(param.requires_grad) for param in bit_params]
            try:
                for param in bit_params:
                    param.requires_grad_(False)
                if getattr(self, "model_level_trainable_selection", None) is not None or self.decoder_param_ids:
                    from compressed_e2e_fintuning.optimizer_grouping import create_decoder_grouped_optimizer

                    main_optimizer = create_decoder_grouped_optimizer(self)
                else:
                    main_optimizer = super().create_optimizer()
            finally:
                for param, requires_grad in zip(bit_params, old_requires_grad):
                    param.requires_grad_(requires_grad)
            optimized_ids = {
                id(param)
                for group in main_optimizer.param_groups
                for param in group.get("params", ())
            }
            overlap = optimized_ids & bit_param_ids
            if overlap:
                raise RuntimeError(f"Main optimizer unexpectedly owns {len(overlap)} Sparse Bit score parameters.")

        self._sparse_bit_main_optimizer = main_optimizer
        self.optimizer = SparseBitCompositeOptimizer(
            main_optimizer=main_optimizer,
            bit_manager=self.sparse_bit_manager.bit_optimizer,
            step_callback=self._sparse_bit_optimizer_step,
        )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.sparse_bit_manager is None:
            return super().create_scheduler(num_training_steps, optimizer=optimizer)
        from sparse_bit_tuning.optimizer import make_noop_scheduler

        self.sparse_bit_manager.configure_schedule(total_optimizer_steps=int(num_training_steps))
        self.sparse_bit_manager.initialize_scores()
        if self._sparse_bit_main_optimizer is None:
            if self.lr_scheduler is None:
                self.lr_scheduler = make_noop_scheduler()
                self._created_lr_scheduler = True
            return self.lr_scheduler
        return super().create_scheduler(
            num_training_steps,
            optimizer=self._sparse_bit_main_optimizer,
        )

    def _save(self, output_dir=None, state_dict=None):
        if self.sparse_bit_manager is None:
            return super()._save(output_dir=output_dir, state_dict=state_dict)
        if state_dict is None:
            state_dict = self.model.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if not str(key).startswith("sparse_bit_tuning.")
        }
        return super()._save(output_dir=output_dir, state_dict=filtered)

    def _save_v6_step_checkpoint(self, model, trial):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        from compressed_e2e_fintuning.v6_runtime_state import collect_e2e_mutable_state
        from train_utils.checkpoint_v6 import save_v6_training_step_payload

        context = dict(self._v6_step_checkpoint_context or {})
        result = super()._save_checkpoint(model, trial)

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(
            run_dir, f"{PREFIX_CHECKPOINT_DIR}-{int(self.state.global_step)}"
        )
        is_main = bool(self.is_world_process_zero())
        distributed_barrier = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            distributed_barrier = torch.distributed.barrier

        if is_main and self.sparse_bit_manager is not None:
            from sparse_bit_tuning.exact_checkpoint import save_exact_sidecar

            save_exact_sidecar(output_dir, self.sparse_bit_manager)

        unwrapped = (
            self.accelerator.unwrap_model(self.model)
            if getattr(self, "accelerator", None) is not None
            else self.model
        )
        selection = getattr(self, "model_level_trainable_selection", None)
        mutable_state, _component_classes, manifest = collect_e2e_mutable_state(
            unwrapped,
            selection=selection,
            selected_vae_modules=self._v6_selected_vae_modules,
        )
        round_base_ref = os.path.relpath(
            str(context["round_base_dir"]),
            start=os.path.abspath(output_dir),
        )
        save_v6_training_step_payload(
            output_dir,
            round_base_ref=round_base_ref,
            round_base_checkpoint_id=str(context["round_base_checkpoint_id"]),
            mutable_state=mutable_state,
            mutable_state_manifest=manifest,
            train_mode=str(context["train_mode"]),
            compressed_targets=tuple(context["compressed_targets"]),
            pending_dense_targets=tuple(context["pending_dense_targets"]),
            skip_targets=tuple(context["skip_targets"]),
            legacy_original_only_sources=tuple(context.get("legacy_original_only_sources", ())),
            norm_train_mode=str(context["norm_train_mode"]),
            lm_head_train_mode=str(context["lm_head_train_mode"]),
            lora_config=(
                None
                if context.get("lora_config") is None
                else dict(context["lora_config"])
            ),
            resolved_learning_rates=dict(context["resolved_learning_rates"]),
            compression_categories=tuple(context.get("compression_categories", ())),
            target_layers=tuple(int(v) for v in context["target_layers"]),
            target_modules=tuple(str(v) for v in context["target_modules"]),
            immutable_resume_contract=dict(context["immutable_resume_contract"]),
            runtime_audit=dict(context.get("runtime_audit", {})),
            base_model_path=str(context["base_model_path"]),
            hf_artifact_refs=dict(context.get("hf_artifact_refs", {})),
            is_main_process=is_main,
            distributed_barrier=distributed_barrier,
        )
        return result

    def _save_checkpoint(self, model, trial):
        if self._v6_step_checkpoint_enabled():
            return self._save_v6_step_checkpoint(model, trial)
        if self.sparse_bit_manager is None and not self.aux_trainable_parameters:
            return super()._save_checkpoint(model, trial)
        packed_snapshot = None
        coverage_snapshot = None
        aux_snapshot = None
        if bool(getattr(self.args, "should_save", True)):
            if self.sparse_bit_manager is not None:
                packed_snapshot = self.sparse_bit_manager.checkpoint_packed_snapshot()
                coverage_snapshot = self.sparse_bit_manager.coverage_metadata()
            if self.aux_trainable_parameters:
                from compressed_e2e_fintuning.aux_trainables import snapshot_auxiliary_trainables

                aux_snapshot = snapshot_auxiliary_trainables(self.aux_trainable_parameters)
        result = super()._save_checkpoint(model, trial)
        if packed_snapshot is not None:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            from sparse_bit_tuning.checkpoint import save_sidecar

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(
                run_dir, f"{PREFIX_CHECKPOINT_DIR}-{int(self.state.global_step)}"
            )
            save_sidecar(
                output_dir,
                packed_banks=packed_snapshot,
                coverage=coverage_snapshot,
            )
        if aux_snapshot is not None:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            from compressed_e2e_fintuning.aux_trainables import save_auxiliary_sidecar

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(
                run_dir, f"{PREFIX_CHECKPOINT_DIR}-{int(self.state.global_step)}"
            )
            save_auxiliary_sidecar(output_dir, aux_snapshot)
        return result

    def _load_v6_exact_step_checkpoint(self, resume_from_checkpoint, model=None):
        from compressed_e2e_fintuning.v6_runtime_state import (
            restore_e2e_mutable_state,
            validate_e2e_immutable_resume_contract,
        )
        from train_utils.checkpoint_v6 import (
            load_v6_training_model_state,
            load_v6_training_step_meta,
            resolve_training_step_round_base_ref,
        )

        context = dict(self._v6_step_checkpoint_context or {})
        checkpoint_dir = os.path.abspath(str(resume_from_checkpoint))
        meta = load_v6_training_step_meta(checkpoint_dir)
        _resolved_base, base_meta = resolve_training_step_round_base_ref(checkpoint_dir, meta)
        expected_base_id = str(context["round_base_checkpoint_id"])
        actual_base_id = str(base_meta["checkpoint_id"])
        if actual_base_id != expected_base_id:
            raise ValueError(
                "E2E v6 resume round-base mismatch: "
                f"checkpoint={actual_base_id!r} current={expected_base_id!r}."
            )

        expected_topology = {
            "train_mode": str(context["train_mode"]),
            "compressed_targets": list(context["compressed_targets"]),
            "pending_dense_targets": list(context["pending_dense_targets"]),
            "skip_targets": list(context["skip_targets"]),
            "norm_train_mode": str(context["norm_train_mode"]),
            "lm_head_train_mode": str(context["lm_head_train_mode"]),
            "target_layers": [int(v) for v in context["target_layers"]],
            "target_modules": [str(v) for v in context["target_modules"]],
        }
        for key, expected in expected_topology.items():
            actual = meta.get(key)
            if actual != expected:
                raise ValueError(
                    f"E2E v6 resume topology mismatch for {key}: "
                    f"checkpoint={actual!r} current={expected!r}."
                )
        validate_e2e_immutable_resume_contract(
            meta.get("immutable_resume_contract") or {},
            context["immutable_resume_contract"],
        )

        checkpoint_state, checkpoint_manifest = load_v6_training_model_state(
            checkpoint_dir,
            map_location="cpu",
        )
        target_model = self.model if model is None else model
        unwrapped = (
            self.accelerator.unwrap_model(target_model)
            if getattr(self, "accelerator", None) is not None
            else target_model
        )
        restore_e2e_mutable_state(
            unwrapped,
            selection=getattr(self, "model_level_trainable_selection", None),
            selected_vae_modules=self._v6_selected_vae_modules,
            checkpoint_state=checkpoint_state,
            checkpoint_manifest=checkpoint_manifest,
        )

        sparse_dir = os.path.join(checkpoint_dir, "sparse_bit_tuning")
        if self.sparse_bit_manager is not None:
            from sparse_bit_tuning.exact_checkpoint import (
                exact_sidecar_complete,
                restore_exact_sidecar,
            )

            if not exact_sidecar_complete(checkpoint_dir):
                raise FileNotFoundError(
                    f"Sparse Bit exact resume requires exact sidecar under {sparse_dir}."
                )
            restore_exact_sidecar(checkpoint_dir, self.sparse_bit_manager)
        elif os.path.isdir(sparse_dir):
            raise RuntimeError(
                "E2E v6 checkpoint contains Sparse Bit state but current train_mode "
                "does not enable Sparse Bit."
            )

        self._v6_exact_resume_loaded = True
        return None

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if self._v6_step_checkpoint_enabled():
            return self._load_v6_exact_step_checkpoint(resume_from_checkpoint, model=model)
        sidecar_path = os.path.join(str(resume_from_checkpoint), "sparse_bit_tuning")
        aux_path = os.path.join(str(resume_from_checkpoint), "e2e_aux_trainables.pt")
        if self.sparse_bit_manager is None:
            if os.path.isdir(sidecar_path):
                raise RuntimeError(
                    "Checkpoint contains Sparse Bit sidecar but sparse_bit_tuning=false. "
                    "Resume from a final committed compressed model instead of a Bit intermediate checkpoint."
                )
            if not self.aux_trainable_parameters:
                if os.path.isfile(aux_path):
                    raise RuntimeError(
                        "Checkpoint contains auxiliary sidecar but no auxiliary trainables are enabled."
                    )
                return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            if not os.path.isfile(aux_path):
                raise RuntimeError(f"Compressed LoRA auxiliary resume requires sidecar: {aux_path}.")
            result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            from compressed_e2e_fintuning.aux_trainables import (
                load_auxiliary_sidecar,
                restore_auxiliary_trainables,
            )

            aux_payload = load_auxiliary_sidecar(str(resume_from_checkpoint))
            restore_auxiliary_trainables(self.aux_trainable_parameters, aux_payload)
            return result

        from sparse_bit_tuning.checkpoint import load_sidecar, sidecar_complete

        if not sidecar_complete(str(resume_from_checkpoint)):
            raise RuntimeError(
                f"Sparse Bit resume requires complete sidecar under {sidecar_path}."
            )
        if self.aux_trainable_parameters and not os.path.isfile(aux_path):
            raise RuntimeError(f"Compressed LoRA auxiliary resume requires sidecar: {aux_path}.")
        if not self.aux_trainable_parameters and os.path.isfile(aux_path):
            raise RuntimeError(
                "Checkpoint contains auxiliary sidecar but no auxiliary trainables are enabled."
            )
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        packed_banks, coverage = load_sidecar(str(resume_from_checkpoint))
        self.sparse_bit_manager.restore_checkpoint_packed(packed_banks)
        self.sparse_bit_manager.restore_coverage_metadata(coverage)
        if self.aux_trainable_parameters:
            from compressed_e2e_fintuning.aux_trainables import (
                load_auxiliary_sidecar,
                restore_auxiliary_trainables,
            )

            aux_payload = load_auxiliary_sidecar(str(resume_from_checkpoint))
            restore_auxiliary_trainables(self.aux_trainable_parameters, aux_payload)
        return result

    def _issue_warnings_after_load(self, load_result):
        if self.sparse_bit_manager is None:
            return super()._issue_warnings_after_load(load_result)
        missing_keys = getattr(load_result, "missing_keys", None)
        if isinstance(missing_keys, list):
            missing_keys[:] = [
                key
                for key in missing_keys
                if not str(key).startswith("sparse_bit_tuning.")
            ]
        return super()._issue_warnings_after_load(load_result)

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

    @contextmanager
    def _student_hif4_act_context(self):
        controller = self.distill_hif4_act_controller
        if controller is None:
            yield
            return
        previous = bool(getattr(controller, "enabled", False))
        controller.enabled = True
        try:
            yield
        finally:
            controller.enabled = previous

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

    def log(self, logs, start_time=None):
        for key, value in getattr(self, "_last_loss_parts", {}).items():
            logs.setdefault(key, value)
        for key, value in getattr(self, "_last_sparse_bit_telemetry", {}).items():
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
            selective_topk = None
            if logits_required:
                teacher_logits = get_output_logits(teacher_outputs)
                if self.selective_student_topk:
                    selective_topk = extract_teacher_topk_targets(
                        teacher_logits,
                        k=parse_selective_student_topk_k(self.loss_type, top_k=self.top_k),
                        sequence_chunk_size=self.teacher_output_chunk_tokens,
                        pin_memory=self.teacher_output_pin_memory,
                    )
                else:
                    logits_cpu = copy_detached_tensor_to_cpu(
                        teacher_logits,
                        pin_memory=self.teacher_output_pin_memory,
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
                selective_topk=selective_topk,
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

    def _compute_legacy_dense_loss(self, model, inputs, return_outputs: bool):
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("model-level loss requires input_ids.")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("model-level loss requires labels.")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("model-level loss requires attention_mask.")

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
        with self._student_hif4_act_context():
            with offload_context, student_pre_mlp_capture_ctx as captured_student_pre_mlp:
                outputs = model(**student_inputs, output_hidden_states=hidden_loss_enabled)
        logits = get_output_logits(outputs)

        canonical_loss = normalize_model_level_loss_type(self.loss_config.loss_type)
        resolved_top_k = int(self.loss_config.top_k)
        need_teacher_logits = canonical_loss != "sft"
        teacher_outputs = None
        captured_teacher_pre_mlp = None
        teacher_logits = None

        if need_teacher_logits or hidden_loss_enabled or pre_mlp_hidden_loss_enabled:
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
            if need_teacher_logits:
                teacher_logits = get_output_logits(teacher_outputs).to(device=logits.device)

        distill_loss = compute_dense_loss_from_logits(
            loss_type=canonical_loss,
            student_logits=logits,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            teacher_logits=teacher_logits,
            temperature=float(self.loss_config.temperature),
            alpha=float(self.loss_config.alpha),
            top_k=resolved_top_k,
            prompt_loss_weight=float(self.loss_config.prompt_loss_weight),
        )
        hidden_loss = None
        pre_mlp_hidden_loss = None
        loss = distill_loss
        if hidden_loss_enabled:
            if teacher_outputs is None:
                raise RuntimeError("hidden alignment requires teacher outputs.")
            hidden_loss = self._compute_hidden_alignment_loss(
                teacher_outputs,
                outputs,
                inputs,
                loss_device=distill_loss.device,
            )
            loss = loss + float(self.hidden_loss_weight) * hidden_loss
        if pre_mlp_hidden_loss_enabled:
            if teacher_outputs is None or captured_teacher_pre_mlp is None:
                raise RuntimeError("pre-MLP hidden alignment requires teacher captures.")
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
        canonical_loss = normalize_model_level_loss_type(self.loss_config.loss_type)
        resolved_top_k = int(self.loss_config.top_k)
        hidden_required = float(self.loss_config.hidden_loss_weight) > 0.0
        pre_mlp_hidden_required = float(self.loss_config.pre_mlp_hidden_loss_weight) > 0.0
        logits_required = canonical_loss != "sft"
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
                )

            selective_indices = None
            selective_teacher_logits = None
            if self.selective_student_topk:
                if targets is None or targets.selective_topk is None:
                    raise RuntimeError("selective student top-k requires compact teacher top-k targets.")
                input_tensor = next(
                    value for value in student_inputs.values() if torch.is_tensor(value)
                )
                selective_indices, selective_teacher_logits = move_teacher_topk_targets_to_device(
                    targets.selective_topk,
                    device=input_tensor.device,
                )
                selective_head_ctx = selective_student_lm_head(
                    model,
                    teacher_topk_indices=selective_indices,
                    chunk_rows=self.selective_student_topk_chunk_rows,
                )
            else:
                selective_head_ctx = nullcontext()

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
                self._student_hif4_act_context(),
                selective_head_ctx,
                student_collector_ctx as student_collector,
                student_pre_mlp_capture_ctx as captured_student_pre_mlp,
            ):
                outputs = model(**student_inputs, output_hidden_states=False)
            logits = get_output_logits(outputs)

            input_ids = inputs.get("input_ids")
            labels = inputs.get("labels")
            attention_mask = inputs.get("attention_mask")
            if not isinstance(input_ids, torch.Tensor):
                raise ValueError("model-level loss requires input_ids.")
            if not isinstance(labels, torch.Tensor):
                raise ValueError("model-level loss requires labels.")
            if not isinstance(attention_mask, torch.Tensor):
                raise ValueError("model-level loss requires attention_mask.")

            if self.selective_student_topk:
                if selective_teacher_logits is None:
                    raise RuntimeError("selective student top-k teacher logits are missing.")
                distill_loss = compute_selected_kl_top_model_level_loss(
                    student_selected_logits=logits,
                    teacher_selected_logits=selective_teacher_logits,
                    labels=labels,
                    attention_mask=attention_mask,
                    temperature=float(self.loss_config.temperature),
                    prompt_loss_weight=float(self.loss_config.prompt_loss_weight),
                )
            elif canonical_loss == "sft":
                if targets is not None and targets.logits_cpu is not None:
                    raise RuntimeError("sft must not cache teacher logits.")
                distill_loss = compute_dense_loss_from_logits(
                    loss_type="sft",
                    student_logits=logits,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    temperature=float(self.loss_config.temperature),
                    prompt_loss_weight=float(self.loss_config.prompt_loss_weight),
                )
            else:
                if targets is None or targets.logits_cpu is None:
                    raise RuntimeError(
                        "Dense distillation with teacher_output_offload=cpu requires teacher logits on CPU."
                    )
                distill_loss = compute_dense_loss_from_offloaded_teacher(
                    loss_type=canonical_loss,
                    student_logits=logits,
                    teacher_logits_cpu=targets.logits_cpu,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    temperature=float(self.loss_config.temperature),
                    alpha=float(self.loss_config.alpha),
                    top_k=resolved_top_k,
                    prompt_loss_weight=float(self.loss_config.prompt_loss_weight),
                    teacher_output_chunk_tokens=int(self.teacher_output_chunk_tokens),
                )

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
        # dispatching to the dense/CPU paths.
        if bool(getattr(model, "training", False)):
            original_labels = inputs.get("labels")
            if isinstance(original_labels, torch.Tensor):
                self.distill_token_stats.update(
                    original_labels, inputs.get("attention_mask")
                )
        if self.selective_student_topk:
            return self._compute_teacher_first_cpu_loss(model, inputs, return_outputs=bool(return_outputs))
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
