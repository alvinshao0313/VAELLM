"""Canonical CAT after-category model-level recovery helpers.

This module owns canonical after-category mode/config resolution and the shared
trainer/runtime used by every supported mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import TrainingArguments

from e2e_common.full_lora import (
    collect_exact_peft_lora_config,
    finalize_model_level_lora,
    iter_named_peft_lora_layers,
)
from e2e_common.lazy_datasets import default_dataloader_num_workers
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import NamedVAELinearTarget
from train_utils.config.configs import (
    AfterCategoryResolvedConfig,
)
from train_utils.config.targets import (
    discover_cat_projection_name_inventory,
    parse_skip_layers,
    parse_target_layers,
)
from train_utils.decoder_execution import prime_named_vae_linear_cache_with_group_fallback
from train_utils.distill_data import build_distill_data_collator, build_distill_dataset
from train_utils.distill_decoder import NamedMainDecoderTarget, finalize_main_decoder_targets
from train_utils.hif4_act import build_hif4_act_controller, register_hif4_act_hooks, remove_hif4_act_hooks
from train_utils.lora_training import CustomSFTTrainer
from train_utils.lora_utils import (
    _LoraDistillTokenStatsCallback,
    _LoraTrainerLogCallback,
    _ensure_lora_stack_available,
    _ensure_lora_tokenizer_ready,
    _replace_progress_log_callback,
    distill_distributed_barrier,
    is_distill_distributed,
    resolve_distill_train_device,
)
from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract
from train_utils.model_level_trainables import (
    build_model_level_trainable_selection,
    finalize_lm_head_linear_if_needed,
)


@dataclass(frozen=True)
class ResolvedCatAfterCategoryStage:
    mode: str
    config: AfterCategoryResolvedConfig
    train_device: str
    base_seed: int
    stage_seed: int
    output_dir: str
    deterministic: bool
    fp16: bool
    bf16: bool
    reset_completed: bool
    save_strategy: str = "steps"
    save_steps: float = 500
    save_total_limit: Optional[int] = None
    save_only_model: bool = False
    ignore_data_skip: bool = False


@dataclass(frozen=True)
class CanonicalCurrentDecoderResult:
    model: nn.Module
    did_train: bool
    current_lora_target_count: int
    decoder_target_count: int
    resolved_learning_rate: Optional[float]
    resolved_decoder_lr: Optional[float]
    distill_meta: Dict[str, object]


@dataclass(frozen=True)
class CanonicalRemainingFamilyResult:
    model: nn.Module
    did_train: bool
    remaining_lora_target_count: int
    decoder_target_count: int
    resolved_learning_rate: Optional[float]
    resolved_decoder_lr: Optional[float]
    distill_meta: Dict[str, object]


def resolve_canonical_after_category_mode(cat_args) -> str:
    """Resolve the canonical common CAT mode."""
    return str(getattr(cat_args, "after_category_mode", "none")).strip().lower()


def resolve_cat_after_category_stage(
    cat_args,
    training_args,
    *,
    category: str,
    round_idx: int,
) -> ResolvedCatAfterCategoryStage:
    mode = resolve_canonical_after_category_mode(cat_args)
    resolver = getattr(cat_args, "resolve_after_category_config", None)
    if not callable(resolver):
        raise TypeError("CAT runtime must provide the common resolve_after_category_config callback.")
    config = resolver(str(category))
    base_seed = int(config.data.seed)
    if not isinstance(config, AfterCategoryResolvedConfig):
        raise TypeError(
            "CAT after-category resolver must return AfterCategoryResolvedConfig, "
            f"got {type(config)}."
        )
    return ResolvedCatAfterCategoryStage(
        mode=str(mode),
        config=config,
        train_device=resolve_distill_train_device(str(getattr(cat_args, "train_device", "cuda"))),
        base_seed=base_seed,
        stage_seed=int(base_seed + int(round_idx)),
        output_dir=str(getattr(cat_args, "output_dir", ".result/catlora")),
        deterministic=bool(getattr(cat_args, "deterministic", False)),
        fp16=bool(getattr(training_args, "fp16", False)),
        bf16=bool(getattr(training_args, "bf16", False)),
        reset_completed=bool(getattr(cat_args, "distill_reset_completed", False)),
        save_strategy=str(getattr(training_args, "save_strategy", "steps")),
        save_steps=float(getattr(training_args, "save_steps", 500)),
        save_total_limit=getattr(training_args, "save_total_limit", None),
        save_only_model=bool(getattr(training_args, "save_only_model", False)),
        ignore_data_skip=bool(getattr(training_args, "ignore_data_skip", False)),
    )


def resolve_exact_current_compressed_targets(
    model: nn.Module,
    *,
    category: str,
    target_names: Sequence[str],
) -> Tuple[Tuple[str, VAELinear], ...]:
    names = tuple(str(name) for name in target_names)
    if len(set(names)) != len(names):
        raise ValueError("current-category target inventory contains duplicate module names.")
    modules = dict(model.named_modules())
    resolved = []
    for name in names:
        if name.rsplit(".", 1)[-1] != str(category):
            raise ValueError(
                f"Current-category target {name!r} does not belong to category {category!r}."
            )
        module = modules.get(name)
        if not isinstance(module, VAELinear):
            raise TypeError(
                f"Current-category target {name!r} must resolve to VAELinear after conversion, "
                f"got {type(module)}."
            )
        if bool(getattr(module, "always_use_original", False)):
            raise ValueError(f"Current-category target {name!r} is original-only, not compressed.")
        resolved.append((name, module))
    return tuple(resolved)


def _resolve_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    current = model
    for token in str(module_name).split("."):
        try:
            current = getattr(current, token)
        except AttributeError as exc:
            raise ValueError(
                f"Failed to resolve CAT inventory module {module_name!r}: missing token {token!r}."
            ) from exc
    if not isinstance(current, nn.Module):
        raise TypeError(f"CAT inventory object {module_name!r} is not nn.Module: {type(current)}.")
    return current


def get_or_build_cat_projection_name_inventory(
    model: nn.Module,
    *,
    vae_args,
    compression_categories: Sequence[str],
) -> Dict[Tuple[int, str], str]:
    """Build CAT logical projection inventory once and reuse it across recovery rounds."""
    categories = tuple(str(category) for category in compression_categories)
    cache = getattr(vae_args, "_canonical_cat_projection_name_inventory", None)
    cache_categories = getattr(vae_args, "_canonical_cat_projection_inventory_categories", None)
    if cache is None:
        cache = discover_cat_projection_name_inventory(
            model,
            compression_categories=categories,
        )
        setattr(vae_args, "_canonical_cat_projection_name_inventory", dict(cache))
        setattr(vae_args, "_canonical_cat_projection_inventory_categories", categories)
    else:
        if tuple(cache_categories or ()) != categories:
            raise ValueError(
                "Cached CAT projection inventory categories mismatch: "
                f"cached={tuple(cache_categories or ())} current={categories}."
            )
        if not isinstance(cache, dict):
            raise TypeError("Cached CAT projection name inventory must be dict.")
    return dict(cache)


def _coerce_target_layers(raw):
    if raw == "all":
        return "all"
    if isinstance(raw, tuple):
        return tuple(int(v) for v in raw)
    return parse_target_layers(raw)


def _coerce_skip_layers(raw):
    if raw is None or raw == "":
        return frozenset()
    if isinstance(raw, (set, frozenset)):
        return frozenset((int(layer), str(category)) for layer, category in raw)
    return parse_skip_layers(raw)


def select_remaining_dense_names_from_inventory(
    model: nn.Module,
    *,
    inventory: Dict[Tuple[int, str], str],
    remaining_categories: Sequence[str],
    target_layers,
    skip_layers,
) -> Tuple[str, ...]:
    remaining_set = {str(category) for category in remaining_categories}
    resolved_layers = _coerce_target_layers(target_layers)
    allowed_layers = None if resolved_layers == "all" else {int(v) for v in resolved_layers}
    skipped = _coerce_skip_layers(skip_layers)
    selected = []
    for (layer_idx, category), name in inventory.items():
        key = (int(layer_idx), str(category))
        if str(category) not in remaining_set:
            continue
        if allowed_layers is not None and int(layer_idx) not in allowed_layers:
            continue
        if key in skipped:
            continue
        module = _resolve_module_by_name(model, name)
        if not isinstance(module, nn.Linear) or isinstance(module, VAELinear):
            raise TypeError(
                "Remaining CAT target must still be ordinary nn.Linear before LoRA: "
                f"name={name!r} key={key} got={type(module)}."
            )
        selected.append(str(name))
    return tuple(selected)


def select_compressed_decoder_targets_from_inventory(
    model: nn.Module,
    *,
    inventory: Dict[Tuple[int, str], str],
    decoder_categories: Sequence[str],
    target_layers,
    skip_layers,
) -> Tuple[Tuple[str, VAELinear], ...]:
    category_set = {str(category) for category in decoder_categories}
    resolved_layers = _coerce_target_layers(target_layers)
    allowed_layers = None if resolved_layers == "all" else {int(v) for v in resolved_layers}
    skipped = _coerce_skip_layers(skip_layers)
    selected = []
    for (layer_idx, category), name in inventory.items():
        key = (int(layer_idx), str(category))
        if str(category) not in category_set:
            continue
        if allowed_layers is not None and int(layer_idx) not in allowed_layers:
            continue
        if key in skipped:
            continue
        module = _resolve_module_by_name(model, name)
        if isinstance(module, nn.Linear) and not isinstance(module, VAELinear):
            # A skipped/tail target has no compressed decoder and is therefore not
            # part of prefix-decoder recovery.
            continue
        if not isinstance(module, VAELinear):
            raise TypeError(
                "CAT decoder inventory target must be VAELinear or an uncompressed nn.Linear: "
                f"name={name!r} key={key} got={type(module)}."
            )
        if bool(getattr(module, "always_use_original", False)):
            raise ValueError(f"CAT decoder target {name!r} is original-only.")
        selected.append((str(name), module))
    return tuple(selected)


def _collect_current_full_low_rank_payloads(
    targets: Sequence[Tuple[str, VAELinear]],
    *,
    reset_completed: bool,
) -> Tuple[Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], bool]:
    present = []
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, module in targets:
        has_a = getattr(module, "low_rank_a", None) is not None
        has_b = getattr(module, "low_rank_b", None) is not None
        if has_a != has_b:
            raise ValueError(f"{name}: existing current-category low-rank payload is incomplete.")
        present.append(bool(has_a))
        if not has_a:
            continue
        low_rank_a = module.low_rank_a.detach().to("cpu").clone().contiguous()
        low_rank_b = module.low_rank_b.detach().to("cpu").clone().contiguous()
        module._validate_low_rank_payload_tensors(low_rank_a, low_rank_b)
        payloads[str(name)] = (low_rank_a, low_rank_b)

    if any(present) and not all(present):
        raise ValueError(
            "Current-category full LoRA payload presence is partial across exact targets; "
            "cannot decide skip/resume semantics."
        )
    if present and all(present) and not bool(reset_completed):
        return payloads, True
    return (payloads if payloads else None), False


class CanonicalCatSFTTrainer(CustomSFTTrainer):
    """CAT shared Trainer with canonical teacher output residency controls."""

    def __init__(
        self,
        *args,
        teacher_output_offload: str,
        teacher_output_pin_memory: bool,
        teacher_output_chunk_tokens: int,
        **kwargs,
    ):
        offload = str(teacher_output_offload).strip().lower()
        if offload not in {"none", "cpu"}:
            raise ValueError("teacher_output_offload must be none or cpu.")
        chunk_tokens = int(teacher_output_chunk_tokens)
        if chunk_tokens < 1:
            raise ValueError("teacher_output_chunk_tokens must be >= 1.")
        kwargs["teacher_logits_cpu_staging"] = offload == "cpu"
        kwargs["selective_teacher_topk_chunk_tokens"] = chunk_tokens
        super().__init__(*args, **kwargs)
        self.teacher_output_offload = offload
        self.teacher_output_pin_memory = bool(teacher_output_pin_memory)
        self.teacher_output_chunk_tokens = chunk_tokens
        self._v6_step_checkpoint_context = None
        self._v6_selected_vae_modules = ()
        self._v6_exact_resume_loaded = False

    def _must_stage_teacher_targets_to_cpu(self) -> bool:
        return bool(
            self.teacher_output_offload == "cpu"
            or super()._must_stage_teacher_targets_to_cpu()
        )

    def _normalize_topk_residency(self, targets):
        compact = getattr(targets, "selective_topk", None)
        if compact is None:
            return targets
        compact.transfer_chunk_size = int(self.teacher_output_chunk_tokens)
        should_pin = bool(
            self.teacher_output_offload == "cpu"
            and self.teacher_output_pin_memory
            and torch.cuda.is_available()
        )
        for attr in ("indices_cpu", "logits_cpu"):
            tensor = getattr(compact, attr)
            if should_pin and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            elif not should_pin and tensor.is_pinned():
                copied = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
                copied.copy_(tensor)
                tensor = copied
            setattr(compact, attr, tensor)
        return targets

    def _run_teacher_forward(self, **kwargs):
        return self._normalize_topk_residency(super()._run_teacher_forward(**kwargs))

    def _stage_teacher_logits(self, logits: torch.Tensor) -> torch.Tensor:
        staged = super()._stage_teacher_logits(logits)
        if staged.device.type != "cpu":
            return staged
        should_pin = bool(
            self.teacher_output_offload == "cpu"
            and self.teacher_output_pin_memory
            and torch.cuda.is_available()
        )
        if should_pin and not staged.is_pinned():
            return staged.pin_memory()
        if not should_pin and staged.is_pinned():
            copied = torch.empty(staged.shape, dtype=staged.dtype, device="cpu")
            copied.copy_(staged)
            return copied
        return staged

    def _teacher_logits_for_loss(
        self,
        staged_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        if staged_logits.device.type != "cpu" or student_logits.device.type != "cuda":
            return super()._teacher_logits_for_loss(staged_logits, student_logits)
        if staged_logits.ndim != 3:
            raise ValueError(
                f"teacher logits must have shape [B,L,V], got {tuple(staged_logits.shape)}."
            )
        target = torch.empty(
            staged_logits.shape,
            dtype=staged_logits.dtype,
            device=student_logits.device,
        )
        non_blocking = bool(staged_logits.is_pinned())
        seq_len = int(staged_logits.shape[1])
        for start in range(0, seq_len, int(self.teacher_output_chunk_tokens)):
            end = min(seq_len, start + int(self.teacher_output_chunk_tokens))
            target[:, start:end, :].copy_(
                staged_logits[:, start:end, :],
                non_blocking=non_blocking,
            )
        return target

    def configure_v6_step_checkpoint(self, *, context: dict, selected_vae_modules) -> None:
        if not isinstance(context, dict):
            raise TypeError(f"CAT v6 step checkpoint context must be dict, got {type(context)}.")
        required = {
            "round_base_dir",
            "round_base_checkpoint_id",
            "active_category",
            "after_category_mode",
            "compressed_targets",
            "pending_dense_targets",
            "skip_targets",
            "completed_categories",
            "compression_categories",
            "target_layers",
            "target_modules",
            "lora_config",
            "resolved_learning_rates",
            "immutable_resume_contract",
            "base_model_path",
        }
        missing = sorted(required - set(context))
        if missing:
            raise ValueError(f"CAT v6 step checkpoint context missing required fields: {missing}")
        round_base_dir = os.path.abspath(str(context["round_base_dir"]))
        if not os.path.isdir(round_base_dir):
            raise FileNotFoundError(f"CAT v6 round_base_dir does not exist: {round_base_dir}")
        if bool(getattr(self.args, "save_only_model", False)):
            raise ValueError("CAT v6 exact-step resume requires save_only_model=false.")
        if bool(getattr(self.args, "ignore_data_skip", False)):
            raise ValueError("CAT v6 exact-step resume requires ignore_data_skip=false.")
        normalized = dict(context)
        normalized["round_base_dir"] = round_base_dir
        self._v6_step_checkpoint_context = normalized
        self._v6_selected_vae_modules = tuple(selected_vae_modules or ())
        self._v6_exact_resume_loaded = False

    def _v6_step_checkpoint_enabled(self) -> bool:
        return isinstance(self._v6_step_checkpoint_context, dict)

    def save_model(self, output_dir=None, _internal_call: bool = False):
        if self._v6_step_checkpoint_enabled() and bool(_internal_call):
            target = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(target, exist_ok=True)
            return None
        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _save_checkpoint(self, model, trial):
        if not self._v6_step_checkpoint_enabled():
            return super()._save_checkpoint(model, trial)
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        from train_utils.checkpoint_v6 import save_v6_training_step_payload
        from train_utils.model_level_checkpoint_state import collect_model_level_mutable_state

        context = dict(self._v6_step_checkpoint_context or {})
        result = super()._save_checkpoint(model, trial)
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, f"{PREFIX_CHECKPOINT_DIR}-{int(self.state.global_step)}")
        unwrapped = (
            self.accelerator.unwrap_model(self.model)
            if getattr(self, "accelerator", None) is not None
            else self.model
        )
        selection = getattr(self, "model_level_trainable_selection", None)
        mutable_state, _component_classes, manifest = collect_model_level_mutable_state(
            unwrapped,
            selection=selection,
            selected_vae_modules=self._v6_selected_vae_modules,
        )
        round_base_ref = os.path.relpath(
            str(context["round_base_dir"]),
            start=os.path.abspath(output_dir),
        )
        is_main = bool(self.is_world_process_zero())
        distributed_barrier = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            distributed_barrier = torch.distributed.barrier
        save_v6_training_step_payload(
            output_dir,
            round_base_ref=round_base_ref,
            round_base_checkpoint_id=str(context["round_base_checkpoint_id"]),
            mutable_state=mutable_state,
            mutable_state_manifest=manifest,
            train_mode="none",
            after_category_mode=str(context["after_category_mode"]),
            compressed_targets=tuple(context["compressed_targets"]),
            pending_dense_targets=tuple(context["pending_dense_targets"]),
            skip_targets=tuple(context["skip_targets"]),
            norm_train_mode=str(context.get("norm_train_mode", "none")),
            lm_head_train_mode=str(context.get("lm_head_train_mode", "none")),
            lora_config=(
                None
                if context.get("lora_config") is None
                else dict(context["lora_config"])
            ),
            resolved_learning_rates=dict(context["resolved_learning_rates"]),
            completed_categories=tuple(context["completed_categories"]),
            compression_categories=tuple(context["compression_categories"]),
            target_layers=context.get("target_layers"),
            target_modules=tuple(context["target_modules"]),
            immutable_resume_contract=dict(context["immutable_resume_contract"]),
            runtime_audit=dict(context.get("runtime_audit", {})),
            base_model_path=str(context["base_model_path"]),
            extra_meta={
                "active_category": str(context["active_category"]),
                "distill_stage_history": [dict(item) for item in context.get("distill_stage_history", ())],
                "round_idx": int(context.get("round_idx", 0)),
            },
            is_main_process=is_main,
            distributed_barrier=distributed_barrier,
        )
        return result

    def _load_v6_exact_step_checkpoint(self, resume_from_checkpoint, model=None):
        from train_utils.cat_step_resume_v6 import validate_cat_step_immutable_resume_contract
        from train_utils.checkpoint_v6 import (
            load_v6_training_model_state,
            load_v6_training_step_meta,
            resolve_training_step_round_base_ref,
        )
        from train_utils.model_level_checkpoint_state import restore_model_level_mutable_state

        context = dict(self._v6_step_checkpoint_context or {})
        checkpoint_dir = os.path.abspath(str(resume_from_checkpoint))
        meta = load_v6_training_step_meta(checkpoint_dir)
        _round_base_dir, base_meta = resolve_training_step_round_base_ref(checkpoint_dir, meta)
        if str(base_meta["checkpoint_id"]) != str(context["round_base_checkpoint_id"]):
            raise ValueError("CAT v6 resume round-base checkpoint_id mismatch.")
        expected = {
            "after_category_mode": str(context["after_category_mode"]),
            "compressed_targets": list(context["compressed_targets"]),
            "pending_dense_targets": list(context["pending_dense_targets"]),
            "skip_targets": list(context["skip_targets"]),
            "completed_categories": list(context["completed_categories"]),
            "compression_categories": list(context["compression_categories"]),
            "target_layers": None if context.get("target_layers") is None else list(context["target_layers"]),
            "target_modules": list(context["target_modules"]),
        }
        for key, expected_value in expected.items():
            actual = meta.get(key)
            if actual != expected_value:
                raise ValueError(
                    f"CAT v6 resume topology mismatch for {key}: checkpoint={actual!r} current={expected_value!r}."
                )
        extra_meta = meta.get("extra_meta") or {}
        if extra_meta.get("active_category") != str(context["active_category"]):
            raise ValueError("CAT v6 resume active_category mismatch.")
        validate_cat_step_immutable_resume_contract(
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
        restore_model_level_mutable_state(
            unwrapped,
            selection=getattr(self, "model_level_trainable_selection", None),
            selected_vae_modules=self._v6_selected_vae_modules,
            checkpoint_state=checkpoint_state,
            checkpoint_manifest=checkpoint_manifest,
        )
        self._v6_exact_resume_loaded = True
        return None

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if self._v6_step_checkpoint_enabled():
            return self._load_v6_exact_step_checkpoint(resume_from_checkpoint, model=model)
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)


def _build_training_arguments(
    stage: ResolvedCatAfterCategoryStage,
    *,
    is_iterable: bool,
    group_by_length: bool,
) -> TrainingArguments:
    cfg = stage.config
    kwargs = dict(
        output_dir=os.path.join(stage.output_dir, "after_category_trainer_state"),
        per_device_train_batch_size=int(cfg.opt.batch_size),
        gradient_accumulation_steps=int(cfg.opt.gradient_accumulation_steps),
        gradient_checkpointing=bool(cfg.opt.gradient_checkpointing),
        gradient_checkpointing_kwargs=dict(cfg.opt.gradient_checkpointing_kwargs),
        optim=str(cfg.opt.optim),
        logging_strategy="steps",
        logging_steps=max(1, int(cfg.opt.logging_steps)),
        logging_first_step=True,
        learning_rate=float(cfg.opt.learning_rate),
        weight_decay=float(cfg.opt.weight_decay),
        fp16=bool(stage.fp16),
        bf16=bool(stage.bf16),
        max_grad_norm=float(cfg.opt.max_grad_norm),
        max_steps=int(cfg.opt.steps),
        warmup_ratio=float(cfg.opt.warmup_ratio),
        group_by_length=bool(group_by_length),
        lr_scheduler_type=str(cfg.opt.lr_scheduler_type),
        report_to=[],
        disable_tqdm=bool(os.environ.get("RANK", "0") != "0"),
        log_level="info" if os.environ.get("RANK", "0") == "0" else "error",
        log_level_replica="error",
        save_strategy=str(stage.save_strategy),
        save_steps=float(stage.save_steps),
        save_total_limit=(None if stage.save_total_limit is None else int(stage.save_total_limit)),
        save_only_model=bool(stage.save_only_model),
        ignore_data_skip=bool(stage.ignore_data_skip),
        save_safetensors=False,
        seed=int(stage.stage_seed),
        data_seed=int(cfg.data.data_seed),
        full_determinism=bool(stage.deterministic),
        dataloader_num_workers=int(default_dataloader_num_workers()),
        dataloader_pin_memory=True,
    )
    if bool(is_iterable):
        kwargs["group_by_length"] = False
        kwargs["accelerator_config"] = {"dispatch_batches": False, "split_batches": False}
    if is_distill_distributed():
        kwargs["ddp_find_unused_parameters"] = True
    return TrainingArguments(**kwargs)


def _prepare_after_category_dataset(
    *,
    model: nn.Module,
    vae_args,
    cfg: AfterCategoryResolvedConfig,
):
    _ensure_lora_stack_available()
    _ensure_lora_tokenizer_ready(vae_args=vae_args, model=model)
    tokenizer = getattr(vae_args, "_cached_lora_tokenizer", None)
    if tokenizer is None:
        raise ValueError("CAT after-category recovery requires the shared cached distill tokenizer.")
    dataset_cache = getattr(vae_args, "_cached_canonical_after_category_datasets", None)
    if not isinstance(dataset_cache, dict):
        dataset_cache = {}
        setattr(vae_args, "_cached_canonical_after_category_datasets", dataset_cache)
    cache_key = (
        cfg.data.dataset_mix,
        cfg.data.dataset_task,
        int(cfg.data.model_max_length),
        int(cfg.data.seed),
        int(cfg.data.data_seed),
        id(tokenizer),
    )
    bundle = dataset_cache.get(cache_key)
    if bundle is None:
        bundle = build_distill_dataset(cfg.data, tokenizer)
        dataset_cache[cache_key] = bundle
    return tokenizer, bundle


def _train_model_level_selection(
    *,
    selection,
    stage: ResolvedCatAfterCategoryStage,
    tokenizer,
    bundle,
    teacher_runtime,
    logger,
    v6_step_checkpoint: Optional[dict] = None,
):
    cfg = stage.config
    model = selection.peft_model if selection.peft_model is not None else None
    if not isinstance(model, nn.Module):
        raise TypeError("ModelLevelTrainableSelection.peft_model must be an nn.Module.")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.to(stage.train_device)
    model.train()

    training_args = _build_training_arguments(
        stage,
        is_iterable=bool(bundle.is_iterable),
        group_by_length=bool(bundle.group_by_length),
    )
    if v6_step_checkpoint is not None:
        trainer_output_dir = v6_step_checkpoint.get("trainer_output_dir")
        if not isinstance(trainer_output_dir, str) or not trainer_output_dir.strip():
            raise ValueError("CAT v6 step checkpoint context requires trainer_output_dir.")
        training_args.output_dir = os.path.abspath(trainer_output_dir)
        if bool(getattr(training_args, "save_only_model", False)):
            raise ValueError("CAT exact-step resume requires save_only_model=false.")
        if bool(getattr(training_args, "ignore_data_skip", False)):
            raise ValueError("CAT exact-step resume requires ignore_data_skip=false.")
    controller = build_hif4_act_controller(bool(cfg.runtime.distill_hif4_act))
    trainer = CanonicalCatSFTTrainer(
        model=model,
        train_dataset=bundle.train_dataset,
        eval_dataset=bundle.eval_dataset,
        args=training_args,
        callbacks=[_LoraTrainerLogCallback(logger=logger)],
        processing_class=tokenizer,
        data_collator=build_distill_data_collator(
            tokenizer,
            model_max_length=int(cfg.data.model_max_length),
            dynamic_padding=bool(cfg.data.dynamic_padding),
        ),
        loss_config=cfg.loss,
        teacher_runtime=teacher_runtime,
        distill_hif4_act_controller=controller,
        teacher_output_offload=str(cfg.runtime.teacher_output_offload),
        teacher_output_pin_memory=bool(cfg.runtime.teacher_output_pin_memory),
        teacher_output_chunk_tokens=int(cfg.runtime.teacher_output_chunk_tokens),
    )
    trainer = _replace_progress_log_callback(trainer)
    trainer.add_callback(_LoraDistillTokenStatsCallback(trainer=trainer, logger=logger))
    attach_model_level_optimizer_contract(
        trainer,
        selection=selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=float(cfg.opt.learning_rate),
            weight_decay=float(cfg.opt.weight_decay),
            decoder_lr=(
                float(cfg.opt.resolved_decoder_lr())
                if selection.decoder_parameters
                else None
            ),
            norm_lr=cfg.aux.norm_lr,
            lm_head_lr=cfg.aux.lm_head_lr,
        ),
    )

    resume_from_checkpoint = None
    if v6_step_checkpoint is not None:
        from train_utils.cat_step_resume_v6 import (
            build_cat_step_immutable_resume_contract,
            build_distill_dataset_identity,
            model_identity,
        )
        from train_utils.distill_teacher import resolve_distill_teacher_required

        round_base_meta = v6_step_checkpoint.get("round_base_meta")
        if not isinstance(round_base_meta, dict):
            raise TypeError("CAT v6 step checkpoint context requires round_base_meta dict.")
        teacher_identity = None
        teacher_required = resolve_distill_teacher_required(
            loss_type=str(cfg.loss.loss_type),
            hidden_loss_weight=float(cfg.loss.hidden_loss_weight),
            pre_mlp_hidden_loss_weight=float(cfg.loss.pre_mlp_hidden_loss_weight),
        )
        if teacher_required:
            if teacher_runtime is None:
                raise RuntimeError("CAT exact-step distillation requires teacher_runtime.")
            teacher_model = teacher_runtime.get_or_load()
            teacher_identity = model_identity(teacher_model, str(teacher_runtime.model_path))
        exact_lora_config = collect_exact_peft_lora_config(
            trainer.model,
            default_rank=int(cfg.lora.rank),
            alpha=float(cfg.lora.alpha),
            dropout=float(cfg.lora.dropout),
        )
        immutable_contract = build_cat_step_immutable_resume_contract(
            stage=stage,
            trainer_args=training_args,
            tokenizer=tokenizer,
            round_base_checkpoint_id=str(v6_step_checkpoint["round_base_checkpoint_id"]),
            active_category=str(v6_step_checkpoint["active_category"]),
            round_base_meta=round_base_meta,
            lora_target_names=tuple(v6_step_checkpoint.get("lora_target_names", ())),
            decoder_target_names=tuple(v6_step_checkpoint.get("decoder_target_names", ())),
            teacher_identity=teacher_identity,
            dataset_identity=build_distill_dataset_identity(bundle),
            lora_config=exact_lora_config,
        )
        checkpoint_context = dict(v6_step_checkpoint)
        checkpoint_context.update(
            {
                "after_category_mode": str(stage.mode),
                "compressed_targets": tuple(round_base_meta.get("compressed_targets") or ()),
                "pending_dense_targets": tuple(round_base_meta.get("pending_dense_targets") or ()),
                "skip_targets": tuple(round_base_meta.get("skip_targets") or ()),
                "completed_categories": tuple(round_base_meta.get("completed_categories") or ()),
                "compression_categories": tuple(round_base_meta.get("compression_categories") or ()),
                "target_layers": (
                    None
                    if round_base_meta.get("target_layers") is None
                    else tuple(int(v) for v in round_base_meta.get("target_layers") or ())
                ),
                "target_modules": tuple(round_base_meta.get("target_modules") or ()),
                "norm_train_mode": str(cfg.aux.norm_train_mode),
                "lm_head_train_mode": str(cfg.aux.lm_head_train_mode),
                "lora_config": exact_lora_config,
                "resolved_learning_rates": {
                    "learning_rate": float(cfg.opt.learning_rate),
                    "decoder_lr": (
                        float(cfg.opt.resolved_decoder_lr()) if selection.decoder_parameters else None
                    ),
                    "norm_lr": None if cfg.aux.norm_lr is None else float(cfg.aux.norm_lr),
                    "lm_head_lr": None if cfg.aux.lm_head_lr is None else float(cfg.aux.lm_head_lr),
                },
                "immutable_resume_contract": immutable_contract,
                "runtime_audit": {
                    "runtime": "train_utils.cat_after_category_common",
                    "active_category": str(v6_step_checkpoint["active_category"]),
                    "after_category_mode": str(stage.mode),
                    "recovery_lora_config": exact_lora_config,
                },
            }
        )
        trainer.configure_v6_step_checkpoint(
            context=checkpoint_context,
            selected_vae_modules=tuple(v6_step_checkpoint.get("selected_vae_modules", ())),
        )
        resume_from_checkpoint = v6_step_checkpoint.get("resume_from_checkpoint")

    hif4_handles = []
    if controller is not None:
        hif4_handles = register_hif4_act_hooks(trainer.model, controller)
        if not hif4_handles:
            raise RuntimeError(
                f"CAT {stage.mode} enabled HiF4 activation but registered no student hooks."
            )
        controller.enabled = True
    if selection.decoder_parameters:
        VAELinear.reset_fuse_stats()
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        if controller is not None:
            controller.enabled = False
        remove_hif4_act_hooks(hif4_handles)
    distill_distributed_barrier()
    return trainer.model


def _prewarm_non_current_vae_linears(
    model: nn.Module,
    *,
    selected_names: Sequence[str],
    compute_device: str,
    bf16: bool,
    logger,
) -> None:
    selected = set(str(name) for name in selected_names)
    prewarm = [
        NamedVAELinearTarget(name=str(name), base_layer=module)
        for name, module in model.named_modules()
        if isinstance(module, VAELinear) and str(name) not in selected
    ]
    if not prewarm:
        return
    stats, resolved_group = prime_named_vae_linear_cache_with_group_fallback(
        prewarm,
        dtype=torch.bfloat16 if bool(bf16) else None,
        clear_existing=False,
        compute_device=str(compute_device),
        logger=logger,
    )
    logger.info(
        "CAT current_decoder: prewarm non-current VAELinear stats=%s group_size=%d reason=%s",
        str(stats),
        int(resolved_group.group_size),
        str(resolved_group.fallback_reason),
    )


def _run_canonical_current_family(
    *,
    model: nn.Module,
    category: str,
    current_target_names: Sequence[str],
    newly_compressed_target_count: int,
    stage: ResolvedCatAfterCategoryStage,
    vae_args,
    logger,
    teacher_runtime=None,
    v6_step_checkpoint: Optional[dict] = None,
    expected_mode: str,
    train_lora: bool,
    train_decoder: bool,
) -> CanonicalCurrentDecoderResult:
    if stage.mode != str(expected_mode):
        raise ValueError(
            f"canonical current-family runner expected mode={expected_mode!r}, got {stage.mode!r}."
        )
    cfg = stage.config
    targets = resolve_exact_current_compressed_targets(
        model,
        category=str(category),
        target_names=current_target_names,
    )
    if int(newly_compressed_target_count) != len(targets):
        raise RuntimeError(
            "Current-family exact inventory mismatch: "
            f"mode={expected_mode} newly_compressed_target_count={int(newly_compressed_target_count)} "
            f"resolved_targets={len(targets)}."
        )
    effective_train_lora = bool(train_lora and targets)
    effective_train_decoder = bool(train_decoder and targets)
    aux_active = bool(
        str(cfg.aux.norm_train_mode) != "none"
        or str(cfg.aux.lm_head_train_mode) != "none"
    )
    if int(cfg.opt.steps) <= 0 or not (effective_train_lora or effective_train_decoder or aux_active):
        return CanonicalCurrentDecoderResult(
            model=model,
            did_train=False,
            current_lora_target_count=len(targets) if effective_train_lora else 0,
            decoder_target_count=len(targets) if effective_train_decoder else 0,
            resolved_learning_rate=float(cfg.opt.learning_rate) if effective_train_lora else None,
            resolved_decoder_lr=float(cfg.opt.resolved_decoder_lr()) if effective_train_decoder else None,
            distill_meta={
                "mode": str(expected_mode),
                "category": str(category),
                "did_train": False,
                "newly_compressed_target_count": int(newly_compressed_target_count),
                "current_lora_target_count": len(targets) if effective_train_lora else 0,
                "current_lora_targets": [name for name, _module in targets] if effective_train_lora else [],
                "remaining_lora_target_count": 0,
                "remaining_lora_targets": [],
                "decoder_target_count": len(targets) if effective_train_decoder else 0,
                "decoder_targets": [name for name, _module in targets] if effective_train_decoder else [],
            },
        )

    initial_low_rank_payloads = None
    if effective_train_lora:
        initial_low_rank_payloads, skip_completed = _collect_current_full_low_rank_payloads(
            targets,
            reset_completed=bool(stage.reset_completed),
        )
        if skip_completed:
            return CanonicalCurrentDecoderResult(
                model=model,
                did_train=False,
                current_lora_target_count=len(targets),
                decoder_target_count=len(targets) if effective_train_decoder else 0,
                resolved_learning_rate=float(cfg.opt.learning_rate),
                resolved_decoder_lr=float(cfg.opt.resolved_decoder_lr()) if effective_train_decoder else None,
                distill_meta={
                    "mode": str(expected_mode),
                    "category": str(category),
                    "did_train": False,
                    "skip_reason": "current_full_lora_already_finalized",
                    "newly_compressed_target_count": int(newly_compressed_target_count),
                    "current_lora_target_count": len(targets),
                    "current_lora_targets": [name for name, _module in targets],
                    "remaining_lora_target_count": 0,
                    "remaining_lora_targets": [],
                    "decoder_target_count": len(targets) if effective_train_decoder else 0,
                    "decoder_targets": [name for name, _module in targets] if effective_train_decoder else [],
                },
            )

    tokenizer, bundle = _prepare_after_category_dataset(
        model=model,
        vae_args=vae_args,
        cfg=cfg,
    )

    previous_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        previous_use_cache = bool(model.config.use_cache)
        model.config.use_cache = False

    _prewarm_non_current_vae_linears(
        model,
        selected_names=([name for name, _ in targets] if effective_train_decoder else ()),
        compute_device=stage.train_device,
        bf16=stage.bf16,
        logger=logger,
    )

    selection = build_model_level_trainable_selection(
        model,
        aux=cfg.aux,
        compressed_modules=targets,
        rank=int(cfg.lora.rank),
        alpha=float(cfg.lora.alpha),
        dropout=float(cfg.lora.dropout),
        rank_explicit=bool(cfg.lora.rank_explicit),
        initial_low_rank_payloads=initial_low_rank_payloads,
        train_decoder=effective_train_decoder,
        train_lora=effective_train_lora,
        freeze=True,
    )
    if effective_train_lora and not selection.lora_parameters:
        raise RuntimeError(f"{expected_mode} resolved no current-category LoRA trainable parameters.")
    if not effective_train_lora and selection.lora_parameters:
        raise RuntimeError(f"{expected_mode} must not create backbone/current LoRA trainables.")
    if effective_train_decoder and not selection.decoder_parameters:
        raise RuntimeError(f"{expected_mode} resolved no decoder trainable parameters.")
    if not effective_train_decoder and selection.decoder_parameters:
        raise RuntimeError(f"{expected_mode} unexpectedly created decoder trainables.")

    stage_v6_checkpoint = None
    if v6_step_checkpoint is not None:
        stage_v6_checkpoint = dict(v6_step_checkpoint)
        stage_v6_checkpoint.update(
            {
                "lora_target_names": tuple(name for name, _module in targets) if effective_train_lora else (),
                "decoder_target_names": tuple(name for name, _module in targets) if effective_train_decoder else (),
                "selected_vae_modules": tuple(targets) if effective_train_decoder else (),
            }
        )
    model = _train_model_level_selection(
        selection=selection,
        stage=stage,
        tokenizer=tokenizer,
        bundle=bundle,
        teacher_runtime=teacher_runtime,
        logger=logger,
        v6_step_checkpoint=stage_v6_checkpoint,
    )
    decoder_targets = tuple(
        NamedMainDecoderTarget(name=name, base_layer=module)
        for name, module in targets
    ) if effective_train_decoder else ()
    if decoder_targets:
        finalized = finalize_main_decoder_targets(decoder_targets)
        if int(finalized) != len(decoder_targets):
            raise RuntimeError(
                f"{expected_mode} finalized={int(finalized)} decoder_targets={len(decoder_targets)}."
            )

    if list(iter_named_peft_lora_layers(model)):
        model = finalize_model_level_lora(
            model,
            compressed_proxy_names=([name for name, _ in targets] if effective_train_lora else None),
        )
    finalize_lm_head_linear_if_needed(
        model,
        lm_head_train_mode=str(cfg.aux.lm_head_train_mode),
    )
    for param in model.parameters():
        param.requires_grad_(False)
    if previous_use_cache is not None and hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = bool(previous_use_cache)

    if effective_train_decoder:
        fuse_stats = VAELinear.get_fuse_stats()
        logger.info(
            "CAT %s: category=%s fuse_stats hit=%d miss=%d reasons=%s",
            str(expected_mode),
            str(category),
            int(fuse_stats["hit"]),
            int(fuse_stats["miss"]),
            str(fuse_stats["miss_reasons"]),
        )
    if not is_distill_distributed():
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    distill_distributed_barrier()

    resolved_learning_rate = float(cfg.opt.learning_rate) if effective_train_lora else None
    resolved_decoder_lr = float(cfg.opt.resolved_decoder_lr()) if effective_train_decoder else None
    return CanonicalCurrentDecoderResult(
        model=model,
        did_train=True,
        current_lora_target_count=len(targets) if effective_train_lora else 0,
        decoder_target_count=len(decoder_targets),
        resolved_learning_rate=resolved_learning_rate,
        resolved_decoder_lr=resolved_decoder_lr,
        distill_meta={
            "mode": str(expected_mode),
            "category": str(category),
            "did_train": True,
            "newly_compressed_target_count": int(newly_compressed_target_count),
            "current_lora_target_count": len(targets) if effective_train_lora else 0,
            "current_lora_targets": [name for name, _module in targets] if effective_train_lora else [],
            "remaining_lora_target_count": 0,
            "remaining_lora_targets": [],
            "decoder_target_count": len(decoder_targets),
            "decoder_targets": [target.name for target in decoder_targets],
            "resolved_distill_lr": float(cfg.opt.learning_rate),
            "resolved_decoder_lr": resolved_decoder_lr,
            "resolved_distill_weight_decay": float(cfg.opt.weight_decay),
            "decoder_weight_decay": 0.0 if effective_train_decoder else None,
            "norm_train_mode": str(cfg.aux.norm_train_mode),
            "norm_lr": None if cfg.aux.norm_lr is None else float(cfg.aux.norm_lr),
            "lm_head_train_mode": str(cfg.aux.lm_head_train_mode),
            "lm_head_lr": None if cfg.aux.lm_head_lr is None else float(cfg.aux.lm_head_lr),
            "teacher_output_offload": str(cfg.runtime.teacher_output_offload),
            "teacher_output_pin_memory": bool(cfg.runtime.teacher_output_pin_memory),
            "teacher_output_chunk_tokens": int(cfg.runtime.teacher_output_chunk_tokens),
        },
    )


def run_canonical_current_decoder(
    *,
    model: nn.Module,
    category: str,
    current_target_names: Sequence[str],
    newly_compressed_target_count: int,
    stage: ResolvedCatAfterCategoryStage,
    vae_args,
    logger,
    teacher_runtime=None,
    v6_step_checkpoint: Optional[dict] = None,
) -> CanonicalCurrentDecoderResult:
    return _run_canonical_current_family(
        model=model,
        category=category,
        current_target_names=current_target_names,
        newly_compressed_target_count=newly_compressed_target_count,
        stage=stage,
        vae_args=vae_args,
        logger=logger,
        teacher_runtime=teacher_runtime,
        v6_step_checkpoint=v6_step_checkpoint,
        expected_mode="current_decoder",
        train_lora=False,
        train_decoder=True,
    )


def run_canonical_current_lora(
    *,
    model: nn.Module,
    category: str,
    current_target_names: Sequence[str],
    newly_compressed_target_count: int,
    stage: ResolvedCatAfterCategoryStage,
    vae_args,
    logger,
    teacher_runtime=None,
    v6_step_checkpoint: Optional[dict] = None,
) -> CanonicalCurrentDecoderResult:
    return _run_canonical_current_family(
        model=model,
        category=category,
        current_target_names=current_target_names,
        newly_compressed_target_count=newly_compressed_target_count,
        stage=stage,
        vae_args=vae_args,
        logger=logger,
        teacher_runtime=teacher_runtime,
        v6_step_checkpoint=v6_step_checkpoint,
        expected_mode="current_lora",
        train_lora=True,
        train_decoder=False,
    )


def run_canonical_current_lora_decoder(
    *,
    model: nn.Module,
    category: str,
    current_target_names: Sequence[str],
    newly_compressed_target_count: int,
    stage: ResolvedCatAfterCategoryStage,
    vae_args,
    logger,
    teacher_runtime=None,
    v6_step_checkpoint: Optional[dict] = None,
) -> CanonicalCurrentDecoderResult:
    return _run_canonical_current_family(
        model=model,
        category=category,
        current_target_names=current_target_names,
        newly_compressed_target_count=newly_compressed_target_count,
        stage=stage,
        vae_args=vae_args,
        logger=logger,
        teacher_runtime=teacher_runtime,
        v6_step_checkpoint=v6_step_checkpoint,
        expected_mode="current_lora_decoder",
        train_lora=True,
        train_decoder=True,
    )


def _run_canonical_remaining_family(
    *,
    model: nn.Module,
    category: str,
    compression_categories: Sequence[str],
    newly_compressed_target_count: int,
    stage: ResolvedCatAfterCategoryStage,
    vae_args,
    logger,
    teacher_runtime=None,
    target_layers="all",
    skip_layers=(),
    v6_step_checkpoint: Optional[dict] = None,
    expected_mode: str,
) -> CanonicalRemainingFamilyResult:
    if stage.mode != str(expected_mode):
        raise ValueError(
            f"canonical remaining-family runner expected mode={expected_mode!r}, got {stage.mode!r}."
        )
    cfg = stage.config
    categories = tuple(str(item) for item in compression_categories)
    if len(categories) != len(set(categories)):
        raise ValueError("compression_categories contains duplicates.")
    if str(category) not in categories:
        raise ValueError(
            f"Current category {category!r} is not present in compression_categories={categories}."
        )
    current_idx = categories.index(str(category))
    remaining_categories = categories[current_idx + 1 :]
    inventory = get_or_build_cat_projection_name_inventory(
        model,
        vae_args=vae_args,
        compression_categories=categories,
    )
    remaining_names = select_remaining_dense_names_from_inventory(
        model,
        inventory=inventory,
        remaining_categories=remaining_categories,
        target_layers=target_layers,
        skip_layers=skip_layers,
    )

    current_decoder_targets = select_compressed_decoder_targets_from_inventory(
        model,
        inventory=inventory,
        decoder_categories=[str(category)],
        target_layers=target_layers,
        skip_layers=skip_layers,
    )
    if int(newly_compressed_target_count) > 0 and len(current_decoder_targets) != int(newly_compressed_target_count):
        raise RuntimeError(
            "Current-category compressed decoder inventory mismatch before remaining recovery: "
            f"category={category} newly_compressed_target_count={int(newly_compressed_target_count)} "
            f"resolved_current_decoders={len(current_decoder_targets)}."
        )

    if expected_mode == "remaining_lora":
        decoder_targets: Tuple[Tuple[str, VAELinear], ...] = ()
    elif expected_mode == "remaining_lora_current_decoder":
        decoder_targets = current_decoder_targets
    elif expected_mode == "remaining_lora_prefix_decoder":
        decoder_targets = select_compressed_decoder_targets_from_inventory(
            model,
            inventory=inventory,
            decoder_categories=categories[: current_idx + 1],
            target_layers=target_layers,
            skip_layers=skip_layers,
        )
    else:
        raise ValueError(f"Unsupported canonical remaining mode: {expected_mode!r}.")

    aux_active = bool(
        str(cfg.aux.norm_train_mode) != "none"
        or str(cfg.aux.lm_head_train_mode) != "none"
    )
    has_stage_trainables = bool(remaining_names or decoder_targets or aux_active)
    if not has_stage_trainables or int(cfg.opt.steps) <= 0:
        return CanonicalRemainingFamilyResult(
            model=model,
            did_train=False,
            remaining_lora_target_count=len(remaining_names),
            decoder_target_count=len(decoder_targets),
            resolved_learning_rate=float(cfg.opt.learning_rate) if remaining_names else None,
            resolved_decoder_lr=float(cfg.opt.resolved_decoder_lr()) if decoder_targets else None,
            distill_meta={
                "mode": str(expected_mode),
                "category": str(category),
                "did_train": False,
                "newly_compressed_target_count": int(newly_compressed_target_count),
                "current_lora_target_count": 0,
                "current_lora_targets": [],
                "remaining_lora_target_count": len(remaining_names),
                "remaining_lora_targets": list(remaining_names),
                "decoder_target_count": len(decoder_targets),
                "decoder_targets": [name for name, _module in decoder_targets],
            },
        )

    tokenizer, bundle = _prepare_after_category_dataset(
        model=model,
        vae_args=vae_args,
        cfg=cfg,
    )
    previous_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        previous_use_cache = bool(model.config.use_cache)
        model.config.use_cache = False

    _prewarm_non_current_vae_linears(
        model,
        selected_names=[name for name, _module in decoder_targets],
        compute_device=stage.train_device,
        bf16=stage.bf16,
        logger=logger,
    )

    selection = build_model_level_trainable_selection(
        model,
        aux=cfg.aux,
        compressed_modules=(),
        dense_target_modules=remaining_names,
        decoder_modules=decoder_targets,
        rank=int(cfg.lora.rank),
        alpha=float(cfg.lora.alpha),
        dropout=float(cfg.lora.dropout),
        rank_explicit=bool(cfg.lora.rank_explicit),
        initial_low_rank_payloads=None,
        train_decoder=bool(decoder_targets),
        train_lora=bool(remaining_names),
        freeze=True,
    )
    if selection.compressed_lora_targets:
        raise RuntimeError(
            f"{expected_mode} must not create compressed/current LoRA targets: "
            f"{selection.compressed_lora_targets}."
        )
    if remaining_names and not selection.lora_parameters:
        raise RuntimeError(f"{expected_mode} resolved no remaining dense LoRA parameters.")
    if not remaining_names and selection.lora_parameters:
        raise RuntimeError(f"{expected_mode} created unexpected backbone LoRA parameters.")
    if decoder_targets and not selection.decoder_parameters:
        raise RuntimeError(f"{expected_mode} resolved no decoder parameters.")
    if not decoder_targets and selection.decoder_parameters:
        raise RuntimeError(f"{expected_mode} created unexpected decoder parameters.")

    stage_v6_checkpoint = None
    if v6_step_checkpoint is not None:
        stage_v6_checkpoint = dict(v6_step_checkpoint)
        stage_v6_checkpoint.update(
            {
                "lora_target_names": tuple(str(name) for name in remaining_names),
                "decoder_target_names": tuple(name for name, _module in decoder_targets),
                "selected_vae_modules": tuple(decoder_targets),
            }
        )
    model = _train_model_level_selection(
        selection=selection,
        stage=stage,
        tokenizer=tokenizer,
        bundle=bundle,
        teacher_runtime=teacher_runtime,
        logger=logger,
        v6_step_checkpoint=stage_v6_checkpoint,
    )

    named_decoder_targets = tuple(
        NamedMainDecoderTarget(name=name, base_layer=module)
        for name, module in decoder_targets
    )
    if named_decoder_targets:
        finalized = finalize_main_decoder_targets(named_decoder_targets)
        if int(finalized) != len(named_decoder_targets):
            raise RuntimeError(
                f"{expected_mode} finalized={int(finalized)} decoder_targets={len(named_decoder_targets)}."
            )
    if list(iter_named_peft_lora_layers(model)):
        # remaining family contains no compressed carrier LoRA, so this is the
        # standard PEFT dense merge-and-unload path.
        model = finalize_model_level_lora(model, compressed_proxy_names=None)
    finalize_lm_head_linear_if_needed(
        model,
        lm_head_train_mode=str(cfg.aux.lm_head_train_mode),
    )
    for param in model.parameters():
        param.requires_grad_(False)
    if previous_use_cache is not None and hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = bool(previous_use_cache)

    if decoder_targets:
        fuse_stats = VAELinear.get_fuse_stats()
        logger.info(
            "CAT %s: category=%s fuse_stats hit=%d miss=%d reasons=%s",
            str(expected_mode),
            str(category),
            int(fuse_stats["hit"]),
            int(fuse_stats["miss"]),
            str(fuse_stats["miss_reasons"]),
        )
    if not is_distill_distributed():
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    distill_distributed_barrier()

    resolved_learning_rate = float(cfg.opt.learning_rate) if remaining_names else None
    resolved_decoder_lr = float(cfg.opt.resolved_decoder_lr()) if decoder_targets else None
    return CanonicalRemainingFamilyResult(
        model=model,
        did_train=True,
        remaining_lora_target_count=len(remaining_names),
        decoder_target_count=len(decoder_targets),
        resolved_learning_rate=resolved_learning_rate,
        resolved_decoder_lr=resolved_decoder_lr,
        distill_meta={
            "mode": str(expected_mode),
            "category": str(category),
            "did_train": True,
            "newly_compressed_target_count": int(newly_compressed_target_count),
            "current_lora_target_count": 0,
            "current_lora_targets": [],
            "remaining_lora_target_count": len(remaining_names),
            "remaining_lora_targets": list(remaining_names),
            "decoder_target_count": len(decoder_targets),
            "decoder_targets": [name for name, _module in decoder_targets],
            "resolved_distill_lr": float(cfg.opt.learning_rate),
            "resolved_decoder_lr": resolved_decoder_lr,
            "resolved_distill_weight_decay": float(cfg.opt.weight_decay),
            "decoder_weight_decay": 0.0 if decoder_targets else None,
            "norm_train_mode": str(cfg.aux.norm_train_mode),
            "norm_lr": None if cfg.aux.norm_lr is None else float(cfg.aux.norm_lr),
            "lm_head_train_mode": str(cfg.aux.lm_head_train_mode),
            "lm_head_lr": None if cfg.aux.lm_head_lr is None else float(cfg.aux.lm_head_lr),
            "teacher_output_offload": str(cfg.runtime.teacher_output_offload),
            "teacher_output_pin_memory": bool(cfg.runtime.teacher_output_pin_memory),
            "teacher_output_chunk_tokens": int(cfg.runtime.teacher_output_chunk_tokens),
        },
    )


def run_canonical_remaining_lora(**kwargs) -> CanonicalRemainingFamilyResult:
    return _run_canonical_remaining_family(expected_mode="remaining_lora", **kwargs)


def run_canonical_remaining_lora_current_decoder(**kwargs) -> CanonicalRemainingFamilyResult:
    return _run_canonical_remaining_family(
        expected_mode="remaining_lora_current_decoder",
        **kwargs,
    )


def run_canonical_remaining_lora_prefix_decoder(**kwargs) -> CanonicalRemainingFamilyResult:
    return _run_canonical_remaining_family(
        expected_mode="remaining_lora_prefix_decoder",
        **kwargs,
    )


__all__ = [
    "CanonicalCatSFTTrainer",
    "CanonicalCurrentDecoderResult",
    "CanonicalRemainingFamilyResult",
    "ResolvedCatAfterCategoryStage",
    "get_or_build_cat_projection_name_inventory",
    "resolve_canonical_after_category_mode",
    "resolve_cat_after_category_stage",
    "resolve_exact_current_compressed_targets",
    "run_canonical_current_decoder",
    "run_canonical_current_lora",
    "run_canonical_current_lora_decoder",
    "run_canonical_remaining_lora",
    "run_canonical_remaining_lora_current_decoder",
    "run_canonical_remaining_lora_prefix_decoder",
    "select_compressed_decoder_targets_from_inventory",
    "select_remaining_dense_names_from_inventory",
]
