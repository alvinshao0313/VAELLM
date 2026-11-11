"""Build the actual PUBLIC native KD trajectory from its own fitted round_base."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch

from e2e_common.full_lora import collect_exact_peft_lora_config
from train_utils import cat_after_category_common as canonical
from train_utils import checkpoint_v6 as v6
from train_utils.cat_step_resume_v6 import build_cat_step_immutable_resume_contract, model_identity
from train_utils.config.configs import (AfterCategoryResolvedConfig, AuxTrainableConfig, DistillDataConfig,
    DistillLossConfig, DistillOptimizationConfig, DistillRuntimeConfig, LoRAConfig)
from train_utils.distill_precision import install_precision_runtime
from train_utils.distill_teacher import DistillTeacherRuntime
from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract

from .contracts import digest, require, validate_inventory
from .data import MIX, UpdateObserver, build_update_bundle
from .live import architecture, source_pins
from .mutable import NativeSelection, select_native_scope
from .nodes import RecoverySettings
from .recovery import NativeNodeTrainer, recovery_seeds
from .replay import StopAtSelectedNode


@dataclass(frozen=True)
class RecoveryExecutionPolicy:
    train_device: str
    attention_implementation: str
    teacher_model_offload: str
    teacher_output_offload: str
    teacher_output_pin_memory: bool = True
    teacher_output_chunk_tokens: int = 8
    deterministic: bool = False

    def __post_init__(self):
        require(self.train_device == "cpu" or self.train_device.startswith("cuda"), "Use an explicit native CPU/CUDA device.")
        require(self.attention_implementation in ("eager", "sdpa", "flash_attention_2"), "The model adapter must select a supported actual attention policy.")
        require(self.teacher_model_offload in ("none", "cpu") and self.teacher_output_offload in ("none", "cpu"), "Keep original teacher residency modes.")
        require(isinstance(self.teacher_output_pin_memory, bool) and self.teacher_output_chunk_tokens > 0, "Specify actual teacher output residency parameters.")


def fixed_recovery_config(settings: RecoverySettings, seeds: Mapping[str, int], policy: RecoveryExecutionPolicy) -> AfterCategoryResolvedConfig:
    """The protocol constants are explicit; only confirmed settings vary."""
    data = DistillDataConfig(dataset_mix=",".join(f"{alias}={weight:.3f}" for alias, weight in MIX.items()),
        dataset_task="sft", model_max_length=1024, dynamic_padding=True,
        seed=seeds["data"], data_seed=seeds["data"], group_by_length=False)
    loss = DistillLossConfig(loss_type="kl_top_mass", top_k=100, temperature=1.,
        prompt_loss_weight=.3, hidden_loss_weight=0., pre_mlp_hidden_loss_weight=0.,
        selective_student_topk=False)
    opt = DistillOptimizationConfig(distill_fp32_components=("decoder", "lora"),
        steps=settings.schedule_max, batch_size=4, learning_rate=settings.lora_lr,
        decoder_lr=settings.decoder_lr, weight_decay=0., gradient_accumulation_steps=1,
        max_grad_norm=1.5, warmup_ratio=.01, lr_scheduler_type="cosine", optim="adamw_torch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoRAConfig(rank=12, alpha=24., dropout=.1,
        rank_explicit=True, alpha_explicit=True, dropout_explicit=True)
    aux = AuxTrainableConfig(norm_train_mode="none", lm_head_train_mode="none", residual_lora_mode="none")
    runtime = DistillRuntimeConfig(teacher_model_offload=policy.teacher_model_offload,
        teacher_output_offload=policy.teacher_output_offload,
        teacher_output_pin_memory=policy.teacher_output_pin_memory,
        teacher_output_chunk_tokens=policy.teacher_output_chunk_tokens, distill_hif4_act=False)
    for config in (data, loss, opt, lora, aux, runtime):
        config.validate()
    return AfterCategoryResolvedConfig(data=data, loss=loss, opt=opt, lora=lora, aux=aux, runtime=runtime)


@dataclass
class NativeRecoveryRun:
    trainer: NativeNodeTrainer
    teacher: DistillTeacherRuntime
    selection: NativeSelection
    stage: canonical.ResolvedCatAfterCategoryStage
    observer: UpdateObserver
    provenance: dict
    expected_stop: int

    def train(self) -> dict:
        """Execute only when the caller has scheduled this already-built trajectory."""
        require(self.trainer.state.global_step == 0, "This entry starts from the raw; it is not an optimizer-resume implementation.")
        result = self.trainer.train()
        require(self.trainer.state.global_step == self.expected_stop, "The native trajectory did not reach its declared unchanged-schedule endpoint.")
        require(len(self.observer.observed) == self.expected_stop, "Actual KD batches did not match the entire declared data prefix.")
        return dict(status="completed_native_KD_trajectory", actual_steps=int(self.trainer.state.global_step),
            schedule_max=self.stage.config.opt.steps, provenance=self.provenance,
            actual_data=self.observer.report(), save_audits=list(self.trainer.node_save_audits),
            trainer_metrics=result.metrics, formal_eligible=False,
            boundary="A completed trajectory still requires independent node exports/full-eight selection/fresh/boundary acceptance.")

    def close(self) -> None:
        canonical._release_cat_trainer_training_state(self.trainer, logger=logging.getLogger(__name__))
        self.teacher.offload_for_eval()
        self.trainer.teacher_runtime = None


def make_native_recovery_trainer(*, round_base_dir: str | Path, inventory: Mapping,
        category_id: str, settings: RecoverySettings, prepared_data: Mapping,
        sources: Sequence, tokenizer, output_dir: str | Path,
        execution_policy: RecoveryExecutionPolicy, phase: str = "formal",
        technical_stop_after: int | None = None, replay_selected_step: int | None = None,
        access_token: str | None = None) -> NativeRecoveryRun:
    """Load real model/teacher, select true owners, and attach original KD/Adam/v6."""
    pins = source_pins()
    inventory = validate_inventory(inventory)
    require(phase in ("formal", "technical_validation", "cpu_fixture"), "Label formal, technical or CPU construction explicitly.")
    require(not (technical_stop_after is not None and replay_selected_step is not None), "A technical short run and a fixed-recipe replay are separate purposes.")
    require(technical_stop_after in (None, 2), "The save-return technical run is exactly two original-schedule updates.")
    require(phase != "formal" or technical_stop_after is None, "A two-step technical fixture cannot be declared a formal search trajectory.")
    require(phase != "cpu_fixture" or execution_policy.train_device == "cpu", "A CPU fixture cannot allocate a CUDA model.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if execution_policy.train_device == "cpu":
        require(visible == "" and not torch.cuda.is_initialized(), "Construct CPU trajectories with CUDA explicitly hidden before original TrainingArguments device setup.")
    else:
        require(bool(visible) and "," not in visible and execution_policy.train_device in ("cuda", "cuda:0"),
            "This per-candidate native builder requires one explicitly assigned visible GPU and local cuda:0.")
    require(phase == "cpu_fixture" or inventory["model_identity"].get("schema") == "dense_model_content_identity_v0",
        "Formal/model-scale technical construction requires the verified path-independent BF16 content identity.")
    if replay_selected_step is not None:
        require(replay_selected_step in settings.node_steps, "Replay stops at a frozen measured node without changing schedule_max.")
    round_base_dir, output_dir = Path(round_base_dir).resolve(), Path(output_dir).resolve()
    require(not output_dir.exists(), "Use a new isolated trajectory output directory.")
    raw = v6.validate_v6_meta(v6.load_v6_meta(str(round_base_dir)), expected_kind="round_base")
    require(phase != "formal" or (raw.get("extra_meta") or {}).get("cat_auto_phase") != "cpu_fixture",
        "A CPU mechanism raw cannot become a formal recovery trajectory.")
    order, completed = tuple(raw["compression_categories"]), tuple(raw["completed_categories"])
    require(set(order) == set(inventory["category_ids"]) and completed == order[:len(completed)]
        and len(completed) < len(order) and order[len(completed)] == category_id,
        "The fitted raw must contain the next category of its own frozen single chain.")
    current = tuple(row["actual_name"] for row in inventory["projections"] if row["category_id"] == category_id)
    expected_prefix = {row["actual_name"] for row in inventory["projections"] if row["category_id"] in order[:len(completed) + 1]}
    expected_future = {row["actual_name"] for row in inventory["projections"] if row["category_id"] in order[len(completed) + 1:]}
    require(set(raw["compressed_targets"]) == expected_prefix and set(raw["pending_dense_targets"]) == expected_future
        and not raw["skip_targets"], "Preserve the complete real prefix/current/future projection inventory.")
    manifest = prepared_data["manifest"]
    require(manifest["model_identity"] == inventory["model_identity"], "KD data must belong to the same BF16 content identity.")
    seeds = recovery_seeds(inventory["model_identity"], category_id, manifest["search_seed"])
    bundle = build_update_bundle(prepared_data, sources, tokenizer, category_id)
    observer = UpdateObserver(manifest["updates"][category_id])
    require(observer.manifest["data_seed"] == seeds["data"] and len(observer.manifest["batches"]) >= settings.schedule_max,
        "The dataset itself must use the semantic data seed and reserve every legal schedule prefix.")
    config = fixed_recovery_config(settings, seeds, execution_policy)
    stage = canonical.ResolvedCatAfterCategoryStage(mode=settings.scope, config=config,
        train_device=execution_policy.train_device, base_seed=manifest["search_seed"], stage_seed=seeds["dropout"],
        output_dir=str(output_dir), deterministic=execution_policy.deterministic,
        fp16=False, bf16=True, reset_completed=False, save_strategy="steps", save_steps=125,
        save_total_limit=None, save_only_model=False, ignore_data_skip=False)
    args = canonical._build_training_arguments(stage, is_iterable=True, group_by_length=False)
    args.output_dir = str(output_dir / "trainer_state")
    args.adam_beta1, args.adam_beta2, args.adam_epsilon = .9, .999, 1e-8
    require(args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
        and args.dataloader_num_workers == 16 and not args.group_by_length and args.max_grad_norm == 1.5
        and args.max_steps == settings.schedule_max and args.warmup_ratio == .01
        and str(getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)) == "cosine",
        "The actual original Trainer arguments must preserve B4/GAS1/16workers/clip1.5/cosineT/warmup.01.")
    model, loaded_raw, loaded = v6.load_v6_full_checkpoint_into_model(
        architecture(round_base_dir, attn_implementation=execution_policy.attention_implementation),
        str(round_base_dir), expected_kind="round_base", strict=True)
    require(loaded_raw["checkpoint_id"] == raw["checkpoint_id"] and not loaded.missing_keys and not loaded.unexpected_keys,
        "Reconstruct the exact own fitted native round_base before selecting trainables.")
    original_use_cache = bool(model.config.use_cache)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    install_precision_runtime(model, torch.bfloat16)
    selection = select_native_scope(model, settings=settings, prefix_targets=raw["compressed_targets"],
        current_targets=current, future_dense_targets=raw["pending_dense_targets"],
        precision=stage, lora_init_seed=seeds["lora_init"], aux=config.aux,
        fp32_components=config.opt.distill_fp32_components)
    canonical._prewarm_non_current_vae_linears(selection.model,
        selected_names=[name for name, _layer in selection.decoder_targets],
        compute_device=stage.train_device, bf16=True, logger=logging.getLogger(__name__))
    if hasattr(selection.model, "enable_input_require_grads"):
        selection.model.enable_input_require_grads()
    selection.model.to(stage.train_device).train()
    teacher_path = Path(inventory["source_evidence"]["path"]).resolve()
    require(teacher_path.is_dir() and (teacher_path / "config.json").is_file(), "The verified BF16 teacher must be a real local source directory.")
    teacher = DistillTeacherRuntime(model_path=str(teacher_path), access_token=access_token,
        forward_device=stage.train_device, dtype=torch.bfloat16,
        model_offload=execution_policy.teacher_model_offload, logger=logging.getLogger(__name__))
    teacher_model = teacher.get_or_load()
    require(teacher_model.config.model_type == inventory["model_identity"]["model_type"]
        and teacher_model.config.model_type == model.config.model_type
        and teacher_model.config.hidden_size == model.config.hidden_size
        and teacher_model.config.num_hidden_layers == model.config.num_hidden_layers
        and teacher_model.config.vocab_size == model.config.vocab_size,
        "The independent teacher must be the same BF16 root architecture, never another donor.")
    require(not teacher_model.training and all(not parameter.requires_grad and parameter.dtype == torch.bfloat16
        for parameter in teacher_model.parameters()), "Keep the independent original BF16 teacher frozen.")
    require(teacher_model.config._attn_implementation == execution_policy.attention_implementation,
        "The actual teacher loader changed the explicitly selected attention implementation.")
    callbacks = [] if replay_selected_step is None else [StopAtSelectedNode(schedule_max=settings.schedule_max, selected_step=replay_selected_step)]
    trainer = NativeNodeTrainer(model=selection.model, train_dataset=bundle.train_dataset,
        eval_dataset=bundle.eval_dataset, args=args, processing_class=tokenizer,
        data_collator=canonical.build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True),
        loss_config=config.loss, teacher_runtime=teacher, callbacks=callbacks,
        teacher_output_offload=execution_policy.teacher_output_offload,
        teacher_output_pin_memory=execution_policy.teacher_output_pin_memory,
        teacher_output_chunk_tokens=execution_policy.teacher_output_chunk_tokens,
        recovery_settings=settings, semantic_seeds=seeds, input_observer=observer,
        technical_stop_after=technical_stop_after, bound_iterable_loader=True)
    attach_model_level_optimizer_contract(trainer, selection=selection.selection,
        lr_config=ModelLevelOptimizerLRConfig(learning_rate=settings.lora_lr, decoder_lr=settings.decoder_lr, weight_decay=0.))
    lora = None if not selection.dense_targets else collect_exact_peft_lora_config(trainer.model, default_rank=12, alpha=24., dropout=.1)
    teacher_binding = dict(content_identity=inventory["model_identity"], actual_native_identity=model_identity(teacher_model, str(teacher_path)))
    dataset_binding = dict(root_data_identity=prepared_data["data_identity"],
        source_identities=manifest["sources"], tokenizer=manifest["tokenizer"],
        semantic_category_id=category_id, update_manifest=observer.manifest,
        actual_train_bundle=canonical.__name__ + ":DistillDatasetBundle", group_by_length=False)
    immutable = build_cat_step_immutable_resume_contract(stage=stage, trainer_args=args, tokenizer=tokenizer,
        round_base_checkpoint_id=raw["checkpoint_id"], active_category=category_id, round_base_meta=raw,
        lora_target_names=selection.dense_targets, decoder_target_names=[name for name, _layer in selection.decoder_targets],
        teacher_identity=teacher_binding, dataset_identity=dataset_binding, lora_config=lora)
    provenance = dict(phase=phase, round_base_id=raw["checkpoint_id"], category_id=category_id,
        semantic_seeds=seeds, settings=asdict(settings), execution_policy=asdict(execution_policy),
        original_use_cache=original_use_cache, original_KD_implementation=str(Path(canonical.__file__).resolve()),
        source_pins=pins, root_data_identity=prepared_data["data_identity"],
        original_Adam=dict(implementation="torch.optim.AdamW", betas=[.9, .999], eps=1e-8, weight_decay=0.),
        teacher_content_identity=inventory["model_identity"], training_exact_resume=False)
    context = dict(round_base_dir=str(round_base_dir), round_base_checkpoint_id=raw["checkpoint_id"],
        active_category=category_id, after_category_mode=settings.scope,
        compressed_targets=raw["compressed_targets"], pending_dense_targets=raw["pending_dense_targets"],
        skip_targets=raw["skip_targets"], completed_categories=raw["completed_categories"],
        compression_categories=raw["compression_categories"], target_layers=raw["target_layers"],
        target_modules=raw["target_modules"], lora_config=lora,
        resolved_learning_rates=dict(learning_rate=settings.lora_lr, decoder_lr=settings.decoder_lr, norm_lr=None, lm_head_lr=None),
        immutable_resume_contract=immutable, base_model_path=str(teacher_path),
        round_idx=len(completed), distill_stage_history=(raw.get("extra_meta") or {}).get("distill_stage_history", []),
        runtime_audit=dict(cat_auto_native_builder=provenance))
    trainer.configure_v6_step_checkpoint(context=context, selected_vae_modules=selection.decoder_targets)
    trainer.create_optimizer()
    require(type(trainer.optimizer) is torch.optim.AdamW
        and trainer.optimizer.defaults["betas"] == (.9, .999) and trainer.optimizer.defaults["eps"] == 1e-8,
        "The actual native recovery optimizer must use original torch AdamW, not VAE Adam or Muon defaults.")
    expected_lr = {"decoder": settings.decoder_lr, **({"lora": settings.lora_lr} if selection.dense_targets else {})}
    require({group["group_name"]: group["lr"] for group in trainer.optimizer.param_groups} == expected_lr
        and all(group["weight_decay"] == 0. for group in trainer.optimizer.param_groups),
        "Actual component-owned optimizer groups changed the selected decoder or future-LoRA LR/WD.")
    return NativeRecoveryRun(trainer, teacher, selection, stage, observer, provenance,
        technical_stop_after or replay_selected_step or settings.schedule_max)
