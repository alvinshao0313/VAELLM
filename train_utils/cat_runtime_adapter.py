"""Single internal adapter from the canonical CAT CLI config to the existing runtime views."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from transformers import HfArgumentParser

from train_utils.cat_category_runtime import CatTrainHFTrainingArguments
from train_utils.config.cli import CatCLIConfig, parse_cat_cli
from train_utils.config.overrides import OverrideTable
from train_utils.train_args import HFArguments


def _constant_table(arg_name: str, value, *, after: bool = False) -> OverrideTable:
    selectors = ("default", "after") if after else ("default", "cat")
    return OverrideTable(
        arg_name=arg_name,
        allowed_selectors=selectors,
        has_default=True,
        default=value,
    )


def _format_target_layers(value) -> str:
    if value == "all":
        return "all"
    return ",".join(str(int(item)) for item in value)


def _format_skip_layers(values) -> str:
    return ",".join(f"{int(layer)}.{name}" for layer, name in sorted(values))


def adapt_cat_cli_config(cfg: CatCLIConfig):
    """Build private namespace-style runtime views without parsing legacy CLI names."""
    hf_parser = HfArgumentParser((HFArguments, CatTrainHFTrainingArguments))
    hf_args, training_args = hf_parser.parse_args_into_dataclasses(args=list(cfg.remaining_argv))

    compression_categories = ",".join(cfg.compression_categories)
    cat_args = argparse.Namespace(
        _common_cat_config=cfg,
        resolve_after_category_config=cfg.resolve_after_category_config,
        after_category_mode=str(cfg.after_category_mode),
        compression_categories=compression_categories,
        target_layers=_format_target_layers(cfg.target_layers),
        skip_layers=_format_skip_layers(cfg.skip_layers),
        transpose_modules=str(cfg.core_template.transpose_modules),
        include_all_linears=False,
        linear_group_size=int(cfg.core_template.linear_group_size),
        intra_parallel=cfg.intra_parallel,
        intra_part_sort_mode=_constant_table("--intra_part_sort_mode", "none"),
        batch_size=int(cfg.vae_opt_template.vae_batch_size),
        gpu_resident_data=bool(cfg.vae_opt_template.gpu_resident_data),
        log_every=int(cfg.vae_opt_template.log_every),
        eval_every=int(cfg.vae_opt_template.eval_every),
        eval_blocks=int(cfg.vae_opt_template.eval_blocks),
        channel_protect_count=(
            cfg.channel_protect_count_table
            if cfg.channel_protect_count_table is not None
            else _constant_table("--channel_protect_count", 0)
        ),
        channel_protect_count_ratio=cfg.channel_protect_count_ratio,
        channel_min_per_layer=int(cfg.channel_min_per_layer),
        channel_protect_mode=str(cfg.channel_protect_mode),
        channel_scope=str(cfg.channel_scope),
        channel_rank_metric=str(cfg.channel_rank_metric),
        channel_mlp_rank_metric=str(cfg.channel_mlp_rank_metric),
        channel_mlp_fuse_weights=tuple(cfg.channel_mlp_fuse_weights),
        channel_axis=str(cfg.channel_axis),
        channel_quant=str(cfg.channel_quant),
        activation_calib_dataset=str(cfg.activation_calib_dataset),
        activation_calib_nsamples=int(cfg.activation_calib_nsamples),
        activation_calib_seqlen=int(cfg.activation_calib_seqlen),
        activation_calib_seed=int(cfg.activation_calib_seed),
        activation_calib_device=str(cfg.activation_calib_device),
        activation_calib_log_every=int(cfg.activation_calib_log_every),
        eval_ppl=not bool(cfg.skip_ppl_eval),
        eval_tasks=str(cfg.eval_tasks or ""),
        ppl_limit=int(cfg.ppl_limit),
        eval_hif4_act=bool(cfg.eval_hif4_act),
        seed=int(cfg.data.seed),
        deterministic=bool(cfg.deterministic),
        train_device=str(cfg.train_device),
        rot_llm=bool(cfg.rot_llm),
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        convert=bool(cfg.convert),
        convert_device=str(cfg.convert_device),
        save_model=bool(cfg.save_model),
        save_candidate_artifact=bool(cfg.save_candidate_artifact),
        candidate_artifact_spec=cfg.candidate_artifact_spec,
        candidate_artifact_output_dir=cfg.candidate_artifact_output_dir,
        distill_reset_completed=bool(cfg.distill_reset_completed),
        distill_independent_categories=bool(cfg.distill_independent_categories),
        output_dir=str(cfg.output_dir),
        allow_tail_group=bool(cfg.core_template.allow_tail_group),
    )

    training_args.distill_model_max_length = int(cfg.data.model_max_length)
    training_args.distill_dynamic_padding = bool(cfg.data.dynamic_padding)
    training_args.distill_gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
    training_args.distill_optim = str(cfg.optim)
    training_args.distill_max_grad_norm = float(cfg.max_grad_norm)
    training_args.distill_warmup_ratio = float(cfg.warmup_ratio)
    training_args.distill_group_by_length = bool(cfg.data.group_by_length)
    training_args.distill_lr_scheduler_type = str(cfg.lr_scheduler_type)
    training_args.distill_gradient_checkpointing = bool(cfg.gradient_checkpointing)
    training_args.distill_gradient_checkpointing_kwargs = json.dumps(
        cfg.gradient_checkpointing_kwargs, sort_keys=True
    )
    training_args.distill_hif4_act = bool(cfg.runtime.distill_hif4_act)
    training_args.distill_teacher_logits_cpu_staging = (
        str(cfg.runtime.teacher_output_offload) == "cpu"
    )
    training_args.distill_teacher_model_offload = str(cfg.runtime.teacher_model_offload)
    training_args.distill_selective_student_topk = bool(cfg.selective_student_topk)
    training_args.distill_selective_student_topk_chunk_rows = int(
        cfg.selective_student_topk_chunk_rows
    )

    weight_dtype = "bf16" if bool(training_args.bf16) else "fp32"
    vae_args = argparse.Namespace(
        model_path=str(cfg.model_path),
        codebook_bits=cfg.codebook_bits,
        codebook_dim=cfg.codebook_dim,
        residual_stages=cfg.residual_stages,
        base_ch=cfg.base_ch,
        num_res_blocks=cfg.num_res_blocks,
        decoder_base_ch=cfg.decoder_base_ch,
        decoder_num_res_blocks=cfg.decoder_num_res_blocks,
        recon_loss_type=cfg.recon_loss_type,
        norm_type=cfg.norm_type,
        activation_type=cfg.activation_type,
        decoder_type=cfg.decoder_type,
        quantizer_type=str(cfg.core_template.quantizer_type),
        gamma0=float(cfg.core_template.gamma0),
        gamma=float(cfg.core_template.gamma),
        zeta=float(cfg.core_template.zeta),
        inv_temperature=float(cfg.core_template.inv_temperature),
        normalize_weight=bool(cfg.core_template.normalize_weight),
        new_quant=bool(cfg.core_template.new_quant),
        lr=float(cfg.vae_opt_template.vae_learning_rate),
        weight_decay=float(cfg.vae_opt_template.vae_weight_decay),
        optimizer=str(cfg.vae_opt_template.vae_optim),
        lr_scheduler=str(cfg.vae_opt_template.vae_lr_scheduler_type),
        lr_warmup_steps=0,
        beta1=float(cfg.vae_opt_template.beta1),
        beta2=float(cfg.vae_opt_template.beta2),
        l1_weight=float(cfg.vae_opt_template.l1_weight),
        lfq_weight=float(cfg.vae_opt_template.lfq_weight),
        commitment_loss_weight=float(cfg.vae_opt_template.commitment_loss_weight),
        entropy_loss_weight=float(cfg.vae_opt_template.entropy_loss_weight),
        vae_decoder_checkpoint=bool(cfg.runtime.vae_decoder_checkpoint),
        vae_weight_dtype=weight_dtype,
        vae_autocast_dtype=weight_dtype,
        access_token=hf_args.access_token,
    )
    return cat_args, hf_args, training_args, vae_args


def parse_cat_runtime_args(argv: Optional[Sequence[str]] = None):
    return adapt_cat_cli_config(parse_cat_cli(argv))


__all__ = ["adapt_cat_cli_config", "parse_cat_runtime_args"]
