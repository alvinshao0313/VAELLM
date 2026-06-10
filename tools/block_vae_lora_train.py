#!/usr/bin/env python
import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tools.cat_train as cat_train_impl
from e2e_common.checkpoint_io import save_e2e_model_checkpoint
from e2e_common.post_norm_head import fuse_post_norm_head_linear
from litebsq.vae_linear import clear_model_vae_linear_cache
from train_utils.block_distill import (
    BlockDistillConfig,
    block_student_weight_scope,
    build_initial_hidden_states,
    mark_untrained_block_targets_original_only,
    prepare_block_eval_decoded_weights,
    train_block_lora_distill,
    validate_block_categories,
    validate_final_block_checkpoint,
    validate_qwen3_model,
    run_qwen3_block,
)
from train_utils.block_vae_lora_args import (
    BlockVaeLoraArgs,
    format_skip_layers,
    parse_block_layers,
    parse_block_vae_lora_args,
    parse_transpose_modules,
    resolve_block_runtime_configs,
    validate_skip_layers_with_block_layers,
)
from train_utils.block_vae_lora_checkpoint import (
    load_block_resume_model,
    load_block_resume_state,
    prune_block_layer_checkpoints,
    save_block_layer_checkpoint,
)
from train_utils.block_vae_cache import (
    build_category_pretrain_tasks,
    compute_block_vae_category_pretrain_hash,
    collect_block_linear_refs as _collect_block_linear_refs,
    load_block_vae_category_pretrained_model,
    planned_block_groups as _planned_block_groups,
    run_block_vae_category_pretrain,
    validate_block_vae_category_pretrained_meta,
)
from train_utils.cat_train_eval import eval_after_category
from train_utils.cat_train_args import parse_skip_layers
from train_utils.cat_train_runtime import (
    load_model_for_cat_train,
)
from train_utils.hif4_act import applied_hif4_act
from train_utils.lora_data import build_calibration_input_ids
from train_utils.model_checkpoint_io import _build_run_output_dir
from train_utils.utils import (
    LinearRef,
    configure_deterministic_mode,
    format_namespace,
    get_logger,
    set_seed,
)


def _to_jsonable(value):
    if hasattr(value, "to_jsonable") and callable(getattr(value, "to_jsonable")):
        return value.to_jsonable()
    if is_dataclass(value):
        return {key: _to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, argparse.Namespace):
        return {key: _to_jsonable(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _build_internal_vae_namespace(args: BlockVaeLoraArgs, training_args) -> argparse.Namespace:
    if bool(getattr(training_args, "fp16", False)):
        raise ValueError("block_vae_lora_train does not support --fp16=true for VAE training; use --bf16=true or fp32.")
    vae_dtype = "bf16" if bool(getattr(training_args, "bf16", False)) else "fp32"
    return argparse.Namespace(
        model_path=str(args.model_path),
        vae_weight_dtype=vae_dtype,
        vae_autocast_dtype=vae_dtype,
        codebook_bits=args.codebook_bits,
        codebook_dim=args.codebook_dim,
        residual_stages=args.residual_stages,
        base_ch=args.base_ch,
        num_res_blocks=args.num_res_blocks,
        decoder_base_ch=args.decoder_base_ch,
        decoder_num_res_blocks=args.decoder_num_res_blocks,
        norm_type=args.norm_type,
        decoder_type=args.decoder_type,
        recon_loss_type=args.recon_loss_type,
        quantizer_type=str(args.quantizer_type),
        gamma0=float(args.gamma0),
        gamma=float(args.gamma),
        zeta=float(args.zeta),
        inv_temperature=float(args.inv_temperature),
        lr=float(args.lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        weight_decay=float(args.weight_decay),
        optimizer=str(args.optimizer),
        lr_scheduler=str(args.lr_scheduler),
        lr_warmup_steps=int(args.lr_warmup_steps),
        l1_weight=float(args.l1_weight),
        lfq_weight=float(args.lfq_weight),
        commitment_loss_weight=float(args.commitment_loss_weight),
        entropy_loss_weight=float(args.entropy_loss_weight),
        diversity_gamma=float(args.diversity_gamma),
        normalize_weight=bool(args.normalize_weight),
        use_checkpoint=bool(args.use_checkpoint),
        new_quant=bool(args.new_quant),
    )


def _build_internal_cat_namespace(args: BlockVaeLoraArgs) -> argparse.Namespace:
    # eval_blocks 只保留给 cat_train helper 兼容；VAE 训练中 eval 现在固定扫描完整 residual stage。
    eval_blocks = 2**63 - 1
    return argparse.Namespace(
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        train_device=str(args.train_device),
        convert_device=str(args.convert_device),
        convert=True,
        save_model=True,
        unload_vae_original_weights_on_final_save=bool(args.unload_vae_original_weights_on_final_save),
        rot_llm=False,
        resume_from_checkpoint=None,
        batch_size=str(args.vae_batch_size),
        gpu_resident_data=bool(args.vae_gpu_resident_data),
        log_every=int(args.vae_log_every),
        eval_every=int(args.vae_eval_every),
        eval_blocks=int(eval_blocks),
    )


def _save_block_args_snapshot(
    run_output_dir: str,
    *,
    args: BlockVaeLoraArgs,
    hf_args,
    training_args,
    resolved_cfgs,
) -> str:
    path = os.path.join(run_output_dir, "normalized_block_vae_lora_args.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": _to_jsonable(args),
                "hf_args": _to_jsonable(hf_args),
                "training_args": _to_jsonable(training_args),
                "resolved_runtime": _to_jsonable(resolved_cfgs),
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    return path


def _set_cat_train_logger(logger) -> None:
    cat_train_impl.log = logger


def _train_block_vae_groups(
    *,
    model: nn.Module,
    layer_idx: int,
    refs_by_category: Dict[str, LinearRef],
    resolved_cfgs,
    cat_args,
    vae_args,
    training_args,
    logger,
) -> None:
    for group_idx, group_refs in enumerate(_planned_block_groups(refs_by_category, resolved_cfgs)):
        categories = ",".join(ref.category for ref in group_refs)
        runtime_cfg = resolved_cfgs[group_refs[0].category]
        logger.info(
            "[block %d] VAE group %d start: categories=%s linears=%d steps=%d codebook=%d/%d",
            int(layer_idx),
            int(group_idx),
            categories,
            len(group_refs),
            int(runtime_cfg.steps),
            int(runtime_cfg.codebook_bits),
            int(runtime_cfg.codebook_dim),
        )
        cat_train_impl._train_group_vae_and_replace(
            model=model,
            group_refs=group_refs,
            group_tag=f"block{int(layer_idx)}.group{int(group_idx)}.{categories}",
            runtime_cfg=runtime_cfg,
            vae_args=vae_args,
            training_args=training_args,
            train_device=cat_args.train_device,
            convert_device=cat_args.convert_device,
            do_convert=True,
            batch_size=cat_args.batch_size,
            log_every=cat_args.log_every,
            eval_every=cat_args.eval_every,
            eval_blocks=cat_args.eval_blocks,
            gpu_resident_data=bool(getattr(cat_args, "gpu_resident_data", False)),
            skip_layer_keys=set(),
            activation_runtime=None,
            outlier_protect_mode="none",
            outlier_residual_score="abs",
            outlier_residual_min_abs=0.0,
            outlier_protect_axis="input",
            outlier_residual_codec="coo_fp16",
            outlier_residual_index_bits=8,
            outlier_residual_value_bits=8,
            outlier_residual_block_shape=(256, 256),
            sort_executor=None,
            sort_prep_workers_resolved=1,
            deterministic=bool(cat_args.deterministic),
            shuffle_seed=int(cat_args.seed) + int(layer_idx) * 100000 + int(group_idx) * 1000,
        )


@torch.no_grad()
def _advance_block_hidden_states(
    *,
    model: nn.Module,
    layer_idx: int,
    teacher_hiddens_cpu: Sequence[torch.Tensor],
    student_hiddens_cpu: Sequence[torch.Tensor],
    device: str,
    student_hif4_act: bool,
    active_block_targets: Sequence[Tuple[int, str]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    run_device = torch.device(device)
    layer = model.model.layers[int(layer_idx)].to(run_device)
    layer.eval()
    next_teacher: List[torch.Tensor] = []
    next_student: List[torch.Tensor] = []
    with applied_hif4_act(model, enabled=bool(student_hif4_act), require_targets=False) as hif4_ctx:
        hif4_controller = hif4_ctx.get("controller")
        for teacher_cpu, student_cpu in zip(teacher_hiddens_cpu, student_hiddens_cpu):
            teacher_in = teacher_cpu.to(device=run_device, non_blocking=True)
            student_in = student_cpu.to(device=run_device, non_blocking=True)
            if hif4_controller is not None:
                hif4_controller.enabled = False
            with block_student_weight_scope(model, set()):
                teacher_next = run_qwen3_block(model, int(layer_idx), teacher_in, output_attentions=False)
            if hif4_controller is not None:
                hif4_controller.enabled = True
            with block_student_weight_scope(model, active_block_targets):
                student_next = run_qwen3_block(model, int(layer_idx), student_in, output_attentions=False)
            next_teacher.append(teacher_next.detach().to(device="cpu", dtype=torch.bfloat16).contiguous())
            next_student.append(student_next.detach().to(device="cpu", dtype=torch.bfloat16).contiguous())
    layer.to("cpu")
    if run_device.type == "cuda":
        torch.cuda.empty_cache()
    return next_teacher, next_student


def _active_categories_for_layer(
    layer_idx: int,
    *,
    configured_categories: Sequence[str],
    skip_layer_keys: Sequence[Tuple[int, str]],
) -> List[str]:
    skip_set = set((int(skip_layer_idx), str(category)) for skip_layer_idx, category in skip_layer_keys)
    return [
        str(category)
        for category in configured_categories
        if (int(layer_idx), str(category)) not in skip_set
    ]


def _active_targets_for_layers(
    layer_indices: Sequence[int],
    *,
    configured_categories: Sequence[str],
    skip_layer_keys: Sequence[Tuple[int, str]],
) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for layer_idx in sorted(int(value) for value in layer_indices):
        for category in _active_categories_for_layer(
            int(layer_idx),
            configured_categories=configured_categories,
            skip_layer_keys=skip_layer_keys,
        ):
            out.append((int(layer_idx), str(category)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    args, hf_args, training_args = parse_block_vae_lora_args(argv)
    cat_args = _build_internal_cat_namespace(args)
    vae_args = _build_internal_vae_namespace(args, training_args)

    configure_deterministic_mode(bool(args.deterministic))
    set_seed(int(args.seed))

    os.makedirs(args.output_dir, exist_ok=True)
    run_output_dir = _build_run_output_dir(args.output_dir, args.model_path)
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "block_vae_lora.log")
    logger = get_logger("block_vae_lora")
    _set_cat_train_logger(logger)
    cat_args.output_dir = run_output_dir
    block_vae_pretrain_output_dir = os.path.join(run_output_dir, "block_vae_cache")

    logger.info("Run output directory: %s", run_output_dir)
    logger.info(
        "Args:\nscript=%s\nvae=%s\ntraining=%s",
        format_namespace(cat_args),
        format_namespace(vae_args),
        format_namespace(training_args),
    )
    pipeline_mode = str(args.block_vae_pipeline_mode).strip().lower()
    if pipeline_mode in {"pretrain", "pretrain_distill"} and args.block_resume_from_checkpoint is not None:
        raise ValueError(
            f"--block_vae_pipeline_mode {pipeline_mode} does not support --block_resume_from_checkpoint."
        )

    loaded_pretrain_meta = None
    loaded_resume_meta = None
    if args.block_resume_from_checkpoint is None:
        if pipeline_mode == "distill":
            model, loaded_pretrain_meta, pretrain_load_result = load_block_vae_category_pretrained_model(
                str(args.vae_pretrained_checkpoint),
                access_token=hf_args.access_token,
                proxy_group_size=int(args.block_decode_group_size),
                proxy_compute_device=str(args.convert_device),
                logger=logger,
            )
            logger.info(
                "Loaded block VAE category-pretrained checkpoint: %s missing_keys=%d unexpected_keys=%d",
                str(args.vae_pretrained_checkpoint),
                len(getattr(pretrain_load_result, "missing_keys", [])),
                len(getattr(pretrain_load_result, "unexpected_keys", [])),
            )
        else:
            model = load_model_for_cat_train(cat_args=cat_args, hf_args=hf_args, vae_args=vae_args)
    else:
        model, resume_checkpoint_dir, resume_load_meta, resume_load_result = load_block_resume_model(
            str(args.block_resume_from_checkpoint),
            access_token=hf_args.access_token,
            proxy_group_size=int(args.block_decode_group_size),
            proxy_compute_device=str(args.convert_device),
            logger=logger,
        )
        loaded_resume_meta = resume_load_meta
        logger.info("Resuming block run from checkpoint: %s", resume_checkpoint_dir)
        logger.info(
            "Resume checkpoint loaded. missing_keys=%d unexpected_keys=%d converted_module_count=%s adapter_module_count=%s",
            len(getattr(resume_load_result, "missing_keys", [])),
            len(getattr(resume_load_result, "unexpected_keys", [])),
            str(resume_load_meta.get("converted_module_count")),
            str(resume_load_meta.get("adapter_module_count")),
        )
    setattr(model, "_e2e_vae_lora_tune_bias", str(args.block_lora_bias) == "lora_only")
    validate_qwen3_model(model)

    transpose_modules = parse_transpose_modules(str(args.transpose_modules))
    configured_categories = list(args.block_vae_categories)
    resolved_cfgs = resolve_block_runtime_configs(args, configured_categories)
    block_snapshot = _save_block_args_snapshot(
        run_output_dir=run_output_dir,
        args=args,
        hf_args=hf_args,
        training_args=training_args,
        resolved_cfgs=resolved_cfgs,
    )
    logger.info("Saved block parameter snapshot: %s", block_snapshot)

    num_layers = int(model.config.num_hidden_layers)
    try:
        selected_layers = set(parse_block_layers(str(args.block_layers), num_layers=int(num_layers)))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not selected_layers:
        raise ValueError("--block_layers resolved to an empty layer set.")
    skip_layer_keys = parse_skip_layers(str(args.skip_layers))
    validate_skip_layers_with_block_layers(
        skip_layer_keys=sorted(skip_layer_keys),
        selected_layers=sorted(selected_layers),
    )
    logger.info(
        "Selected block layers: %s",
        ",".join(str(idx) for idx in sorted(selected_layers)),
    )
    if skip_layer_keys:
        logger.info("Block skip_layers: %s", ",".join(format_skip_layers(sorted(skip_layer_keys))))
    expected_count = sum(
        1
        for layer_idx in selected_layers
        for category in configured_categories
        if (int(layer_idx), str(category)) not in skip_layer_keys
    )
    if int(expected_count) <= 0:
        raise ValueError(
            "No effective block VAE targets remain after applying --block_layers, "
            "--block_vae_categories, and --skip_layers."
        )
    block_vae_pretrain_manifest_hash = ""
    if pipeline_mode == "distill":
        if loaded_pretrain_meta is not None:
            block_vae_pretrain_manifest_hash = validate_block_vae_category_pretrained_meta(
                loaded_pretrain_meta,
                args=args,
                selected_layers=sorted(selected_layers),
                skip_layer_keys=sorted(skip_layer_keys),
                transpose_modules=transpose_modules,
                resolved_cfgs=resolved_cfgs,
            )
            logger.info(
                "Validated block VAE category-pretrained checkpoint: manifest_hash=%s",
                block_vae_pretrain_manifest_hash,
            )
        else:
            extra_meta = loaded_resume_meta.get("extra_meta", {}) if isinstance(loaded_resume_meta, dict) else {}
            block_vae_pretrain_manifest_hash = compute_block_vae_category_pretrain_hash(
                args=args,
                selected_layers=sorted(selected_layers),
                skip_layer_keys=sorted(skip_layer_keys),
                transpose_modules=transpose_modules,
                resolved_cfgs=resolved_cfgs,
            )
            checkpoint_hash = str(extra_meta.get("block_vae_cache_manifest_hash", ""))
            if checkpoint_hash != block_vae_pretrain_manifest_hash:
                raise ValueError(
                    "Resume checkpoint block VAE pretrain manifest hash mismatch: "
                    f"checkpoint={checkpoint_hash!r} current={block_vae_pretrain_manifest_hash!r}."
                )
    resume_start_layer = 0
    completed_block_layers: List[int] = []
    if args.block_resume_from_checkpoint is not None:
        resume_state = load_block_resume_state(
            str(args.block_resume_from_checkpoint),
            current_args=args,
            selected_layers=sorted(selected_layers),
            skip_layer_keys=sorted(skip_layer_keys),
        )
        resume_start_layer = int(resume_state.next_block_layer_idx)
        if int(resume_start_layer) >= int(num_layers):
            raise ValueError(
                f"Resume checkpoint already reached layer {int(resume_state.completed_block_layer_idx)}, "
                f"but model only has {int(num_layers)} layers."
            )
        completed_block_layers = list(resume_state.completed_block_layers)
        logger.info(
            "Block resume state: completed_layer=%d next_layer=%d completed_layers=%s",
            int(resume_state.completed_block_layer_idx),
            int(resume_start_layer),
            ",".join(str(idx) for idx in completed_block_layers),
        )
    completed_block_targets = set(
        _active_targets_for_layers(
            completed_block_layers,
            configured_categories=configured_categories,
            skip_layer_keys=sorted(skip_layer_keys),
        )
    )
    if pipeline_mode in {"pretrain", "pretrain_distill"}:
        category_tasks, block_vae_pretrain_manifest_hash = build_category_pretrain_tasks(
            model=model,
            args=args,
            selected_layers=sorted(selected_layers),
            skip_layer_keys=sorted(skip_layer_keys),
            transpose_modules=transpose_modules,
            resolved_cfgs=resolved_cfgs,
        )
        logger.info(
            "Block VAE category pretrain tasks ready: mode=%s tasks=%d manifest_hash=%s",
            pipeline_mode,
            len(category_tasks),
            block_vae_pretrain_manifest_hash,
        )
        run_block_vae_category_pretrain(
            model=model,
            tasks=category_tasks,
            pretrain_hash=block_vae_pretrain_manifest_hash,
            output_dir=block_vae_pretrain_output_dir,
            args=args,
            hf_args=hf_args,
            training_args=training_args,
            vae_args=vae_args,
            cat_args=cat_args,
            transpose_modules=transpose_modules,
            resolved_cfgs=resolved_cfgs,
            selected_layers=sorted(selected_layers),
            skip_layer_keys=sorted(skip_layer_keys),
            logger=logger,
        )
        if pipeline_mode == "pretrain":
            logger.info("Done.")
            return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, token=hf_args.access_token)
    input_blocks = build_calibration_input_ids(
        args.block_distill_dataset,
        tokenizer=tokenizer,
        nsamples=int(args.block_distill_nsamples),
        seqlen=int(args.block_distill_seqlen),
        seed=int(args.seed),
    )
    logger.info(
        "Built block distill calibration blocks: nsamples=%d seqlen=%d dataset=%s",
        len(input_blocks),
        int(args.block_distill_seqlen),
        str(args.block_distill_dataset),
    )

    teacher_hiddens, student_hiddens = build_initial_hidden_states(
        model,
        input_blocks,
        device=str(args.train_device),
    )
    distill_cfg = BlockDistillConfig(
        steps=int(args.block_distill_steps),
        seqlen=int(args.block_distill_seqlen),
        rank=int(args.block_lora_rank),
        lr=float(args.block_lora_lr),
        lora_variant=str(args.block_lora_variant),
        lora_alpha=float(args.block_lora_alpha),
        lora_dropout=float(args.block_lora_dropout),
        lora_bias=str(args.block_lora_bias),
        lora_hif4_act=bool(args.block_lora_hif4_act),
        adalora_init_rank=int(args.block_adalora_init_rank),
        adalora_tinit=int(args.block_adalora_tinit),
        adalora_tfinal=int(args.block_adalora_tfinal),
        adalora_delta_t=int(args.block_adalora_delta_t),
        adalora_beta1=float(args.block_adalora_beta1),
        adalora_beta2=float(args.block_adalora_beta2),
        adalora_orth_reg_weight=float(args.block_adalora_orth_reg_weight),
        alpha=float(args.block_loss_alpha),
        beta=float(args.block_loss_beta),
        attn_query_chunk_size=int(args.block_attn_query_chunk_size),
        log_every=int(args.block_distill_log_every),
        device=str(args.train_device),
        train_mode=str(args.block_distill_train_mode),
        decode_group_size=int(args.block_decode_group_size),
    )

    for layer_idx in range(num_layers):
        if int(layer_idx) < int(resume_start_layer):
            logger.info("[block %d] resume prefix; advancing hidden states only.", int(layer_idx))
            teacher_hiddens, student_hiddens = _advance_block_hidden_states(
                model=model,
                layer_idx=int(layer_idx),
                teacher_hiddens_cpu=teacher_hiddens,
                student_hiddens_cpu=student_hiddens,
                device=str(args.train_device),
                student_hif4_act=bool(args.block_lora_hif4_act),
                active_block_targets=sorted(completed_block_targets),
            )
            continue
        if int(layer_idx) not in selected_layers:
            logger.info("[block %d] skipped compression; advancing hidden states only.", int(layer_idx))
            teacher_hiddens, student_hiddens = _advance_block_hidden_states(
                model=model,
                layer_idx=int(layer_idx),
                teacher_hiddens_cpu=teacher_hiddens,
                student_hiddens_cpu=student_hiddens,
                device=str(args.train_device),
                student_hif4_act=bool(args.block_lora_hif4_act),
                active_block_targets=sorted(completed_block_targets),
            )
            continue
        validate_block_categories(model, layer_idx)
        layer_skip_categories = {
            category
            for skip_layer_idx, category in skip_layer_keys
            if int(skip_layer_idx) == int(layer_idx)
        }
        if layer_skip_categories:
            logger.info(
                "[block %d] skip_layers keeps original Linear weights: %s",
                int(layer_idx),
                ",".join(sorted(layer_skip_categories)),
            )
        active_categories = [
            category
            for category in configured_categories
            if category not in layer_skip_categories
        ]
        active_refs_by_category: Dict[str, LinearRef] = {}
        if pipeline_mode == "inline":
            refs_by_category = _collect_block_linear_refs(
                model,
                layer_idx=int(layer_idx),
                transpose_modules=transpose_modules,
            )
            active_refs_by_category = {
                category: ref
                for category, ref in refs_by_category.items()
                if category in active_categories
            }
            active_categories = list(active_refs_by_category.keys())
        if not active_categories:
            logger.info("[block %d] all target Linear modules are skipped; advancing hidden states only.", int(layer_idx))
            teacher_hiddens, student_hiddens = _advance_block_hidden_states(
                model=model,
                layer_idx=int(layer_idx),
                teacher_hiddens_cpu=teacher_hiddens,
                student_hiddens_cpu=student_hiddens,
                device=str(args.train_device),
                student_hif4_act=bool(args.block_lora_hif4_act),
                active_block_targets=sorted(completed_block_targets),
            )
            continue
        if pipeline_mode == "inline":
            _train_block_vae_groups(
                model=model,
                layer_idx=int(layer_idx),
                refs_by_category=active_refs_by_category,
                resolved_cfgs=resolved_cfgs,
                cat_args=cat_args,
                vae_args=vae_args,
                training_args=training_args,
                logger=logger,
            )
        validate_block_categories(model, layer_idx)
        teacher_hiddens, student_hiddens = train_block_lora_distill(
            model=model,
            layer_idx=int(layer_idx),
            teacher_hiddens_cpu=teacher_hiddens,
            student_hiddens_cpu=student_hiddens,
            config=distill_cfg,
            target_categories=list(active_categories),
            logger=logger,
        )
        if int(layer_idx) not in completed_block_layers:
            completed_block_layers.append(int(layer_idx))
            completed_block_layers.sort()
        for category in active_categories:
            completed_block_targets.add((int(layer_idx), str(category)))
        if int(args.block_keep_last_checkpoints) > 0:
            logger.info("[block %d] start saving layer checkpoint.", int(layer_idx))
            save_paths = save_block_layer_checkpoint(
                model=model,
                run_output_dir=run_output_dir,
                args=args,
                completed_block_layer_idx=int(layer_idx),
                completed_block_layers=completed_block_layers,
                selected_layers=sorted(selected_layers),
                skip_layer_keys=sorted(skip_layer_keys),
                target_module_count=expected_count,
                block_vae_cache_manifest_hash=block_vae_pretrain_manifest_hash,
            )
            logger.info("[block %d] finished saving layer checkpoint: %s", int(layer_idx), save_paths["output_dir"])
            removed_checkpoints = prune_block_layer_checkpoints(
                run_output_dir=run_output_dir,
                keep_last=int(args.block_keep_last_checkpoints),
            )
            if removed_checkpoints:
                logger.info(
                    "[block %d] pruned old layer checkpoints: %s",
                    int(layer_idx),
                    ",".join(removed_checkpoints),
                )
        if bool(args.block_eval_after_each_layer):
            eval_device = str(args.block_eval_device or args.train_device)
            with prepare_block_eval_decoded_weights(
                model=model,
                eval_device=eval_device,
                group_size=int(args.block_decode_group_size),
                train_mode=str(args.block_distill_train_mode),
                active_block_targets=sorted(completed_block_targets),
                logger=logger,
            ):
                eval_after_category(
                    model=model,
                    vae_args=vae_args,
                    ppl_limit=int(args.block_eval_ppl_limit),
                    category=f"block_{int(layer_idx)}",
                    logger=logger,
                    eval_device=eval_device,
                    eval_hif4_act=bool(args.block_eval_hif4_act),
                    eval_ppl=bool(args.block_eval_ppl),
                    eval_tasks=str(args.block_eval_tasks),
                    tokenizer=tokenizer,
                    move_model_to_cpu_after_eval=False,
                )
        logger.info("[block %d] finished.", int(layer_idx))

    final_original_only_count = mark_untrained_block_targets_original_only(
        model,
        sorted(completed_block_targets),
    )
    if final_original_only_count:
        logger.info(
            "Final save: marked untrained block targets original-only: %d",
            int(final_original_only_count),
        )
    validate_final_block_checkpoint(
        model,
        expected_rank=int(args.block_lora_rank),
        expected_init_rank=int(args.block_adalora_init_rank),
        expected_count=expected_count,
        lora_variant=str(args.block_lora_variant),
        train_mode=str(args.block_distill_train_mode),
    )
    fused_post_norm_head = fuse_post_norm_head_linear(model)
    if fused_post_norm_head:
        logger.info("Final save: fused post_norm_linear into lm_head.weight.")
    cleared = clear_model_vae_linear_cache(model)
    logger.info("Final save: cleared decoded cache for %d VAELinear modules.", cleared)
    model_out = os.path.join(run_output_dir, "final_model")
    save_paths = save_e2e_model_checkpoint(
        model,
        model_out,
        base_model_path=args.model_path,
        tokenizer=tokenizer,
        save_config=True,
        extra_meta={
            "stage": "block_vae_lora_final",
            "block_distill": _to_jsonable(args),
            "block_distill_train_mode": str(args.block_distill_train_mode),
            "selected_block_layers": sorted(int(idx) for idx in selected_layers),
            "skip_layers": format_skip_layers(sorted(skip_layer_keys)),
            "target_module_count": expected_count,
            "completed_block_layers": sorted(int(idx) for idx in completed_block_layers),
            "resume_from_checkpoint": args.block_resume_from_checkpoint,
            "block_vae_pipeline_mode": pipeline_mode,
            "block_vae_categories": [str(category) for category in args.block_vae_categories],
            "block_vae_pretrain_manifest_hash": block_vae_pretrain_manifest_hash,
            "block_vae_cache_manifest_hash": block_vae_pretrain_manifest_hash,
        },
        unload_vae_original_weights=bool(args.unload_vae_original_weights_on_final_save),
        compact_unload_vae_original_weights=True,
    )
    logger.info("Saved final model to %s", save_paths["output_dir"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
