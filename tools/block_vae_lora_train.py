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
    QWEN3_BLOCK_CATEGORIES,
    build_initial_hidden_states,
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
from train_utils.cat_train_eval import eval_after_category
from train_utils.cat_train_args import parse_skip_layers
from train_utils.cat_train_runtime import (
    load_model_for_cat_train,
)
from train_utils.hif4_act import applied_hif4_act
from train_utils.lora_data import build_calibration_input_ids
from train_utils.model_checkpoint_io import _build_run_output_dir
from e2e_common.temporary_mode import set_model_temporary
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
    eval_blocks = 2**63 - 1 if str(args.vae_batch_size).strip().lower() == "all" else int(args.vae_batch_size)
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
        log_every=int(args.vae_log_every),
        eval_every=int(args.vae_eval_every),
        eval_blocks=int(eval_blocks),
    )


def _module_shape_key(ref: LinearRef, runtime_cfg) -> Tuple[object, ...]:
    weight = ref.module.weight
    effective = weight.t() if bool(ref.transpose) else weight
    row_parts, col_parts = tuple(runtime_cfg.intra_parallel)
    return (
        int(effective.numel()),
        int(row_parts) * int(col_parts),
        int(runtime_cfg.residual_stages),
        int(runtime_cfg.steps),
        int(runtime_cfg.joint_decoder_steps),
        float(runtime_cfg.joint_decoder_lr),
        int(runtime_cfg.joint_decoder_group_size),
        None if runtime_cfg.joint_decoder_batch_size is None else int(runtime_cfg.joint_decoder_batch_size),
        tuple(runtime_cfg.intra_parallel),
        str(runtime_cfg.intra_part_sort_mode),
        int(runtime_cfg.codebook_bits),
        int(runtime_cfg.codebook_dim),
        str(runtime_cfg.recon_loss_type),
        int(runtime_cfg.base_ch),
        int(runtime_cfg.num_res_blocks),
        None if runtime_cfg.decoder_base_ch is None else int(runtime_cfg.decoder_base_ch),
        None if runtime_cfg.decoder_num_res_blocks is None else int(runtime_cfg.decoder_num_res_blocks),
        str(runtime_cfg.norm_type),
        str(runtime_cfg.decoder_type),
    )


def _collect_block_linear_refs(
    model: nn.Module,
    *,
    layer_idx: int,
    transpose_modules: Sequence[str],
) -> Dict[str, LinearRef]:
    transpose_set = set(str(item) for item in transpose_modules)
    layer = model.model.layers[int(layer_idx)]
    refs = {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
    }
    out: Dict[str, LinearRef] = {}
    for category, module in refs.items():
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Layer {layer_idx}.{category} must be nn.Linear before VAE conversion, got {type(module)}."
            )
        name = f"model.layers.{int(layer_idx)}."
        if category in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            name += f"self_attn.{category}"
        else:
            name += f"mlp.{category}"
        out[category] = LinearRef(
            name=name,
            module=module,
            category=category,
            transpose=category in transpose_set,
        )
    return out


def _planned_block_groups(
    refs_by_category: Dict[str, LinearRef],
    resolved_cfgs,
) -> List[List[LinearRef]]:
    preferred_groups = (
        ("q_proj", "o_proj"),
        ("k_proj", "v_proj"),
        ("gate_proj", "up_proj", "down_proj"),
    )
    groups: List[List[LinearRef]] = []
    for categories in preferred_groups:
        by_key: Dict[Tuple[object, ...], List[LinearRef]] = {}
        for category in categories:
            if category not in refs_by_category:
                continue
            ref = refs_by_category[category]
            key = _module_shape_key(ref, resolved_cfgs[category])
            by_key.setdefault(key, []).append(ref)
        groups.extend(by_key.values())
    return groups


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
            set_model_temporary(model, False)
            teacher_next = run_qwen3_block(model, int(layer_idx), teacher_in, output_attentions=False)
            if hif4_controller is not None:
                hif4_controller.enabled = True
            set_model_temporary(model, True)
            student_next = run_qwen3_block(model, int(layer_idx), student_in, output_attentions=False)
            next_teacher.append(teacher_next.detach().to(device="cpu", dtype=torch.bfloat16).contiguous())
            next_student.append(student_next.detach().to(device="cpu", dtype=torch.bfloat16).contiguous())
    layer.to("cpu")
    if run_device.type == "cuda":
        torch.cuda.empty_cache()
    return next_teacher, next_student


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

    logger.info("Run output directory: %s", run_output_dir)
    logger.info(
        "Args:\nscript=%s\nvae=%s\ntraining=%s",
        format_namespace(cat_args),
        format_namespace(vae_args),
        format_namespace(training_args),
    )

    model = load_model_for_cat_train(cat_args=cat_args, hf_args=hf_args, vae_args=vae_args)
    setattr(model, "_e2e_vae_lora_tune_bias", str(args.block_lora_bias) == "lora_only")
    validate_qwen3_model(model)

    transpose_modules = parse_transpose_modules(str(args.transpose_modules))
    active_categories = list(QWEN3_BLOCK_CATEGORIES)
    resolved_cfgs = resolve_block_runtime_configs(args, active_categories)
    block_snapshot = _save_block_args_snapshot(
        run_output_dir=run_output_dir,
        args=args,
        hf_args=hf_args,
        training_args=training_args,
        resolved_cfgs=resolved_cfgs,
    )
    logger.info("Saved block parameter snapshot: %s", block_snapshot)

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
        if int(layer_idx) not in selected_layers:
            logger.info("[block %d] skipped compression; advancing hidden states only.", int(layer_idx))
            teacher_hiddens, student_hiddens = _advance_block_hidden_states(
                model=model,
                layer_idx=int(layer_idx),
                teacher_hiddens_cpu=teacher_hiddens,
                student_hiddens_cpu=student_hiddens,
                device=str(args.train_device),
                student_hif4_act=bool(args.block_lora_hif4_act),
            )
            continue
        validate_block_categories(model, layer_idx)
        refs_by_category = _collect_block_linear_refs(
            model,
            layer_idx=int(layer_idx),
            transpose_modules=transpose_modules,
        )
        layer_skip_categories = {
            category
            for skip_layer_idx, category in skip_layer_keys
            if int(skip_layer_idx) == int(layer_idx)
        }
        active_refs_by_category = {
            category: ref
            for category, ref in refs_by_category.items()
            if category not in layer_skip_categories
        }
        if layer_skip_categories:
            logger.info(
                "[block %d] skip_layers keeps original Linear weights: %s",
                int(layer_idx),
                ",".join(sorted(layer_skip_categories)),
            )
        if not active_refs_by_category:
            logger.info("[block %d] all target Linear modules are skipped; advancing hidden states only.", int(layer_idx))
            teacher_hiddens, student_hiddens = _advance_block_hidden_states(
                model=model,
                layer_idx=int(layer_idx),
                teacher_hiddens_cpu=teacher_hiddens,
                student_hiddens_cpu=student_hiddens,
                device=str(args.train_device),
                student_hif4_act=bool(args.block_lora_hif4_act),
            )
            continue
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
            target_categories=list(active_refs_by_category.keys()),
            logger=logger,
        )
        if bool(args.block_eval_after_each_layer):
            eval_device = str(args.block_eval_device or args.train_device)
            with prepare_block_eval_decoded_weights(
                model=model,
                eval_device=eval_device,
                group_size=int(args.block_decode_group_size),
                train_mode=str(args.block_distill_train_mode),
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
                )
        logger.info("[block %d] finished.", int(layer_idx))

    expected_count = sum(
        1
        for layer_idx in selected_layers
        for category in QWEN3_BLOCK_CATEGORIES
        if (int(layer_idx), str(category)) not in skip_layer_keys
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
        },
        unload_vae_original_weights=bool(args.unload_vae_original_weights_on_final_save),
        compact_unload_vae_original_weights=True,
    )
    logger.info("Saved final model to %s", save_paths["output_dir"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
