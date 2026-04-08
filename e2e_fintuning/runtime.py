import json
import os
import argparse
import gc
from typing import Optional

import torch
from torch import nn

from transformers import default_data_collator

from distill_utils.layerwise_distill_runtime import resolve_checkpoint_dir
from e2e_fintuning.checkpoint_io import load_e2e_model_checkpoint, save_e2e_model_checkpoint
from e2e_fintuning.args import needs_teacher
from e2e_fintuning.data import build_datasets, build_tokenizer
from e2e_fintuning.lora import merge_and_unload_extra_lora_modules, merge_extra_lora_state_dict
from e2e_fintuning.peft_proxy import (
    convert_peft_vae_proxy_modules_to_lora,
    ensure_peft_vae_proxy_lora,
    initialize_peft_vae_proxy_lora_from_teacher_residual,
)
from e2e_fintuning.trainables import resolve_target_layer_ids, select_e2e_trainables_peft_proxy
from e2e_fintuning.trainer import (
    E2EFinetuneTrainer,
    E2EFSDPFinetuneTrainer,
    register_lora_hif4_act_hooks,
    remove_lora_hif4_act_hooks,
    set_model_temporary,
)
from litebsq.vae_linear import clear_model_vae_linear_cache
from rotation.common import separate_embeddings_and_lm_head
from rotation.model_utils import get_layers, get_model
from train_utils.eval_utils import calculate_ppl
from train_utils.model_checkpoint_io import _build_run_output_dir, unload_vae_original_linear_weights
from train_utils.utils import get_logger, pt_fsdp_state_dict


_E2E_FINETUNE_MODE = "vae_lora"


def _unwrap_model(trainer, model):
    if getattr(trainer, "accelerator", None) is None:
        return model
    return trainer.accelerator.unwrap_model(model)


def _uses_fsdp(training_args) -> bool:
    fsdp = getattr(training_args, "fsdp", "")
    return not (fsdp is None or fsdp == "" or fsdp == [])


def _ensure_student_mode(model) -> None:
    set_model_temporary(model, True)


def _embedding_and_lm_head_are_tied(model: nn.Module) -> bool:
    embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if not isinstance(embedding, nn.Embedding) or not isinstance(lm_head, nn.Linear):
        return False
    return embedding.weight.data_ptr() == lm_head.weight.data_ptr()


def _resolve_teacher_model_path(*, args, meta) -> Optional[str]:
    teacher_model_path = None if args.teacher_model_path is None else str(args.teacher_model_path).strip()
    if teacher_model_path:
        return teacher_model_path
    meta_path = meta.get("base_model_path")
    if meta_path:
        return str(meta_path)
    return None


def _load_external_teacher_model(*, teacher_model_path: str, hf_args, log) -> nn.Module:
    log.info("Loading external teacher model from %s", teacher_model_path)
    teacher_model = get_model(str(teacher_model_path), hf_args.access_token)
    teacher_model.eval()
    if hasattr(teacher_model, "config"):
        teacher_model.config.use_cache = False
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model


def _load_teacher_for_e2e(*, args, hf_args, meta, log, require_for_init: bool):
    require_for_training = needs_teacher(args.loss_type)
    if not require_for_init and not require_for_training:
        return None, "student", False

    teacher_model_path = _resolve_teacher_model_path(args=args, meta=meta)
    if not teacher_model_path:
        raise ValueError(
            "当前运行需要外部 teacher（蒸馏或 residual_svd 初始化），"
            "但既没有 --teacher_model_path，也无法从 checkpoint meta 里解析 base_model_path。"
        )

    teacher_model = _load_external_teacher_model(
        teacher_model_path=str(teacher_model_path),
        hf_args=hf_args,
        log=log,
    )
    teacher_source = "external_teacher" if require_for_training else "external_teacher_init_only"
    return teacher_model, teacher_source, require_for_training


def _release_init_only_teacher(teacher_model, log):
    if teacher_model is None:
        return None
    try:
        teacher_model.to("cpu")
    finally:
        del teacher_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log.info("Released init-only external teacher model after residual_svd initialization.")
    return None


def _normalize_module_names(names):
    if names is None:
        return None
    values = [str(name).strip().lower() for name in names if str(name).strip()]
    return sorted(values) if values else None


def _validate_resume_checkpoint_config(*, args, meta, decoder_layer_ids) -> None:
    extra_meta = meta.get("extra_meta", {}) if isinstance(meta.get("extra_meta"), dict) else {}
    if str(extra_meta.get("stage", "")).strip().lower() != "e2e_fintuning":
        raise ValueError("resume checkpoint 缺少有效的 e2e_fintuning stage 元信息。")

    expected_layers = extra_meta.get("target_decoder_layers")
    if expected_layers is not None:
        expected_layers = [int(idx) for idx in expected_layers]
        current_layers = [int(idx) for idx in decoder_layer_ids]
        if expected_layers != current_layers:
            raise ValueError(
                f"resume checkpoint 的 target_decoder_layers={expected_layers} 与当前参数 {current_layers} 不一致。"
            )

    expected_modules = _normalize_module_names(extra_meta.get("target_module_names"))
    current_modules = _normalize_module_names(args.target_module_names)
    if expected_modules != current_modules:
        raise ValueError(
            f"resume checkpoint 的 target_module_names={expected_modules} 与当前参数 {current_modules} 不一致。"
        )

    if "vae_lora_rank" in extra_meta and int(extra_meta["vae_lora_rank"]) != int(args.vae_lora_rank):
        raise ValueError(
            f"resume checkpoint 的 vae_lora_rank={extra_meta['vae_lora_rank']} 与当前参数 {args.vae_lora_rank} 不一致。"
        )
    if "vae_lora_alpha" in extra_meta and float(extra_meta["vae_lora_alpha"]) != float(args.vae_lora_alpha):
        raise ValueError(
            f"resume checkpoint 的 vae_lora_alpha={extra_meta['vae_lora_alpha']} 与当前参数 {args.vae_lora_alpha} 不一致。"
        )
    if "vae_lora_dropout" in extra_meta and float(extra_meta["vae_lora_dropout"]) != float(args.vae_lora_dropout):
        raise ValueError(
            f"resume checkpoint 的 vae_lora_dropout={extra_meta['vae_lora_dropout']} 与当前参数 {args.vae_lora_dropout} 不一致。"
        )
    if "lora_embedding" in extra_meta and bool(extra_meta["lora_embedding"]) != bool(args.lora_embedding):
        raise ValueError(
            f"resume checkpoint 的 lora_embedding={extra_meta['lora_embedding']} 与当前参数 {args.lora_embedding} 不一致。"
        )
    if "lora_lm_head" in extra_meta and bool(extra_meta["lora_lm_head"]) != bool(args.lora_lm_head):
        raise ValueError(
            f"resume checkpoint 的 lora_lm_head={extra_meta['lora_lm_head']} 与当前参数 {args.lora_lm_head} 不一致。"
        )


def _checkpoint_has_peft_proxy_lora(meta) -> bool:
    adapter_modules = meta.get("adapter_modules", [])
    if not isinstance(adapter_modules, list):
        return False
    for spec in adapter_modules:
        if isinstance(spec, dict) and str(spec.get("adapter_type")) == "peft_proxy_lora":
            return True
    return False


def _should_initialize_vae_lora_residual_svd(*, args, selection, resume_from_checkpoint) -> bool:
    return (
        str(getattr(args, "vae_lora_init_mode", "zero")).strip().lower() == "residual_svd"
        and not bool(resume_from_checkpoint)
        and bool(getattr(selection, "peft_proxy_modules", []))
    )


def _resolve_saved_vae_lora_init_mode(*, args, meta, resume_from_checkpoint) -> Optional[str]:
    if resume_from_checkpoint:
        extra_meta = meta.get("extra_meta", {}) if isinstance(meta.get("extra_meta"), dict) else {}
        saved_mode = extra_meta.get("vae_lora_init_mode")
        if saved_mode is None:
            return None
        return str(saved_mode).strip().lower()
    return str(getattr(args, "vae_lora_init_mode", "zero")).strip().lower()


def _eval_final_ppl(*, model, args, model_path: str, output_dir: str, log):
    if bool(getattr(args, "skip_ppl_eval", False)):
        log.info("Skipping final PPL evaluation because --skip_ppl_eval=true.")
        return None

    ppl_args = argparse.Namespace(
        model_path=str(model_path),
        seqlen=int(getattr(args, "ppl_seqlen", 2048)),
        limit=int(getattr(args, "ppl_limit", -1)),
    )
    log.info(
        "Start final PPL eval (seqlen=%d, limit=%d)...",
        int(ppl_args.seqlen),
        int(ppl_args.limit),
    )
    with torch.no_grad():
        ppl_result = calculate_ppl(model, ppl_args)

    result = {
        "wiki_ppl": float(ppl_result.get("wiki_ppl", float("nan"))),
        "nsamples": int(ppl_result.get("nsamples", 0)),
        "seqlen": int(ppl_result.get("seqlen", int(ppl_args.seqlen))),
    }
    ppl_path = os.path.join(output_dir, "final_ppl.json")
    with open(ppl_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    log.info(
        "Final PPL=%.4f (nsamples=%d, seqlen=%d) saved to %s",
        result["wiki_ppl"],
        result["nsamples"],
        result["seqlen"],
        ppl_path,
    )
    return {
        "result": result,
        "path": ppl_path,
    }


def _collect_trainable_params(model: nn.Module):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def run(args, hf_args, training_args):
    student_checkpoint_dir = resolve_checkpoint_dir(args.student_checkpoint_dir)
    run_output_dir = _build_run_output_dir(args.run_root_dir, os.path.basename(student_checkpoint_dir))
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "e2e_fintuning.log")
    log = get_logger("e2e_fintuning")
    resume_from_checkpoint = None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint).strip()

    log.info("Run output directory: %s", run_output_dir)
    log.info("Input e2e args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    log.info("Resolved student checkpoint directory: %s", student_checkpoint_dir)
    if resume_from_checkpoint:
        log.info("Resuming Trainer state from checkpoint: %s", resume_from_checkpoint)

    load_checkpoint_dir = resume_from_checkpoint or student_checkpoint_dir
    model, meta, load_result = load_e2e_model_checkpoint(
        load_checkpoint_dir,
        access_token=hf_args.access_token,
        base_model_path=None if resume_from_checkpoint else args.teacher_model_path,
        map_location="cpu",
        strict=True,
    )
    log.info(
        "Student checkpoint loaded from %s. missing_keys=%d unexpected_keys=%d converted_module_count=%s adapter_module_count=%s",
        load_checkpoint_dir,
        len(getattr(load_result, "missing_keys", [])),
        len(getattr(load_result, "unexpected_keys", [])),
        str(meta.get("converted_module_count")),
        str(meta.get("adapter_module_count", 0)),
    )

    base_model_path = meta.get("base_model_path") or args.teacher_model_path
    if not base_model_path:
        raise ValueError("Cannot resolve tokenizer/base model path from checkpoint meta.")

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if bool(args.lora_embedding) or bool(args.lora_lm_head):
        if _embedding_and_lm_head_are_tied(model):
            log.info("Detected tied word embeddings; separating embedding and lm_head before LoRA wrapping.")
            separate_embeddings_and_lm_head(model)

    layers = list(get_layers(model))
    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(layers))
    if resume_from_checkpoint:
        _validate_resume_checkpoint_config(
            args=args,
            meta=meta,
            decoder_layer_ids=decoder_layer_ids,
        )
    if (
        not resume_from_checkpoint
        and str(args.vae_lora_init_mode) == "residual_svd"
        and _checkpoint_has_peft_proxy_lora(meta)
    ):
        raise ValueError("Fresh e2e 训练遇到已包含 peft_proxy_lora 的 checkpoint，拒绝再次执行 residual_svd 初始化。")
    selection = select_e2e_trainables_peft_proxy(
        model,
        decoder_layer_ids=decoder_layer_ids,
        target_module_names=args.target_module_names,
        vae_lora_rank=int(args.vae_lora_rank),
        vae_lora_alpha=float(args.vae_lora_alpha),
        vae_lora_dropout=float(args.vae_lora_dropout),
        lora_embedding=bool(args.lora_embedding),
        lora_lm_head=bool(args.lora_lm_head),
    )
    injected_proxy_count = 0
    if selection.peft_proxy_modules:
        injected_proxy_count = ensure_peft_vae_proxy_lora(
            model,
            rank=int(args.vae_lora_rank),
            alpha=float(args.vae_lora_alpha),
            dropout=float(args.vae_lora_dropout),
            use_rslora=False,
        )
    need_residual_svd_init = _should_initialize_vae_lora_residual_svd(
        args=args,
        selection=selection,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    teacher_model, teacher_source, keep_teacher_for_training = _load_teacher_for_e2e(
        args=args,
        hf_args=hf_args,
        meta=meta,
        log=log,
        require_for_init=need_residual_svd_init,
    )
    if need_residual_svd_init:
        initialized_proxy_count = initialize_peft_vae_proxy_lora_from_teacher_residual(
            model,
            teacher_model,
        )
        log.info(
            "Initialized %d PEFT VAELinear proxy LoRA modules with residual_svd.",
            initialized_proxy_count,
        )
        if not keep_teacher_for_training:
            teacher_model = _release_init_only_teacher(teacher_model, log)
    trainable_params = _collect_trainable_params(model)
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for requested decoder layers.")
    setattr(model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)
    log.info(
        "Selected trainables: mode=%s layers=%s modules=%d adapters=%d peft_proxy=%d trainable_tensors=%d total_params=%d cacheable=%d",
        _E2E_FINETUNE_MODE,
        selection.decoder_layer_ids,
        len(selection.target_modules),
        len(selection.adapter_modules),
        injected_proxy_count,
        len(trainable_params),
        int(sum(int(param.numel()) for _name, param in trainable_params)),
        len(selection.frozen_cacheable_vae_modules),
    )

    tokenizer = build_tokenizer(str(base_model_path), access_token=hf_args.access_token)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset, eval_dataset, data_info = build_datasets(args, training_args, tokenizer)
    if len(train_dataset) < 1:
        raise ValueError(
            "Packed training dataset is empty. Increase input text volume or lower --model_max_length."
        )
    if eval_dataset is not None and len(eval_dataset) < 1:
        eval_dataset = None
    log.info(
        "Prepared datasets: train=%d eval=%s block_size=%d",
        len(train_dataset),
        "none" if eval_dataset is None else str(len(eval_dataset)),
        int(data_info["block_size"]),
    )

    log.info("Teacher source: %s", teacher_source)

    training_args.output_dir = os.path.join(run_output_dir, "trainer_state")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    trainer_cls = E2EFSDPFinetuneTrainer if _uses_fsdp(training_args) else E2EFinetuneTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        loss_type=args.loss_type,
        teacher_model=teacher_model,
        distill_temperature=args.distill_temperature,
        distill_alpha=args.distill_alpha,
        post_attn=bool(args.post_attn),
        lora_hif4_act=bool(args.lora_hif4_act),
        prewarm_frozen_vae=bool(args.prewarm_frozen_vae),
        prewarm_log_every=int(args.prewarm_log_every),
    )
    checkpoint_extra_meta = {
        "stage": "e2e_fintuning",
        "source_checkpoint_dir": student_checkpoint_dir,
        "teacher_source": teacher_source,
        "target_decoder_layers": list(selection.decoder_layer_ids),
        "target_module_names": None if args.target_module_names is None else list(args.target_module_names),
        "loss_type": str(args.loss_type),
        "post_attn": bool(args.post_attn),
        "lora_embedding": bool(args.lora_embedding),
        "lora_lm_head": bool(args.lora_lm_head),
        "lora_hif4_act": bool(args.lora_hif4_act),
        "finetune_mode": _E2E_FINETUNE_MODE,
        "prewarm_frozen_vae": bool(args.prewarm_frozen_vae),
        "vae_lora_rank": int(args.vae_lora_rank),
        "vae_lora_alpha": float(args.vae_lora_alpha),
        "vae_lora_dropout": float(args.vae_lora_dropout),
    }
    saved_init_mode = _resolve_saved_vae_lora_init_mode(
        args=args,
        meta=meta,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    if saved_init_mode is not None:
        checkpoint_extra_meta["vae_lora_init_mode"] = saved_init_mode
    trainer._e2e_base_model_path = str(base_model_path)
    trainer._e2e_checkpoint_extra_meta = checkpoint_extra_meta
    hif4_act_handles = []
    if trainer.lora_hif4_act_controller is not None:
        hif4_act_handles = register_lora_hif4_act_hooks(trainer.model, trainer.lora_hif4_act_controller)
        if not hif4_act_handles:
            raise RuntimeError("启用 --lora_hif4_act 失败：未找到可注册 hook 的逻辑线性层。")
        trainer.lora_hif4_act_controller.enabled = True
        log.info("Registered %d HiFloat4 activation hooks for e2e LoRA training.", len(hif4_act_handles))
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    finally:
        if trainer.lora_hif4_act_controller is not None:
            trainer.lora_hif4_act_controller.enabled = False
        remove_lora_hif4_act_hooks(hif4_act_handles)

    final_model = _unwrap_model(trainer, trainer.model)
    setattr(final_model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)
    _ensure_student_mode(final_model)
    converted_proxy_count = convert_peft_vae_proxy_modules_to_lora(final_model)
    if converted_proxy_count > 0:
        log.info("Converted %d PEFT VAELinear proxy modules back to LoRAVAELinear before final save.", converted_proxy_count)
    merged_extra_lora_count = 0
    merged_state_dict = None
    if _uses_fsdp(training_args):
        if bool(args.unload_vae_original_weights_on_save):
            unload_vae_original_linear_weights(final_model)
        merged_state_dict, merged_extra_lora_count = merge_extra_lora_state_dict(
            final_model,
            pt_fsdp_state_dict(trainer.model),
        )
        final_model, _ = merge_and_unload_extra_lora_modules(final_model)
    else:
        final_model, merged_extra_lora_count = merge_and_unload_extra_lora_modules(final_model)
    clear_model_vae_linear_cache(final_model)
    if teacher_model is not None:
        teacher_model.to("cpu")
    if merged_extra_lora_count > 0:
        log.info("Merged and unloaded %d embedding/lm_head LoRA modules before final save.", merged_extra_lora_count)

    final_dir = os.path.join(run_output_dir, "final_model")
    extra_meta = dict(checkpoint_extra_meta)

    if _uses_fsdp(training_args):
        save_paths = save_e2e_model_checkpoint(
            final_model,
            final_dir,
            base_model_path=str(base_model_path),
            tokenizer=tokenizer if bool(args.save_tokenizer) else None,
            save_config=True,
            extra_meta=extra_meta,
            state_dict=merged_state_dict,
        )
    else:
        save_paths = save_e2e_model_checkpoint(
            final_model,
            final_dir,
            base_model_path=str(base_model_path),
            tokenizer=tokenizer if bool(args.save_tokenizer) else None,
            save_config=True,
            extra_meta=extra_meta,
            unload_vae_original_weights=bool(args.unload_vae_original_weights_on_save),
        )

    log.info("Saved final model to %s", save_paths["output_dir"])
    ppl_eval = _eval_final_ppl(
        model=final_model,
        args=args,
        model_path=str(base_model_path),
        output_dir=run_output_dir,
        log=log,
    )
    clear_model_vae_linear_cache(final_model)
    return {
        "run_output_dir": run_output_dir,
        "saved_model_dir": save_paths["output_dir"],
        "teacher_source": teacher_source,
        "final_ppl": None if ppl_eval is None else ppl_eval["result"],
        "final_ppl_path": None if ppl_eval is None else ppl_eval["path"],
    }
