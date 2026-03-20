import json
import os
import argparse

import torch

from transformers import default_data_collator

from distill_utils.layerwise_distill_runtime import resolve_checkpoint_dir
from e2e_fintuning.checkpoint_io import load_e2e_model_checkpoint, save_e2e_model_checkpoint
from e2e_fintuning.data import build_datasets, build_tokenizer
from e2e_fintuning.trainables import resolve_target_layer_ids, select_e2e_trainables
from e2e_fintuning.trainer import (
    E2EFinetuneTrainer,
    E2EFSDPFinetuneTrainer,
    model_requires_external_teacher,
    set_model_temporary,
)
from litebsq.vae_linear import clear_model_vae_linear_cache
from rotation.model_utils import get_layers, get_model
from train_utils.eval_utils import calculate_ppl
from train_utils.model_checkpoint_io import _build_run_output_dir, unload_vae_original_linear_weights
from train_utils.utils import get_logger, pt_fsdp_state_dict


def _unwrap_model(trainer, model):
    if getattr(trainer, "accelerator", None) is None:
        return model
    return trainer.accelerator.unwrap_model(model)


def _uses_fsdp(training_args) -> bool:
    fsdp = getattr(training_args, "fsdp", "")
    return not (fsdp is None or fsdp == "" or fsdp == [])


def _ensure_student_mode(model) -> None:
    set_model_temporary(model, True)


def _load_teacher_if_needed(*, model, args, hf_args, meta, log):
    if args.loss_type in {"sft", "origin"}:
        return None, "student"
    if not model_requires_external_teacher(model):
        return None, "student_original_weights"

    teacher_model_path = args.teacher_model_path or meta.get("base_model_path")
    if not teacher_model_path:
        raise ValueError(
            "Teacher is required because current checkpoint lacks original weights, "
            "but neither --teacher_model_path nor checkpoint meta base_model_path is available."
        )

    log.info("Loading external teacher model from %s", teacher_model_path)
    teacher_model = get_model(str(teacher_model_path), hf_args.access_token)
    teacher_model.eval()
    if hasattr(teacher_model, "config"):
        teacher_model.config.use_cache = False
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model, "external_teacher"


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


def run(args, hf_args, training_args):
    student_checkpoint_dir = resolve_checkpoint_dir(args.student_checkpoint_dir)
    run_output_dir = _build_run_output_dir(args.run_root_dir, os.path.basename(student_checkpoint_dir))
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "e2e_fintuning.log")
    log = get_logger("e2e_fintuning")

    log.info("Run output directory: %s", run_output_dir)
    log.info("Input e2e args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    log.info("Resolved student checkpoint directory: %s", student_checkpoint_dir)

    model, meta, load_result = load_e2e_model_checkpoint(
        student_checkpoint_dir,
        access_token=hf_args.access_token,
        base_model_path=args.teacher_model_path,
        map_location="cpu",
        strict=True,
    )
    log.info(
        "Student checkpoint loaded. missing_keys=%d unexpected_keys=%d converted_module_count=%s adapter_module_count=%s",
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

    layers = list(get_layers(model))
    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(layers))
    selection = select_e2e_trainables(
        model,
        decoder_layer_ids=decoder_layer_ids,
        target_module_names=args.target_module_names,
        train_protected_outliers=bool(args.train_protected_outliers),
        finetune_mode=str(args.finetune_mode),
        vae_lora_rank=int(args.vae_lora_rank),
        vae_lora_alpha=float(args.vae_lora_alpha),
        vae_lora_dropout=float(args.vae_lora_dropout),
    )
    if not selection.trainable_params:
        raise RuntimeError("No trainable parameters found for requested decoder layers.")
    setattr(model, "_e2e_finetune_mode", str(args.finetune_mode))
    log.info(
        "Selected trainables: mode=%s layers=%s modules=%d full_tensors=%d lora_tensors=%d total_params=%d cacheable=%d non_cacheable=%d protected=%d",
        selection.finetune_mode,
        selection.decoder_layer_ids,
        len(selection.target_modules),
        len(selection.full_trainable_params),
        len(selection.lora_trainable_params),
        selection.trainable_param_count,
        len(selection.frozen_cacheable_vae_modules),
        len(selection.non_cacheable_vae_modules),
        len(selection.protected_param_names),
    )

    tokenizer = build_tokenizer(str(base_model_path), access_token=hf_args.access_token)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset, eval_dataset, data_info = build_datasets(args, training_args, tokenizer)
    if len(train_dataset) < 1:
        raise ValueError(
            "Packed training dataset is empty. Increase input text volume or lower --packing_block_size."
        )
    if eval_dataset is not None and len(eval_dataset) < 1:
        eval_dataset = None
    log.info(
        "Prepared datasets: train=%d eval=%s block_size=%d",
        len(train_dataset),
        "none" if eval_dataset is None else str(len(eval_dataset)),
        int(data_info["block_size"]),
    )

    teacher_model, teacher_source = _load_teacher_if_needed(
        model=model,
        args=args,
        hf_args=hf_args,
        meta=meta,
        log=log,
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
        prewarm_frozen_vae=bool(args.prewarm_frozen_vae),
        prewarm_log_every=int(args.prewarm_log_every),
    )
    trainer.train()

    final_model = _unwrap_model(trainer, trainer.model)
    setattr(final_model, "_e2e_finetune_mode", str(args.finetune_mode))
    _ensure_student_mode(final_model)
    clear_model_vae_linear_cache(final_model)
    if teacher_model is not None:
        teacher_model.to("cpu")

    final_dir = os.path.join(run_output_dir, "final_model")
    extra_meta = {
        "stage": "e2e_fintuning",
        "source_checkpoint_dir": student_checkpoint_dir,
        "teacher_source": teacher_source,
        "target_decoder_layers": list(selection.decoder_layer_ids),
        "target_module_names": None if args.target_module_names is None else list(args.target_module_names),
        "train_protected_outliers": bool(args.train_protected_outliers),
        "loss_type": str(args.loss_type),
        "finetune_mode": str(args.finetune_mode),
        "prewarm_frozen_vae": bool(args.prewarm_frozen_vae),
    }

    if _uses_fsdp(training_args):
        if bool(args.unload_vae_original_weights_on_save):
            unload_vae_original_linear_weights(final_model)
        save_paths = save_e2e_model_checkpoint(
            final_model,
            final_dir,
            base_model_path=str(base_model_path),
            tokenizer=tokenizer if bool(args.save_tokenizer) else None,
            save_config=True,
            extra_meta=extra_meta,
            state_dict=pt_fsdp_state_dict(trainer.model),
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
