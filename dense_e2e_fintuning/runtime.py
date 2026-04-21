import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import default_data_collator

from dense_e2e_fintuning.args import needs_teacher
from dense_e2e_fintuning.checkpoint_bridge import (
    build_dense_model_from_checkpoint,
    export_dense_peft_to_compact_checkpoint,
    rebuild_dense_peft_model_for_export,
    resolve_base_model_path,
    resolve_decode_device,
)
from dense_e2e_fintuning.trainables import inject_dense_peft_adapters, resolve_target_layer_ids
from dense_e2e_fintuning.trainer import (
    DenseAdaLoraCallback,
    DenseFinetuneTrainer,
    DenseFSDPFinetuneTrainer,
)
from e2e_common.data import build_datasets, build_tokenizer
from e2e_common.post_norm_head import ensure_post_norm_head_linear
from rotation.model_utils import get_layers, get_model
from train_utils.eval_utils import calculate_ppl
from train_utils.hif4_act import (
    applied_hif4_act,
    register_hif4_act_hooks,
    remove_hif4_act_hooks,
)
from train_utils.model_checkpoint_io import _build_run_output_dir
from train_utils.utils import get_logger, pt_fsdp_state_dict


def _unwrap_model(trainer, model):
    if getattr(trainer, "accelerator", None) is None:
        return model
    return trainer.accelerator.unwrap_model(model)


def _uses_fsdp(training_args) -> bool:
    fsdp = getattr(training_args, "fsdp", "")
    return not (fsdp is None or fsdp == "" or fsdp == [])


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return str(value)


def _namespace_to_dict(ns) -> Dict[str, Any]:
    if ns is None:
        return {}
    if is_dataclass(ns):
        return {str(k): _jsonable(v) for k, v in asdict(ns).items()}
    if hasattr(ns, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(ns).items()}
    return {"value": _jsonable(ns)}


def _build_run_meta(
    *,
    dense_args,
    hf_args,
    training_args,
    data_info: Dict[str, Any],
    trainable_info: Dict[str, Any],
    teacher_source: str,
    student_checkpoint_dir: str,
    base_model_path: str,
    decode_device_requested: str,
    decode_device_resolved: str,
    decode_group_size: int,
    global_step: int,
) -> Dict[str, Any]:
    return {
        "format": "dense_e2e_run_meta",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "dense_e2e_fintuning",
        "teacher_source": str(teacher_source),
        "student_checkpoint_dir": str(student_checkpoint_dir),
        "base_model_path": str(base_model_path),
        "decode_device_requested": str(decode_device_requested),
        "decode_device_resolved": str(decode_device_resolved),
        "decode_group_size": int(decode_group_size),
        "global_step": int(global_step),
        "dense_args": _namespace_to_dict(dense_args),
        "hf_args": _namespace_to_dict(hf_args),
        "training_args": _namespace_to_dict(training_args),
        "dataset": _jsonable(data_info),
        "trainables": _jsonable(trainable_info),
    }


def _selection_to_meta(selection) -> Dict[str, object]:
    data = asdict(selection)
    data["target_module_count"] = len(selection.target_modules)
    data["decoder_layer_count"] = len(selection.decoder_layer_ids)
    return data


def _load_teacher_for_dense(*, args, hf_args, base_model_path: str, log):
    requires_teacher = needs_teacher(args.loss_type)
    teacher_path = None if args.teacher_model_path is None else str(args.teacher_model_path).strip()
    if not teacher_path:
        teacher_path = str(base_model_path)
    if not requires_teacher:
        return None, "disabled"
    log.info("Loading teacher model from %s", teacher_path)
    teacher_model = get_model(teacher_path, hf_args.access_token)
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
    with applied_hif4_act(
        model,
        enabled=bool(getattr(args, "eval_hif4_act", False)),
        logger=log,
        log_prefix="[final_ppl] ",
    ):
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


def _save_dense_adapter(
    *,
    model,
    output_dir: str,
    tokenizer,
    save_tokenizer: bool,
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    if bool(save_tokenizer) and tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    return output_dir


def run(args, hf_args, training_args):
    run_output_dir = _build_run_output_dir(args.run_root_dir, os.path.basename(args.student_checkpoint_dir))
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "dense_e2e_fintuning.log")
    log = get_logger("dense_e2e_fintuning")
    resume_from_checkpoint = None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint).strip()

    log.info("Run output directory: %s", run_output_dir)
    log.info("Resolved student checkpoint directory: %s", args.student_checkpoint_dir)
    log.info("Input dense e2e args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    if resume_from_checkpoint:
        log.info("Resuming Trainer state from checkpoint: %s", resume_from_checkpoint)
    requested_decode_device = str(getattr(args, "decode_device", "auto"))
    resolved_decode_device = resolve_decode_device(requested_decode_device)
    log.info(
        "Decode config: requested_device=%s resolved_device=%s group_size=%d",
        requested_decode_device,
        resolved_decode_device,
        int(args.decode_group_size),
    )

    dense_model, meta, _resolved_dir = build_dense_model_from_checkpoint(
        args.student_checkpoint_dir,
        access_token=hf_args.access_token,
        logger=log,
        decode_group_size=int(args.decode_group_size),
        decode_device=requested_decode_device,
    )
    base_model_path = resolve_base_model_path(meta, args.teacher_model_path)

    if hasattr(dense_model, "config"):
        dense_model.config.use_cache = False
    if hasattr(dense_model, "enable_input_require_grads"):
        dense_model.enable_input_require_grads()
    if bool(args.use_post_norm_head_linear):
        ensure_post_norm_head_linear(dense_model)

    layers = list(get_layers(dense_model))
    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(layers))
    peft_model, selection = inject_dense_peft_adapters(
        dense_model,
        args=args,
        decoder_layer_ids=decoder_layer_ids,
        total_step=int(training_args.max_steps),
    )
    log.info(
        "Selected trainables: variant=%s layers=%s targets=%d target_suffixes=%s modules_to_save=%s trainable_tensors=%d trainable_params=%d",
        str(args.lora_variant),
        selection.decoder_layer_ids,
        len(selection.target_modules),
        selection.target_module_suffixes,
        selection.modules_to_save,
        len(selection.trainable_parameter_names),
        int(selection.trainable_parameter_count),
    )

    tokenizer = build_tokenizer(str(base_model_path), access_token=hf_args.access_token)
    if getattr(peft_model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        peft_model.config.pad_token_id = tokenizer.pad_token_id

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
    if str(data_info.get("dataset_mode", "single")) == "mix":
        log.info(
            "Mixed dataset config: spec=%s target_examples=%d required_examples=%d",
            str(data_info["dataset_mix_spec"]),
            int(data_info["dataset_mix_target_examples"]),
            int(data_info["required_train_examples"]),
        )
        for source_stat in data_info.get("source_stats", []):
            log.info(
                "Mixed dataset source: alias=%s weight=%.6f raw_rows=%d text_rows=%d packed_rows=%d target_rows=%d repeat_factor=%.4f eval_packed_rows=%d",
                str(source_stat["alias"]),
                float(source_stat["weight"]),
                int(source_stat["raw_rows"]),
                int(source_stat["text_rows"]),
                int(source_stat["packed_rows"]),
                int(source_stat["target_rows"]),
                float(source_stat["repeat_factor"]),
                int(source_stat.get("eval_packed_rows", 0)),
            )

    teacher_model, teacher_source = _load_teacher_for_dense(
        args=args,
        hf_args=hf_args,
        base_model_path=str(base_model_path),
        log=log,
    )
    log.info("Teacher source: %s", teacher_source)

    training_args.output_dir = os.path.join(run_output_dir, "trainer_state")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    trainer_cls = DenseFSDPFinetuneTrainer if _uses_fsdp(training_args) else DenseFinetuneTrainer
    trainer = trainer_cls(
        model=peft_model,
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
    )
    if str(args.lora_variant) == "adalora":
        trainer.add_callback(DenseAdaLoraCallback(trainer))

    hif4_act_handles = []
    if trainer.lora_hif4_act_controller is not None:
        hif4_act_handles = register_hif4_act_hooks(trainer.model, trainer.lora_hif4_act_controller)
        if not hif4_act_handles:
            raise RuntimeError("启用 HiFloat4 激活量化失败：未找到可注册 hook 的线性层。")
        trainer.lora_hif4_act_controller.enabled = True
        log.info("Registered %d HiFloat4 activation hooks for dense LoRA training.", len(hif4_act_handles))

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    finally:
        if trainer.lora_hif4_act_controller is not None:
            trainer.lora_hif4_act_controller.enabled = False
        remove_hif4_act_hooks(hif4_act_handles)

    final_model = _unwrap_model(trainer, trainer.model)
    final_model.eval()
    if teacher_model is not None:
        teacher_model.to("cpu")

    ppl_eval = _eval_final_ppl(
        model=final_model,
        args=args,
        model_path=str(base_model_path),
        output_dir=run_output_dir,
        log=log,
    )

    should_save = bool(getattr(training_args, "should_save", True))
    final_state_dict = pt_fsdp_state_dict(trainer.model) if _uses_fsdp(training_args) else None
    export_model = final_model
    if should_save and final_state_dict is not None:
        log.info("Rebuilding dense PEFT model on CPU from FSDP state_dict for final export.")
        export_model, _meta, _selection = rebuild_dense_peft_model_for_export(
            args.student_checkpoint_dir,
            access_token=hf_args.access_token,
            args=args,
            training_args=training_args,
            state_dict=final_state_dict,
            decode_group_size=int(args.decode_group_size),
            decode_device=requested_decode_device,
            logger=log,
        )
    elif should_save:
        export_model.to("cpu")

    adapter_dir = None
    compact_dir = None
    run_meta_path = None
    if should_save:
        adapter_dir = _save_dense_adapter(
            model=export_model,
            output_dir=os.path.join(run_output_dir, "final_adapter"),
            tokenizer=tokenizer,
            save_tokenizer=bool(args.save_tokenizer),
        )
        compact_extra_meta = {
            "stage": "dense_e2e_fintuning",
            "teacher_source": str(teacher_source),
            "source_checkpoint_dir": str(args.student_checkpoint_dir),
            "target_decoder_layers": list(selection.decoder_layer_ids),
            "target_module_names": None if args.target_module_names is None else list(args.target_module_names),
            "lora_variant": str(args.lora_variant),
            "lora_rank": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_tune_bias": bool(args.lora_tune_bias),
            "tune_final_norm": bool(args.tune_final_norm),
            "use_post_norm_head_linear": bool(args.use_post_norm_head_linear),
        }
        compact_dir = export_dense_peft_to_compact_checkpoint(
            export_model,
            student_checkpoint_dir=args.student_checkpoint_dir,
            access_token=hf_args.access_token,
            output_dir=os.path.join(run_output_dir, "final_model"),
            args=args,
            training_args=training_args,
            base_model_path=str(base_model_path),
            tokenizer=tokenizer,
            save_tokenizer=bool(args.save_tokenizer),
            extra_meta=compact_extra_meta,
            decode_group_size=int(args.decode_group_size),
            decode_device=requested_decode_device,
            logger=log,
        )["output_dir"]

        run_meta = _build_run_meta(
            dense_args=args,
            hf_args=hf_args,
            training_args=training_args,
            data_info=data_info,
            trainable_info=_selection_to_meta(selection),
            teacher_source=teacher_source,
            student_checkpoint_dir=str(args.student_checkpoint_dir),
            base_model_path=str(base_model_path),
            decode_device_requested=requested_decode_device,
            decode_device_resolved=resolved_decode_device,
            decode_group_size=int(args.decode_group_size),
            global_step=int(getattr(trainer.state, "global_step", 0)),
        )
        run_meta_path = os.path.join(run_output_dir, "run_meta.json")
        with open(run_meta_path, "w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, ensure_ascii=False, indent=2)
        log.info(
            "Saved final artifacts: adapter=%s compact=%s run_meta=%s",
            adapter_dir,
            compact_dir,
            run_meta_path,
        )
    else:
        log.info("Skipping final artifact save on this rank because should_save=false.")

    return {
        "run_output_dir": run_output_dir,
        "saved_adapter_dir": adapter_dir,
        "saved_compact_dir": compact_dir,
        "teacher_source": teacher_source,
        "final_ppl": None if ppl_eval is None else ppl_eval["result"],
        "final_ppl_path": None if ppl_eval is None else ppl_eval["path"],
        "run_meta_path": run_meta_path,
    }
