import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, default_data_collator

from dense_e2e_fintuning.args import needs_teacher
from dense_e2e_fintuning.checkpoint_bridge import (
    load_compressed_student_checkpoint,
    resolve_base_model_path,
)
from dense_e2e_fintuning.runtime import _build_datasets_with_main_process_first, _eval_final_ppl
from e2e_common.data import build_tokenizer
from e2e_common.post_norm_head import ensure_post_norm_head_linear
from rotation.model_utils import get_layers, get_model
from train_utils.model_checkpoint_io import (
    STATE_DICT_FILENAME,
    _build_distributed_run_output_dir,
    save_model_checkpoint,
)
from train_utils.utils import get_logger
from vae_e2e_fintuning.device_map import apply_boundary_device_map, apply_layer_device_map, resolve_layer_device_map
from vae_e2e_fintuning.offload import (
    SavedTensorOffloadContext,
    unwrap_streaming_offload_layers,
    validate_streaming_layer_devices,
    wrap_model_layers_for_streaming_offload,
)
from vae_e2e_fintuning.trainables import (
    resolve_target_layer_ids,
    select_vae_decoder_trainables,
    unpack_parallel_stage_decoders,
)
from vae_e2e_fintuning.trainer import VAEDecoderE2ETrainer


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


def _selection_to_meta(selection) -> Dict[str, object]:
    data = asdict(selection)
    data["target_module_count"] = len(selection.target_modules)
    data["decoder_layer_count"] = len(selection.decoder_layer_ids)
    return data


def _load_teacher(*, args, hf_args, base_model_path: str, log):
    if not needs_teacher(args.loss_type):
        return None, "disabled"
    teacher_path = str(args.teacher_model_path or base_model_path)
    log.info("Loading teacher model from %s", teacher_path)
    teacher_model = get_model(teacher_path, hf_args.access_token)
    teacher_model.eval()
    if hasattr(teacher_model, "config"):
        teacher_model.config.use_cache = False
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model, "external_teacher"


def _resolve_eval_device(requested_device: str) -> str:
    device = str(requested_device or "").strip()
    if not device:
        raise ValueError("--eval_device cannot be empty.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested eval_device={device}, but CUDA is not available.")
    return device


def _disable_trainable_decode_for_eval(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        disable_fn = getattr(module, "disable_trainable_decode", None)
        if callable(disable_fn):
            disable_fn()
            count += 1
    return count


def _run_final_lm_eval(*, model, tokenizer, args, base_model_path: str, output_dir: str, log):
    tasks = None if args.eval_tasks is None else str(args.eval_tasks).strip()
    if not tasks:
        return None

    device = _resolve_eval_device(str(args.eval_device))
    eval_log_dir = os.path.join(output_dir, "lm_eval")
    os.makedirs(eval_log_dir, exist_ok=True)
    log.info("[lm_eval] Moving final saved model to %s ...", device)
    model.to(device)

    from train_utils.eval_utils import run_lm_eval
    from train_utils.hif4_act import applied_hif4_act

    lm_args = argparse.Namespace(
        tasks=tasks,
        num_fewshot=int(args.eval_num_fewshot),
        batch_size=str(args.eval_lm_batch_size),
        lm_limit=args.eval_lm_limit,
        model_path=str(base_model_path),
        eval_log_dir=eval_log_dir,
        eval_run_ts="final",
    )
    log.info(
        "[lm_eval] tasks=%s fewshot=%d batch_size=%s limit=%s",
        tasks,
        int(args.eval_num_fewshot),
        str(args.eval_lm_batch_size),
        str(args.eval_lm_limit),
    )
    with applied_hif4_act(
        model,
        enabled=bool(args.eval_hif4_act),
        logger=log,
        log_prefix="[lm_eval] ",
    ):
        result = run_lm_eval(model, tokenizer, lm_args)

    table = str(result.get("summary_table", "")).strip()
    if table:
        log.info("[lm_eval] Summary table:\n%s", table)
    return {
        "result": result,
        "json_path": os.path.join(eval_log_dir, "lm_eval_results_final.json"),
        "summary_path": os.path.join(eval_log_dir, "lm_eval_summary_final.md"),
    }


def _build_run_meta(
    *,
    args,
    hf_args,
    training_args,
    data_info: Dict[str, Any],
    trainable_info: Dict[str, Any],
    teacher_source: str,
    resolved_student_checkpoint_dir: str,
    base_model_path: str,
    layer_device_map: Dict[str, str],
    offload_mode: str,
    global_step: int,
) -> Dict[str, Any]:
    return {
        "format": "vae_decoder_e2e_run_meta",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "vae_e2e_fintuning",
        "teacher_source": str(teacher_source),
        "student_checkpoint_dir": str(resolved_student_checkpoint_dir),
        "base_model_path": str(base_model_path),
        "source_checkpoint_state_dict_file": STATE_DICT_FILENAME,
        "layer_device_map": dict(layer_device_map),
        "offload_mode": str(offload_mode),
        "global_step": int(global_step),
        "vae_e2e_args": _namespace_to_dict(args),
        "hf_args": _namespace_to_dict(hf_args),
        "training_args": _namespace_to_dict(training_args),
        "dataset": _jsonable(data_info),
        "trainables": _jsonable(trainable_info),
    }


def run(args, hf_args, training_args):
    run_output_dir = _build_distributed_run_output_dir(
        args.run_root_dir,
        os.path.basename(args.student_checkpoint_dir),
    )
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "vae_e2e_fintuning.log")
    log = get_logger("vae_e2e_fintuning")
    resume_from_checkpoint = None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint).strip()

    log.info("Run output directory: %s", run_output_dir)
    log.info("Resolved student checkpoint directory: %s", args.student_checkpoint_dir)
    log.info("Input VAE decoder e2e args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    model, meta, resolved_student_checkpoint_dir = load_compressed_student_checkpoint(
        args.student_checkpoint_dir,
        access_token=hf_args.access_token,
        logger=log,
    )
    base_model_path = resolve_base_model_path(meta, args.teacher_model_path)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if bool(args.use_post_norm_head_linear):
        ensure_post_norm_head_linear(model)

    layers = list(get_layers(model))
    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(layers))
    selection = select_vae_decoder_trainables(
        model,
        decoder_layer_ids=decoder_layer_ids,
        target_module_names=args.target_module_names,
        parallel_stage_decode=bool(args.parallel_stage_decode),
        tune_final_norm=bool(args.tune_final_norm),
        use_post_norm_head_linear=bool(args.use_post_norm_head_linear),
        vae_tune_bias=bool(args.vae_tune_bias),
    )
    log.info(
        "Selected VAE decoder trainables: layers=%s targets=%d suffixes=%s bias_modules=%d final_norm=%s post_norm_head=%s trainable_tensors=%d trainable_params=%d parallel_stage_decode=%s",
        selection.decoder_layer_ids,
        len(selection.target_modules),
        selection.target_module_suffixes,
        len(selection.bias_modules),
        selection.final_norm_modules,
        selection.post_norm_head_modules,
        len(selection.trainable_parameter_names),
        int(selection.trainable_parameter_count),
        str(selection.parallel_stage_decode).lower(),
    )

    resolved_layer_device_map = resolve_layer_device_map(args.layer_device_map, len(layers))
    offload_mode = str(args.offload_mode).strip().lower()
    streaming_manager = None
    saved_tensor_offload = None
    if offload_mode == "streaming":
        validate_streaming_layer_devices(resolved_layer_device_map)
        if bool(getattr(training_args, "gradient_checkpointing", False)):
            log.info(
                "offload_mode=streaming manages layer checkpointing itself; overriding HF gradient_checkpointing=false."
            )
            training_args.gradient_checkpointing = False
        hook_handles, boundary_map = apply_boundary_device_map(model, layer_device_map=resolved_layer_device_map)
        streaming_manager, streaming_map = wrap_model_layers_for_streaming_offload(
            model,
            layer_devices=resolved_layer_device_map,
            prefetch_distance=int(args.offload_prefetch_distance),
            checkpoint_layers=bool(args.offload_checkpoint),
        )
        hf_device_map = {**boundary_map, **streaming_map}
    else:
        hook_handles, hf_device_map = apply_layer_device_map(model, layer_device_map=resolved_layer_device_map)

    if offload_mode in {"saved_tensors", "streaming"}:
        saved_tensor_offload = SavedTensorOffloadContext(
            enabled=True,
            min_tensor_bytes=int(args.offload_min_tensor_bytes),
            pin_memory=bool(args.offload_pin_memory),
        )

    log.info(
        "Applied layer device/offload config: offload_mode=%s offload_checkpoint=%s map=%s saved_tensor_min_bytes=%d pin_memory=%s",
        offload_mode,
        str(bool(args.offload_checkpoint)).lower(),
        json.dumps(hf_device_map, ensure_ascii=False, sort_keys=True),
        int(args.offload_min_tensor_bytes),
        str(bool(args.offload_pin_memory)).lower(),
    )

    tokenizer = build_tokenizer(str(base_model_path), access_token=hf_args.access_token)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset, eval_dataset, data_info = _build_datasets_with_main_process_first(
        args,
        training_args,
        tokenizer,
        log,
    )
    if len(train_dataset) < 1:
        raise ValueError("Packed training dataset is empty. Increase input text volume or lower --model_max_length.")
    if eval_dataset is not None and len(eval_dataset) < 1:
        eval_dataset = None
    log.info(
        "Prepared datasets: train=%d eval=%s block_size=%d",
        len(train_dataset),
        "none" if eval_dataset is None else str(len(eval_dataset)),
        int(data_info["block_size"]),
    )

    teacher_model, teacher_source = _load_teacher(
        args=args,
        hf_args=hf_args,
        base_model_path=str(base_model_path),
        log=log,
    )
    log.info("Teacher source: %s", teacher_source)

    training_args.output_dir = os.path.join(run_output_dir, "trainer_state")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    trainer = VAEDecoderE2ETrainer(
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
        saved_tensor_offload=saved_tensor_offload,
        streaming_offload_manager=streaming_manager,
    )
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    except Exception:
        if streaming_manager is not None:
            streaming_manager.offload_all(synchronize=True)
        for handle in hook_handles:
            handle.remove()
        raise

    final_model = trainer.accelerator.unwrap_model(trainer.model) if getattr(trainer, "accelerator", None) else trainer.model
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

    for handle in hook_handles:
        handle.remove()
    if streaming_manager is not None:
        streaming_manager.offload_all(synchronize=True)
        unwrapped_streaming = unwrap_streaming_offload_layers(final_model)
        log.info("Unwrapped %d streaming offload layers before final save.", unwrapped_streaming)
    final_model.to("cpu")
    unpacked = unpack_parallel_stage_decoders(final_model)
    log.info("Unpacked %d parallel stage decoder modules before final save.", unpacked)
    disabled_decode = _disable_trainable_decode_for_eval(final_model)
    log.info("Disabled trainable decode mode on %d VAELinear modules before final save/eval.", disabled_decode)

    model_out = None
    run_meta_path = None
    lm_eval = None
    if bool(getattr(training_args, "should_save", True)):
        model_out = os.path.join(run_output_dir, "final_model")
        tok = None
        if bool(args.save_tokenizer):
            tok = AutoTokenizer.from_pretrained(str(base_model_path), use_fast=True, token=hf_args.access_token)
        save_paths = save_model_checkpoint(
            final_model,
            model_out,
            base_model_path=str(base_model_path),
            tokenizer=tok,
            save_config=True,
            extra_meta={
                "stage": "vae_e2e_fintuning",
                "tune_final_norm": bool(args.tune_final_norm),
                "use_post_norm_head_linear": bool(args.use_post_norm_head_linear),
                "vae_tune_bias": bool(args.vae_tune_bias),
            },
            unload_vae_original_weights=False,
        )
        run_meta = _build_run_meta(
            args=args,
            hf_args=hf_args,
            training_args=training_args,
            data_info=data_info,
            trainable_info=_selection_to_meta(selection),
            teacher_source=teacher_source,
            resolved_student_checkpoint_dir=str(resolved_student_checkpoint_dir),
            base_model_path=str(base_model_path),
            layer_device_map=hf_device_map,
            offload_mode=offload_mode,
            global_step=int(getattr(trainer.state, "global_step", 0)),
        )
        run_meta_path = os.path.join(run_output_dir, "run_meta.json")
        with open(run_meta_path, "w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, ensure_ascii=False, indent=2)
        log.info("Saved final compressed model to %s", save_paths["output_dir"])
        lm_eval = _run_final_lm_eval(
            model=final_model,
            tokenizer=tokenizer,
            args=args,
            base_model_path=str(base_model_path),
            output_dir=run_output_dir,
            log=log,
        )
    else:
        log.info("Skipping final model save on this rank because should_save=false.")

    return {
        "run_output_dir": run_output_dir,
        "saved_model_dir": model_out,
        "teacher_source": teacher_source,
        "final_ppl": None if ppl_eval is None else ppl_eval["result"],
        "final_ppl_path": None if ppl_eval is None else ppl_eval["path"],
        "final_lm_eval_path": None if lm_eval is None else lm_eval["json_path"],
        "final_lm_eval_summary_path": None if lm_eval is None else lm_eval["summary_path"],
        "run_meta_path": run_meta_path,
    }
