import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Optional

import torch
from torch import nn
from transformers import default_data_collator

from e2e_common.data import build_datasets, build_tokenizer
from e2e_common.post_norm_head import ensure_post_norm_head_linear
from raw_e2e_fintuning.args import needs_teacher
from raw_e2e_fintuning.checkpoint_io import save_final_artifacts
from raw_e2e_fintuning.trainables import inject_raw_peft_adapters, resolve_target_layer_ids
from raw_e2e_fintuning.trainer import (
    RawAdaLoraCallback,
    RawFinetuneTrainer,
    RawFSDPFinetuneTrainer,
)
from rotation.model_utils import get_layers, get_model
from train_utils.eval_utils import calculate_ppl
from train_utils.hif4_act import (
    applied_hif4_act,
    register_hif4_act_hooks,
    remove_hif4_act_hooks,
)
from train_utils.model_checkpoint_io import _build_distributed_run_output_dir
from train_utils.utils import get_logger, pt_fsdp_state_dict


def _unwrap_model(trainer, model):
    if getattr(trainer, "accelerator", None) is None:
        return model
    return trainer.accelerator.unwrap_model(model)


def _uses_fsdp(training_args) -> bool:
    fsdp = getattr(training_args, "fsdp", "")
    return not (fsdp is None or fsdp == "" or fsdp == [])


def _load_teacher_for_raw(*, args, hf_args, log):
    requires_teacher = needs_teacher(args.loss_type)
    teacher_path = None if args.teacher_model_path is None else str(args.teacher_model_path).strip()
    if not requires_teacher:
        return None, "disabled"
    if not teacher_path:
        raise ValueError(
            "当前 loss_type 需要 teacher，请显式传 --teacher_model_path。"
        )
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


def _selection_to_meta(selection) -> Dict[str, object]:
    data = asdict(selection)
    data["target_module_count"] = len(selection.target_modules)
    data["decoder_layer_count"] = len(selection.decoder_layer_ids)
    return data


def _normalized_eval_strategy(training_args) -> str:
    eval_strategy = getattr(training_args, "eval_strategy", None)
    normalized = getattr(eval_strategy, "value", eval_strategy)
    if normalized is None:
        return "none"
    return str(normalized).strip().lower()


def _build_datasets_with_main_process_first(args, training_args, tokenizer, log):
    eval_strategy = _normalized_eval_strategy(training_args)
    skip_eval_preprocessing = eval_strategy == "no"
    log.info(
        "Dataset preprocess config: dataset_num_proc=%d eval_strategy=%s skip_eval_preprocessing=%s main_process_first=%s",
        int(getattr(args, "dataset_num_proc", 1)),
        eval_strategy,
        str(skip_eval_preprocessing).lower(),
        "true",
    )
    with training_args.main_process_first(local=False, desc="dataset preprocessing"):
        return build_datasets(args, training_args, tokenizer)


def run(args, hf_args, training_args):
    run_output_dir = _build_distributed_run_output_dir(args.run_root_dir, args.student_model_path)
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "e2e_raw_fintuning.log")
    log = get_logger("e2e_raw_fintuning")
    resume_from_checkpoint = None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint).strip()

    log.info("Run output directory: %s", run_output_dir)
    log.info("Input raw e2e args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    if resume_from_checkpoint:
        log.info("Resuming Trainer state from checkpoint: %s", resume_from_checkpoint)

    model = get_model(str(args.student_model_path), hf_args.access_token)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if bool(args.use_post_norm_head_linear):
        ensure_post_norm_head_linear(model)

    layers = list(get_layers(model))
    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(layers))
    peft_model, selection = inject_raw_peft_adapters(
        model,
        args=args,
        decoder_layer_ids=decoder_layer_ids,
        total_step=int(training_args.max_steps),
    )
    log.info(
        "Selected trainables: variant=%s layers=%s targets=%d target_suffixes=%s modules_to_save=%s lora_smooth_modules=%d lora_smooth_params=%d trainable_tensors=%d trainable_params=%d",
        str(args.lora_variant),
        selection.decoder_layer_ids,
        len(selection.target_modules),
        selection.target_module_suffixes,
        selection.modules_to_save,
        len(selection.lora_smooth_modules),
        int(selection.lora_smooth_parameter_count),
        len(selection.trainable_parameter_names),
        int(selection.trainable_parameter_count),
    )

    tokenizer = build_tokenizer(str(args.student_model_path), access_token=hf_args.access_token)
    if getattr(peft_model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        peft_model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset, eval_dataset, data_info = _build_datasets_with_main_process_first(
        args,
        training_args,
        tokenizer,
        log,
    )
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

    teacher_model, teacher_source = _load_teacher_for_raw(
        args=args,
        hf_args=hf_args,
        log=log,
    )
    log.info("Teacher source: %s", teacher_source)

    training_args.output_dir = os.path.join(run_output_dir, "trainer_state")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    trainer_cls = RawFSDPFinetuneTrainer if _uses_fsdp(training_args) else RawFinetuneTrainer
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
        trainer.add_callback(RawAdaLoraCallback(trainer))

    hif4_act_handles = []
    if trainer.lora_hif4_act_controller is not None:
        hif4_act_handles = register_hif4_act_hooks(trainer.model, trainer.lora_hif4_act_controller)
        if not hif4_act_handles:
            raise RuntimeError("启用 HiFloat4 激活量化失败：未找到可注册 hook 的线性层。")
        trainer.lora_hif4_act_controller.enabled = True
        log.info("Registered %d HiFloat4 activation hooks for raw LoRA training.", len(hif4_act_handles))

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
        model_path=str(args.student_model_path),
        output_dir=run_output_dir,
        log=log,
    )

    final_state_dict = pt_fsdp_state_dict(trainer.model) if _uses_fsdp(training_args) else None
    save_paths = save_final_artifacts(
        model=final_model,
        run_output_dir=run_output_dir,
        tokenizer=tokenizer,
        raw_args=args,
        hf_args=hf_args,
        training_args=training_args,
        data_info=data_info,
        trainable_info=_selection_to_meta(selection),
        teacher_source=teacher_source,
        global_step=int(getattr(trainer.state, "global_step", 0)),
        should_save=bool(getattr(training_args, "should_save", True)),
        state_dict=final_state_dict,
    )
    if save_paths["adapter_dir"] is not None:
        log.info(
            "Saved final artifacts: adapter=%s tokenizer=%s merged=%s run_meta=%s",
            save_paths["adapter_dir"],
            save_paths["tokenizer_dir"],
            save_paths["merged_dir"],
            save_paths["run_meta_path"],
        )
    else:
        log.info("Skipping final artifact save on this rank because should_save=false.")

    return {
        "run_output_dir": run_output_dir,
        "saved_adapter_dir": save_paths["adapter_dir"],
        "saved_merged_dir": save_paths["merged_dir"],
        "teacher_source": teacher_source,
        "final_ppl": None if ppl_eval is None else ppl_eval["result"],
        "final_ppl_path": None if ppl_eval is None else ppl_eval["path"],
    }
