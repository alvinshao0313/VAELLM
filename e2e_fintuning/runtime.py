import argparse
import gc
import json
import os
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import default_data_collator

from e2e_fintuning.args import needs_teacher
from e2e_fintuning.checkpoint_io import load_e2e_model_checkpoint, save_e2e_model_checkpoint
from e2e_fintuning.data import build_datasets, build_tokenizer
from e2e_fintuning.peft_proxy import (
    ensure_peft_vae_proxy_adapter,
    initialize_peft_vae_proxy_lora_from_teacher_residual,
    sync_peft_vae_proxy_lora_weights,
)
from e2e_fintuning.trainables import resolve_target_layer_ids, select_e2e_trainables_peft_proxy
from e2e_fintuning.trainer import (
    E2EAdaLoraCallback,
    E2EFinetuneTrainer,
    E2EFSDPFinetuneTrainer,
    set_model_temporary,
)
from litebsq.vae_linear import clear_model_vae_linear_cache
from rotation.model_utils import get_layers, get_model
from train_utils.eval_utils import calculate_ppl
from train_utils.hif4_act import (
    applied_hif4_act,
    register_hif4_act_hooks,
    remove_hif4_act_hooks,
)
from train_utils.model_checkpoint_io import (
    _build_run_output_dir,
    resolve_checkpoint_dir,
    unload_vae_original_linear_weights,
)
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


def _variant_flags(variant: str) -> Tuple[bool, bool]:
    norm = str(variant).strip().lower()
    return norm == "rslora", norm == "dora"


def _infer_vae_lora_variant_from_meta(meta: Dict[str, object]) -> Optional[str]:
    extra_meta = meta.get("extra_meta", {}) if isinstance(meta.get("extra_meta"), dict) else {}
    saved_variant = extra_meta.get("vae_lora_variant")
    if saved_variant:
        return str(saved_variant).strip().lower()

    adapter_modules = meta.get("adapter_modules", [])
    if not isinstance(adapter_modules, list):
        return None
    for spec in adapter_modules:
        adapter_type = str(spec.get("adapter_type"))
        if adapter_type == "peft_proxy_adalora":
            return "adalora"
        if adapter_type == "peft_proxy_lora":
            if bool(spec.get("use_dora", False)):
                return "dora"
            if bool(spec.get("use_rslora", False)):
                return "rslora"
            return "plain"
        if adapter_type == "vae_lora":
            return "plain"
    return None


def _first_adapter_spec(meta: Dict[str, object], adapter_type: str) -> Optional[Dict[str, object]]:
    adapter_modules = meta.get("adapter_modules", [])
    if not isinstance(adapter_modules, list):
        return None
    for spec in adapter_modules:
        if str(spec.get("adapter_type")) == str(adapter_type):
            return spec
    return None


def _validate_resume_checkpoint_config(*, args, meta, decoder_layer_ids, training_args) -> None:
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

    saved_variant = _infer_vae_lora_variant_from_meta(meta)
    if saved_variant is not None and saved_variant != str(args.vae_lora_variant):
        raise ValueError(
            f"resume checkpoint 的 vae_lora_variant={saved_variant} 与当前参数 {args.vae_lora_variant} 不一致。"
        )

    if "vae_lora_init_mode" in extra_meta and str(extra_meta["vae_lora_init_mode"]) != str(args.vae_lora_init_mode):
        raise ValueError(
            f"resume checkpoint 的 vae_lora_init_mode={extra_meta['vae_lora_init_mode']} "
            f"与当前参数 {args.vae_lora_init_mode} 不一致。"
        )

    expected_rslora, expected_dora = _variant_flags(args.vae_lora_variant)
    if "vae_lora_use_rslora" in extra_meta and bool(extra_meta["vae_lora_use_rslora"]) != bool(expected_rslora):
        raise ValueError("resume checkpoint 的 vae_lora_use_rslora 与当前参数不一致。")
    if "vae_lora_use_dora" in extra_meta and bool(extra_meta["vae_lora_use_dora"]) != bool(expected_dora):
        raise ValueError("resume checkpoint 的 vae_lora_use_dora 与当前参数不一致。")

    if str(args.vae_lora_variant) != "adalora":
        return

    spec = _first_adapter_spec(meta, "peft_proxy_adalora")
    saved_target_r = extra_meta.get("vae_adalora_target_r", None if spec is None else spec.get("target_r"))
    saved_init_r = extra_meta.get("vae_adalora_init_r", None if spec is None else spec.get("init_r", spec.get("r")))
    saved_tinit = extra_meta.get("vae_adalora_tinit", None if spec is None else spec.get("tinit"))
    saved_tfinal = extra_meta.get("vae_adalora_tfinal", None if spec is None else spec.get("tfinal"))
    saved_delta_t = extra_meta.get("vae_adalora_delta_t", None if spec is None else spec.get("delta_t"))
    saved_beta1 = extra_meta.get("vae_adalora_beta1", None if spec is None else spec.get("beta1"))
    saved_beta2 = extra_meta.get("vae_adalora_beta2", None if spec is None else spec.get("beta2"))
    saved_orth = extra_meta.get("vae_adalora_orth_reg_weight", None if spec is None else spec.get("orth_reg_weight"))
    saved_total_step = extra_meta.get("vae_adalora_total_step", None if spec is None else spec.get("total_step"))

    if saved_target_r is not None and int(saved_target_r) != int(args.vae_adalora_target_r):
        raise ValueError("resume checkpoint 的 vae_adalora_target_r 与当前参数不一致。")
    if saved_init_r is not None and int(saved_init_r) != int(args.vae_adalora_init_r):
        raise ValueError("resume checkpoint 的 vae_adalora_init_r 与当前参数不一致。")
    if saved_tinit is not None and int(saved_tinit) != int(args.vae_adalora_tinit):
        raise ValueError("resume checkpoint 的 vae_adalora_tinit 与当前参数不一致。")
    if saved_tfinal is not None and int(saved_tfinal) != int(args.vae_adalora_tfinal):
        raise ValueError("resume checkpoint 的 vae_adalora_tfinal 与当前参数不一致。")
    if saved_delta_t is not None and int(saved_delta_t) != int(args.vae_adalora_delta_t):
        raise ValueError("resume checkpoint 的 vae_adalora_delta_t 与当前参数不一致。")
    if saved_beta1 is not None and float(saved_beta1) != float(args.vae_adalora_beta1):
        raise ValueError("resume checkpoint 的 vae_adalora_beta1 与当前参数不一致。")
    if saved_beta2 is not None and float(saved_beta2) != float(args.vae_adalora_beta2):
        raise ValueError("resume checkpoint 的 vae_adalora_beta2 与当前参数不一致。")
    if saved_orth is not None and float(saved_orth) != float(args.vae_adalora_orth_reg_weight):
        raise ValueError("resume checkpoint 的 vae_adalora_orth_reg_weight 与当前参数不一致。")
    if saved_total_step is not None and int(saved_total_step) != int(training_args.max_steps):
        raise ValueError("resume checkpoint 的 vae_adalora_total_step 与当前 TrainingArguments.max_steps 不一致。")


def _checkpoint_has_peft_proxy_adapter(meta) -> bool:
    adapter_modules = meta.get("adapter_modules", [])
    if not isinstance(adapter_modules, list):
        return False
    for spec in adapter_modules:
        if isinstance(spec, dict) and str(spec.get("adapter_type")) in {"peft_proxy_lora", "peft_proxy_adalora"}:
            return True
    return False


def _should_initialize_vae_lora_residual_svd(*, args, selection, resume_from_checkpoint) -> bool:
    return (
        str(getattr(args, "vae_lora_variant", "plain")).strip().lower() != "adalora"
        and str(getattr(args, "vae_lora_init_mode", "zero")).strip().lower() == "residual_svd"
        and not bool(resume_from_checkpoint)
        and bool(getattr(selection, "peft_proxy_modules", []))
    )


def _resolve_residual_svd_runtime(training_args) -> Tuple[torch.device, int, int]:
    training_args._setup_devices
    device = torch.device(training_args.device)
    if device.type != "cuda":
        raise ValueError("residual_svd 初始化已改为 batched GPU SVD，当前运行必须提供 CUDA device。")

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env <= 1:
        return device, 0, 1

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "检测到 WORLD_SIZE > 1，但 torch.distributed 在 residual_svd 初始化前没有完成初始化。"
        )
    return device, int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())


def _resolve_saved_vae_lora_init_mode(*, args, meta, resume_from_checkpoint) -> Optional[str]:
    if resume_from_checkpoint:
        extra_meta = meta.get("extra_meta", {}) if isinstance(meta.get("extra_meta"), dict) else {}
        saved_mode = extra_meta.get("vae_lora_init_mode")
        if saved_mode is None:
            return None
        return str(saved_mode).strip().lower()
    return str(getattr(args, "vae_lora_init_mode", "zero")).strip().lower()


def _resolve_saved_vae_lora_variant(*, args, meta, resume_from_checkpoint) -> str:
    if resume_from_checkpoint:
        saved_variant = _infer_vae_lora_variant_from_meta(meta)
        if saved_variant:
            return str(saved_variant)
    return str(getattr(args, "vae_lora_variant", "plain")).strip().lower()


def _build_checkpoint_extra_meta(
    *,
    args,
    selection,
    student_checkpoint_dir: str,
    teacher_source: str,
    saved_variant: str,
    saved_init_mode: Optional[str],
    training_args,
) -> Dict[str, object]:
    use_rslora, use_dora = _variant_flags(saved_variant)
    extra_meta: Dict[str, object] = {
        "stage": "e2e_fintuning",
        "source_checkpoint_dir": student_checkpoint_dir,
        "teacher_source": teacher_source,
        "target_decoder_layers": list(selection.decoder_layer_ids),
        "target_module_names": None if args.target_module_names is None else list(args.target_module_names),
        "loss_type": str(args.loss_type),
        "post_attn": bool(args.post_attn),
        "lora_hif4_act": bool(args.lora_hif4_act),
        "finetune_mode": _E2E_FINETUNE_MODE,
        "prewarm_frozen_vae": bool(args.prewarm_frozen_vae),
        "vae_lora_variant": str(saved_variant),
        "vae_lora_rank": int(args.vae_lora_rank),
        "vae_lora_alpha": float(args.vae_lora_alpha),
        "vae_lora_dropout": float(args.vae_lora_dropout),
        "vae_lora_use_rslora": bool(use_rslora),
        "vae_lora_use_dora": bool(use_dora),
    }
    if saved_init_mode is not None:
        extra_meta["vae_lora_init_mode"] = str(saved_init_mode)
    if str(saved_variant) == "adalora":
        extra_meta.update(
            {
                "vae_adalora_target_r": int(args.vae_adalora_target_r),
                "vae_adalora_init_r": int(args.vae_adalora_init_r),
                "vae_adalora_tinit": int(args.vae_adalora_tinit),
                "vae_adalora_tfinal": int(args.vae_adalora_tfinal),
                "vae_adalora_delta_t": int(args.vae_adalora_delta_t),
                "vae_adalora_beta1": float(args.vae_adalora_beta1),
                "vae_adalora_beta2": float(args.vae_adalora_beta2),
                "vae_adalora_orth_reg_weight": float(args.vae_adalora_orth_reg_weight),
                "vae_adalora_total_step": int(training_args.max_steps),
            }
        )
    return extra_meta


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


def _save_final_e2e_model(
    *,
    final_model: nn.Module,
    final_dir: str,
    base_model_path: str,
    tokenizer,
    extra_meta: Dict[str, object],
    save_tokenizer: bool,
    should_save: bool,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
):
    if not bool(should_save):
        return None

    save_kwargs = {
        "base_model_path": str(base_model_path),
        "tokenizer": tokenizer if bool(save_tokenizer) else None,
        "save_config": True,
        "extra_meta": extra_meta,
        "compact_unload_vae_original_weights": True,
    }
    if state_dict is not None:
        save_kwargs["state_dict"] = state_dict

    return save_e2e_model_checkpoint(
        final_model,
        final_dir,
        **save_kwargs,
    )


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
        materialize_proxy_decoded_linears=True,
        proxy_group_size=int(args.prewarm_group_size),
        proxy_compute_device=str(training_args.device),
        proxy_logger=log,
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

    layers = list(get_layers(model))
    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(layers))
    if resume_from_checkpoint:
        _validate_resume_checkpoint_config(
            args=args,
            meta=meta,
            decoder_layer_ids=decoder_layer_ids,
            training_args=training_args,
        )
    if (
        not resume_from_checkpoint
        and str(args.vae_lora_init_mode) == "residual_svd"
        and _checkpoint_has_peft_proxy_adapter(meta)
    ):
        raise ValueError("Fresh e2e 训练遇到已包含 peft_proxy adapter 的 checkpoint，拒绝再次执行 residual_svd 初始化。")

    selection = select_e2e_trainables_peft_proxy(
        model,
        decoder_layer_ids=decoder_layer_ids,
        target_module_names=args.target_module_names,
    )

    injected_proxy_count = 0
    if selection.peft_proxy_modules:
        injected_proxy_count = ensure_peft_vae_proxy_adapter(
            model,
            variant=str(args.vae_lora_variant),
            rank=int(args.vae_lora_rank),
            alpha=float(args.vae_lora_alpha),
            dropout=float(args.vae_lora_dropout),
            init_mode=str(args.vae_lora_init_mode),
            total_step=int(training_args.max_steps) if str(args.vae_lora_variant) == "adalora" else None,
            adalora_target_r=int(args.vae_adalora_target_r),
            adalora_init_r=int(args.vae_adalora_init_r),
            adalora_tinit=int(args.vae_adalora_tinit),
            adalora_tfinal=int(args.vae_adalora_tfinal),
            adalora_delta_t=int(args.vae_adalora_delta_t),
            adalora_beta1=float(args.vae_adalora_beta1),
            adalora_beta2=float(args.vae_adalora_beta2),
            adalora_orth_reg_weight=float(args.vae_adalora_orth_reg_weight),
            materialize_before_inject=True,
            materialize_group_size=int(args.prewarm_group_size),
            materialize_compute_device=str(training_args.device),
            materialize_logger=log,
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
        residual_svd_device, residual_svd_rank, residual_svd_world_size = _resolve_residual_svd_runtime(training_args)
        if residual_svd_rank == 0:
            initialized_proxy_count = initialize_peft_vae_proxy_lora_from_teacher_residual(
                model,
                teacher_model,
                batch_device=residual_svd_device,
            )
            log.info(
                "Initialized %d PEFT VAELinear proxy LoRA modules with residual_svd on rank0 device=%s.",
                initialized_proxy_count,
                str(residual_svd_device),
            )
        else:
            initialized_proxy_count = int(len(selection.peft_proxy_modules))
        if residual_svd_world_size > 1:
            synced_proxy_count = sync_peft_vae_proxy_lora_weights(
                model,
                sync_device=residual_svd_device,
                src_rank=0,
            )
            torch.distributed.barrier()
            if synced_proxy_count != int(len(selection.peft_proxy_modules)):
                raise RuntimeError(
                    f"PEFT proxy LoRA sync count mismatch: synced={synced_proxy_count} "
                    f"expected={len(selection.peft_proxy_modules)}"
                )
            if residual_svd_rank == 0:
                log.info(
                    "Synchronized %d PEFT VAELinear proxy LoRA modules from rank0 to %d ranks.",
                    synced_proxy_count,
                    residual_svd_world_size,
                )
        if not keep_teacher_for_training:
            teacher_model = _release_init_only_teacher(teacher_model, log)

    trainable_params = _collect_trainable_params(model)
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for requested decoder layers.")
    setattr(model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)
    log.info(
        "Selected trainables: mode=%s variant=%s layers=%s modules=%d adapters=%d peft_proxy=%d trainable_tensors=%d total_params=%d cacheable=%d",
        _E2E_FINETUNE_MODE,
        str(args.vae_lora_variant),
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
        prewarm_group_size=int(args.prewarm_group_size),
        prewarm_module_names=selection.frozen_cacheable_vae_modules,
    )
    if str(args.vae_lora_variant) == "adalora":
        trainer.add_callback(E2EAdaLoraCallback(trainer))

    saved_variant = _resolve_saved_vae_lora_variant(
        args=args,
        meta=meta,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    saved_init_mode = _resolve_saved_vae_lora_init_mode(
        args=args,
        meta=meta,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    checkpoint_extra_meta = _build_checkpoint_extra_meta(
        args=args,
        selection=selection,
        student_checkpoint_dir=student_checkpoint_dir,
        teacher_source=teacher_source,
        saved_variant=saved_variant,
        saved_init_mode=saved_init_mode,
        training_args=training_args,
    )
    trainer._e2e_base_model_path = str(base_model_path)
    trainer._e2e_checkpoint_extra_meta = checkpoint_extra_meta

    hif4_act_handles = []
    if trainer.lora_hif4_act_controller is not None:
        hif4_act_handles = register_hif4_act_hooks(trainer.model, trainer.lora_hif4_act_controller)
        if not hif4_act_handles:
            raise RuntimeError("启用 HiFloat4 激活量化失败：未找到可注册 hook 的逻辑线性层。")
        trainer.lora_hif4_act_controller.enabled = True
        log.info("Registered %d HiFloat4 activation hooks for e2e LoRA training.", len(hif4_act_handles))
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    finally:
        if trainer.lora_hif4_act_controller is not None:
            trainer.lora_hif4_act_controller.enabled = False
        remove_hif4_act_hooks(hif4_act_handles)

    final_model = _unwrap_model(trainer, trainer.model)
    setattr(final_model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)
    _ensure_student_mode(final_model)
    clear_model_vae_linear_cache(final_model)
    if bool(args.unload_vae_original_weights_on_save):
        unload_vae_original_linear_weights(final_model)
    if teacher_model is not None:
        teacher_model.to("cpu")

    final_dir = os.path.join(run_output_dir, "final_model")
    final_state_dict = pt_fsdp_state_dict(trainer.model) if _uses_fsdp(training_args) else None
    save_paths = _save_final_e2e_model(
        final_model=final_model,
        final_dir=final_dir,
        base_model_path=str(base_model_path),
        tokenizer=tokenizer,
        extra_meta=dict(checkpoint_extra_meta),
        save_tokenizer=bool(args.save_tokenizer),
        should_save=bool(getattr(training_args, "should_save", True)),
        state_dict=final_state_dict,
    )
    if save_paths is not None:
        log.info("Saved final model to %s", save_paths["output_dir"])
    else:
        log.info("Skipping final model save on this rank because should_save=false.")
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
        "saved_model_dir": None if save_paths is None else save_paths["output_dir"],
        "teacher_source": teacher_source,
        "final_ppl": None if ppl_eval is None else ppl_eval["result"],
        "final_ppl_path": None if ppl_eval is None else ppl_eval["path"],
    }
