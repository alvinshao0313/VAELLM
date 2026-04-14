import logging
import os
import sys
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer, TrainerCallback
from transformers.trainer import SCHEDULER_NAME, reissue_pt_warnings, save_fsdp_optimizer

from e2e_fintuning.checkpoint_io import save_e2e_model_checkpoint
from e2e_fintuning.lora import LoRAVAELinear, iter_named_vae_module_refs
from e2e_fintuning.peft_proxy import (
    PeftVAELinearProxy,
    is_peft_proxy_adapter_linear,
    update_peft_vae_proxy_adalora,
)
from litebsq.vae_linear import NamedVAELinearTarget, VAELinear, prime_named_vae_linear_cache
from train_utils.distill_losses import (
    build_distill_token_mask,
    compute_dual_kl_loss,
    compute_dual_kl_topk_loss,
    compute_dual_rkl_loss,
    compute_dual_rkl_topk_loss,
)
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.model_checkpoint_io import META_FILENAME, STATE_DICT_FILENAME
from train_utils.utils import pt_fsdp_state_dict


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HIF4_GPU_ROOT = os.path.join(_REPO_ROOT, "HiFloat4", "hif4_gpu")
_HIF4_ACT_QUANTIZER: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class _E2EHif4ActController:
    def __init__(self, quantize: Callable[[torch.Tensor], torch.Tensor]):
        self.quantize = quantize
        self.enabled = False


def _iter_parent_names(name: str):
    parts = [part for part in str(name).split(".") if part]
    for idx in range(len(parts) - 1, 0, -1):
        yield ".".join(parts[:idx])


def _load_lora_hif4_act_quantizer() -> Callable[[torch.Tensor], torch.Tensor]:
    global _HIF4_ACT_QUANTIZER
    if _HIF4_ACT_QUANTIZER is not None:
        return _HIF4_ACT_QUANTIZER
    if not os.path.isdir(_HIF4_GPU_ROOT):
        raise ImportError(
            "启用 --lora_hif4_act 失败：未找到 HiFloat4 GPU 目录。"
            f" 期望路径: {_HIF4_GPU_ROOT}"
        )
    if _HIF4_GPU_ROOT not in sys.path:
        sys.path.insert(0, _HIF4_GPU_ROOT)
    try:
        from quant_cy import QType, quant_func
    except Exception as exc:
        raise ImportError(
            "启用 --lora_hif4_act 失败：无法导入 HiFloat4 quant_cy。"
            f" 请确认已构建 {_HIF4_GPU_ROOT}/build.sh。原始错误: {exc}"
        ) from exc

    quant_type = QType("hifx4").dim(-1)

    def quantize(x: torch.Tensor) -> torch.Tensor:
        return quant_func(x, quant_type, force_py=False, force_fp32=True)

    _HIF4_ACT_QUANTIZER = quantize
    return _HIF4_ACT_QUANTIZER


def _collect_lora_hif4_act_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    module_map = dict(model.named_modules())
    targets: List[Tuple[str, nn.Module]] = []
    for name, module in module_map.items():
        if not name:
            continue
        has_wrapped_parent = any(
            isinstance(module_map[parent_name], (LoRAVAELinear, PeftVAELinearProxy, VAELinear))
            or is_peft_proxy_adapter_linear(module_map[parent_name])
            for parent_name in _iter_parent_names(name)
        )
        if has_wrapped_parent:
            continue
        if isinstance(module, (LoRAVAELinear, VAELinear)) or is_peft_proxy_adapter_linear(module):
            targets.append((name, module))
            continue
        if not isinstance(module, nn.Linear):
            continue
        targets.append((name, module))
    return targets


def _make_lora_hif4_act_pre_hook(controller: _E2EHif4ActController):
    def hook(_module, args, kwargs):
        if not controller.enabled or not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return None
        return (controller.quantize(x),) + tuple(args[1:]), kwargs

    return hook


def register_lora_hif4_act_hooks(
    model: nn.Module,
    controller: _E2EHif4ActController,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    seen: set[int] = set()
    hook = _make_lora_hif4_act_pre_hook(controller)
    for _name, module in _collect_lora_hif4_act_modules(model):
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


def remove_lora_hif4_act_hooks(handles: Sequence[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def _iter_named_temporary_modules(model: nn.Module):
    skip_prefixes = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if isinstance(module, LoRAVAELinear):
            skip_prefixes.append(f"{name}.base_layer")
        if isinstance(module, PeftVAELinearProxy):
            skip_prefixes.append(f"{name}.base_layer")
            skip_prefixes.append(f"{name}.per_decoded_linear")
        if callable(getattr(module, "set_temporary", None)):
            yield name, module


def set_model_temporary(model: nn.Module, temporary: bool) -> None:
    for _name, module in _iter_named_temporary_modules(model):
        module.set_temporary(bool(temporary))


class E2EAdaLoraCallback(TrainerCallback):
    def __init__(self, trainer: "_E2ELossMixin"):
        self.trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        update_peft_vae_proxy_adalora(
            self.trainer._unwrap_student_model(),
            global_step=int(state.global_step) + 1,
        )
        return control


def _get_output_logits(outputs) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    raise AttributeError("Model outputs do not contain `logits`.")


def compute_e2e_loss_from_logits(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    ce_loss: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    alpha: float = 0.5,
    post_attn: bool = False,
) -> torch.Tensor:
    norm = str(loss_type or "").strip().lower()
    if norm in {"sft", "origin"}:
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        return ce_loss
    if teacher_logits is None:
        raise ValueError(f"loss_type={norm} requires teacher_logits.")

    if norm == "rkl":
        return F.kl_div(
            F.log_softmax(teacher_logits.flatten(0, -2), dim=-1),
            F.softmax(student_logits.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm == "dual_rkl":
        return compute_dual_rkl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
    if norm == "kl":
        return F.kl_div(
            F.log_softmax(student_logits.flatten(0, -2), dim=-1),
            F.softmax(teacher_logits.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm == "dual_kl":
        return compute_dual_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
    if norm.startswith("r_kl_top"):
        k = _parse_topk(norm, prefix="r_kl_top", default_k=1000)
        k = min(int(k), int(student_logits.shape[-1]))
        top_student, indices = student_logits.topk(k, dim=-1, sorted=False)
        top_teacher = teacher_logits.gather(-1, indices)
        return F.kl_div(
            F.log_softmax(top_teacher.flatten(0, -2), dim=-1),
            F.softmax(top_student.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm.startswith("dual_r_kl_top"):
        k = _parse_topk(norm, prefix="dual_r_kl_top", default_k=1000)
        return compute_dual_rkl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            post_attn=bool(post_attn),
        )
    if norm.startswith("kl_top"):
        k = _parse_topk(norm, prefix="kl_top", default_k=1000)
        k = min(int(k), int(teacher_logits.shape[-1]))
        top_teacher, indices = teacher_logits.topk(k, dim=-1, sorted=False)
        if bool(post_attn):
            ref = F.softmax(teacher_logits, dim=-1).gather(-1, indices).flatten(0, -2)
            can = F.log_softmax(student_logits, dim=-1).gather(-1, indices).flatten(0, -2)
            return F.kl_div(can, ref, reduction="batchmean")
        top_student = student_logits.gather(-1, indices)
        return F.kl_div(
            F.log_softmax(top_student.flatten(0, -2), dim=-1),
            F.softmax(top_teacher.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm.startswith("kd_top"):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        k = _parse_topk(norm, prefix="kd_top", default_k=1000)
        k = min(int(k), int(teacher_logits.shape[-1]))
        top_teacher, indices = teacher_logits.topk(k, dim=-1, sorted=False)
        temperature = float(temperature)
        if bool(post_attn):
            ref = F.softmax(teacher_logits / temperature, dim=-1).gather(-1, indices).flatten(0, -2)
            can = F.log_softmax(student_logits / temperature, dim=-1).gather(-1, indices).flatten(0, -2)
            kd_loss = F.kl_div(can, ref, reduction="batchmean")
        else:
            top_student = student_logits.gather(-1, indices)
            kd_loss = F.kl_div(
                F.log_softmax((top_student / temperature).flatten(0, -2), dim=-1),
                F.softmax((top_teacher / temperature).flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * (float(alpha) * temperature * temperature)
    if norm.startswith("dual_kl_top"):
        k = _parse_topk(norm, prefix="dual_kl_top", default_k=1000)
        return compute_dual_kl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            post_attn=bool(post_attn),
        )
    if norm == "mse":
        return F.mse_loss(student_logits, teacher_logits)
    if norm == "kd":
        if ce_loss is None:
            raise ValueError("loss_type=kd requires ce_loss.")
        temperature = float(temperature)
        kd_loss = F.kl_div(
            F.log_softmax((student_logits / temperature).flatten(0, -2), dim=-1),
            F.softmax((teacher_logits / temperature).flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * (float(alpha) * temperature * temperature)
    if norm == "dual_kd":
        if ce_loss is None:
            raise ValueError("loss_type=dual_kd requires ce_loss.")
        kd_loss = compute_dual_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * float(alpha)
    if norm.startswith("dual_kd_top"):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        k = _parse_topk(norm, prefix="dual_kd_top", default_k=1000)
        kd_loss = compute_dual_kl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            post_attn=bool(post_attn),
        )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * float(alpha)

    raise ValueError(
        f"Unsupported e2e loss type: {loss_type}. "
        "Supported: sft/origin, kl, rkl, dual_rkl, mse, kd, kd_top[_K], dual_kd_top[_K], "
        "dual_kl, dual_kd, kl_top[_K], r_kl_top[_K], dual_r_kl_top[_K], dual_kl_top[_K]."
    )


def _parse_topk(value: str, *, prefix: str, default_k: int) -> int:
    if value == prefix:
        return int(default_k)
    suffix = value[len(prefix):]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    if not suffix:
        return int(default_k)
    return max(1, int(suffix))


class _E2ELossMixin:
    def __init__(
        self,
        *args,
        loss_type: str = "sft",
        teacher_model: Optional[nn.Module] = None,
        distill_temperature: float = 1.0,
        distill_alpha: float = 0.5,
        post_attn: bool = False,
        lora_hif4_act: bool = False,
        prewarm_frozen_vae: bool = True,
        prewarm_log_every: int = 32,
        prewarm_group_size: int = 8,
        prewarm_module_names: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        self.loss_type = str(loss_type).strip().lower()
        self.teacher_model = teacher_model
        self.distill_temperature = float(distill_temperature)
        self.distill_alpha = float(distill_alpha)
        self.post_attn = bool(post_attn)
        self.lora_hif4_act = bool(lora_hif4_act)
        self.lora_hif4_act_controller = (
            _E2EHif4ActController(_load_lora_hif4_act_quantizer())
            if self.lora_hif4_act
            else None
        )
        self.prewarm_frozen_vae = bool(prewarm_frozen_vae)
        self.prewarm_log_every = max(1, int(prewarm_log_every))
        self.prewarm_group_size = max(1, int(prewarm_group_size))
        self.prewarm_module_names = (
            None
            if prewarm_module_names is None
            else {str(name) for name in prewarm_module_names if str(name)}
        )
        self._teacher_device = None
        self._vae_cache_prepared = False
        self._logger = logging.getLogger("e2e_fintuning")
        super().__init__(*args, **kwargs)

    def _unwrap_student_model(self):
        model = self.model
        if getattr(self, "accelerator", None) is not None:
            model = self.accelerator.unwrap_model(model)
        return model

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        base_model_path = getattr(self, "_e2e_base_model_path", None)
        extra_meta = getattr(self, "_e2e_checkpoint_extra_meta", None)
        model = self._unwrap_student_model()

        if getattr(self, "is_fsdp_enabled", False):
            state_dict = pt_fsdp_state_dict(self.model)
            if self.args.should_save:
                save_e2e_model_checkpoint(
                    model,
                    output_dir,
                    base_model_path=base_model_path,
                    save_config=False,
                    extra_meta=extra_meta,
                    state_dict=state_dict,
                    compact_unload_vae_original_weights=True,
                )
            return

        if self.args.should_save:
            save_e2e_model_checkpoint(
                model,
                output_dir,
                base_model_path=base_model_path,
                save_config=False,
                extra_meta=extra_meta,
                state_dict=model.state_dict(),
                compact_unload_vae_original_weights=True,
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if (
            os.path.isdir(resume_from_checkpoint)
            and os.path.exists(os.path.join(resume_from_checkpoint, META_FILENAME))
            and os.path.exists(os.path.join(resume_from_checkpoint, STATE_DICT_FILENAME))
        ):
            self._logger.info(
                "Skipping Trainer model reload for compact e2e checkpoint: %s",
                resume_from_checkpoint,
            )
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _set_hif4_act_enabled(self, enabled: bool) -> None:
        if self.lora_hif4_act_controller is not None:
            self.lora_hif4_act_controller.enabled = bool(enabled)

    def _ensure_teacher_device(self, device: torch.device) -> None:
        if self.teacher_model is None:
            return
        if self._teacher_device == device:
            return
        self.teacher_model.to(device)
        self.teacher_model.eval()
        self._teacher_device = device

    def _infer_cache_dtype(self, model: nn.Module) -> torch.dtype:
        for param in model.parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.float32

    def prepare_frozen_vae_cache_once(self, model: nn.Module) -> None:
        if self._vae_cache_prepared or not self.prewarm_frozen_vae:
            return

        target_dtype = self._infer_cache_dtype(model)
        named_targets: List[NamedVAELinearTarget] = []
        for ref in iter_named_vae_module_refs(model):
            if isinstance(ref.module, PeftVAELinearProxy):
                continue
            if self.prewarm_module_names is not None and ref.name not in self.prewarm_module_names:
                continue
            named_targets.append(NamedVAELinearTarget(name=ref.name, base_layer=ref.base_layer))

        self._logger.info(
            "Start VAELinear prewarm: total=%d dtype=%s prewarm_group_size=%d",
            len(named_targets),
            str(target_dtype),
            int(self.prewarm_group_size),
        )
        stats = prime_named_vae_linear_cache(
            named_targets,
            dtype=target_dtype,
            clear_existing=True,
            group_size=int(self.prewarm_group_size),
            logger=self._logger,
        )

        self._vae_cache_prepared = True
        self._logger.info(
            "VAELinear prewarm complete: total=%d warmed=%d skipped=%d failed=%d dtype=%s prewarm_group_size=%d",
            int(stats.get("total", 0)),
            int(stats.get("warmed", 0)),
            int(stats.get("skipped", 0)),
            int(stats.get("failed", 0)),
            str(target_dtype),
            int(self.prewarm_group_size),
        )
        self._logger.info("Finished VAELinear prewarm.")

    def _compute_teacher_outputs(
        self,
        teacher_inputs: Dict[str, torch.Tensor],
    ):
        self._set_hif4_act_enabled(False)
        if self.teacher_model is None:
            raise RuntimeError("当前 e2e 蒸馏已强制使用外部 teacher，但 trainer.teacher_model 为空。")
        input_tensor = next(value for value in teacher_inputs.values() if torch.is_tensor(value))
        self._ensure_teacher_device(device=input_tensor.device)
        with torch.no_grad():
            return self.teacher_model(**teacher_inputs, output_hidden_states=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        unwrapped_model = model
        if getattr(self, "accelerator", None) is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
        self.prepare_frozen_vae_cache_once(unwrapped_model)
        previous_hif4_enabled = bool(getattr(self.lora_hif4_act_controller, "enabled", False))

        loss_type = self.loss_type
        try:
            if loss_type in {"sft", "origin"}:
                try:
                    return super().compute_loss(
                        model,
                        inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                        **kwargs,
                    )
                except TypeError:
                    return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

            teacher_inputs = dict(inputs)
            teacher_inputs.pop("labels", None)
            student_inputs = dict(inputs)
            if not (
                loss_type in {"kd", "dual_kd"}
                or loss_type.startswith("kd_top")
                or loss_type.startswith("dual_kd_top")
            ):
                student_inputs.pop("labels", None)
            full_inputs = dict(inputs)

            teacher_outputs = self._compute_teacher_outputs(
                teacher_inputs,
            )
            self._set_hif4_act_enabled(previous_hif4_enabled)
            if (
                loss_type in {"kd", "dual_kd"}
                or loss_type.startswith("kd_top")
                or loss_type.startswith("dual_kd_top")
            ):
                outputs = model(**full_inputs)
                ce_loss = outputs["loss"]
            else:
                outputs = model(**student_inputs)
                ce_loss = None
            logits = _get_output_logits(outputs)
            token_mask = build_distill_token_mask(
                labels=inputs.get("labels"),
                attention_mask=inputs.get("attention_mask"),
                reference_logits=logits,
            )

            loss = compute_e2e_loss_from_logits(
                loss_type=loss_type,
                student_logits=logits,
                teacher_logits=_get_output_logits(teacher_outputs),
                ce_loss=ce_loss,
                mask=token_mask,
                temperature=self.distill_temperature,
                alpha=self.distill_alpha,
                post_attn=self.post_attn,
            )
            return (loss, outputs) if return_outputs else loss
        finally:
            self._set_hif4_act_enabled(previous_hif4_enabled)


class E2EFinetuneTrainer(_E2ELossMixin, Trainer):
    pass


class E2EFSDPFinetuneTrainer(_E2ELossMixin, FSDPTrainer):
    def _save_optimizer_and_scheduler(self, output_dir):
        if self.args.should_save:
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.optimizer,
                self.model,
                output_dir,
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
