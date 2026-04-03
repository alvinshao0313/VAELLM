import os
import sys
from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from litebsq.vae_linear import VAELinear
from train_utils.distill_losses import (
    build_distill_token_mask,
    compute_dual_kl_loss,
    compute_dual_kl_topk_loss,
    compute_dual_rkl_loss,
    compute_dual_rkl_topk_loss,
)

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None

try:
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
except ImportError:
    DataCollatorForCompletionOnlyLM = None
    SFTTrainer = None


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HIF4_GPU_ROOT = os.path.join(_REPO_ROOT, "HiFloat4", "hif4_gpu")
_PEFT_LORA_LINEAR_TYPE = None
_HIF4_ACT_QUANTIZER: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class LoraHif4ActController:
    def __init__(self, quantize: Callable[[torch.Tensor], torch.Tensor]):
        self.quantize = quantize
        self.enabled = False


def ensure_lora_training_stack_available() -> None:
    if LoraConfig is None or TaskType is None or get_peft_model is None:
        raise ImportError("未安装 peft。请先安装：pip install peft")
    if SFTTrainer is None or DataCollatorForCompletionOnlyLM is None:
        raise ImportError("未安装 trl。请先安装：pip install trl")


def create_lora_adapters(
    model: nn.Module,
    *,
    target_names: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    use_dora: bool,
):
    unique_target_names = sorted(set(str(name) for name in target_names if str(name).strip()))
    if not unique_target_names:
        return model, None, unique_target_names

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        target_modules=unique_target_names,
        inference_mode=False,
        bias="none",
        use_dora=bool(use_dora),
    )
    return get_peft_model(model, lora_config), lora_config, unique_target_names


def merge_all_lora(model: nn.Module) -> Tuple[nn.Module, int]:
    if PeftModel is None or not isinstance(model, PeftModel):
        return model, 0
    trainable_count = 0
    for name, _ in model.named_parameters():
        if "lora_" in name:
            trainable_count += 1
    merged_model = model.merge_and_unload()
    return merged_model, trainable_count


def build_lora_hif4_act_controller(enabled: bool) -> Optional[LoraHif4ActController]:
    if not enabled:
        return None
    return LoraHif4ActController(load_lora_hif4_act_quantizer())


def _get_peft_lora_linear_type():
    global _PEFT_LORA_LINEAR_TYPE
    if _PEFT_LORA_LINEAR_TYPE is not None:
        return _PEFT_LORA_LINEAR_TYPE
    from peft.tuners.lora.layer import Linear as PeftLoraLinear

    _PEFT_LORA_LINEAR_TYPE = PeftLoraLinear
    return _PEFT_LORA_LINEAR_TYPE


def _is_peft_lora_linear(module: nn.Module) -> bool:
    peft_linear_type = _get_peft_lora_linear_type()
    return isinstance(module, peft_linear_type)


def _iter_parent_names(name: str):
    parts = [part for part in str(name).split(".") if part]
    for idx in range(len(parts) - 1, 0, -1):
        yield ".".join(parts[:idx])


def load_lora_hif4_act_quantizer() -> Callable[[torch.Tensor], torch.Tensor]:
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
        if isinstance(module, VAELinear) or _is_peft_lora_linear(module):
            targets.append((name, module))
            continue
        if not isinstance(module, nn.Linear):
            continue
        if any(
            isinstance(module_map[parent_name], VAELinear) or _is_peft_lora_linear(module_map[parent_name])
            for parent_name in _iter_parent_names(name)
        ):
            continue
        targets.append((name, module))
    return targets


def _make_lora_hif4_act_pre_hook(controller: LoraHif4ActController):
    def hook(_module, args, kwargs):
        if not controller.enabled or not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return None
        new_args = (controller.quantize(x),) + tuple(args[1:])
        return new_args, kwargs

    return hook


def register_lora_hif4_act_hooks(
    model: nn.Module,
    controller: LoraHif4ActController,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    seen: Set[int] = set()
    hook = _make_lora_hif4_act_pre_hook(controller)
    for _name, module in _collect_lora_hif4_act_modules(model):
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


def remove_hook_handles(handles: Sequence[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


if SFTTrainer is None:
    class CustomSFTTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("未安装 trl。请先安装：pip install trl")
else:
    class CustomSFTTrainer(SFTTrainer):
        def __init__(
            self,
            *args,
            loss_type: str = "r_kl_top_1000",
            temperature: float = 1.0,
            loss_alpha: float = 0.5,
            lora_hif4_act_controller: Optional[LoraHif4ActController] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.loss_type = str(loss_type).strip().lower()
            self.temperature = float(temperature)
            self.loss_alpha = float(loss_alpha)
            self.lora_hif4_act_controller = lora_hif4_act_controller

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            args = self.args
            loss_type = self.loss_type
            teacher_inputs = dict(inputs)
            teacher_inputs.pop("labels", None)
            student_inputs = dict(inputs)
            uses_ce_loss = (
                loss_type == "kd"
                or loss_type == "dual_kd"
                or loss_type.startswith("kd_top")
                or loss_type.startswith("dual_kd_top")
            )
            if not uses_ce_loss:
                student_inputs.pop("labels", None)
            full_inputs = dict(inputs)

            unwrapped_model = model
            if getattr(self, "accelerator", None) is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
            temporary_modules = [
                module
                for module in unwrapped_model.modules()
                if callable(getattr(module, "set_temporary", None))
            ]
            previous_temporary = [getattr(module, "temporary", None) for module in temporary_modules]
            hif4_act_controller = self.lora_hif4_act_controller
            previous_hif4_enabled = bool(getattr(hif4_act_controller, "enabled", False))
            peft_model_for_teacher = unwrapped_model if isinstance(unwrapped_model, PeftModel) else model

            def set_temporary(temporary: bool) -> None:
                for module in temporary_modules:
                    module.set_temporary(temporary)

            def restore_temporary() -> None:
                for module, previous in zip(temporary_modules, previous_temporary):
                    module.set_temporary(True if previous is None else bool(previous))

            def set_hif4_act_enabled(enabled: bool) -> None:
                if hif4_act_controller is not None:
                    hif4_act_controller.enabled = bool(enabled)

            def prepare_student_path() -> None:
                set_temporary(True)
                set_hif4_act_enabled(previous_hif4_enabled)

            def parse_k(prefix: str, default_k: int = 1000) -> int:
                if loss_type == prefix:
                    return default_k
                suffix = loss_type[len(prefix):]
                if suffix.startswith("_"):
                    suffix = suffix[1:]
                if not suffix:
                    return default_k
                return max(1, int(suffix))

            def use_post_attn() -> bool:
                return bool(getattr(args, "lora_post_attn", False))

            @torch.no_grad()
            def get_ori_outputs():
                set_temporary(False)
                set_hif4_act_enabled(False)
                adapter_context = (
                    peft_model_for_teacher.disable_adapter()
                    if hasattr(peft_model_for_teacher, "disable_adapter")
                    else nullcontext()
                )
                with adapter_context:
                    outputs = model(**teacher_inputs, output_hidden_states=False)
                return outputs

            try:
                if loss_type in {"origin", "sft"}:
                    try:
                        return super().compute_loss(
                            model,
                            full_inputs,
                            return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch,
                        )
                    except TypeError:
                        return super().compute_loss(
                            model,
                            full_inputs,
                            return_outputs=return_outputs,
                        )

                prepare_student_path()

                if loss_type == "rkl":
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**student_inputs)
                    logits = outputs.logits
                    loss = F.kl_div(
                        F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
                        F.softmax(logits, dim=-1).flatten(0, -2),
                        reduction="batchmean",
                    )
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_rkl":
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**student_inputs)
                    logits = outputs.logits
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_rkl_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                    )
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "kl":
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**student_inputs)
                    logits = outputs.logits
                    loss = F.kl_div(
                        F.log_softmax(logits.flatten(0, -2), dim=-1),
                        F.softmax(ori_logits, dim=-1).flatten(0, -2),
                        reduction="batchmean",
                    )
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("r_kl_top"):
                    k = parse_k("r_kl_top", default_k=1000)
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**student_inputs)
                    logits = outputs.logits
                    k = min(k, int(logits.shape[-1]))
                    top_logits, indices = logits.topk(k, dim=-1, sorted=False)
                    top_ori_logits = ori_logits.gather(-1, indices)
                    loss = F.kl_div(
                        F.log_softmax(top_ori_logits.flatten(0, -2), dim=-1),
                        F.softmax(top_logits.flatten(0, -2), dim=-1),
                        reduction="batchmean",
                    )
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_r_kl_top"):
                    k = parse_k("dual_r_kl_top", default_k=1000)
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**student_inputs)
                    logits = outputs.logits
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_rkl_topk_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                        k=k,
                        post_attn=use_post_attn(),
                    )
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("kl_top"):
                    k = parse_k("kl_top", default_k=1000)
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**student_inputs)
                    logits = outputs.logits
                    k = min(k, int(ori_logits.shape[-1]))
                    top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
                    if use_post_attn():
                        ref = F.softmax(ori_logits, dim=-1).gather(-1, indices).flatten(0, -2)
                        can = F.log_softmax(logits, dim=-1).gather(-1, indices).flatten(0, -2)
                        loss = F.kl_div(can, ref, reduction="batchmean")
                    else:
                        top_logits = logits.gather(-1, indices)
                        loss = F.kl_div(
                            F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                            F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                            reduction="batchmean",
                        )
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("kd_top"):
                    k = parse_k("kd_top", default_k=1000)
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**full_inputs)
                    logits = outputs.logits
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    k = min(k, int(ori_logits.shape[-1]))
                    top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
                    if use_post_attn():
                        ref = F.softmax(ori_logits / T, dim=-1).gather(-1, indices).flatten(0, -2)
                        can = F.log_softmax(logits / T, dim=-1).gather(-1, indices).flatten(0, -2)
                        distill_loss = F.kl_div(can, ref, reduction="batchmean")
                    else:
                        top_logits = logits.gather(-1, indices)
                        distill_loss = F.kl_div(
                            F.log_softmax(top_logits / T, dim=-1).flatten(0, -2),
                            F.softmax(top_ori_logits / T, dim=-1).flatten(0, -2),
                            reduction="batchmean",
                        )
                    loss = ori_loss * (1 - alpha) + distill_loss * (alpha * T * T)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "mse":
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**student_inputs)
                    logits = outputs.logits
                    loss = F.mse_loss(logits, ori_logits)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "kd":
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**full_inputs)
                    logits = outputs.logits
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    logits = logits.view(-1, logits.size(-1))
                    ori_logits = ori_logits.view(-1, ori_logits.size(-1))
                    distill_loss = F.kl_div(
                        F.log_softmax(logits / T, dim=-1).flatten(0, -2),
                        F.softmax(ori_logits / T, dim=-1).flatten(0, -2),
                        reduction="batchmean",
                    )
                    loss = ori_loss * (1 - alpha) + distill_loss * (alpha * T * T)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_kl":
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**student_inputs)
                    logits = outputs.logits
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_kl_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                    )
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_kl_top"):
                    k = parse_k("dual_kl_top", default_k=1000)
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**student_inputs)
                    logits = outputs.logits
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_kl_topk_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                        k=k,
                        post_attn=use_post_attn(),
                    )
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_kd_top"):
                    k = parse_k("dual_kd_top", default_k=1000)
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**full_inputs)
                    logits = outputs.logits
                    ori_loss = outputs["loss"]
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    distill_loss = compute_dual_kl_topk_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                        k=k,
                        post_attn=use_post_attn(),
                    )
                    alpha = self.loss_alpha
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_kd":
                    ori_logits = get_ori_outputs().logits
                    prepare_student_path()
                    outputs = model(**full_inputs)
                    logits = outputs.logits
                    ori_loss = outputs["loss"]
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    distill_loss = compute_dual_kl_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                    )
                    alpha = self.loss_alpha
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    return (loss, outputs) if return_outputs else loss

                raise ValueError(
                    f"Unsupported lora loss type: {loss_type}. "
                    f"Supported: sft/origin, rkl, dual_rkl, kl, r_kl_top[_K], dual_r_kl_top[_K], "
                    f"kl_top[_K], kd_top[_K], dual_kl, dual_kd, dual_kl_top[_K], dual_kd_top[_K], mse, kd."
                )
            finally:
                restore_temporary()
                set_hif4_act_enabled(previous_hif4_enabled)
