import os
import sys
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HIF4_GPU_ROOT = os.path.join(_REPO_ROOT, "HiFloat4", "hif4_gpu")
_HIF4_ACT_QUANTIZER: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
_VAE_LINEAR_TYPE = None
_PEFT_LORA_LINEAR_TYPE = None
_PEFT_VAE_PROXY_TYPE = None
_COMPRESSED_SUBSPACE_PEFT_PROXY_TYPE = None
_PEFT_PROXY_ADAPTER_PREDICATE = None


class Hif4ActController:
    def __init__(self, quantize: Callable[[torch.Tensor], torch.Tensor]):
        self.quantize = quantize
        self.enabled = False


def build_hif4_act_controller(enabled: bool) -> Optional[Hif4ActController]:
    if not enabled:
        return None
    return Hif4ActController(load_hif4_act_quantizer())


def load_hif4_act_quantizer() -> Callable[[torch.Tensor], torch.Tensor]:
    global _HIF4_ACT_QUANTIZER
    if _HIF4_ACT_QUANTIZER is not None:
        return _HIF4_ACT_QUANTIZER
    if not os.path.isdir(_HIF4_GPU_ROOT):
        raise ImportError(
            "启用 HiFloat4 激活量化失败：未找到 HiFloat4 GPU 目录。"
            f" 期望路径: {_HIF4_GPU_ROOT}"
        )
    if _HIF4_GPU_ROOT not in sys.path:
        sys.path.insert(0, _HIF4_GPU_ROOT)
    try:
        from quant_cy import QType, quant_func
    except Exception as exc:
        raise ImportError(
            "启用 HiFloat4 激活量化失败：无法导入 HiFloat4 quant_cy。"
            f" 请确认已构建 {_HIF4_GPU_ROOT}/build.sh。原始错误: {exc}"
        ) from exc

    quant_type = QType("hifx4").dim(-1)

    def quantize(x: torch.Tensor) -> torch.Tensor:
        return quant_func(x, quant_type, force_py=False, force_fp32=True)

    _HIF4_ACT_QUANTIZER = quantize
    return _HIF4_ACT_QUANTIZER


def _iter_parent_names(name: str):
    parts = [part for part in str(name).split(".") if part]
    for idx in range(len(parts) - 1, 0, -1):
        yield ".".join(parts[:idx])


def is_hif4_act_excluded_module_name(name: str) -> bool:
    parts = [part for part in str(name).split(".") if part]
    return "lm_head" in parts


def _get_vae_linear_type():
    global _VAE_LINEAR_TYPE
    if _VAE_LINEAR_TYPE is None:
        try:
            from litebsq.vae_linear import VAELinear
        except Exception:
            _VAE_LINEAR_TYPE = False
        else:
            _VAE_LINEAR_TYPE = VAELinear
    return None if _VAE_LINEAR_TYPE is False else _VAE_LINEAR_TYPE


def _get_peft_lora_linear_type():
    global _PEFT_LORA_LINEAR_TYPE
    if _PEFT_LORA_LINEAR_TYPE is None:
        try:
            from peft.tuners.lora.layer import Linear as PeftLoraLinear
        except Exception:
            _PEFT_LORA_LINEAR_TYPE = False
        else:
            _PEFT_LORA_LINEAR_TYPE = PeftLoraLinear
    return None if _PEFT_LORA_LINEAR_TYPE is False else _PEFT_LORA_LINEAR_TYPE


def _is_peft_lora_linear(module: nn.Module) -> bool:
    peft_linear_type = _get_peft_lora_linear_type()
    return peft_linear_type is not None and isinstance(module, peft_linear_type)


def _get_peft_vae_proxy_type():
    global _PEFT_VAE_PROXY_TYPE
    if _PEFT_VAE_PROXY_TYPE is None:
        try:
            from e2e_common.peft_proxy import PeftVAELinearProxy
        except Exception:
            _PEFT_VAE_PROXY_TYPE = False
        else:
            _PEFT_VAE_PROXY_TYPE = PeftVAELinearProxy
    return None if _PEFT_VAE_PROXY_TYPE is False else _PEFT_VAE_PROXY_TYPE


def _get_compressed_subspace_peft_proxy_type():
    global _COMPRESSED_SUBSPACE_PEFT_PROXY_TYPE
    if _COMPRESSED_SUBSPACE_PEFT_PROXY_TYPE is None:
        try:
            from e2e_common.compressed_subspace_lora import CompressedSubspacePeftProxy
        except Exception:
            _COMPRESSED_SUBSPACE_PEFT_PROXY_TYPE = False
        else:
            _COMPRESSED_SUBSPACE_PEFT_PROXY_TYPE = CompressedSubspacePeftProxy
    return None if _COMPRESSED_SUBSPACE_PEFT_PROXY_TYPE is False else _COMPRESSED_SUBSPACE_PEFT_PROXY_TYPE


def _get_peft_proxy_adapter_predicate():
    global _PEFT_PROXY_ADAPTER_PREDICATE
    if _PEFT_PROXY_ADAPTER_PREDICATE is None:
        try:
            from e2e_common.peft_proxy import is_peft_proxy_adapter_linear
        except Exception:
            _PEFT_PROXY_ADAPTER_PREDICATE = False
        else:
            _PEFT_PROXY_ADAPTER_PREDICATE = is_peft_proxy_adapter_linear
    return None if _PEFT_PROXY_ADAPTER_PREDICATE is False else _PEFT_PROXY_ADAPTER_PREDICATE


def _is_peft_proxy_adapter_linear(module: nn.Module) -> bool:
    predicate = _get_peft_proxy_adapter_predicate()
    return False if predicate is None else bool(predicate(module))


def _is_hif4_wrapped_module(module: nn.Module) -> bool:
    vae_linear_type = _get_vae_linear_type()
    if vae_linear_type is not None and isinstance(module, vae_linear_type):
        return True
    peft_vae_proxy_type = _get_peft_vae_proxy_type()
    if peft_vae_proxy_type is not None and isinstance(module, peft_vae_proxy_type):
        return True
    subspace_proxy_type = _get_compressed_subspace_peft_proxy_type()
    if subspace_proxy_type is not None and isinstance(module, subspace_proxy_type):
        return True
    return _is_peft_lora_linear(module) or _is_peft_proxy_adapter_linear(module)


def collect_hif4_act_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    module_map = dict(model.named_modules())
    targets: List[Tuple[str, nn.Module]] = []
    for name, module in module_map.items():
        if not name:
            continue
        if is_hif4_act_excluded_module_name(name):
            continue
        if any(_is_hif4_wrapped_module(module_map[parent_name]) for parent_name in _iter_parent_names(name)):
            continue
        if _is_hif4_wrapped_module(module):
            targets.append((name, module))
            continue
        if not isinstance(module, nn.Linear):
            continue
        targets.append((name, module))
    return targets


def _make_hif4_act_pre_hook(controller: Hif4ActController):
    def hook(_module, args, kwargs):
        if not controller.enabled or not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return None
        return (controller.quantize(x),) + tuple(args[1:]), kwargs

    return hook


def register_hif4_act_hooks(
    model: nn.Module,
    controller: Hif4ActController,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    seen = set()
    hook = _make_hif4_act_pre_hook(controller)
    for _name, module in collect_hif4_act_modules(model):
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


def remove_hif4_act_hooks(handles: Sequence[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


@contextmanager
def applied_hif4_act(
    model: nn.Module,
    *,
    enabled: bool,
    controller: Optional[Hif4ActController] = None,
    require_targets: bool = True,
    logger=None,
    log_prefix: str = "",
) -> Iterator[dict]:
    if not enabled:
        yield {"enabled": False, "hook_count": 0, "controller": controller}
        return

    resolved_controller = controller or build_hif4_act_controller(True)
    if resolved_controller is None:
        yield {"enabled": False, "hook_count": 0, "controller": None}
        return

    handles = register_hif4_act_hooks(model, resolved_controller)
    if require_targets and not handles:
        raise RuntimeError("启用 HiFloat4 激活量化失败：未找到可注册 hook 的逻辑线性层。")

    previous_enabled = bool(resolved_controller.enabled)
    resolved_controller.enabled = True
    if logger is not None:
        logger.info(
            "%s启用 HiFloat4 激活量化，hook 模块数=%d",
            str(log_prefix),
            len(handles),
        )
    try:
        yield {
            "enabled": True,
            "hook_count": len(handles),
            "controller": resolved_controller,
        }
    finally:
        resolved_controller.enabled = previous_enabled
        remove_hif4_act_hooks(handles)
