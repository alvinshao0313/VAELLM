from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from litebsq.vae_linear import VAELinear
from train_utils.utils import extract_layer_idx


def resolve_target_devices(
    targets: Sequence[Tuple[str, VAELinear]],
    *,
    parallel_mode: str,
    dp_local_device: Optional[torch.device],
    offload_mode: str,
    layer_device_map: Optional[Dict[int, torch.device]],
) -> Dict[str, torch.device]:
    """Resolve the fixed score device for each logical VAELinear target."""
    devices: Dict[str, torch.device] = {}
    for module_name, module in targets:
        name = str(module_name)
        if str(parallel_mode).strip().lower() == "dp":
            if dp_local_device is None:
                raise RuntimeError("Sparse Bit DP target-device resolution requires dp_local_device.")
            device = torch.device(dp_local_device)
        elif str(offload_mode).strip().lower() == "streaming":
            layer_idx = extract_layer_idx(name)
            if layer_idx is None or layer_device_map is None or int(layer_idx) not in layer_device_map:
                raise RuntimeError(f"{name}: cannot resolve streaming Sparse Bit target layer device.")
            device = torch.device(layer_device_map[int(layer_idx)])
        else:
            param = next(module.parameters(), None)
            if param is not None:
                device = torch.device(param.device)
            else:
                storage = module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)
                device = torch.device(storage.device)
        if device.type != "cuda":
            raise RuntimeError(
                f"{name}: Sparse Bit production training requires a CUDA target device, got {device}."
            )
        devices[name] = device
    if len(devices) != len(targets):
        raise RuntimeError("Sparse Bit target-device map size mismatch.")
    return devices


def collect_packed_payloads(
    targets: Sequence[Tuple[str, VAELinear]],
) -> Dict[str, Dict[Tuple[int, int], torch.Tensor]]:
    payloads: Dict[str, Dict[Tuple[int, int], torch.Tensor]] = {}
    for module_name, module in targets:
        per_bank: Dict[Tuple[int, int], torch.Tensor] = {}
        for stage_idx in range(int(module.residual_stages)):
            for part_idx in range(int(module.parallel_parts)):
                storage = module.get_stage_part_vq_storage(stage_idx=stage_idx, part_idx=part_idx)
                per_bank[(int(stage_idx), int(part_idx))] = (
                    storage.detach().to(device="cpu", dtype=torch.uint8).clone().contiguous()
                )
        payloads[str(module_name)] = per_bank
    return payloads


@torch.no_grad()
def write_packed_payloads(
    model,
    payloads: Dict[str, Dict[Tuple[int, int], torch.Tensor]],
) -> int:
    refs = {str(name): module for name, module in model.named_modules() if isinstance(module, VAELinear)}
    missing = sorted(set(payloads) - set(refs))
    if missing:
        raise RuntimeError(f"Sparse Bit export model is missing VAELinear targets: {missing}")
    written = 0
    for module_name, banks in payloads.items():
        module = refs[module_name]
        for (stage_idx, part_idx), source in banks.items():
            target = module.get_stage_part_vq_storage(stage_idx=stage_idx, part_idx=part_idx)
            if target.dtype != torch.uint8 or tuple(target.shape) != tuple(source.shape):
                raise RuntimeError(
                    f"{module_name}|stage={stage_idx}|part={part_idx}: packed payload shape/dtype mismatch: "
                    f"target={target.dtype}/{tuple(target.shape)} source={source.dtype}/{tuple(source.shape)}."
                )
            target.copy_(source.to(device=target.device, dtype=torch.uint8))
            written += 1
        module.clear_decoded_weight_cache()
    return int(written)
