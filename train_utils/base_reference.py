import hashlib
import time
from typing import Optional, Union

import torch
from torch import nn

import rotation.model_utils as model_utils


def load_frozen_base_reference_model(
    model_path: str,
    *,
    access_token: Optional[str],
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    model = model_utils.get_model(model_path, access_token)
    if dtype is not None:
        model.to(dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    if hasattr(getattr(model, "config", None), "use_cache"):
        model.config.use_cache = False
    model.to(device)
    return model


def _first_floating_dtype(model: nn.Module) -> torch.dtype:
    for tensor in model.parameters():
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in model.buffers():
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def _load_reference_config(model_path: str, access_token: Optional[str]):
    from transformers import AutoConfig

    kwargs = {"trust_remote_code": False}
    if access_token:
        kwargs["token"] = access_token
    return AutoConfig.from_pretrained(model_path, **kwargs)


def _resolve_reference_dtype(config, requested_dtype: Optional[torch.dtype]) -> Optional[torch.dtype]:
    if requested_dtype is not None:
        return requested_dtype
    config_dtype = getattr(config, "torch_dtype", None)
    if isinstance(config_dtype, torch.dtype):
        return config_dtype
    if isinstance(config_dtype, str):
        value = getattr(torch, config_dtype, None)
        if isinstance(value, torch.dtype):
            return value
    return None


def _reference_attn_implementation(model_path: str) -> Optional[str]:
    if "llama" in model_path.lower():
        return "sdpa"
    if "Qwen3" in model_path:
        return "flash_attention_2"
    return None


def _reference_seqlen(model_path: str, config) -> int:
    if (
        "llama" in model_path.lower()
        or "mistral" in model_path.lower()
        or "Qwen2" in model_path
        or "Qwen3" in model_path
    ):
        return 2048
    max_pos = getattr(config, "max_position_embeddings", None)
    return int(max_pos) if isinstance(max_pos, int) and max_pos > 0 else 2048


def _build_empty_reference_model(
    model_path: str,
    config,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    model_kwargs = {
        "trust_remote_code": False,
        "torch_dtype": dtype,
    }
    attn_implementation = _reference_attn_implementation(model_path)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    model.to(dtype=dtype)
    model.to_empty(device=device)
    model.tie_weights()
    model.seqlen = _reference_seqlen(model_path, config)
    return model


def _reference_broadcast_entries(model: nn.Module):
    entries = []
    entries.extend((f"param:{name}", tensor.data) for name, tensor in model.named_parameters(remove_duplicate=False))
    entries.extend((f"buffer:{name}", tensor.data) for name, tensor in model.named_buffers(remove_duplicate=False))
    return entries


def _reference_entry_signature(entries, *, device: torch.device) -> torch.Tensor:
    digest = hashlib.sha256()
    total_numel = 0
    for name, tensor in entries:
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        total_numel += int(tensor.numel())
    digest_value = int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)
    return torch.tensor(
        [len(entries), total_numel, digest_value],
        device=device,
        dtype=torch.int64,
    )


def _validate_reference_broadcast_layout(model: nn.Module, *, device: torch.device, world_size: int):
    entries = _reference_broadcast_entries(model)
    local_signature = _reference_entry_signature(entries, device=device)
    signatures = [torch.empty_like(local_signature) for _ in range(world_size)]
    torch.distributed.all_gather(signatures, local_signature)
    if any(not torch.equal(signatures[0], signature) for signature in signatures[1:]):
        values = [tuple(int(x) for x in signature.cpu().tolist()) for signature in signatures]
        raise RuntimeError(f"Distributed teacher tensor layout differs across ranks: {values}")
    return entries


def _broadcast_reference_model_tensors(entries, *, src: int = 0) -> None:
    for _name, tensor in entries:
        if tensor.numel() > 0:
            torch.distributed.broadcast(tensor, src=src)


def load_frozen_base_reference_model_distributed(
    model_path: str,
    *,
    access_token: Optional[str],
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    logger=None,
) -> nn.Module:
    """Load teacher weight shards only on rank 0, then copy the weights to other CUDA DP ranks."""
    target_device = torch.device(device)
    distributed_ready = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and target_device.type == "cuda"
        and str(torch.distributed.get_backend()).lower() == "nccl"
    )
    if not distributed_ready:
        return load_frozen_base_reference_model(
            model_path,
            access_token=access_token,
            device=target_device,
            dtype=dtype,
        )

    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    started = time.perf_counter()

    config = _load_reference_config(model_path, access_token)
    resolved_dtype = _resolve_reference_dtype(config, dtype)
    if resolved_dtype is None:
        if logger is not None and rank == 0:
            logger.warning(
                "Distributed teacher load disabled because config does not declare torch_dtype; "
                "falling back to per-rank checkpoint loading."
            )
        return load_frozen_base_reference_model(
            model_path,
            access_token=access_token,
            device=target_device,
            dtype=dtype,
        )

    model: Optional[nn.Module] = None
    local_error: Optional[BaseException] = None
    try:
        if rank == 0:
            model = load_frozen_base_reference_model(
                model_path,
                access_token=access_token,
                device=target_device,
                dtype=resolved_dtype,
            )
        else:
            model = _build_empty_reference_model(
                model_path,
                config,
                dtype=resolved_dtype,
                device=target_device,
            )
    except BaseException as exc:
        local_error = exc

    ready = torch.tensor([0 if local_error is not None else 1], device=target_device, dtype=torch.int32)
    torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
    if int(ready.item()) != 1:
        if local_error is not None:
            raise local_error
        raise RuntimeError("Another rank failed while preparing the distributed distill teacher.")

    assert model is not None
    entries = _validate_reference_broadcast_layout(model, device=target_device, world_size=world_size)
    _broadcast_reference_model_tensors(entries, src=0)
    model.requires_grad_(False)
    model.eval()
    if hasattr(getattr(model, "config", None), "use_cache"):
        model.config.use_cache = False
    if logger is not None and rank == 0:
        logger.info(
            "Loaded distributed distill teacher with rank0 weight-shard I/O + NCCL broadcast: "
            "world_size=%d device=%s dtype=%s elapsed=%.2fs",
            world_size,
            str(target_device),
            str(_first_floating_dtype(model)),
            time.perf_counter() - started,
        )
    return model


def load_frozen_base_reference_model_distributed_from_hf_args(
    model_path: str,
    hf_args,
    *,
    device: Union[str, torch.device],
    logger=None,
) -> nn.Module:
    return load_frozen_base_reference_model_distributed(
        model_path,
        access_token=vars(hf_args).get("access_token"),
        device=device,
        dtype=None,
        logger=logger,
    )


def get_reference_module(model: nn.Module, module_name: str) -> nn.Module:
    current: object = model
    for part in str(module_name).split("."):
        if not part:
            raise ValueError(f"Invalid empty path segment in reference module path: {module_name}")
        if part.isdigit():
            try:
                current = current[int(part)]  # type: ignore[index]
                continue
            except (TypeError, IndexError, KeyError):
                pass
        if isinstance(current, nn.Module) and part in current._modules:
            current = current._modules[part]
            continue
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        raise ValueError(f"Reference module path not found: {module_name}")
    if not isinstance(current, nn.Module):
        raise ValueError(f"Reference module path does not resolve to nn.Module: {module_name}")
    return current


def clone_frozen_linear_from_reference(
    reference_model: nn.Module,
    module_name: str,
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> nn.Linear:
    source = get_reference_module(reference_model, module_name)
    if not isinstance(source, nn.Linear):
        raise ValueError(f"Reference module is not nn.Linear: {module_name}")

    target_dtype = dtype if dtype is not None else source.weight.dtype
    cloned = nn.Linear(
        source.in_features,
        source.out_features,
        bias=source.bias is not None,
        device=device,
        dtype=target_dtype,
    )
    cloned.weight = nn.Parameter(
        source.weight.detach().clone().to(device=device, dtype=target_dtype),
        requires_grad=False,
    )
    if source.bias is not None:
        cloned.bias = nn.Parameter(
            source.bias.detach().clone().to(device=device, dtype=target_dtype),
            requires_grad=False,
        )
    cloned.eval()
    return cloned
