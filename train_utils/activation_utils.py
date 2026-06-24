import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from e2e_common.data import normalize_dataset_mix_spec
from train_utils.lora_data import build_calibration_input_ids


@dataclass
class ActivationCalibrationCache:
    dataset: str
    model_path: str
    nsamples: int
    seqlen: int
    seed: int
    input_ids: List[torch.Tensor]


def _normalize_calibration_dataset_mix(dataset: str) -> str:
    _sources, _weights, normalized_spec = normalize_dataset_mix_spec(dataset)
    return str(normalized_spec)


def _build_calibration_cache(
    *,
    dataset: str,
    model_path: str,
    access_token: Optional[str],
    nsamples: int,
    seqlen: int,
    seed: int,
) -> ActivationCalibrationCache:
    dataset_key = _normalize_calibration_dataset_mix(dataset)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        token=access_token,
    )
    input_ids = build_calibration_input_ids(
        dataset_name=dataset_key,
        tokenizer=tokenizer,
        nsamples=int(nsamples),
        seqlen=int(seqlen),
        seed=int(seed),
    )
    return ActivationCalibrationCache(
        dataset=dataset_key,
        model_path=str(model_path),
        nsamples=int(nsamples),
        seqlen=int(seqlen),
        seed=int(seed),
        input_ids=input_ids,
    )


def _cache_matches(
    cache: ActivationCalibrationCache,
    *,
    dataset: str,
    model_path: str,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> bool:
    return (
        str(cache.dataset) == _normalize_calibration_dataset_mix(dataset)
        and str(cache.model_path) == str(model_path)
        and int(cache.nsamples) == int(nsamples)
        and int(cache.seqlen) == int(seqlen)
        and int(cache.seed) == int(seed)
    )


def collect_activation_stats_for_linears(
    *,
    model: nn.Module,
    linear_items: Sequence[Tuple[str, nn.Module]],
    model_path: str,
    access_token: Optional[str],
    dataset: str = "",
    nsamples: int = 512,
    seqlen: int = 512,
    seed: int = 0,
    device: str = "cuda",
    cache: Optional[ActivationCalibrationCache] = None,
    log_every: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Dict[str, object]], ActivationCalibrationCache]:
    if not linear_items:
        return {}, cache if cache is not None else _build_calibration_cache(
            dataset=dataset,
            model_path=model_path,
            access_token=access_token,
            nsamples=nsamples,
            seqlen=seqlen,
            seed=seed,
        )

    if cache is None or not _cache_matches(
        cache,
        dataset=dataset,
        model_path=model_path,
        nsamples=nsamples,
        seqlen=seqlen,
        seed=seed,
    ):
        cache = _build_calibration_cache(
            dataset=dataset,
            model_path=model_path,
            access_token=access_token,
            nsamples=nsamples,
            seqlen=seqlen,
            seed=seed,
        )

    run_device = str(device)
    if run_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device={run_device}, but CUDA is not available.")
    device_obj = torch.device(run_device)

    absmax_by_linear: Dict[str, torch.Tensor] = {}
    sqsum_by_linear: Dict[str, torch.Tensor] = {}
    count_by_linear: Dict[str, int] = {}
    handles = []
    for name, module in linear_items:
        if not isinstance(module, nn.Module) or not hasattr(module, "in_features"):
            raise TypeError(f"Target module for {name} must be an nn.Module with in_features, got {type(module)}")
        in_features = int(getattr(module, "in_features"))
        absmax_by_linear[name] = torch.zeros(in_features, dtype=torch.float32, device="cpu")
        sqsum_by_linear[name] = torch.zeros(in_features, dtype=torch.float32, device="cpu")
        count_by_linear[name] = 0

        def _hook_factory(one_name: str, hook_in_features: int):
            def _hook(_module: nn.Module, inputs, _output):
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.numel() == 0:
                    return
                if int(x.shape[-1]) != int(hook_in_features):
                    return
                x_flat = x.detach().reshape(-1, int(hook_in_features)).to(dtype=torch.float32)
                cur = x_flat.abs().amax(dim=0).to(dtype=torch.float32, device="cpu")
                absmax_by_linear[one_name] = torch.maximum(absmax_by_linear[one_name], cur)
                sqsum_by_linear[one_name] += x_flat.pow(2).sum(dim=0).to(dtype=torch.float32, device="cpu")
                count_by_linear[one_name] += int(x_flat.shape[0])

            return _hook

        handles.append(module.register_forward_hook(_hook_factory(name, in_features)))

    was_training = bool(model.training)
    use_cache_cfg = getattr(model.config, "use_cache", None)
    param = next(model.parameters(), None)
    original_device = param.device if param is not None else torch.device("cpu")

    try:
        if original_device != device_obj:
            model.to(device_obj)
        model.eval()
        if use_cache_cfg is not None:
            model.config.use_cache = False

        total = len(cache.input_ids)
        with torch.no_grad():
            for i, inp in enumerate(cache.input_ids, start=1):
                _ = model(input_ids=inp.to(device_obj, non_blocking=True))
                if logger is not None and int(log_every) > 0 and (i % int(log_every) == 0 or i == total):
                    logger.info("act_max recalib progress: %d/%d", i, total)
    finally:
        for h in handles:
            h.remove()
        if use_cache_cfg is not None:
            model.config.use_cache = use_cache_cfg
        if original_device != device_obj:
            model.to(original_device)
            if device_obj.type == "cuda" and original_device.type == "cpu":
                torch.cuda.empty_cache()
        if was_training:
            model.train()

    stats_by_linear: Dict[str, Dict[str, object]] = {}
    for name, _module in linear_items:
        num_tokens = int(count_by_linear[name])
        if num_tokens <= 0:
            sq_mean = torch.zeros_like(sqsum_by_linear[name])
        else:
            sq_mean = (sqsum_by_linear[name] / float(num_tokens)).contiguous()
        stats_by_linear[name] = {
            "max": absmax_by_linear[name].contiguous(),
            "sq_mean": sq_mean,
            "rms": torch.sqrt(sq_mean.clamp_min(0.0)).contiguous(),
            "num_tokens": int(num_tokens),
        }

    return stats_by_linear, cache


def collect_act_max_for_linears(
    *,
    model: nn.Module,
    linear_items: Sequence[Tuple[str, nn.Module]],
    model_path: str,
    access_token: Optional[str],
    dataset: str = "",
    nsamples: int = 512,
    seqlen: int = 512,
    seed: int = 0,
    device: str = "cuda",
    cache: Optional[ActivationCalibrationCache] = None,
    log_every: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, torch.Tensor], ActivationCalibrationCache]:
    stats_by_linear, cache = collect_activation_stats_for_linears(
        model=model,
        linear_items=linear_items,
        model_path=model_path,
        access_token=access_token,
        dataset=dataset,
        nsamples=nsamples,
        seqlen=seqlen,
        seed=seed,
        device=device,
        cache=cache,
        log_every=log_every,
        logger=logger,
    )
    return {
        name: stats["max"]
        for name, stats in stats_by_linear.items()
        if isinstance(stats.get("max"), torch.Tensor)
    }, cache
