import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
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
    abssum_by_linear: Dict[str, torch.Tensor] = {}
    sqsum_by_linear: Dict[str, torch.Tensor] = {}
    count_by_linear: Dict[str, int] = {}
    handles = []
    for name, module in linear_items:
        if not isinstance(module, nn.Module) or not hasattr(module, "in_features"):
            raise TypeError(f"Target module for {name} must be an nn.Module with in_features, got {type(module)}")
        in_features = int(getattr(module, "in_features"))
        absmax_by_linear[name] = torch.zeros(in_features, dtype=torch.float32, device="cpu")
        abssum_by_linear[name] = torch.zeros(in_features, dtype=torch.float32, device="cpu")
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
                x_abs = x_flat.abs()
                cur = x_abs.amax(dim=0).to(dtype=torch.float32, device="cpu")
                absmax_by_linear[one_name] = torch.maximum(absmax_by_linear[one_name], cur)
                abssum_by_linear[one_name] += x_abs.sum(dim=0).to(dtype=torch.float32, device="cpu")
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
                    logger.info("activation stats recalib progress: %d/%d", i, total)
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
            abs_mean = torch.zeros_like(abssum_by_linear[name])
            sq_mean = torch.zeros_like(sqsum_by_linear[name])
        else:
            abs_mean = (abssum_by_linear[name] / float(num_tokens)).contiguous()
            sq_mean = (sqsum_by_linear[name] / float(num_tokens)).contiguous()
        stats_by_linear[name] = {
            "max": absmax_by_linear[name].contiguous(),
            "abs_mean": abs_mean,
            "sq_mean": sq_mean,
            "rms": torch.sqrt(sq_mean.clamp_min(0.0)).contiguous(),
            "num_tokens": int(num_tokens),
        }

    return stats_by_linear, cache


def subset_activation_stats(
    stats_by_linear: Dict[str, Dict[str, object]],
    linear_names: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    missing = [str(name) for name in linear_names if str(name) not in stats_by_linear]
    if missing:
        raise KeyError(
            "Missing precomputed activation stats for linears: " + ",".join(missing)
        )
    return {str(name): stats_by_linear[str(name)] for name in linear_names}


def activation_stats_to_views(
    stats: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    max_by_name: Dict[str, torch.Tensor] = {}
    abs_mean_by_name: Dict[str, torch.Tensor] = {}
    sq_mean_by_name: Dict[str, torch.Tensor] = {}
    for name, entry in stats.items():
        act_max = entry.get("max")
        if isinstance(act_max, torch.Tensor):
            max_by_name[str(name)] = act_max
        act_mean = entry.get("abs_mean")
        if isinstance(act_mean, torch.Tensor):
            abs_mean_by_name[str(name)] = act_mean
        act_sq = entry.get("sq_mean")
        if isinstance(act_sq, torch.Tensor):
            sq_mean_by_name[str(name)] = act_sq
    return max_by_name, abs_mean_by_name, sq_mean_by_name


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


@dataclass
class MlpBlockActivationStats:
    sum_abs_in: torch.Tensor
    sum_sq_in: torch.Tensor
    sum_abs_mid: torch.Tensor
    sum_sq_mid: torch.Tensor
    num_tokens: int

    def to_stats_dict(self, *, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
        if int(self.num_tokens) <= 0:
            raise ValueError("MLP block activation stats have zero tokens.")
        denom = float(self.num_tokens)
        abs_mean_in = (self.sum_abs_in / denom).contiguous()
        sq_mean_in = (self.sum_sq_in / denom).contiguous()
        abs_mean_mid = (self.sum_abs_mid / denom).contiguous()
        sq_mean_mid = (self.sum_sq_mid / denom).contiguous()
        a_in = torch.sqrt(sq_mean_in.clamp_min(0.0) + eps).contiguous()
        a_mid = torch.sqrt(sq_mean_mid.clamp_min(0.0) + eps).contiguous()
        return {
            "abs_mean_in": abs_mean_in,
            "sq_mean_in": sq_mean_in,
            "abs_mean_mid": abs_mean_mid,
            "sq_mean_mid": sq_mean_mid,
            "a_in": a_in,
            "a_mid": a_mid,
            "num_tokens": torch.tensor(int(self.num_tokens), dtype=torch.long),
        }


def _resolve_transformer_layers(model: nn.Module):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model must expose model.model.layers for MLP block activation stats.")
    return model.model.layers


def collect_mlp_block_activation_stats(
    *,
    model: nn.Module,
    layer_indices: Sequence[int],
    model_path: str,
    access_token: Optional[str],
    dataset: str = "",
    nsamples: int = 512,
    seqlen: int = 512,
    seed: int = 0,
    device: str = "cuda",
    cache: Optional[ActivationCalibrationCache] = None,
    skip_layer_keys: Optional[Set[Tuple[int, str]]] = None,
    log_every: int = 0,
    logger: Optional[logging.Logger] = None,
    eps: float = 1e-8,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], ActivationCalibrationCache]:
    requested = sorted({int(idx) for idx in layer_indices})
    if not requested:
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

    layers = _resolve_transformer_layers(model)
    skipped = skip_layer_keys or set()
    sum_sq_in_by_layer: Dict[int, torch.Tensor] = {}
    sum_sq_mid_by_layer: Dict[int, torch.Tensor] = {}
    sum_abs_in_by_layer: Dict[int, torch.Tensor] = {}
    sum_abs_mid_by_layer: Dict[int, torch.Tensor] = {}
    num_tokens_by_layer: Dict[int, int] = {}
    handles = []

    for layer_idx in requested:
        if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
            raise ValueError(f"layer_idx={layer_idx} is out of range for model with {len(layers)} layers.")
        if any((int(layer_idx), cat) in skipped for cat in ("gate_proj", "up_proj", "down_proj")):
            continue
        layer = layers[int(layer_idx)]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
            continue
        d_model = int(up.in_features)
        d_ffn = int(up.out_features)
        if int(gate.in_features) != d_model or int(down.out_features) != d_model or int(down.in_features) != d_ffn:
            raise ValueError(
                f"Layer {layer_idx} MLP shape mismatch: "
                f"up=({int(up.out_features)}, {int(up.in_features)}), "
                f"gate=({int(gate.out_features)}, {int(gate.in_features)}), "
                f"down=({int(down.out_features)}, {int(down.in_features)})."
            )
        sum_sq_in_by_layer[int(layer_idx)] = torch.zeros(d_model, dtype=torch.float32, device="cpu")
        sum_sq_mid_by_layer[int(layer_idx)] = torch.zeros(d_ffn, dtype=torch.float32, device="cpu")
        sum_abs_in_by_layer[int(layer_idx)] = torch.zeros(d_model, dtype=torch.float32, device="cpu")
        sum_abs_mid_by_layer[int(layer_idx)] = torch.zeros(d_ffn, dtype=torch.float32, device="cpu")
        num_tokens_by_layer[int(layer_idx)] = 0

        def _hook_factory(one_layer_idx: int, hook_d_model: int, hook_gate: nn.Linear, hook_up: nn.Linear):
            def _hook(_module: nn.Module, inputs, _output):
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.numel() == 0:
                    return
                if int(x.shape[-1]) != int(hook_d_model):
                    return
                x_flat = x.detach().reshape(-1, int(hook_d_model)).to(dtype=torch.float32)
                sum_sq_in_by_layer[one_layer_idx] += x_flat.pow(2).sum(dim=0).to(
                    dtype=torch.float32,
                    device="cpu",
                )
                sum_abs_in_by_layer[one_layer_idx] += x_flat.abs().sum(dim=0).to(
                    dtype=torch.float32,
                    device="cpu",
                )
                W_gate = hook_gate.weight.detach().to(dtype=torch.float32)
                W_up = hook_up.weight.detach().to(dtype=torch.float32)
                u = x_flat @ W_up.t()
                g = F.silu(x_flat @ W_gate.t())
                z = g * u
                sum_sq_mid_by_layer[one_layer_idx] += z.pow(2).sum(dim=0).to(
                    dtype=torch.float32,
                    device="cpu",
                )
                sum_abs_mid_by_layer[one_layer_idx] += z.abs().sum(dim=0).to(
                    dtype=torch.float32,
                    device="cpu",
                )
                num_tokens_by_layer[one_layer_idx] += int(x_flat.shape[0])

            return _hook

        handles.append(mlp.register_forward_hook(_hook_factory(int(layer_idx), d_model, gate, up)))

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
                    logger.info("MLP block activation stats progress: %d/%d", i, total)
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

    stats_by_mlp_block: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx, sum_sq_in in sum_sq_in_by_layer.items():
        num_tokens = int(num_tokens_by_layer[layer_idx])
        if num_tokens <= 0:
            raise ValueError(f"MLP block activation stats collected zero tokens for layer_idx={layer_idx}.")
        block_stats = MlpBlockActivationStats(
            sum_abs_in=sum_abs_in_by_layer[layer_idx],
            sum_sq_in=sum_sq_in,
            sum_abs_mid=sum_abs_mid_by_layer[layer_idx],
            sum_sq_mid=sum_sq_mid_by_layer[layer_idx],
            num_tokens=num_tokens,
        )
        stats_by_mlp_block[int(layer_idx)] = block_stats.to_stats_dict(eps=eps)

    return stats_by_mlp_block, cache
