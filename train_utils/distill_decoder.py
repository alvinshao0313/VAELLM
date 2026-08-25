from dataclasses import dataclass
from typing import Sequence, Tuple

from torch import nn

from litebsq.vae_linear import VAELinear


@dataclass(frozen=True)
class NamedMainDecoderTarget:
    name: str
    base_layer: VAELinear


def iter_main_decoder_modules(base_layer: VAELinear) -> Tuple[nn.Module, ...]:
    packed = getattr(base_layer, "_parallel_stage_decoder", None)
    if packed is not None:
        return (packed,)

    modules = []
    seen: set[int] = set()
    residual_stages = int(getattr(base_layer, "residual_stages", 1))
    parallel_parts = int(getattr(base_layer, "parallel_parts", 1))
    for stage_idx in range(residual_stages):
        for part_idx in range(parallel_parts):
            decoder = base_layer.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
            decoder_id = id(decoder)
            if decoder_id in seen:
                continue
            seen.add(decoder_id)
            modules.append(decoder)
    return tuple(modules)


def enable_main_decoder_targets(
    targets: Sequence[NamedMainDecoderTarget],
) -> Tuple[nn.Parameter, ...]:
    params = []
    seen: set[int] = set()
    empty_targets = []

    for target in targets:
        target.base_layer.enable_trainable_decode(parallel_stage_decode=True)
        target_params = []
        for decoder in iter_main_decoder_modules(target.base_layer):
            decoder.requires_grad_(True)
            for _param_name, param in decoder.named_parameters():
                target_params.append(param)
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        if not target_params:
            empty_targets.append(str(target.name))

    if empty_targets:
        raise RuntimeError(
            "Selected VAELinear main decoder has no trainable parameters: "
            + ", ".join(empty_targets)
        )
    return tuple(params)


def finalize_main_decoder_targets(
    targets: Sequence[NamedMainDecoderTarget],
) -> int:
    finalized = 0
    for target in targets:
        for decoder in iter_main_decoder_modules(target.base_layer):
            decoder.requires_grad_(False)
        target.base_layer.disable_trainable_decode()
        target.base_layer.clear_decoded_weight_cache()
        finalized += 1
    return int(finalized)
