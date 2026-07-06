import argparse
import json
from dataclasses import dataclass
from typing import Optional, Sequence


_AUTOENCODER_NORM_CHOICES = ("group", "batch", "layer", "rms", "no")
_AUTOENCODER_ACTIVATION_CHOICES = ("swish", "relu", "none", "sigmoid", "gelu", "hard_swish")
_AUTOENCODER_DECODER_CHOICES = ("linear", "symmetric", "asymmetric")
_DYNAMIC_ARCH_FIELDS = (
    "base_ch",
    "num_res_blocks",
    "decoder_type",
    "decoder_base_ch",
    "decoder_num_res_blocks",
)


@dataclass(frozen=True)
class AutoEncoderArchSpec:
    codebook_bits: int
    codebook_dim: int
    encoder_hidden_dim: int
    encoder_num_res_blocks: int
    decoder_hidden_dim: int
    decoder_num_res_blocks: int
    decoder_type: str
    norm_type: str
    activation_type: str
    use_checkpoint: bool


def _parse_positive_int_like(value, *, arg_name: str) -> int:
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{value}'. Expected positive integer.")
    try:
        out = int(value)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{value}'. Expected positive integer.") from e
    if out < 1:
        raise argparse.ArgumentTypeError(f"{arg_name} must be >= 1, got {out}.")
    return int(out)


def _parse_non_negative_int_like(value, *, arg_name: str) -> int:
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{value}'. Expected non-negative integer.")
    try:
        out = int(value)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(
            f"Invalid {arg_name} value '{value}'. Expected non-negative integer."
        ) from e
    if out < 0:
        raise argparse.ArgumentTypeError(f"{arg_name} must be >= 0, got {out}.")
    return int(out)


def _parse_stage_list_or_scalar(
    value,
    *,
    arg_name: str,
    item_parser,
):
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raw = str(value).strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"Invalid {arg_name} list '{value}'. Expected valid JSON list."
                ) from e
            if not isinstance(parsed, list):
                raise argparse.ArgumentTypeError(
                    f"Invalid {arg_name} value '{value}'. JSON form must be a list."
                )
            raw_items = parsed
        else:
            return item_parser(value)

    if len(raw_items) == 0:
        raise argparse.ArgumentTypeError(f"{arg_name} list cannot be empty.")
    return [item_parser(v) for v in raw_items]


def _parse_choice_like(value, *, arg_name: str, choices: Sequence[str]) -> str:
    raw = str(value).strip().lower()
    allowed = {str(c).strip().lower() for c in choices}
    if raw not in allowed:
        raise argparse.ArgumentTypeError(
            f"Invalid {arg_name} value '{value}'. Supported: {','.join(sorted(allowed))}."
        )
    return raw


def _parse_choice_or_stage_list(value, *, arg_name: str, choices: Sequence[str]):
    return _parse_stage_list_or_scalar(
        value,
        arg_name=arg_name,
        item_parser=lambda v: _parse_choice_like(v, arg_name=arg_name, choices=choices),
    )


def _parse_positive_int_or_stage_list(value, *, arg_name: str):
    return _parse_stage_list_or_scalar(
        value,
        arg_name=arg_name,
        item_parser=lambda v: _parse_positive_int_like(v, arg_name=arg_name),
    )


def _parse_non_negative_int_or_stage_list(value, *, arg_name: str):
    return _parse_stage_list_or_scalar(
        value,
        arg_name=arg_name,
        item_parser=lambda v: _parse_non_negative_int_like(v, arg_name=arg_name),
    )


def _parse_positive_int_or_category_schedule(value, *, arg_name: str):
    if value is None:
        raise argparse.ArgumentTypeError(f"{arg_name} cannot be empty.")

    if isinstance(value, int):
        return _parse_positive_int_like(value, arg_name=arg_name)

    parsed_obj = None
    if isinstance(value, dict):
        parsed_obj = value
    else:
        raw = str(value).strip()
        if not raw:
            raise argparse.ArgumentTypeError(f"{arg_name} cannot be empty.")
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed_obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(
                    f"Invalid {arg_name} dict '{value}'. "
                    "Please pass valid JSON, for example: "
                    '\'{"default":16,"q_proj":24}\'.'
                ) from e
        else:
            return _parse_positive_int_like(raw, arg_name=arg_name)

    if not isinstance(parsed_obj, dict):
        raise argparse.ArgumentTypeError(
            f"Invalid {arg_name} value '{value}'. "
            "JSON form must be an object/dict."
        )
    if not parsed_obj:
        raise argparse.ArgumentTypeError(f"{arg_name} dict cannot be empty.")

    out = {}
    for k, v in parsed_obj.items():
        key = str(k).strip()
        if not key:
            raise argparse.ArgumentTypeError(
                f"Invalid {arg_name} key in '{value}': key cannot be empty."
            )
        out[key] = _parse_positive_int_like(v, arg_name=f"{arg_name}[{key}]")
    return out


def _parse_positive_int_or_category_schedule_or_stage_list(value, *, arg_name: str):
    return _parse_stage_list_or_scalar(
        value,
        arg_name=arg_name,
        item_parser=lambda v: _parse_positive_int_or_category_schedule(v, arg_name=arg_name),
    )


def _resolve_positive_int(value, *, default: int, arg_name: str, allow_zero: bool = False) -> int:
    out = default if value is None else int(value)
    min_value = 0 if allow_zero else 1
    if out < min_value:
        raise ValueError(f"{arg_name} must be >= {min_value}, got {out}")
    return int(out)


def _has_dynamic_arch_fields(args) -> bool:
    return any(isinstance(getattr(args, key, None), (list, tuple)) for key in _DYNAMIC_ARCH_FIELDS)


def add_autoencoder_model_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--codebook_bits",
        type=lambda v: _parse_positive_int_or_category_schedule_or_stage_list(v, arg_name="--codebook_bits"),
        default=16,
    )
    parser.add_argument(
        "--codebook_dim",
        type=lambda v: _parse_positive_int_or_category_schedule_or_stage_list(v, arg_name="--codebook_dim"),
        default=8,
    )
    parser.add_argument(
        "--residual_stages",
        type=lambda v: _parse_positive_int_like(v, arg_name="--residual_stages"),
        default=1,
        help="Number of residual quantization stages. 1 keeps the original single-stage behavior.",
    )
    parser.add_argument(
        "--base_ch",
        type=lambda v: _parse_positive_int_or_stage_list(v, arg_name="--base_ch"),
        default=128,
    )
    parser.add_argument(
        "--num_res_blocks",
        type=lambda v: _parse_non_negative_int_or_stage_list(v, arg_name="--num_res_blocks"),
        default=1,
    )
    parser.add_argument(
        "--decoder_base_ch",
        "--decoder_hidden_dim",
        dest="decoder_base_ch",
        type=lambda v: _parse_positive_int_or_stage_list(v, arg_name="--decoder_base_ch"),
        default=None,
        help="Decoder hidden dim for --decoder_type asymmetric. Default: --base_ch",
    )
    parser.add_argument(
        "--decoder_num_res_blocks",
        type=lambda v: _parse_non_negative_int_or_stage_list(v, arg_name="--decoder_num_res_blocks"),
        default=None,
        help="Decoder residual blocks for --decoder_type asymmetric. Default: --num_res_blocks",
    )
    parser.add_argument("--quantizer_type", type=str, default="BSQ")
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--inv_temperature", type=float, default=100.0)
    parser.add_argument(
        "--norm_type",
        type=lambda v: _parse_choice_or_stage_list(v, arg_name="--norm_type", choices=_AUTOENCODER_NORM_CHOICES),
        default="group",
    )
    parser.add_argument(
        "--activation_type",
        type=lambda v: _parse_choice_or_stage_list(
            v,
            arg_name="--activation_type",
            choices=_AUTOENCODER_ACTIVATION_CHOICES,
        ),
        default="swish",
    )
    parser.add_argument(
        "--decoder_type",
        type=lambda v: _parse_choice_or_stage_list(
            v,
            arg_name="--decoder_type",
            choices=_AUTOENCODER_DECODER_CHOICES,
        ),
        default="linear",
    )
    return parser


def resolve_autoencoder_arch_spec(args) -> AutoEncoderArchSpec:
    if _has_dynamic_arch_fields(args):
        raise ValueError(
            "resolve_autoencoder_arch_spec requires scalar autoencoder args. "
            "Resolve stage-wise values before building the autoencoder."
        )

    codebook_bits = _parse_positive_int_like(getattr(args, "codebook_bits", 16), arg_name="--codebook_bits")
    codebook_dim = _parse_positive_int_like(getattr(args, "codebook_dim", 8), arg_name="--codebook_dim")
    encoder_hidden_dim = _resolve_positive_int(
        getattr(args, "base_ch", 128),
        default=128,
        arg_name="--base_ch",
        allow_zero=False,
    )
    encoder_num_res_blocks = _resolve_positive_int(
        getattr(args, "num_res_blocks", 1),
        default=1,
        arg_name="--num_res_blocks",
        allow_zero=True,
    )
    decoder_type = _parse_choice_like(
        getattr(args, "decoder_type", "linear"),
        arg_name="--decoder_type",
        choices=_AUTOENCODER_DECODER_CHOICES,
    )
    norm_type = _parse_choice_like(
        getattr(args, "norm_type", "group"),
        arg_name="--norm_type",
        choices=_AUTOENCODER_NORM_CHOICES,
    )
    activation_type = _parse_choice_like(
        getattr(args, "activation_type", "swish"),
        arg_name="--activation_type",
        choices=_AUTOENCODER_ACTIVATION_CHOICES,
    )
    if decoder_type == "asymmetric":
        decoder_hidden_dim = _resolve_positive_int(
            getattr(args, "decoder_base_ch", None),
            default=encoder_hidden_dim,
            arg_name="--decoder_base_ch",
            allow_zero=False,
        )
        decoder_num_res_blocks = _resolve_positive_int(
            getattr(args, "decoder_num_res_blocks", None),
            default=encoder_num_res_blocks,
            arg_name="--decoder_num_res_blocks",
            allow_zero=True,
        )
    else:
        decoder_hidden_dim = int(encoder_hidden_dim)
        decoder_num_res_blocks = int(encoder_num_res_blocks)

    return AutoEncoderArchSpec(
        codebook_bits=int(codebook_bits),
        codebook_dim=int(codebook_dim),
        encoder_hidden_dim=int(encoder_hidden_dim),
        encoder_num_res_blocks=int(encoder_num_res_blocks),
        decoder_hidden_dim=int(decoder_hidden_dim),
        decoder_num_res_blocks=int(decoder_num_res_blocks),
        decoder_type=str(decoder_type),
        norm_type=str(norm_type),
        activation_type=str(activation_type),
        use_checkpoint=bool(getattr(args, "vae_decoder_checkpoint", False)),
    )


def apply_autoencoder_arch_defaults(args):
    if _has_dynamic_arch_fields(args):
        return args

    spec = resolve_autoencoder_arch_spec(args)
    setattr(args, "codebook_bits", int(spec.codebook_bits))
    setattr(args, "codebook_dim", int(spec.codebook_dim))
    setattr(args, "base_ch", int(spec.encoder_hidden_dim))
    setattr(args, "num_res_blocks", int(spec.encoder_num_res_blocks))
    setattr(args, "encoder_base_ch", int(spec.encoder_hidden_dim))
    setattr(args, "encoder_num_res_blocks", int(spec.encoder_num_res_blocks))
    setattr(args, "decoder_base_ch", int(spec.decoder_hidden_dim))
    setattr(args, "decoder_num_res_blocks", int(spec.decoder_num_res_blocks))
    setattr(args, "decoder_type", str(spec.decoder_type))
    setattr(args, "norm_type", str(spec.norm_type))
    setattr(args, "activation_type", str(spec.activation_type))
    return args


__all__ = [
    "AutoEncoderArchSpec",
    "add_autoencoder_model_args",
    "apply_autoencoder_arch_defaults",
    "resolve_autoencoder_arch_spec",
]
