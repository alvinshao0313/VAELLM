
import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import transformers


@dataclass
class HFArguments:
    access_token: Optional[str] = field(
        default=None,
        metadata={"help": "Huggingface access token to access gated repo like Llama"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)"
        },
    )


_SKIP_LAYER_PATTERN = re.compile(r"^(\d+)\.([A-Za-z0-9_]+)$")


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def parse_skip_layers(value: Optional[str]) -> Set[Tuple[int, str]]:
    entries = _split_csv(value)
    out: Set[Tuple[int, str]] = set()
    for item in entries:
        m = _SKIP_LAYER_PATTERN.match(item)
        if not m:
            raise ValueError(
                f"Invalid --skip_layers entry '{item}'. Expected format: <layer_idx>.<category>, "
                "for example 0.down_proj or 30.q_proj."
            )
        out.add((int(m.group(1)), m.group(2)))
    return out


def resolve_skip_layer_matches(
    skip_layers: Optional[str],
    discovered_keys: Sequence[Tuple[int, str]],
) -> Tuple[Set[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    requested = parse_skip_layers(skip_layers)
    discovered_set = {(int(layer_idx), str(category)) for layer_idx, category in discovered_keys}
    matched = sorted(requested & discovered_set)
    missing = sorted(requested - discovered_set)
    return requested, matched, missing


def _parse_lora_loss_type(value: str) -> str:
    raw = str(value).strip().lower()
    static_choices = {"sft", "origin", "rkl", "kl", "mse", "kd", "r_kl_top", "kl_top"}
    if raw in static_choices:
        return raw
    for prefix in ("r_kl_top_", "kl_top_"):
        if raw.startswith(prefix):
            k = raw[len(prefix):]
            if k.isdigit() and int(k) > 0:
                return raw
    raise argparse.ArgumentTypeError(
        "Invalid --lora_loss_type. Supported: sft, origin, rkl, kl, mse, kd, "
        "r_kl_top[_K], kl_top[_K] (K must be a positive integer)."
    )


def _parse_bool_like(value, *, arg_name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{value}'. Expected bool.")
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{value}'. Expected bool.")


def _parse_csv_like_names(value, *, arg_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid {arg_name} value '{value}'. Expected comma-separated names or a JSON list."
            ) from e
        if not isinstance(parsed, list):
            raise argparse.ArgumentTypeError(
                f"Invalid {arg_name} value '{value}'. JSON form must be a list."
            )
        return [str(v).strip() for v in parsed if str(v).strip()]
    return _split_csv(raw)


_LORA_SCHEDULE_KEY_ALIASES = {
    "rank": "rank",
    "r": "rank",
    "lora_rank": "rank",
    "alpha": "alpha",
    "lora_alpha": "alpha",
    "dropout": "dropout",
    "lora_dropout": "dropout",
    "steps": "steps",
    "lora_steps": "steps",
    "batch_size": "batch_size",
    "lora_batch_size": "batch_size",
    "nsamples": "nsamples",
    "lora_nsamples": "nsamples",
    "lr": "lr",
    "lora_lr": "lr",
    "weight_decay": "weight_decay",
    "lora_weight_decay": "weight_decay",
    "log_every": "log_every",
    "lora_log_every": "log_every",
    "tune_norm": "tune_norm",
    "lora_tune_norm": "tune_norm",
    "tune_lm_head": "tune_lm_head",
    "lora_tune_lm_head": "tune_lm_head",
    "tune_bias": "tune_bias",
    "lora_tune_bias": "tune_bias",
    "tune_protected_outliers": "tune_protected_outliers",
    "lora_tune_protected_outliers": "tune_protected_outliers",
    "bias_categories": "bias_categories",
    "lora_bias_categories": "bias_categories",
    "loss_type": "loss_type",
    "lora_loss_type": "loss_type",
    "use_dora": "use_dora",
    "lora_use_dora": "use_dora",
}


def _normalize_lora_schedule_item(*, key: str, value):
    if key == "rank":
        out = int(value)
        if out < 1:
            raise argparse.ArgumentTypeError("lora_schedule.rank must be >= 1.")
        return out
    if key == "alpha":
        out = float(value)
        if out <= 0:
            raise argparse.ArgumentTypeError("lora_schedule.alpha must be > 0.")
        return out
    if key == "dropout":
        out = float(value)
        if out < 0:
            raise argparse.ArgumentTypeError("lora_schedule.dropout must be >= 0.")
        return out
    if key == "steps":
        out = int(value)
        if out < 0:
            raise argparse.ArgumentTypeError("lora_schedule.steps must be >= 0.")
        return out
    if key in {"batch_size", "nsamples", "log_every"}:
        out = int(value)
        if out < 1:
            raise argparse.ArgumentTypeError(f"lora_schedule.{key} must be >= 1.")
        return out
    if key in {"lr", "weight_decay"}:
        return float(value)
    if key in {"tune_norm", "tune_lm_head", "tune_bias", "tune_protected_outliers", "use_dora"}:
        return _parse_bool_like(value, arg_name=f"lora_schedule.{key}")
    if key == "bias_categories":
        return _parse_csv_like_names(value, arg_name="lora_schedule.bias_categories")
    if key == "loss_type":
        return _parse_lora_loss_type(str(value))
    raise argparse.ArgumentTypeError(f"Unsupported lora_schedule key: {key}")


def _parse_lora_schedule(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        raw_obj = value
    else:
        raw = str(value).strip()
        if not raw:
            return {}
        try:
            raw_obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentTypeError(
                "Invalid --lora_schedule JSON. Example: "
                '\'{"default":{"rank":8},"q_proj":{"rank":8,"alpha":16,"steps":1000,"loss_type":"sft"}}\''
            ) from e
    if not isinstance(raw_obj, dict):
        raise argparse.ArgumentTypeError("--lora_schedule must be a JSON object/dict.")
    out: Dict[str, Dict[str, object]] = {}
    for category, cfg in raw_obj.items():
        category_key = str(category).strip()
        if not category_key:
            raise argparse.ArgumentTypeError("Invalid --lora_schedule: empty category key.")
        if not isinstance(cfg, dict):
            raise argparse.ArgumentTypeError(
                f"Invalid --lora_schedule[{category_key}]: each value must be an object/dict."
            )
        normalized_cfg: Dict[str, object] = {}
        for raw_key, raw_value in cfg.items():
            key = _LORA_SCHEDULE_KEY_ALIASES.get(str(raw_key).strip().lower())
            if key is None:
                valid_keys = ",".join(sorted(_LORA_SCHEDULE_KEY_ALIASES.keys()))
                raise argparse.ArgumentTypeError(
                    f"Invalid --lora_schedule key '{raw_key}' in category '{category_key}'. "
                    f"Supported keys: {valid_keys}"
                )
            normalized_cfg[key] = _normalize_lora_schedule_item(key=key, value=raw_value)
        out[category_key] = normalized_cfg
    return out


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


def _parse_ratio_like(value, *, arg_name: str) -> float:
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{value}'. Expected float.")
    try:
        out = float(value)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{value}'. Expected float.") from e
    if out < 0.0 or out >= 1.0:
        raise argparse.ArgumentTypeError(f"{arg_name} must satisfy 0.0 <= value < 1.0, got {out}.")
    return float(out)


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


def _parse_dual_choice_or_scalar(
    value,
    *,
    arg_name: str,
    choices: Sequence[str],
):
    def _parse_single(item):
        return _parse_choice_like(item, arg_name=arg_name, choices=choices)

    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) == 0:
            raise argparse.ArgumentTypeError(f"{arg_name} cannot be empty.")
        if len(items) == 1:
            return _parse_single(items[0])
        if len(items) == 2:
            return (_parse_single(items[0]), _parse_single(items[1]))
        raise argparse.ArgumentTypeError(
            f"Invalid {arg_name} value '{value}'. "
            "Expected one mode or two modes (row_mode,col_mode)."
        )

    raw = str(value).strip()
    if not raw:
        raise argparse.ArgumentTypeError(f"{arg_name} cannot be empty.")
    raw = raw.replace("，", ",")
    if "," not in raw:
        return _parse_single(raw)
    items = [p.strip() for p in raw.split(",") if p.strip()]
    if len(items) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid {arg_name} value '{value}'. "
            "Expected one mode or two comma-separated modes (row_mode,col_mode)."
        )
    return (_parse_single(items[0]), _parse_single(items[1]))


def _parse_dual_choice_or_stage_list(
    value,
    *,
    arg_name: str,
    choices: Sequence[str],
):
    return _parse_stage_list_or_scalar(
        value,
        arg_name=arg_name,
        item_parser=lambda v: _parse_dual_choice_or_scalar(v, arg_name=arg_name, choices=choices),
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

    out: Dict[str, int] = {}
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


def resolve_stage_value(value, stage_idx: int, *, arg_name: str):
    if not isinstance(value, (list, tuple)):
        return value
    values = list(value)
    if len(values) == 0:
        raise ValueError(f"{arg_name} list cannot be empty.")
    idx = int(stage_idx)
    if idx < 0:
        raise ValueError(f"{arg_name} stage_idx must be >=0, got {idx}")
    if len(values) == 1:
        return values[0]
    if idx >= len(values):
        raise ValueError(
            f"{arg_name} list length {len(values)} is smaller than required stage index {idx}."
        )
    return values[idx]


def _resolve_category_override(value, category: str, *, arg_name: str):
    if isinstance(value, dict):
        category_key = str(category)
        if category_key in value:
            return value[category_key]
        if "default" in value:
            return value["default"]
        if "*" in value:
            return value["*"]
        keys = sorted(str(k) for k in value.keys())
        raise ValueError(
            f"{arg_name} dict does not contain category '{category_key}', "
            "and missing fallback key 'default' or '*'. "
            f"Available keys: {keys}"
        )
    return value


def _parse_intra_parallel(value: str):
    def _parse_positive_int(token, *, raw_value: str) -> int:
        try:
            out = int(token)
        except (TypeError, ValueError) as e:
            raise argparse.ArgumentTypeError(
                f"Invalid --intra_parallel value '{raw_value}'. "
                "Expected int, two ints like 2,4, or dict like "
                '\'{"default":[2,1],"q_proj":[4,1]}\'.'
            ) from e
        if out < 1:
            raise argparse.ArgumentTypeError(
                f"--intra_parallel must be >= 1, got {out}."
            )
        return out

    def _parse_intra_parallel_scalar(raw_value: str):
        raw = str(raw_value).strip()
        if not raw:
            raise argparse.ArgumentTypeError("--intra_parallel cannot be empty.")
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1].strip()

        if "," not in raw:
            return _parse_positive_int(raw, raw_value=raw_value)

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid --intra_parallel value '{raw_value}'. "
                "Expected int, two ints like 2,4, or dict like "
                '\'{"default":[2,1],"q_proj":[4,1]}\'.'
            )
        row_parts = _parse_positive_int(parts[0], raw_value=raw_value)
        col_parts = _parse_positive_int(parts[1], raw_value=raw_value)
        return (row_parts, col_parts)

    def _parse_intra_parallel_item(item, *, raw_value: str):
        if isinstance(item, int):
            return _parse_positive_int(item, raw_value=raw_value)
        if isinstance(item, (list, tuple)):
            values = [_parse_positive_int(v, raw_value=raw_value) for v in item]
            if len(values) == 1:
                return int(values[0])
            if len(values) == 2:
                return int(values[0]), int(values[1])
            raise argparse.ArgumentTypeError(
                f"Invalid --intra_parallel value '{raw_value}'. "
                "List/tuple must contain 1 or 2 integers."
            )
        return _parse_intra_parallel_scalar(str(item))

    raw = str(value).strip()
    if not raw:
        raise argparse.ArgumentTypeError("--intra_parallel cannot be empty.")

    if raw.startswith("{") and raw.endswith("}"):
        try:
            parsed_obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid --intra_parallel dict '{value}'. "
                "Please pass valid JSON, for example: "
                '\'{"default":[2,1],"q_proj":[4,1]}\'.'
            ) from e
        if not isinstance(parsed_obj, dict):
            raise argparse.ArgumentTypeError(
                f"Invalid --intra_parallel value '{value}'. "
                "JSON form must be an object/dict."
            )
        if not parsed_obj:
            raise argparse.ArgumentTypeError(
                "--intra_parallel dict cannot be empty."
            )
        out: Dict[str, Union[int, Tuple[int, int]]] = {}
        for k, v in parsed_obj.items():
            key = str(k).strip()
            if not key:
                raise argparse.ArgumentTypeError(
                    f"Invalid --intra_parallel key in '{value}': key cannot be empty."
                )
            out[key] = _parse_intra_parallel_item(v, raw_value=f"{value}[{k}]")
        return out

    return _parse_intra_parallel_scalar(raw)


def resolve_intra_parallel(value) -> Tuple[int, int]:
    if isinstance(value, dict):
        keys = sorted(str(k) for k in value.keys())
        raise ValueError(
            "intra_parallel dict requires category context. "
            "Use resolve_intra_parallel_for_category(value, category). "
            f"Available keys: {keys}"
        )

    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"intra_parallel must be >= 1, got {value}")
        return int(value), 1

    if isinstance(value, (list, tuple)):
        items = [int(v) for v in value]
        if len(items) == 1:
            if items[0] < 1:
                raise ValueError(f"intra_parallel must be >= 1, got {items[0]}")
            return int(items[0]), 1
        if len(items) == 2:
            if items[0] < 1 or items[1] < 1:
                raise ValueError(f"intra_parallel factors must be >= 1, got {items}")
            return int(items[0]), int(items[1])
        raise ValueError(
            f"intra_parallel list/tuple must have length 1 or 2, got {len(items)}"
        )

    parsed = _parse_intra_parallel(str(value))
    if isinstance(parsed, int):
        return parsed, 1
    return int(parsed[0]), int(parsed[1])


def resolve_intra_parallel_for_category(
    value,
    category: str,
) -> Tuple[int, int]:
    selected = _resolve_category_override(value, category, arg_name="intra_parallel")
    return resolve_intra_parallel(selected)


def resolve_codebook_int_for_category(
    value,
    category: str,
    *,
    arg_name: str,
) -> int:
    selected = _resolve_category_override(value, category, arg_name=arg_name)
    return _parse_positive_int_like(selected, arg_name=arg_name)


def resolve_lora_schedule_for_category(
    schedule,
    category: Optional[str],
) -> Dict[str, object]:
    if not isinstance(schedule, dict) or not schedule:
        return {}
    out: Dict[str, object] = {}
    for fallback_key in ("default", "*"):
        fallback_cfg = schedule.get(fallback_key)
        if isinstance(fallback_cfg, dict):
            out.update(fallback_cfg)
    if category is not None:
        cat_cfg = schedule.get(str(category))
        if isinstance(cat_cfg, dict):
            out.update(cat_cfg)
    return out


def intra_parallel_total(value) -> int:
    row_parts, col_parts = resolve_intra_parallel(value)
    return int(row_parts) * int(col_parts)


def _resolve_positive_int(value, *, default: int, arg_name: str, allow_zero: bool = False) -> int:
    out = default if value is None else int(value)
    min_value = 0 if allow_zero else 1
    if out < min_value:
        raise ValueError(f"{arg_name} must be >= {min_value}, got {out}")
    return int(out)


def resolve_autoencoder_arch_args(args) -> None:
    dynamic_fields = (
        "base_ch",
        "num_res_blocks",
        "decoder_type",
        "decoder_base_ch",
        "decoder_num_res_blocks",
    )
    if any(isinstance(getattr(args, k, None), (list, tuple)) for k in dynamic_fields):
        # Stage-wise overrides are resolved later (e.g. in cat_train per residual stage).
        return

    shared_base_ch = _resolve_positive_int(
        getattr(args, "base_ch", 128),
        default=128,
        arg_name="--base_ch",
        allow_zero=False,
    )
    shared_num_res_blocks = _resolve_positive_int(
        getattr(args, "num_res_blocks", 1),
        default=1,
        arg_name="--num_res_blocks",
        allow_zero=True,
    )

    decoder_type = str(getattr(args, "decoder_type", "linear")).strip().lower()
    if decoder_type == "asymmetric":
        decoder_base_ch = _resolve_positive_int(
            getattr(args, "decoder_base_ch", None),
            default=shared_base_ch,
            arg_name="--decoder_base_ch",
            allow_zero=False,
        )
        decoder_num_res_blocks = _resolve_positive_int(
            getattr(args, "decoder_num_res_blocks", None),
            default=shared_num_res_blocks,
            arg_name="--decoder_num_res_blocks",
            allow_zero=True,
        )
    else:
        # linear/symmetric 都沿用共享结构，保持历史行为兼容。
        decoder_base_ch = int(shared_base_ch)
        decoder_num_res_blocks = int(shared_num_res_blocks)

    # Encoder 始终使用共享参数，保留这两个派生字段便于日志查看。
    setattr(args, "encoder_base_ch", int(shared_base_ch))
    setattr(args, "encoder_num_res_blocks", int(shared_num_res_blocks))
    setattr(args, "decoder_base_ch", int(decoder_base_ch))
    setattr(args, "decoder_num_res_blocks", int(decoder_num_res_blocks))


def add_llm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default='adamw', choices=['adam', 'adamw', 'sgd', 'rmsprop'])
    parser.add_argument("--lr_scheduler", type=str, default='none', choices=['none', 'linear', 'cosine'],
                        help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Warmup steps for scheduler")

    # Training Specific
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Path or HuggingFace ID of the LLM")

    # Data Preprocessing
    parser.add_argument("--normalize_weight", action="store_true",
                        help="Normalize weight (z-score) before training")

    recon_loss_choices = ['mse', 'l1', 'huber', 'relative_l1', 'top_k_mse', 'cosine', 'w_mse', 'w2_mse', 'wa_mse']
    parser.add_argument(
        "--recon_loss_type",
        type=lambda v: _parse_choice_or_stage_list(v, arg_name="--recon_loss_type", choices=recon_loss_choices),
        default='mse',
        help="Type of reconstruction loss to use. Supports scalar or JSON list for residual stages.",
    )
    parser.add_argument("--distil_loss_type", type=str, default='mse',
                        choices=['mse', 'none'],
                        help="Type of distillation loss to use between original and reconstructed weights")
    parser.add_argument("--distil_loss_weight", type=float, default=1.0,
                        help="Weight of the distillation loss")
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--lfq_weight", type=float, default=1.0)
    parser.add_argument("--commitment_loss_weight", type=float, default=0.25)
    parser.add_argument("--entropy_loss_weight", type=float, default=0.1)
    parser.add_argument("--diversity_gamma", type=float, default=1.0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--new_quant", action="store_true")
    parser.add_argument("--w_input_batches", type=int, default=1,
                        help="Split w_input into this many batches for VAE forward to reduce peak memory.")
    return parser


def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--codebook_bits",
        type=lambda v: _parse_positive_int_or_category_schedule_or_stage_list(v, arg_name="--codebook_bits"),
        default=16,
    )  # 2^16 -> 16 bits
    parser.add_argument(
        "--codebook_dim",
        type=lambda v: _parse_positive_int_or_category_schedule_or_stage_list(v, arg_name="--codebook_dim"),
        default=8,
    )  # 这时候它代表 Input Chunk Size
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

    # BSQ / Quantizer 相关参数
    parser.add_argument("--quantizer_type", type=str, default='BSQ')
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--inv_temperature", type=float, default=100.0)

    parser.add_argument(
        "--norm_type",
        type=lambda v: _parse_choice_or_stage_list(v, arg_name="--norm_type", choices=['group', 'batch', 'layer', 'no']),
        default='group',
    )
    parser.add_argument(
        "--decoder_type",
        type=lambda v: _parse_choice_or_stage_list(
            v,
            arg_name="--decoder_type",
            choices=['linear', 'symmetric', 'asymmetric'],
        ),
        default='linear',
    )

    # Multi-Layer Training
    parser.add_argument("--parallel_layers", type=int, default=32, help="Number of layers to train in parallel")

    return parser


def add_lbl_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--layer_indices", type=str, default=None)
    parser.add_argument("--steps_per_layer", type=int, default=None)
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--layer_checkpointing", action="store_true")
    parser.add_argument("--use_output_mse_loss", action="store_true")
    parser.add_argument("--output_mse_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--weight_only",
        action="store_true",
        help="Train only VAE weight recon/commitment losses (skip calibration data forward).",
    )
    parser.add_argument(
        "--skip_ppl_eval",
        action="store_true",
        help="Skip PPL evaluation after each trained layer.",
    )
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    return parser


def parse_lbl_args(argv):
    parser = add_lbl_args(argparse.ArgumentParser(add_help=False))
    return parser.parse_known_args(argv)


def process_args_from(argv):
    parser = argparse.ArgumentParser()
    # 添加模型和LLM相关参数
    parser = add_model_specific_args(parser)
    parser = add_llm_args(parser)
    vae_args, unknown_args = parser.parse_known_args(argv)
    resolve_autoencoder_arch_args(vae_args)
    parser = transformers.HfArgumentParser((HFArguments, TrainingArguments))
    hf_args, training_args = parser.parse_args_into_dataclasses(args=unknown_args)
    use_bf16 = bool(training_args.bf16)
    vae_args.vae_weight_dtype = "bf16" if use_bf16 else "fp32"
    vae_args.vae_autocast_dtype = "bf16" if use_bf16 else "fp32"
    return hf_args, training_args, vae_args


def process_args():
    return process_args_from(None)


def process_all_args(argv):
    lbl_args, remaining = parse_lbl_args(argv)
    hf_args, training_args, vae_args = process_args_from(remaining)
    return lbl_args, hf_args, training_args, vae_args


def build_cat_train_parser() -> argparse.ArgumentParser:
    # 给 tools/cat_train.py 使用的脚本层参数解析器（不含 HF/Training/vae 通用参数）。
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_order", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--transpose_modules", type=str, default="v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument(
        "--projection_suffixes",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="开启 --only_decoder_projections 时，允许参与训练的投影层后缀列表。",
    )
    parser.add_argument(
        "--only_decoder_projections",
        action="store_true",
        default=True,
        help="仅处理 decoder layers 中的投影层 Linear（推荐）。",
    )
    parser.add_argument(
        "--include_all_linears",
        action="store_true",
        default=False,
        help="覆盖 --only_decoder_projections，改为包含模型中全部 nn.Linear。",
    )
    parser.add_argument(
        "--steps_per_category",
        type=lambda v: _parse_positive_int_or_stage_list(v, arg_name="--steps_per_category"),
        default=2000,
    )
    parser.add_argument(
        "--steps_per_group",
        type=lambda v: _parse_positive_int_or_stage_list(v, arg_name="--steps_per_group"),
        default=None,
        help="分组模式下覆盖 steps_per_category。支持标量或 JSON list（按 residual stage）。",
    )
    parser.add_argument(
        "--skip_layers",
        type=str,
        default="",
        help="指定在 LLM 前向中始终使用原始线性权重的层，格式: layer_idx.category，例如 0.down_proj,30.q_proj。",
    )
    parser.add_argument(
        "--linear_group_size",
        type=int,
        default=32,
        help="跨层分组大小：每组同时训练多少个同类 Linear。",
    )
    parser.add_argument(
        "--intra_parallel",
        type=_parse_intra_parallel,
        default=1,
        help=(
            "层内并行切分数。支持两种格式："
            "1) 单个整数 n：仅在主维度切分（与旧行为一致）；"
            "2) 两个整数 a,b：先按主维度切分 a 份，再按另一维切分 b 份；"
            "3) JSON dict：按类别指定，例如 "
            '\'{"default":[2,1],"q_proj":[4,1],"k_proj":2}\'。'
        ),
    )
    parser.add_argument(
        "--intra_part_sort_mode",
        type=lambda v: _parse_dual_choice_or_stage_list(
            v,
            arg_name="--intra_part_sort_mode",
            choices=["none", "l2", "act_l2"],
        ),
        default="l2",
        help=(
            "intra_parallel>1 时切分前的排序模式（转置后生效）："
            "支持单值 mode（两维同模式）或双值 row_mode,col_mode（两维独立）。"
            "可选模式：l2=按启用维度的L2范数排序并按codebook_dim蛇形交错分配，"
            "act_l2=先乘act_max再按启用维度的L2范数排序并按codebook_dim蛇形交错分配，"
            "none=不排序。"
            "示例：l2,l2 或 l2,none。"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_blocks", type=int, default=256)
    parser.add_argument(
        "--activation_weight_path",
        type=str,
        default=None,
        help="Path to activation abs-max dict (*.pt) for --recon_loss_type wa_mse.",
    )
    parser.add_argument(
        "--outlier_protect_ratio",
        type=lambda v: _parse_ratio_like(v, arg_name="--outlier_protect_ratio"),
        default=0.0,
        help="Protect top floor(channel_dim * ratio) channels from VAE compression using act-weighted weight norms.",
    )
    parser.add_argument(
        "--outlier_protect_axis",
        type=str,
        choices=["input", "output"],
        default="input",
        help="Choose whether outlier protection preserves input channels or output channels.",
    )
    parser.add_argument(
        "--wa_mse_act_mode",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="act_max source for wa_mse: dynamic (recompute each group/stage) or static (use --activation_weight_path).",
    )
    parser.add_argument(
        "--wa_mse_calib_dataset",
        type=str,
        default="wikitext2",
        help="Calibration dataset used for dynamic wa_mse act-max recomputation.",
    )
    parser.add_argument(
        "--wa_mse_calib_nsamples",
        type=int,
        default=512,
        help="Calibration sample count used for dynamic wa_mse act-max recomputation.",
    )
    parser.add_argument(
        "--wa_mse_calib_seqlen",
        type=int,
        default=512,
        help="Calibration sequence length used for dynamic wa_mse act-max recomputation.",
    )
    parser.add_argument(
        "--wa_mse_calib_seed",
        type=int,
        default=0,
        help="Calibration sampling seed used for dynamic wa_mse act-max recomputation.",
    )
    parser.add_argument(
        "--wa_mse_calib_device",
        type=str,
        default="",
        help="Device for dynamic wa_mse act-max recomputation. Empty means use --train_device.",
    )
    parser.add_argument(
        "--wa_mse_calib_log_every",
        type=int,
        default=0,
        help="Log interval for dynamic wa_mse act-max recomputation progress (0 to disable).",
    )
    parser.add_argument("--ppl_limit", type=int, default=-1, help="每类训练后 PPL 评估样本上限，-1 为全量。")
    parser.add_argument("--lora_after_category", action="store_true", help="每个类别 VAE 训练后，对剩余类别做一次 LoRA 微调并融合。")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_steps", type=int, default=50)
    parser.add_argument("--lora_batch_size", type=int, default=2)
    parser.add_argument("--lora_nsamples", type=int, default=128)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--lora_weight_decay", type=float, default=0.0)
    parser.add_argument("--lora_log_every", type=int, default=1)
    parser.add_argument(
        "--lora_tune_norm",
        action="store_true",
        default=False,
        help="LoRA 微调时同时训练 norm 参数。",
    )
    parser.add_argument(
        "--lora_tune_lm_head",
        action="store_true",
        default=False,
        help="LoRA 微调时把 lm_head 也加入 LoRA 目标模块。",
    )
    parser.add_argument(
        "--lora_tune_bias",
        type=lambda v: _parse_bool_like(v, arg_name="--lora_tune_bias"),
        default=False,
        help="LoRA 微调时是否额外训练选中 Linear 的 bias（支持 true/false）。默认 false。",
    )
    parser.add_argument(
        "--lora_tune_protected_outliers",
        type=lambda v: _parse_bool_like(v, arg_name="--lora_tune_protected_outliers"),
        default=False,
        help="LoRA 微调时是否额外训练 VAELinear 中被保护的 outlier 权重切片（支持 true/false）。默认 false。",
    )
    parser.add_argument(
        "--lora_bias_categories",
        type=lambda v: _parse_csv_like_names(v, arg_name="--lora_bias_categories"),
        default=[],
        help="允许训练 bias 的 Linear 类别列表（逗号分隔或 JSON 列表）。为空时默认覆盖全部 LoRA 目标 Linear。",
    )
    parser.add_argument(
        "--lora_loss_type",
        type=_parse_lora_loss_type,
        default="sft",
        help="LoRA 损失类型。支持：sft/origin/rkl/kl/mse/kd/r_kl_top[_K]/kl_top[_K]。",
    )
    parser.add_argument(
        "--lora_use_dora",
        type=lambda v: _parse_bool_like(v, arg_name="--lora_use_dora"),
        default=True,
        help="LoRA 微调时是否开启 DoRA（支持 true/false）。默认 true。",
    )
    parser.add_argument(
        "--lora_schedule",
        type=_parse_lora_schedule,
        default=None,
        help=(
            "按“已完成类别”覆盖 LoRA 超参的 JSON。"
            '示例: {"q_proj":{"rank":8,"alpha":16,"steps":1000,"loss_type":"sft","use_dora":false},'
            '"k_proj":{"rank":128,"alpha":256,"steps":2000,"loss_type":"r_kl_top_1000","use_dora":true}}。'
            "支持 default/* 作为兜底键。"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument(
        "--rot_llm",
        action="store_true",
        default=False,
        help="在 VAE 压缩前先对基座 LLM 执行一次离线旋转融合。",
    )
    parser.add_argument("--convert", action="store_true",
                        help="每个类别训练完成后，将 Linear 替换为压缩后的线性层。")
    parser.add_argument("--convert_device", type=str, default="cuda")
    parser.add_argument("--save_model", action="store_true",
                        help="保存最终模型 state_dict/config/tokenizer（需要 --convert）。")
    parser.add_argument(
        "--unload_vae_original_weights_on_final_save",
        action="store_true",
        default=False,
        help="最终保存前卸载 VAELinear 中缓存的原始 Linear 权重，减小保存体积。",
    )
    parser.add_argument("--output_dir", type=str, default="./output_linear_by_category")
    parser.add_argument(
        "--allow_tail_group",
        action="store_true",
        default=True,
        help="允许处理最后一个不足分组大小的尾部分组。",
    )
    return parser


def process_cat_train_args(argv: Optional[Sequence[str]]):
    # 给 tools/cat_train.py 使用：先解析脚本私有参数，再把剩余参数交给 process_args_from。
    if argv is None:
        import sys
        argv = sys.argv[1:]
    parser = build_cat_train_parser()
    script_args, remaining = parser.parse_known_args(list(argv))
    hf_args, training_args, vae_args = process_args_from(remaining)
    return script_args, hf_args, training_args, vae_args


def create_optimizer(params, args, lr):
    opt_name = args.optimizer.lower()
    if opt_name == 'adam':
        return torch.optim.Adam(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif opt_name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
