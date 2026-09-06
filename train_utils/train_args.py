from dataclasses import dataclass, field
from typing import Optional

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


def create_optimizer(params, args, lr):
    opt_name = args.optimizer.lower()
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=args.beta1, weight_decay=args.weight_decay)
    if opt_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")
