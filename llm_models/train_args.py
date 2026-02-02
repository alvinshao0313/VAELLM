
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch
import argparse
import transformers
from bitvae.models.llm_vae import AutoEncoder
from bitvae.utils.llm_arguments import LLMArgs


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


def process_args():
    parser = argparse.ArgumentParser()
    # 添加模型和LLM相关参数
    parser = AutoEncoder.add_model_specific_args(parser)
    parser = LLMArgs.add_llm_args(parser)
    vae_args, unknown_args = parser.parse_known_args()
    parser = transformers.HfArgumentParser((HFArguments, TrainingArguments))
    hf_args, training_args = parser.parse_args_into_dataclasses(args=unknown_args)
    return hf_args, training_args, vae_args


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
