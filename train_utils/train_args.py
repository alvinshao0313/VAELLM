
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch
import argparse
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

    parser.add_argument("--recon_loss_type", type=str, default='mse',
                        choices=['mse', 'l1', 'huber',
                                 'relative_l1', 'top_k_mse'],
                        help="Type of reconstruction loss to use")
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
    parser.add_argument("--codebook_bits", type=int, default=16)  # 2^16 -> 16 bits
    parser.add_argument("--codebook_dim", type=int, default=8)  # 这时候它代表 Input Chunk Size

    parser.add_argument("--base_ch", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=1)

    # BSQ / Quantizer 相关参数
    parser.add_argument("--quantizer_type", type=str, default='BSQ')
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--inv_temperature", type=float, default=100.0)

    parser.add_argument("--norm_type", type=str, default='group', choices=['group', 'batch', 'layer', 'no'])
    parser.add_argument("--decoder_type", type=str, default='linear', choices=['linear', 'symmetric'])

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
