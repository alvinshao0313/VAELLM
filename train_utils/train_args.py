
import argparse
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

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
                                 'relative_l1', 'top_k_mse', 'cosine', 'w_mse', 'w2_mse'],
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
    parser.add_argument("--steps_per_category", type=int, default=2000)
    parser.add_argument("--steps_per_group", type=int, default=None, help="分组模式下覆盖 steps_per_category。")
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
        type=int,
        default=1,
        help="层内并行切分数：每个 Linear 再切成多少份并行训练。",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_blocks", type=int, default=256)
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
        "--lora_loss_type",
        type=_parse_lora_loss_type,
        default="sft",
        help="LoRA 损失类型。支持：sft/origin/rkl/kl/mse/kd/r_kl_top[_K]/kl_top[_K]。",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_device", type=str, default="cuda")
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
