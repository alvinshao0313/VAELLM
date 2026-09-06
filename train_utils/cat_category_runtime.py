"""Canonical CAT category runtime views derived from the common configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from train_utils.config.cli import CatCLIConfig


@dataclass
class CatTrainHFTrainingArguments:
    distill_model_max_length: int = field(default=2048)
    distill_dynamic_padding: bool = field(default=False)
    distill_gradient_accumulation_steps: int = field(default=1)
    distill_optim: str = field(default="paged_adamw_8bit")
    distill_max_grad_norm: float = field(default=0.3)
    distill_warmup_ratio: float = field(default=0.3)
    distill_group_by_length: bool = field(default=True)
    distill_lr_scheduler_type: str = field(default="linear")
    distill_gradient_checkpointing: bool = field(default=False)
    distill_gradient_checkpointing_kwargs: Optional[str] = field(default=None)
    distill_hif4_act: bool = field(default=False)
    distill_teacher_logits_cpu_staging: bool = field(default=True)
    distill_selective_student_topk: bool = field(default=False)
    distill_selective_student_topk_chunk_rows: int = field(default=32)
    distill_teacher_model_offload: str = field(default="none")
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=500)
    save_total_limit: Optional[int] = field(default=None)
    save_only_model: bool = field(default=False)
    ignore_data_skip: bool = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)


@dataclass(frozen=True)
class ResolvedCategoryRuntimeConfig:
    category: str
    residual_stages: int
    steps: int
    intra_part_sort_mode: str
    codebook_bits: int
    codebook_dim: int
    intra_parallel: Tuple[int, int]
    channel_protect_count: int
    recon_loss_type: str
    base_ch: int
    num_res_blocks: int
    decoder_base_ch: Optional[int]
    decoder_num_res_blocks: Optional[int]
    norm_type: str
    activation_type: str
    decoder_type: str


def resolve_category_runtime_configs(
    cat_args,
    vae_args,
    active_categories: Sequence[str],
) -> Dict[str, ResolvedCategoryRuntimeConfig]:
    del vae_args
    cfg = getattr(cat_args, "_common_cat_config", None)
    if not isinstance(cfg, CatCLIConfig):
        raise TypeError("CAT runtime requires a canonical CatCLIConfig.")
    resolved: Dict[str, ResolvedCategoryRuntimeConfig] = {}
    for category in active_categories:
        compression, opt = cfg.resolve_category_config(str(category))
        channel_count = compression.channel.channel_protect_count
        resolved[str(category)] = ResolvedCategoryRuntimeConfig(
            category=str(category),
            residual_stages=int(compression.core.residual_stages),
            steps=int(opt.vae_steps),
            intra_part_sort_mode="none",
            codebook_bits=int(compression.core.codebook_bits),
            codebook_dim=int(compression.core.codebook_dim),
            intra_parallel=tuple(int(v) for v in compression.core.intra_parallel),
            channel_protect_count=(
                0 if isinstance(channel_count, float) else int(channel_count)
            ),
            recon_loss_type=str(compression.recon_loss_type),
            base_ch=int(compression.core.base_ch),
            num_res_blocks=int(compression.core.num_res_blocks),
            decoder_base_ch=compression.decoder.decoder_base_ch,
            decoder_num_res_blocks=compression.decoder.decoder_num_res_blocks,
            norm_type=str(compression.decoder.norm_type),
            activation_type=str(compression.decoder.activation_type),
            decoder_type=str(compression.decoder.decoder_type),
        )
    return resolved


__all__ = [
    "CatTrainHFTrainingArguments",
    "ResolvedCategoryRuntimeConfig",
    "resolve_category_runtime_configs",
]
