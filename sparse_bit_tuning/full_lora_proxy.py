"""Compatibility re-export of shared full-space LoRA helpers.

Canonical implementation lives in `e2e_common.full_lora`.
"""

from e2e_common.full_lora import (
    FullCompressedPeftProxy,
    PeftZeroLinearCarrier,
    assert_exact_adapter_target_set,
    build_full_compressed_peft_model,
    collect_logical_adapter_target_names,
    extract_full_proxy_low_rank_payloads,
    finalize_model_level_lora,
    initialize_full_proxy_lora_from_low_rank,
    iter_named_full_compressed_peft_proxies,
    unwrap_full_compressed_peft_proxies,
    wrap_full_compressed_peft_proxies,
)

__all__ = [
    "FullCompressedPeftProxy",
    "PeftZeroLinearCarrier",
    "assert_exact_adapter_target_set",
    "build_full_compressed_peft_model",
    "collect_logical_adapter_target_names",
    "extract_full_proxy_low_rank_payloads",
    "finalize_model_level_lora",
    "initialize_full_proxy_lora_from_low_rank",
    "iter_named_full_compressed_peft_proxies",
    "unwrap_full_compressed_peft_proxies",
    "wrap_full_compressed_peft_proxies",
]
