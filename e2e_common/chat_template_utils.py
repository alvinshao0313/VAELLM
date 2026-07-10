from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

_ROLE_PROBE_MARKER = "VAELLM_ROLE_PROBE_MARKER"


def infer_chat_family(*, model_path: str, model_type: Optional[str] = None) -> str:
    normalized_type = str(model_type or "").strip().lower()
    if normalized_type in {"qwen", "qwen2", "qwen3", "qwen2_vl", "qwen2_moe", "qwen3_moe"}:
        return "qwen"
    if normalized_type in {"llama", "mistral"}:
        return "llama"

    normalized_path = str(model_path).strip().lower()
    if "llama" in normalized_path or "meta-llama" in normalized_path:
        return "llama"
    if "qwen" in normalized_path:
        return "qwen"
    raise ValueError(
        f"Cannot infer chat family from model_path={model_path!r} and model_type={model_type!r}. "
        "Expected a Qwen or Llama model."
    )


def render_messages(
    messages: Sequence[Dict[str, Any]],
    tokenizer,
    *,
    add_generation_prompt: bool = False,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template; cannot render chat messages.")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer chat_template is not set. Use an instruct/chat checkpoint for edgerazor_messages."
        )
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
    )


def _extract_role_marker(tokenizer, role: str) -> str:
    rendered = render_messages([{"role": str(role), "content": _ROLE_PROBE_MARKER}], tokenizer)
    marker_idx = rendered.find(_ROLE_PROBE_MARKER)
    if marker_idx < 0:
        raise ValueError(
            f"Cannot locate role probe marker for role={role!r} in apply_chat_template output."
        )
    marker = rendered[:marker_idx]
    if not marker:
        raise ValueError(f"Inferred chat role marker for role={role!r} is empty.")
    return marker


def infer_user_instruction_template(tokenizer) -> str:
    return _extract_role_marker(tokenizer, "user")


def infer_assistant_response_template(tokenizer) -> str:
    return _extract_role_marker(tokenizer, "assistant")
