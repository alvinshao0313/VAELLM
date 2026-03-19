import os
from typing import Dict, List, Optional

import psutil
import torch

from train_utils.data_utils import get_wikitext2
from train_utils.model_checkpoint_io import META_FILENAME


def resolve_checkpoint_dir(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        if os.path.basename(abs_path) == META_FILENAME:
            return os.path.dirname(abs_path)
        raise FileNotFoundError(f"Expected {META_FILENAME} file, got: {abs_path}")

    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Path does not exist: {abs_path}")

    direct_meta = os.path.join(abs_path, META_FILENAME)
    if os.path.exists(direct_meta):
        return abs_path

    final_model_meta = os.path.join(abs_path, "final_model", META_FILENAME)
    if os.path.exists(final_model_meta):
        return os.path.join(abs_path, "final_model")

    candidates: List[str] = []
    for child in os.listdir(abs_path):
        child_dir = os.path.join(abs_path, child)
        if not os.path.isdir(child_dir):
            continue
        if os.path.exists(os.path.join(child_dir, META_FILENAME)):
            candidates.append(child_dir)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot find checkpoint metadata under: {abs_path}. "
        f"Please pass a directory containing {META_FILENAME}."
    )


def resolve_distill_device(device: str) -> str:
    d = str(device).strip()
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return d


def resolve_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def get_available_cpu_memory() -> int:
    return int(psutil.virtual_memory().available)


def get_base_model(model):
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise NotImplementedError(f"Unsupported model type without `.model`: {type(model)}")
    required_attrs = ("layers", "embed_tokens", "rotary_emb", "_update_causal_mask")
    for attr in required_attrs:
        if not hasattr(base_model, attr):
            raise NotImplementedError(f"Unsupported decoder model missing `{attr}`: {type(base_model)}")
    return base_model


def estimate_layer_cache_bytes(
    *,
    num_samples: int,
    seqlen: int,
    hidden_size: int,
    num_attention_heads: int,
    cache_dtype: torch.dtype,
    teacher_label_dtype: torch.dtype,
    extra_teacher_out: bool = False,
) -> Dict[str, int]:
    cache_bytes = torch.tensor([], dtype=cache_dtype).element_size()
    label_bytes = torch.tensor([], dtype=teacher_label_dtype).element_size()

    teacher_hidden = int(num_samples) * int(seqlen) * int(hidden_size) * cache_bytes
    student_hidden = int(num_samples) * int(seqlen) * int(hidden_size) * cache_bytes
    teacher_out = int(num_samples) * int(seqlen) * int(hidden_size) * cache_bytes
    teacher_aug_out = teacher_out if bool(extra_teacher_out) else 0
    teacher_attn = int(num_samples) * int(num_attention_heads) * int(seqlen) * int(seqlen) * label_bytes
    teacher_attn_mean = int(num_samples) * int(hidden_size) * label_bytes
    total = teacher_hidden + student_hidden + teacher_out + teacher_aug_out + teacher_attn + teacher_attn_mean

    return {
        "teacher_hidden_cpu": int(teacher_hidden),
        "student_hidden_cpu": int(student_hidden),
        "teacher_out_cpu": int(teacher_out),
        "teacher_aug_out_cpu": int(teacher_aug_out),
        "teacher_attn_cpu": int(teacher_attn),
        "teacher_attn_mean_cpu": int(teacher_attn_mean),
        "total": int(total),
        "available": int(get_available_cpu_memory()),
    }


def build_shared_layer0_inputs(
    *,
    model,
    input_ids: torch.Tensor,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    base_model = get_base_model(model)
    with torch.inference_mode():
        hidden_states = base_model.embed_tokens(input_ids)
    return hidden_states.to(device="cpu", dtype=cache_dtype).contiguous()


def build_layer_runtime_kwargs(
    *,
    model,
    hidden_states: torch.Tensor,
    output_attentions: bool,
) -> Dict[str, torch.Tensor]:
    base_model = get_base_model(model)
    seq_len = int(hidden_states.shape[1])
    batch_size = int(hidden_states.shape[0])
    cache_position = torch.arange(seq_len, device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
    attention_mask = base_model._update_causal_mask(
        None,
        hidden_states,
        cache_position,
        None,
        bool(output_attentions),
    )
    position_embeddings = base_model.rotary_emb(hidden_states, position_ids)
    return {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "position_embeddings": position_embeddings,
    }


def collect_calib_inputs(
    *,
    model_path: str,
    nsamples: int,
    seed: int,
    seqlen: int,
    access_token: Optional[str] = None,
) -> torch.Tensor:
    tokenizer = None
    if access_token:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            token=access_token,
        )
    samples = get_wikitext2(
        nsamples=int(nsamples),
        seed=int(seed),
        seqlen=int(seqlen),
        model=str(model_path),
        tokenizer=tokenizer,
    )
    if not samples:
        raise RuntimeError("Empty calibration set.")
    return torch.cat([item[0] for item in samples], dim=0).contiguous()
