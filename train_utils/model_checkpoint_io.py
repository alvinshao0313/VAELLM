import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from litebsq.misc import set_module_by_name
from litebsq.llm_vae import Decoder
from litebsq.vae_linear import VAELinear
from rotation.model_utils import get_model


STATE_DICT_FILENAME = "pytorch_model.bin"
META_FILENAME = "checkpoint_meta.json"


def _safe_path_token(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown_model"
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/-]+", "_", value)
    value = value.replace("/", "__")
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "unknown_model"


def _build_run_output_dir(root_output_dir: str, model_path: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_tag = _safe_path_token(model_path)
    base_run_dir = os.path.join(root_output_dir, f"{model_tag}_{ts}")
    run_dir = base_run_dir
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = f"{base_run_dir}_{suffix}"
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


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


def _dtype_to_name(dtype: torch.dtype) -> str:
    text = str(dtype)
    if text.startswith("torch."):
        return text[len("torch."):]
    return text


def _name_to_dtype(name: str) -> torch.dtype:
    if not hasattr(torch, name):
        raise ValueError(f"Unknown torch dtype name: {name}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid torch dtype entry: {name}")
    return dtype


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    module: nn.Module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _build_unique_index_placeholder(shape: Sequence[int], *, dtype: torch.dtype, device) -> torch.Tensor:
    if len(shape) != 1:
        raise ValueError(f"Index placeholder shape must be 1D, got {tuple(shape)}")
    # VAELinear validates protected indices during construction before load_state_dict()
    # restores the real values, so the placeholder must already be unique and in range.
    return torch.arange(int(shape[0]), dtype=dtype, device=device)


def _decoder_to_spec(decoder: Decoder) -> Dict[str, Any]:
    if not isinstance(decoder, Decoder):
        raise TypeError(f"Expected Decoder, got {type(decoder)}")

    if decoder.decoder_type not in {"linear", "symmetric", "asymmetric"}:
        raise ValueError(f"Unsupported decoder_type: {decoder.decoder_type}")

    hidden_dim = int(getattr(decoder, "hidden_dim"))
    num_res_blocks = int(getattr(decoder, "num_res_blocks"))
    norm_type = str(getattr(decoder, "norm_type"))

    first_param = next(decoder.parameters(), None)
    param_dtype = _dtype_to_name(first_param.dtype) if first_param is not None else "float32"

    return {
        "in_dim": int(decoder.in_dim),
        "out_dim": int(decoder.out_dim),
        "hidden_dim": int(hidden_dim),
        "num_res_blocks": int(num_res_blocks),
        "norm_type": str(norm_type),
        "decoder_type": str(decoder.decoder_type),
        "use_checkpoint": bool(decoder.use_checkpoint),
        "param_dtype": param_dtype,
    }


def _build_decoder_from_spec(spec: Dict[str, Any]) -> Decoder:
    decoder = Decoder(
        in_dim=int(spec["in_dim"]),
        out_dim=int(spec["out_dim"]),
        hidden_dim=int(spec["hidden_dim"]),
        num_res_blocks=int(spec["num_res_blocks"]),
        norm_type=str(spec["norm_type"]),
        decoder_type=str(spec["decoder_type"]),
        use_checkpoint=bool(spec["use_checkpoint"]),
        num_models=1,
    )
    param_dtype = spec.get("param_dtype")
    if param_dtype:
        decoder = decoder.to(dtype=_name_to_dtype(str(param_dtype)))
    return decoder


def _collect_vae_linear_specs(model: nn.Module) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, VAELinear):
            continue

        parallel_parts = int(module.parallel_parts)
        residual_stages = int(getattr(module, "residual_stages", 1))
        if residual_stages < 1:
            residual_stages = 1
        stage_codebook_dims = [int(v) for v in getattr(module, "stage_codebook_dims", [int(module.codebook_dim)])]
        if len(stage_codebook_dims) == 1 and residual_stages > 1:
            stage_codebook_dims = stage_codebook_dims * residual_stages
        if len(stage_codebook_dims) != residual_stages:
            raise ValueError(
                f"[{name}] stage_codebook_dims length {len(stage_codebook_dims)} != residual_stages {residual_stages}"
            )

        stage_vq_specs: List[Any] = []
        stage_decoder_specs: List[Any] = []
        for stage_idx in range(residual_stages):
            stage_vq_parts = []
            stage_decoder_parts = []
            for part_idx in range(parallel_parts):
                w = module.get_stage_part_vq_weight(stage_idx=stage_idx, part_idx=part_idx)
                stage_vq_parts.append(
                    {
                        "shape": list(w.shape),
                        "dtype": _dtype_to_name(w.dtype),
                    }
                )
                dec = module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
                stage_decoder_parts.append(_decoder_to_spec(dec))
            if parallel_parts == 1:
                stage_vq_specs.append(stage_vq_parts[0])
                stage_decoder_specs.append(stage_decoder_parts[0])
            else:
                stage_vq_specs.append(stage_vq_parts)
                stage_decoder_specs.append(stage_decoder_parts)

        # Keep legacy fields for stage-0 compatibility.
        if parallel_parts == 1:
            vq_specs = [stage_vq_specs[0]]
            decoder_specs = [stage_decoder_specs[0]]
        else:
            vq_specs = list(stage_vq_specs[0])
            decoder_specs = list(stage_decoder_specs[0])

        restore_idx = getattr(module, "restore_row_indices", None)
        restore_spec = None
        if isinstance(restore_idx, torch.Tensor):
            restore_spec = {
                "shape": list(restore_idx.shape),
                "dtype": _dtype_to_name(restore_idx.dtype),
            }
        restore_col_idx = getattr(module, "restore_col_indices", None)
        restore_col_spec = None
        if isinstance(restore_col_idx, torch.Tensor):
            restore_col_spec = {
                "shape": list(restore_col_idx.shape),
                "dtype": _dtype_to_name(restore_col_idx.dtype),
            }
        protected_idx = getattr(module, "protected_input_indices", None)
        protected_idx_spec = None
        if isinstance(protected_idx, torch.Tensor):
            protected_idx_spec = {
                "shape": list(protected_idx.shape),
                "dtype": _dtype_to_name(protected_idx.dtype),
            }
        protected_weight = getattr(module, "protected_input_weight", None)
        protected_weight_spec = None
        if isinstance(protected_weight, torch.Tensor):
            protected_weight_spec = {
                "shape": list(protected_weight.shape),
                "dtype": _dtype_to_name(protected_weight.dtype),
            }
        protected_out_idx = getattr(module, "protected_output_indices", None)
        protected_out_idx_spec = None
        if isinstance(protected_out_idx, torch.Tensor):
            protected_out_idx_spec = {
                "shape": list(protected_out_idx.shape),
                "dtype": _dtype_to_name(protected_out_idx.dtype),
            }
        protected_out_weight = getattr(module, "protected_output_weight", None)
        protected_out_weight_spec = None
        if isinstance(protected_out_weight, torch.Tensor):
            protected_out_weight_spec = {
                "shape": list(protected_out_weight.shape),
                "dtype": _dtype_to_name(protected_out_weight.dtype),
            }
        sparse_row_idx = getattr(module, "sparse_residual_row_indices", None)
        sparse_row_idx_spec = None
        if isinstance(sparse_row_idx, torch.Tensor):
            sparse_row_idx_spec = {
                "shape": list(sparse_row_idx.shape),
                "dtype": _dtype_to_name(sparse_row_idx.dtype),
            }
        sparse_col_idx = getattr(module, "sparse_residual_col_indices", None)
        sparse_col_idx_spec = None
        if isinstance(sparse_col_idx, torch.Tensor):
            sparse_col_idx_spec = {
                "shape": list(sparse_col_idx.shape),
                "dtype": _dtype_to_name(sparse_col_idx.dtype),
            }
        sparse_values = getattr(module, "sparse_residual_values", None)
        sparse_values_spec = None
        if isinstance(sparse_values, torch.Tensor):
            sparse_values_spec = {
                "shape": list(sparse_values.shape),
                "dtype": _dtype_to_name(sparse_values.dtype),
            }
        specs.append(
            {
                "name": name,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "compressed_in_features": int(getattr(module, "compressed_in_features", module.in_features)),
                "compressed_out_features": int(getattr(module, "compressed_out_features", module.out_features)),
                "codebook_dim": int(module.codebook_dim),
                "transpose": bool(module.transpose),
                "parallel_parts": parallel_parts,
                "parallel_rows": int(getattr(module, "parallel_rows", parallel_parts)),
                "parallel_cols": int(getattr(module, "parallel_cols", 1)),
                "residual_stages": residual_stages,
                "stage_codebook_dims": stage_codebook_dims,
                "has_bias": bool(module.bias is not None),
                "has_original_weight": bool(module.original_weight is not None),
                "always_use_original": bool(getattr(module, "always_use_original", False)),
                "protect_original_weight": bool(getattr(module, "protect_original_weight", False)),
                "vq_weights": vq_specs,
                "decoders": decoder_specs,
                "stage_vq_weights": stage_vq_specs if residual_stages > 1 else None,
                "stage_decoders": stage_decoder_specs if residual_stages > 1 else None,
                "restore_row_indices": restore_spec,
                "restore_col_indices": restore_col_spec,
                "protected_input_indices": protected_idx_spec,
                "protected_input_weight": protected_weight_spec,
                "protected_output_indices": protected_out_idx_spec,
                "protected_output_weight": protected_out_weight_spec,
                "sparse_residual_row_indices": sparse_row_idx_spec,
                "sparse_residual_col_indices": sparse_col_idx_spec,
                "sparse_residual_values": sparse_values_spec,
            }
        )
    return specs


def unload_vae_original_linear_weights(model: nn.Module) -> int:
    unloaded = 0
    for module in model.modules():
        if isinstance(module, VAELinear) and module.unload_original_linear():
            unloaded += 1
    return unloaded


def save_model_checkpoint(
    model: nn.Module,
    output_dir: str,
    *,
    base_model_path: Optional[str] = None,
    tokenizer=None,
    save_config: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
    unload_vae_original_weights: bool = False,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    if unload_vae_original_weights:
        unload_vae_original_linear_weights(model)

    state_path = os.path.join(output_dir, STATE_DICT_FILENAME)
    torch.save(model.state_dict(), state_path)

    if save_config and getattr(model, "config", None) is not None:
        model.config.save_pretrained(output_dir)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    if base_model_path is None and getattr(model, "config", None) is not None:
        base_model_path = getattr(model.config, "_name_or_path", None)

    vae_specs = _collect_vae_linear_specs(model)
    meta: Dict[str, Any] = {
        "format": "vaellm_state_dict_with_meta",
        "version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_path": base_model_path,
        "state_dict_file": STATE_DICT_FILENAME,
        "converted_module_count": len(vae_specs),
        "converted_modules": vae_specs,
    }
    if extra_meta:
        meta["extra_meta"] = extra_meta

    meta_path = os.path.join(output_dir, META_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "state_dict": state_path,
        "meta": meta_path,
        "output_dir": output_dir,
    }


def _make_vq_placeholders(vq_specs: Sequence[Dict[str, Any]], device: torch.device) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for spec in vq_specs:
        shape = tuple(int(x) for x in spec["shape"])
        dtype = _name_to_dtype(str(spec["dtype"]))
        tensors.append(torch.zeros(shape, dtype=dtype, device=device))
    return tensors


def _ensure_bias_param(
    old_module: nn.Module,
    out_features: int,
    has_bias: bool,
) -> Optional[nn.Parameter]:
    if not has_bias:
        return None

    old_bias = getattr(old_module, "bias", None)
    if old_bias is not None:
        return nn.Parameter(torch.zeros_like(old_bias.detach()))

    old_weight = getattr(old_module, "weight", None)
    if old_weight is not None:
        return nn.Parameter(
            torch.zeros(
                out_features,
                dtype=old_weight.dtype,
                device=old_weight.device,
            )
        )
    return nn.Parameter(torch.zeros(out_features, dtype=torch.float32))


def _rebuild_converted_modules(model: nn.Module, converted_modules: Sequence[Dict[str, Any]]) -> None:
    for spec in converted_modules:
        name = str(spec["name"])
        old_module = _get_module_by_name(model, name)
        weight = getattr(old_module, "weight", None)
        device = weight.device if weight is not None else torch.device("cpu")

        parallel_parts = int(spec["parallel_parts"])
        residual_stages = int(spec.get("residual_stages", 1))
        if residual_stages < 1:
            residual_stages = 1
        stage_codebook_dims_raw = spec.get("stage_codebook_dims")
        if isinstance(stage_codebook_dims_raw, (list, tuple)) and len(stage_codebook_dims_raw) > 0:
            stage_codebook_dims = [int(v) for v in stage_codebook_dims_raw]
            if len(stage_codebook_dims) == 1 and residual_stages > 1:
                stage_codebook_dims = stage_codebook_dims * residual_stages
            if len(stage_codebook_dims) != residual_stages:
                raise ValueError(
                    f"[{name}] stage_codebook_dims length {len(stage_codebook_dims)} != residual_stages {residual_stages}"
                )
        else:
            stage_codebook_dims = [int(spec["codebook_dim"]) for _ in range(residual_stages)]

        stage_vq_payload = None
        stage_decoder_payload = None
        vq_payload = None
        decoder_payload = None

        if residual_stages > 1:
            stage_vq_specs = spec.get("stage_vq_weights")
            stage_decoder_specs = spec.get("stage_decoders")
            if not isinstance(stage_vq_specs, (list, tuple)):
                raise ValueError(f"[{name}] missing/invalid stage_vq_weights for residual_stages={residual_stages}")
            if not isinstance(stage_decoder_specs, (list, tuple)):
                raise ValueError(f"[{name}] missing/invalid stage_decoders for residual_stages={residual_stages}")
            if len(stage_vq_specs) != residual_stages:
                raise ValueError(
                    f"[{name}] stage_vq_weights length {len(stage_vq_specs)} != residual_stages {residual_stages}"
                )
            if len(stage_decoder_specs) != residual_stages:
                raise ValueError(
                    f"[{name}] stage_decoders length {len(stage_decoder_specs)} != residual_stages {residual_stages}"
                )

            stage_vq_payload = []
            stage_decoder_payload = []
            for stage_idx in range(residual_stages):
                stage_vq_spec = stage_vq_specs[stage_idx]
                stage_decoder_spec = stage_decoder_specs[stage_idx]
                if parallel_parts == 1:
                    if not isinstance(stage_vq_spec, dict):
                        raise ValueError(f"[{name}] stage_vq_weights[{stage_idx}] must be a dict for single-part mode.")
                    if not isinstance(stage_decoder_spec, dict):
                        raise ValueError(f"[{name}] stage_decoders[{stage_idx}] must be a dict for single-part mode.")
                    stage_vq_payload.append(_make_vq_placeholders([stage_vq_spec], device=device)[0])
                    stage_decoder_payload.append(_build_decoder_from_spec(stage_decoder_spec))
                else:
                    if not isinstance(stage_vq_spec, (list, tuple)):
                        raise ValueError(
                            f"[{name}] stage_vq_weights[{stage_idx}] must be list/tuple for parallel_parts={parallel_parts}."
                        )
                    if not isinstance(stage_decoder_spec, (list, tuple)):
                        raise ValueError(
                            f"[{name}] stage_decoders[{stage_idx}] must be list/tuple for parallel_parts={parallel_parts}."
                        )
                    if len(stage_vq_spec) != parallel_parts:
                        raise ValueError(
                            f"[{name}] stage_vq_weights[{stage_idx}] length {len(stage_vq_spec)} != parallel_parts {parallel_parts}"
                        )
                    if len(stage_decoder_spec) != parallel_parts:
                        raise ValueError(
                            f"[{name}] stage_decoders[{stage_idx}] length {len(stage_decoder_spec)} != parallel_parts {parallel_parts}"
                        )
                    stage_vq_payload.append(_make_vq_placeholders(stage_vq_spec, device=device))
                    stage_decoder_payload.append([_build_decoder_from_spec(s) for s in stage_decoder_spec])
        else:
            vq_placeholders = _make_vq_placeholders(spec["vq_weights"], device=device)
            decoders = [_build_decoder_from_spec(s) for s in spec["decoders"]]

            if len(vq_placeholders) != parallel_parts:
                raise ValueError(
                    f"[{name}] vq placeholders count {len(vq_placeholders)} != parallel_parts {parallel_parts}"
                )
            if len(decoders) != parallel_parts:
                raise ValueError(f"[{name}] decoder count {len(decoders)} != parallel_parts {parallel_parts}")

            if parallel_parts == 1:
                vq_payload = vq_placeholders[0]
                decoder_payload = decoders[0]
            else:
                vq_payload = vq_placeholders
                decoder_payload = decoders
        restore_payload = None
        restore_spec = spec.get("restore_row_indices")
        if isinstance(restore_spec, dict):
            shape = tuple(int(v) for v in restore_spec.get("shape", []))
            if len(shape) != 1:
                raise ValueError(f"[{name}] restore_row_indices shape must be 1D, got {shape}")
            restore_dtype = _name_to_dtype(str(restore_spec.get("dtype", "int64")))
            restore_payload = torch.zeros(shape, dtype=restore_dtype, device=device)
        restore_col_payload = None
        restore_col_spec = spec.get("restore_col_indices")
        if isinstance(restore_col_spec, dict):
            shape = tuple(int(v) for v in restore_col_spec.get("shape", []))
            if len(shape) != 1:
                raise ValueError(f"[{name}] restore_col_indices shape must be 1D, got {shape}")
            restore_dtype = _name_to_dtype(str(restore_col_spec.get("dtype", "int64")))
            restore_col_payload = torch.zeros(shape, dtype=restore_dtype, device=device)
        protected_idx_payload = None
        protected_idx_spec = spec.get("protected_input_indices")
        if isinstance(protected_idx_spec, dict):
            shape = tuple(int(v) for v in protected_idx_spec.get("shape", []))
            if len(shape) != 1:
                raise ValueError(f"[{name}] protected_input_indices shape must be 1D, got {shape}")
            protected_idx_dtype = _name_to_dtype(str(protected_idx_spec.get("dtype", "int64")))
            protected_idx_payload = _build_unique_index_placeholder(
                shape,
                dtype=protected_idx_dtype,
                device=device,
            )
        protected_weight_payload = None
        protected_weight_spec = spec.get("protected_input_weight")
        if isinstance(protected_weight_spec, dict):
            shape = tuple(int(v) for v in protected_weight_spec.get("shape", []))
            if len(shape) != 2:
                raise ValueError(f"[{name}] protected_input_weight shape must be 2D, got {shape}")
            protected_weight_dtype = _name_to_dtype(str(protected_weight_spec.get("dtype", "float32")))
            protected_weight_payload = torch.zeros(shape, dtype=protected_weight_dtype, device=device)
        protected_out_idx_payload = None
        protected_out_idx_spec = spec.get("protected_output_indices")
        if isinstance(protected_out_idx_spec, dict):
            shape = tuple(int(v) for v in protected_out_idx_spec.get("shape", []))
            if len(shape) != 1:
                raise ValueError(f"[{name}] protected_output_indices shape must be 1D, got {shape}")
            protected_out_idx_dtype = _name_to_dtype(str(protected_out_idx_spec.get("dtype", "int64")))
            protected_out_idx_payload = _build_unique_index_placeholder(
                shape,
                dtype=protected_out_idx_dtype,
                device=device,
            )
        protected_out_weight_payload = None
        protected_out_weight_spec = spec.get("protected_output_weight")
        if isinstance(protected_out_weight_spec, dict):
            shape = tuple(int(v) for v in protected_out_weight_spec.get("shape", []))
            if len(shape) != 2:
                raise ValueError(f"[{name}] protected_output_weight shape must be 2D, got {shape}")
            protected_out_weight_dtype = _name_to_dtype(str(protected_out_weight_spec.get("dtype", "float32")))
            protected_out_weight_payload = torch.zeros(shape, dtype=protected_out_weight_dtype, device=device)
        sparse_row_idx_payload = None
        sparse_row_idx_spec = spec.get("sparse_residual_row_indices")
        if isinstance(sparse_row_idx_spec, dict):
            shape = tuple(int(v) for v in sparse_row_idx_spec.get("shape", []))
            if len(shape) != 1:
                raise ValueError(f"[{name}] sparse_residual_row_indices shape must be 1D, got {shape}")
            sparse_row_idx_dtype = _name_to_dtype(str(sparse_row_idx_spec.get("dtype", "uint16")))
            sparse_row_idx_payload = torch.zeros(shape, dtype=sparse_row_idx_dtype, device=device)
        sparse_col_idx_payload = None
        sparse_col_idx_spec = spec.get("sparse_residual_col_indices")
        if isinstance(sparse_col_idx_spec, dict):
            shape = tuple(int(v) for v in sparse_col_idx_spec.get("shape", []))
            if len(shape) != 1:
                raise ValueError(f"[{name}] sparse_residual_col_indices shape must be 1D, got {shape}")
            sparse_col_idx_dtype = _name_to_dtype(str(sparse_col_idx_spec.get("dtype", "uint16")))
            sparse_col_idx_payload = torch.zeros(shape, dtype=sparse_col_idx_dtype, device=device)
        sparse_values_payload = None
        sparse_values_spec = spec.get("sparse_residual_values")
        if isinstance(sparse_values_spec, dict):
            shape = tuple(int(v) for v in sparse_values_spec.get("shape", []))
            if len(shape) != 1:
                raise ValueError(f"[{name}] sparse_residual_values shape must be 1D, got {shape}")
            sparse_values_dtype = _name_to_dtype(str(sparse_values_spec.get("dtype", "float16")))
            sparse_values_payload = torch.zeros(shape, dtype=sparse_values_dtype, device=device)

        new_module = VAELinear(
            in_features=int(spec["in_features"]),
            out_features=int(spec["out_features"]),
            bias=_ensure_bias_param(
                old_module=old_module,
                out_features=int(spec["out_features"]),
                has_bias=bool(spec["has_bias"]),
            ),
            original_weight=getattr(old_module, "weight", None) if bool(spec.get("has_original_weight", False)) else None,
            vq_weight=vq_payload,
            decoder=decoder_payload,
            stage_vq_weights=stage_vq_payload,
            stage_decoders=stage_decoder_payload,
            codebook_dim=int(spec["codebook_dim"]),
            stage_codebook_dims=stage_codebook_dims,
            transpose=bool(spec["transpose"]),
            parallel_parts=parallel_parts,
            parallel_rows=int(spec.get("parallel_rows", parallel_parts)),
            parallel_cols=int(spec.get("parallel_cols", 1)),
            restore_row_indices=restore_payload,
            restore_col_indices=restore_col_payload,
            compressed_in_features=int(spec.get("compressed_in_features", spec["in_features"])),
            compressed_out_features=int(spec.get("compressed_out_features", spec["out_features"])),
            protected_input_indices=protected_idx_payload,
            protected_input_weight=protected_weight_payload,
            protected_output_indices=protected_out_idx_payload,
            protected_output_weight=protected_out_weight_payload,
            sparse_residual_row_indices=sparse_row_idx_payload,
            sparse_residual_col_indices=sparse_col_idx_payload,
            sparse_residual_values=sparse_values_payload,
            always_use_original=bool(spec.get("always_use_original", False)),
            protect_original_weight=bool(spec.get("protect_original_weight", False)),
        )
        set_module_by_name(model, name, new_module)


def _torch_load_state_dict(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _materialize_missing_bias_params_from_state_dict(model: nn.Module, state_dict: Dict[str, Any]) -> int:
    created = 0
    for key, value in state_dict.items():
        if not key.endswith(".bias"):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        module_name = key[:-len(".bias")]
        try:
            module = _get_module_by_name(model, module_name)
        except Exception:
            continue
        if not hasattr(module, "bias"):
            continue
        old_bias = getattr(module, "bias")
        if old_bias is not None:
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            device = weight.device
            dtype = weight.dtype
        else:
            device = value.device
            dtype = value.dtype
        setattr(
            module,
            "bias",
            nn.Parameter(torch.zeros(tuple(value.shape), dtype=dtype, device=device)),
        )
        created += 1
    return created


def _remap_legacy_parallel_linear_state_dict_keys(
    state_dict: Dict[str, Any],
    model_state_keys: Sequence[str],
) -> Tuple[Dict[str, Any], int]:
    """Map legacy Linear keys to current ParallelLinear key layout.

    Older checkpoints may store decoder keys like:
      *.linear_in.weight / *.linear_in.bias
    while current modules expect:
      *.linear_in.linear.weight / *.linear_in.linear.bias  (num_models=1)
    or:
      *.linear_in.conv.weight / *.linear_in.conv.bias      (num_models>1)
    """

    key_set = set(model_state_keys)
    remapped: Dict[str, Any] = {}
    converted = 0

    for key, value in state_dict.items():
        target_key = key

        if (key.endswith(".weight") or key.endswith(".bias")) and key.count(".") >= 1:
            stem, suffix = key.rsplit(".", 1)
            cand_linear = f"{stem}.linear.{suffix}"
            cand_conv = f"{stem}.conv.{suffix}"

            # Only remap when original key is no longer expected by model.
            if key not in key_set:
                if cand_linear in key_set:
                    target_key = cand_linear
                    converted += 1
                elif cand_conv in key_set:
                    target_key = cand_conv
                    converted += 1

        remapped[target_key] = value

    return remapped, converted


def load_checkpoint_into_model(
    model: nn.Module,
    model_dir: str,
    *,
    map_location: str = "cpu",
    strict: bool = True,
):
    meta_path = os.path.join(model_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    converted_modules = meta.get("converted_modules", [])
    if converted_modules:
        _rebuild_converted_modules(model, converted_modules)

    state_dict_file = str(meta.get("state_dict_file", STATE_DICT_FILENAME))
    state_dict_path = os.path.join(model_dir, state_dict_file)
    state_dict = _torch_load_state_dict(state_dict_path, map_location=map_location)
    model_state_keys = tuple(model.state_dict().keys())
    state_dict, _remap_count = _remap_legacy_parallel_linear_state_dict_keys(state_dict, model_state_keys)
    _materialize_missing_bias_params_from_state_dict(model, state_dict)

    load_result = model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model, meta, load_result


def load_model_checkpoint(
    model_dir: str,
    *,
    access_token: Optional[str] = None,
    base_model_path: Optional[str] = None,
    map_location: str = "cpu",
    strict: bool = True,
):
    meta_path = os.path.join(model_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    base_path = base_model_path or meta.get("base_model_path")
    if not base_path:
        raise ValueError("base_model_path is required (not found in meta and not provided).")

    model = get_model(base_path, access_token)

    return load_checkpoint_into_model(
        model=model,
        model_dir=model_dir,
        map_location=map_location,
        strict=strict,
    )
