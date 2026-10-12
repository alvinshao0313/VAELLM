"""Single-matrix CAT reconstruction for an isolated, fixed-protection rotation pair."""
from __future__ import annotations

import gc
import time

import torch
from torch import nn

from experiments.rotation_factor_probe.transform import ProbeRotation
from litebsq.protected_channel_quant import decode_protected_channel_weight
from rotation.weight_preprocess import rotation_spec
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_pipeline import apply_group_vae_payload, train_group_vae_payload
from train_utils.utils import LinearRef, configure_deterministic_mode, set_seed


def recipe(model_path: str, category: str, *, steps: int, seed: int):
    if category not in {"gate_proj", "down_proj"} or steps < 1:
        raise ValueError("This paired screen requires gate/down and positive stage steps.")
    args = [
        "--model_path", model_path, "--compression_categories", category,
        "--target_layers", "0", "--after_category_mode", "none",
        "--seed", str(seed), "--data_seed", str(seed), "--bf16", "true", "--fp16", "false",
        "--deterministic", "true", "--codebook_bits", "default=" + ("48" if category == "down_proj" else "32"),
        "--codebook_dim", "default=32", "--residual_stages", "default=2",
        "--base_ch", "default=128", "--num_res_blocks", "default=0",
        "--norm_type", "default=layer", "--activation_type", "default=swish",
        "--decoder_type", "default=symmetric", "--recon_loss_type", "default=mse",
        "--normalize_weight", "--new_quant",
        "--quantizer_type", "BSQ", "--gamma0", "1", "--gamma", "1", "--zeta", "1",
        "--inv_temperature", "100", "--vae_steps", f"default={steps}",
        "--vae_batch_size", "8192", "--vae_learning_rate", "0.003",
        "--vae_weight_decay", "0", "--vae_optim", "adamw",
        "--vae_lr_scheduler_type", "linear", "--vae_warmup_ratio", "0",
        "--beta1", "0.9", "--beta2", "0.95", "--l1_weight", "1",
        "--lfq_weight", "2.5", "--commitment_loss_weight", "0.25",
        "--entropy_loss_weight", "0.01", "--vae_decoder_checkpoint", "true",
        "--channel_protect_mode", "none", "--weight_rotation", "none",
    ]
    cat_args, _, training_args, vae_args = parse_cat_runtime_args(args)
    training_args.seed = seed
    runtime = resolve_category_runtime_configs(cat_args, vae_args, [category])[category]
    return args, runtime, vae_args, training_args


def host_for_weight(weight: torch.Tensor, name: str):
    host, parent = nn.Module(), None
    parent = host
    parts = name.split(".")
    for part in parts[:-1]:
        child = nn.Module()
        parent.add_module(part, child)
        parent = child
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
    parent.add_module(parts[-1], linear)
    return host, linear


def run_matrix(bundle: dict, *, name: str, model_path: str, full: bool,
               factors: dict, steps: int, seed: int, device: torch.device,
               diagnostic_out: dict | None = None):
    """Return an original-coordinate BF16 matrix; no model checkpoint is exported."""
    started = time.perf_counter()
    configure_deterministic_mode(True)
    set_seed(seed)
    category = name.rsplit(".", 1)[-1]
    weight = bundle["reference_weight"]
    indices = bundle["protected_input_indices"]
    required_count = 256 if category == "down_proj" else 64
    if (weight.dtype != torch.bfloat16 or weight.ndim != 2 or indices.dtype != torch.int64
            or indices.numel() != required_count or indices.numel() % 32
            or indices.unique().numel() != indices.numel()
            or bool((indices < 0).any()) or bool((indices >= weight.shape[1]).any())):
        raise ValueError("The fixed original BF16 weight/input protection contract changed.")
    keep = torch.ones(weight.shape[1], dtype=torch.bool)
    keep[indices] = False
    compressed = weight[:, keep].float().contiguous()
    spec = rotation_spec("two_sided", 0 if full else 32, seed, name)
    rotation = ProbeRotation(*compressed.shape, spec, factors).to(device)
    with torch.no_grad():
        rotated = rotation(compressed.to(device)).cpu()
    if rotated.dtype != torch.float32:
        raise RuntimeError("Native CAT rotates the unprotected matrix in FP32 before stage normalization.")
    host, linear = host_for_weight(rotated, name)
    args, runtime, vae_args, training_args = recipe(model_path, category, steps=steps, seed=seed)
    refs = [LinearRef(name, linear, category, category == "down_proj")]
    # Host initialization is outside the matched VAE RNG stream in both arms.
    torch.manual_seed(seed)
    payload = train_group_vae_payload(
        model=host, group_refs=refs, group_tag=f"{name}/{'full' if full else 'block32'}/seed{seed}",
        runtime_cfg=runtime, vae_args=vae_args, training_args=training_args,
        train_device=str(device), convert_device=str(device), do_convert=True,
        batch_size=8192, gpu_resident_data=True, log_every=100, eval_every=0,
        eval_blocks=256, channel_protect_mode="none", channel_plan=None,
        channel_rank_metric="channel_weight_abs", channel_axis="input", channel_quant="none",
        deterministic=True, shuffle_seed=seed,
    )
    apply_group_vae_payload(model=host, group_refs=refs, group_tag=name,
                            payload=payload, convert_device=str(device))
    layer = host.get_submodule(name).to(device)
    layer.eval()
    with torch.no_grad():
        if diagnostic_out is not None:
            diagnostic_out["rotated_weight"] = rotated.clone()
            diagnostic_out["stage_codes"] = [layer.get_stage_part_vq_storage(
                stage_idx=s, part_idx=0).detach().cpu().clone() for s in range(layer.residual_stages)]
            diagnostic_out["stage_decoders"] = [{k: v.detach().cpu().clone() for k, v in
                layer.get_stage_part_decoder(stage_idx=s, part_idx=0).state_dict().items()}
                for s in range(layer.residual_stages)]
        # This follows deployment: BF16 stage sum, FP32 inverse arithmetic with
        # BF16 result, then BF16 protected-column restoration.
        decoded = layer._decode_weight(dtype=torch.bfloat16)
        restored = rotation(decoded, inverse=True)
        if decoded.dtype != torch.bfloat16 or restored.dtype != torch.bfloat16:
            raise RuntimeError("The deployment measurement must retain BF16 rounding boundaries.")
        reconstruction = torch.empty_like(weight, device=device)
        reconstruction[:, keep.to(device)] = restored
        protected = decode_protected_channel_weight(
            bundle["protected_input_qvalues"], bundle["protected_input_scales"],
            quant_format="int8", dtype=torch.bfloat16, device=device,
        )
        reconstruction[:, indices.to(device)] = protected.T
        if not bool(torch.isfinite(reconstruction).all()):
            raise FloatingPointError("Nonfinite reconstructed matrix.")
        state_bits = sum(t.numel() * t.element_size() * 8 for t in layer.state_dict().values())
        code_bits = sum(layer.get_stage_part_vq_storage(stage_idx=s, part_idx=0).numel() * 8
                        for s in range(layer.residual_stages))
        protection_bits = sum(bundle[key].numel() * bundle[key].element_size() * 8 for key in (
            "protected_input_indices", "protected_input_qvalues", "protected_input_scales"))
        rotation_bits = sum(getattr(rotation, key).numel() * 8 for key in ("input_signs", "output_signs"))
        report = {
            "name": name, "rotation": "full" if full else "block32", "rotation_seed": spec["seed"],
            "input_factor": rotation.input_factor, "output_factor": rotation.output_factor,
            "weight_shape": list(weight.shape), "unprotected_shape": list(compressed.shape),
            "protected_input_count": indices.numel(), "seed": seed, "steps_per_stage": steps,
            "cli_recipe": args, "deployment_decode_dtype": str(decoded.dtype),
            "main_code_bits": code_bits, "unprotected_module_bits": state_bits,
            "protection_bits": protection_bits, "rotation_sign_bits": rotation_bits,
            "estimated_projection_bpw": (state_bits + protection_bits + rotation_bits) / weight.numel(),
            "storage_note": "Local payload accounting only; no stable whole-model export or acceptance.",
        }
        reconstruction = reconstruction.cpu()
    torch.cuda.synchronize(device)
    report["elapsed_seconds"] = time.perf_counter() - started
    del host, linear, layer, payload, rotation, rotated, decoded, restored, protected
    gc.collect()
    torch.cuda.empty_cache()
    return reconstruction, report
