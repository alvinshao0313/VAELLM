"""Fixed-code two-stage weight fitting using the existing VAELinear decoder."""
from __future__ import annotations

import hashlib
import math
import time

import torch
from torch.nn import functional as F

from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.transform import ProbeRotation
from litebsq.autoencoder import Decoder
from litebsq.bitpack import build_bitpack_u8_spec
from litebsq.protected_channel_quant import decode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec


STEPS, BATCH_SIZE, SEED, LR, EPS = 500, 8192, 31, 1e-5, 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rebuild(diagnostic, category):
    """Rebuild the actual finalized payload; its input and output scales are fused."""
    target = diagnostic["rotated_weight"]
    require(category in {"gate_proj", "down_proj"} and target.ndim == 2
            and target.dtype == torch.float32, "Expected an FP32 rotated gate/down target.")
    latent = 48 if category == "down_proj" else 32
    blocks = target.numel() // 32
    require(target.numel() % 32 == 0 and len(diagnostic["stage_codes"]) == 2
            and len(diagnostic["stage_decoders"]) == 2, "Expected two D32 stages.")
    decoders, codes, specs = [], [], []
    for code, state in zip(diagnostic["stage_codes"], diagnostic["stage_decoders"], strict=True):
        require(code.dtype == torch.uint8 and tuple(code.shape) == (blocks, 1, latent // 8),
                "Actual packed bank geometry differs from the target.")
        require(all(value.dtype == torch.bfloat16 for value in state.values()),
                "The original finalized decoder must be BF16.")
        # extract_single gives the same plain Linear submodule keys as diagnostic_out.
        decoder = Decoder(in_dim=latent, out_dim=32, hidden_dim=128, num_res_blocks=0,
                          norm_type="layer", activation_type="swish", decoder_type="symmetric",
                          use_checkpoint=True, num_models=1).get_sub_decoder(0).to(torch.bfloat16)
        decoder.load_state_dict(state, strict=True)
        decoder._q_scale_fused = True  # Metadata only: never apply either fusion a second time.
        decoders.append(decoder.eval())
        codes.append(code.clone())
        specs.append(build_bitpack_u8_spec(logical_shape=(blocks, 1, latent)))
    layer = VAELinear(in_features=target.shape[1], out_features=target.shape[0], bias=None,
        stage_vq_weights=codes, stage_vq_storage_specs=specs, stage_decoders=decoders,
        codebook_dim=32, stage_codebook_dims=[32, 32], transpose=category == "down_proj")
    layer.eval().pack_parallel_stage_decoder_(trainable=False)
    require(layer._parallel_stage_layout == [(0, 0), (1, 0)]
            and layer._parallel_stage_restore_identity, "Native stage/block layout changed.")
    for stage in range(2):
        require(torch.equal(layer.get_stage_part_vq_storage(stage, 0), codes[stage]),
                "Rebuild changed a packed bank.")
        restored = layer.get_stage_part_decoder(stage, 0).state_dict()
        require(set(restored) == set(diagnostic["stage_decoders"][stage])
                and all(torch.equal(value, diagnostic["stage_decoders"][stage][key])
                        for key, value in restored.items()), "Rebuild changed finalized decoder values.")
    return layer


def target_blocks(target, transpose):
    return (target.T.contiguous() if transpose else target.contiguous()).view(-1, 32)


def index_batches(num_blocks, *, steps=STEPS, batch_size=BATCH_SIZE, seed=SEED):
    """Whole-population epochs, including the last partial batch, without replacement."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    done, epoch = 0, 0
    while done < steps:
        order = torch.randperm(num_blocks, generator=generator)
        for start in range(0, num_blocks, batch_size):
            yield epoch, order[start:start + batch_size]
            done += 1
            if done == steps:
                return
        epoch += 1


def frozen_record(layer):
    return {name: tensor_record(value) for name, value in layer.state_dict().items()
            if not name.startswith("_parallel_stage_decoder.")}


def schema(layer):
    return {name: [list(value.shape), str(value.dtype)] for name, value in layer.state_dict().items()}


def storage_bits(layer):
    return sum(value.numel() * value.element_size() * 8 for value in layer.state_dict().values())


def stage_norm(values):
    total = torch.zeros(2, dtype=torch.float64, device=values[0].device)
    for value in values:
        require(value.shape[0] % 2 == 0, "Packed stage parameter axis changed.")
        total += value.detach().double().reshape(2, -1).square().sum(dim=1)
    return total.sqrt().cpu().tolist()


def error_stats(candidate, target):
    require(candidate.shape == target.shape, "Reconstruction shape changed.")
    error, signal = 0.0, 0.0
    for start in range(0, target.shape[0], 256):
        x = candidate[start:start + 256].float()
        y = target[start:start + 256].to(device=x.device, dtype=torch.float32)
        error += float((x - y).double().square().sum())
        signal += float(y.double().square().sum())
    require(signal > 0 and math.isfinite(error), "Invalid reconstruction statistics.")
    return {"mse": error / target.numel(), "nmse": error / signal,
            "target_second_moment": signal / target.numel()}


@torch.no_grad()
def decode(layer, *, deployment):
    layer._decoder_compute_dtype = torch.bfloat16
    if deployment:
        layer.disable_trainable_decode()
    else:
        layer.enable_trainable_decode(parallel_stage_decode=True)
    layer.clear_decoded_weight_cache()
    with torch.autocast(device_type=next(layer.parameters()).device.type, enabled=False):
        result = layer._decode_weight(dtype=torch.bfloat16)
    require(result.dtype == torch.bfloat16 and bool(torch.isfinite(result).all()),
            "Expected a finite native BF16 reconstruction.")
    return result


@torch.no_grad()
def restore(bundle, name, rotated, device):
    weight, indices = bundle["reference_weight"], bundle["protected_input_indices"]
    keep = torch.ones(weight.shape[1], dtype=torch.bool)
    keep[indices] = False
    require(rotated.shape == (weight.shape[0], int(keep.sum())) and rotated.dtype == torch.bfloat16,
            "Unexpected unprotected BF16 matrix.")
    rotation = ProbeRotation(*rotated.shape, rotation_spec("two_sided", 32, SEED, name), {}).to(device)
    result = torch.empty_like(weight, device=device)
    result[:, keep.to(device)] = rotation(rotated.to(device), inverse=True)
    protected = decode_protected_channel_weight(bundle["protected_input_qvalues"],
        bundle["protected_input_scales"], quant_format="int8", dtype=torch.bfloat16, device=device)
    result[:, indices.to(device)] = protected.T
    return result.cpu()


def fit(layer, target, progress):
    """Fit both registered stages to original rotated weights with immutable hard codes."""
    device = next(layer.parameters()).device
    require(device.type == "cuda", "The real probe requires native CUDA packed decode.")
    before_frozen, before_schema, before_bits = frozen_record(layer), schema(layer), storage_bits(layer)
    initial = {name: value.detach().clone() for name, value in layer._parallel_stage_decoder.named_parameters()}
    before_deploy = decode(layer, deployment=True)
    report = {"before_deployment": error_stats(before_deploy, target)}
    denominator = report["before_deployment"]["target_second_moment"]
    for parameter in layer._parallel_stage_decoder.parameters():
        parameter.data = parameter.data.float()
    layer.enable_trainable_decode(parallel_stage_decode=True)
    layer._decoder_compute_dtype = torch.bfloat16
    parameters = dict(layer._parallel_stage_decoder.named_parameters())
    identities = {name: id(value) for name, value in parameters.items()}
    require({value.dtype for value in parameters.values()} == {torch.float32}, "Decoder masters must be FP32.")
    before_train = decode(layer, deployment=False)
    report["before_train_decode"] = error_stats(before_train, target)
    report["before_path_delta"] = error_stats(before_train, before_deploy)
    del before_deploy, before_train
    optimizer = torch.optim.AdamW(parameters.values(), lr=LR, betas=(.9, .95), eps=EPS, weight_decay=0)
    targets = target_blocks(target, layer.transpose).to(device)
    grouped = layer._get_parallel_stage_grouped_vq_packed(device=device)
    require(tuple(grouped.shape[:2]) == (targets.shape[0], 2), "Both codes must share target indices.")
    latent = layer._parallel_stage_decoder.in_dim
    trajectory, batch_hash = [], hashlib.sha256()
    stage_gradient_max = [0.0, 0.0]
    started = time.perf_counter()
    for step, (epoch, indices_cpu) in enumerate(index_batches(len(targets)), 1):
        batch_hash.update(indices_cpu.numpy().tobytes())
        indices = indices_cpu.to(device)
        optimizer.param_groups[0]["lr"] = LR * (1.0 - (step - 1) / STEPS)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=False):
            stages = layer._decode_packed_u8_with_decoder(layer._parallel_stage_decoder,
                grouped.index_select(0, indices), logical_shape=(len(indices), 2, latent),
                activation_dtype=torch.bfloat16, sparse_bit_grouped=True)
            require(stages is not None and stages.dtype == torch.bfloat16
                    and tuple(stages.shape) == (len(indices), 2, 32), "Native BF16 packed decode unavailable.")
            summed = stages.sum(dim=1)  # Match native stage sum before converting to FP32.
            raw_mse = F.mse_loss(summed.float(), targets.index_select(0, indices))
            loss = raw_mse / denominator
        require(bool(torch.isfinite(loss)), f"Nonfinite loss at step {step}.")
        loss.backward()
        gradients = [value.grad for value in parameters.values()]
        require(all(value is not None and bool(torch.isfinite(value).all()) for value in gradients),
                f"Missing or nonfinite registered decoder gradient at step {step}.")
        norms = stage_norm(gradients)
        stage_gradient_max = [max(a, b) for a, b in zip(stage_gradient_max, norms, strict=True)]
        absolute = torch.cat([value.detach().abs().reshape(-1) for value in gradients])
        nonzero = absolute[absolute > 0]
        item = {"step": step, "epoch": epoch, "blocks": len(indices), "lr": optimizer.param_groups[0]["lr"],
                "raw_mse": float(raw_mse), "nmse_loss": float(loss), "stage_gradient_l2": norms,
                "gradient_abs_max": float(absolute.max()), "gradient_abs_median": float(absolute.median()),
                "gradient_nonzero_min": float(nonzero.min()) if nonzero.numel() else 0.0,
                "gradient_abs_le_adam_eps_fraction": float((absolute <= EPS).float().mean()),
                "gradient_zero_fraction": float((absolute == 0).float().mean())}
        optimizer.step()
        trajectory.append(item)
        if step == 1 or step % 100 == 0:
            progress(item)
    torch.cuda.synchronize(device)
    report["training_seconds"] = time.perf_counter() - started
    report["after_train_decode"] = error_stats(decode(layer, deployment=False), target)
    delta_fp32 = stage_norm([value.detach() - initial[name].float() for name, value in parameters.items()])
    moments = sorted({str(state[key].dtype) for state in optimizer.state.values()
                      for key in ("exp_avg", "exp_avg_sq")})
    require(moments == ["torch.float32"] and all(value > 0 for value in stage_gradient_max + delta_fp32),
            "Both stages must actually update with FP32 moments.")
    del optimizer, grouped, targets
    for name, value in parameters.items():
        value.grad = None
        value.data = value.data.to(dtype=initial[name].dtype)
    layer._build_parallel_stage_decode_plan()
    after_deploy = decode(layer, deployment=True)
    after_train_cast = decode(layer, deployment=False)
    report["after_deployment"] = error_stats(after_deploy, target)
    report["after_cast_train_decode"] = error_stats(after_train_cast, target)
    report["after_cast_path_delta"] = error_stats(after_train_cast, after_deploy)
    layer.disable_trainable_decode()
    delta_bf16 = stage_norm([value.detach().float() - initial[name].float() for name, value in parameters.items()])
    require(before_frozen == frozen_record(layer) and before_schema == schema(layer)
            and before_bits == storage_bits(layer), "Fixed bank bytes or persistent schema/storage changed.")
    require(identities == {name: id(value) for name, value in layer._parallel_stage_decoder.named_parameters()},
            "Optimization replaced registered parameter objects.")
    report.update(trajectory=trajectory, index_stream_sha256=batch_hash.hexdigest(),
        block_count=int(target.numel() // 32), processed_blocks=sum(item["blocks"] for item in trajectory),
        stage_gradient_l2_max=stage_gradient_max, stage_parameter_delta_l2_fp32=delta_fp32,
        stage_parameter_delta_l2_bf16=delta_bf16, optimizer_moment_dtypes=moments,
        fixed_payload=before_frozen, fixed_payload_unchanged=True, persistent_schema_unchanged=True,
        persistent_bits_before=before_bits, persistent_bits_after=storage_bits(layer),
        final_decoder={name: tensor_record(value) for name, value in layer._parallel_stage_decoder.state_dict().items()},
        settings={"steps": STEPS, "batch_size": BATCH_SIZE, "shuffle_seed": SEED, "optimizer": "AdamW",
            "lr": LR, "betas": [.9, .95], "eps": EPS, "weight_decay": 0, "schedule": "linear", "warmup": 0,
            "decoder_storage_training": "float32", "decoder_storage_deployment": "bfloat16",
            "decoder_compute": "bfloat16", "target_dtype": "float32", "stage_sum_dtype": "bfloat16",
            "normalizer": "Fixed complete original rotated target mean(W^2); no recentering or stage renormalization.",
            "normalizer_value": denominator, "sampling": "Complete shuffled epochs without replacement, retain partial batch."})
    return after_deploy.cpu(), report
