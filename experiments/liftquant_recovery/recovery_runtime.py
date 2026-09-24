"""CPU-resident model, one GPU block at a time; actual HF block arguments."""
import hashlib
from pathlib import Path

import torch

from litebsq.vae_linear import VAELinear


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(tree_to(x, device) for x in value)
    if isinstance(value, list):
        return [tree_to(x, device) for x in value]
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    return value


def clear_caches(block):
    for module in block.modules():
        if isinstance(module, VAELinear):
            module.clear_decoded_weight_cache()


@torch.no_grad()
def first_inputs(teacher, ids, batch_size, device):
    class Captured(Exception):
        pass

    backbone = teacher.model
    capture = {}
    def hook(_module, args, kwargs):
        capture["kwargs"] = tree_to(kwargs, "cpu")
        capture["hidden"] = args[0].detach().cpu()
        raise Captured()

    backbone.embed_tokens.to(device)
    backbone.rotary_emb.to(device)
    handle = backbone.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        try:
            backbone(input_ids=ids[:batch_size].to(device), use_cache=False)
        except Captured:
            pass
        else:
            raise RuntimeError("Could not capture first-block inputs.")
        captured_hidden = capture["hidden"]
        hidden = torch.empty((len(ids), *captured_hidden.shape[1:]), dtype=captured_hidden.dtype, device="cpu")
        hidden[:len(captured_hidden)].copy_(captured_hidden)
        for start in range(batch_size, len(ids), batch_size):
            hidden[start:start + batch_size].copy_(backbone.embed_tokens(ids[start:start + batch_size].to(device)).cpu())
    finally:
        handle.remove()
        backbone.embed_tokens.cpu()
        backbone.rotary_emb.cpu()
    # Fixed length, unpadded sequences; all batches share exact positions/masks.
    kwargs = capture["kwargs"]
    if kwargs.get("attention_mask") is not None:
        raise ValueError("Expected unpadded FlashAttention-2 calibration.")
    return hidden, kwargs


def block_output(block, hidden, kwargs):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = block(hidden, **kwargs)
    return result[0]


@torch.no_grad()
def teacher_outputs(block, hidden, kwargs, batch_size, device):
    block.to(device).eval()
    gpu_kwargs = tree_to(kwargs, device)
    result = torch.empty_like(hidden, device="cpu")
    for start in range(0, len(hidden), batch_size):
        result[start:start + batch_size] = block_output(
            block, hidden[start:start + batch_size].to(device), gpu_kwargs,
        ).cpu()
    block.cpu()
    torch.cuda.empty_cache()
    return result


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def frozen_digest(model, allowed_names):
    digest = hashlib.sha256()
    for name, value in model.state_dict(keep_vars=True).items():
        if name in allowed_names:
            continue
        digest.update(name.encode())
        digest.update(tensor_digest(value).encode())
    return digest.hexdigest()


def targets_by_block(model, meta):
    targets = {}
    for path in meta["compressed_targets"]:
        pieces = path.split(".")
        if pieces[:2] != ["model", "layers"]:
            raise ValueError(f"Unsupported target path: {path}")
        module = model.get_submodule(path)
        if not isinstance(module, VAELinear):
            raise TypeError(path)
        targets.setdefault(int(pieces[2]), []).append((path, module))
    return targets


def audit_topology(model, meta, selected, targets):
    from experiments.liftquant_recovery.all_bits import decoder_for
    owners = {}
    allowed_ids = set()
    for index, modules in targets.items():
        for path, module in modules:
            if module.residual_stages != 1 or module.parallel_parts != 1:
                raise ValueError(f"{path}: expected stage=1, part=1")
            decoder = decoder_for(module)
            for parameter in decoder.parameters():
                owners.setdefault(id(parameter), set()).add(index)
                if index in selected:
                    allowed_ids.add(id(parameter))
            if index in selected:
                allowed_ids.add(id(module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)))
    shared = [sorted(value) for value in owners.values() if len(value) > 1 and value.intersection(selected)]
    if shared:
        raise ValueError(f"Cross-block decoder sharing would violate freezing: {shared[:5]}")
    for name, module in model.named_modules():
        if "lora" in name.lower():
            raise ValueError(f"Unexpected LoRA topology: {name}")
    return allowed_ids


def source_fingerprint(path):
    root = Path(path)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("checkpoint_meta.json", "config.json")}


@torch.no_grad()
def prime_packed_cache(block):
    """Use the SAME native packed-u8 BF16 decoder arithmetic as training.

    Native cached inference otherwise prefers the whole-decoder fused kernel,
    whose intermediate rounding differs. This changes residency/prewarm policy
    only; all weights, topology and serialization remain canonical v6.
    """
    for module in block.modules():
        if isinstance(module, VAELinear):
            if getattr(module, "_recovery_runtime", None) is not None:
                raise ValueError("Prewarm is inference-only; remove the recovery adapter first.")
            module.clear_decoded_weight_cache()
            module.cache_decoded_weight = False
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    module.prime_decoded_weight_cache(dtype=torch.bfloat16)
            finally:
                module.cache_decoded_weight = True
