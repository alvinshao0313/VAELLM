from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


CATEGORIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BLOCK_ROUTES = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-block LiftQuant-style VAELLM smoke harness")
    p.add_argument("--preflight", action="store_true", help="only validate imports and public contracts")
    p.add_argument("--model-path", default="Qwen/Qwen3-8B")
    p.add_argument("--output-root", default=None)
    p.add_argument("--blocks", default="0,1")
    p.add_argument("--calib-dataset", default="redpajama=1.0")
    p.add_argument("--calib-samples", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=64)
    p.add_argument("--seed", type=int, default=31)
    return p.parse_args()


def preflight() -> None:
    from litebsq.vae_linear import VAELinear
    from sparse_bit_tuning.config import SparseBitTuningConfig
    from sparse_bit_tuning.manager import SparseBitTuningManager

    required = (
        "enable_trainable_sparse_bit_decode_graph",
        "get_stage_part_vq_storage",
        "get_stage_part_vq_spec",
    )
    missing = [name for name in required if not hasattr(VAELinear, name)]
    if missing:
        raise RuntimeError(f"VAELinear contract is missing: {missing}")
    cfg = SparseBitTuningConfig(enabled=True, active_ratio=1.0, optimizer="adamw", bit_lr=2e-5)
    cfg.normalized()
    assert SparseBitTuningManager is not None
    print(json.dumps({"status": "preflight_pass", "vae_linear": VAELinear.__name__, "sparse_bit_config": cfg.normalized().__dict__}, sort_keys=True))


def run_stage_a(args: argparse.Namespace, root: Path) -> Path:
    stage_a_root = root / "stage_a"
    stage_a_root.mkdir(parents=True, exist_ok=False)
    cmd = [
        sys.executable, "tools/cat_train.py", "--model_path", args.model_path,
        "--output_dir", str(stage_a_root), "--compression_categories", ",".join(CATEGORIES),
        "--target_layers", "0-1", "--skip_layers", "", "--after_category_mode", "none",
        "--seed", str(args.seed), "--data_seed", str(args.seed), "--deterministic", "false",
        "--train_device", "cuda", "--convert", "true", "--save_model", "true", "--convert_device", "cuda",
        "--allow_tail_group", "true", "--transpose_modules", "q_proj,v_proj,o_proj,down_proj", "--linear_group_size", "36",
        "--vae_steps", "default=1", "--vae_batch_size", "2", "--vae_learning_rate", "3e-3", "--vae_weight_decay", "0",
        "--vae_optim", "adamw", "--vae_lr_scheduler_type", "constant", "--vae_warmup_ratio", "0",
        "--activation_calib_dataset", "redpajama=1.0", "--activation_calib_nsamples", str(max(2, args.calib_samples)),
        "--activation_calib_seqlen", str(args.seqlen), "--activation_calib_seed", str(args.seed), "--activation_calib_device", "",
        "--activation_calib_log_every", "0", "--codebook_bits", "default=32", "--codebook_dim", "default=32",
        "--residual_stages", "default=2", "--base_ch", "default=128", "--num_res_blocks", "default=0",
        "--decoder_base_ch", "default=128", "--decoder_num_res_blocks", "default=1", "--norm_type", "default=layer",
        "--activation_type", "default=swish", "--decoder_type", "default=symmetric", "--recon_loss_type", "default=mse",
        "--quantizer_type", "BSQ", "--gamma0", "1", "--gamma", "1", "--zeta", "1", "--inv_temperature", "100",
        "--beta1", "0.9", "--beta2", "0.95", "--l1_weight", "1", "--lfq_weight", "2.5",
        "--commitment_loss_weight", "0.25", "--entropy_loss_weight", "0.01", "--normalize_weight", "true",
        "--weight_rotation", "none", "--weight_rotation_block_size", "0", "--vae_decoder_checkpoint", "true", "--new_quant", "true",
        "--log_every", "1", "--eval_every", "0", "--eval_blocks", "256", "--skip_ppl_eval", "true", "--eval_tasks", "",
        "--channel_protect_mode", "none", "--channel_rank_metric", "channel_weight_actmean_abs", "--channel_mlp_rank_metric", "none",
        "--channel_mlp_fuse_weights", "1,1,1", "--channel_scope", "layer", "--channel_min_per_layer", "0", "--channel_quant", "none",
        "--channel_axis", "input", "--channel_protect_count", "default=0", "--channel_refresh_after_category", "false",
        "--bf16", "true", "--fp16", "false",
    ]
    env = os.environ.copy(); env["PYTHONPATH"] = "."; env["HF_HUB_OFFLINE"] = "1"; env["HF_DATASETS_OFFLINE"] = "1"
    print("[stage A] starting fresh CAT VAE initialization", flush=True)
    subprocess.run(cmd, check=True, env=env)
    candidates = sorted(stage_a_root.glob("*/final_model"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"stage A did not produce final_model under {stage_a_root}")
    checkpoint = candidates[-1]; meta_path = checkpoint / "checkpoint_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"stage A final model has no checkpoint_meta.json: {checkpoint}")
    meta = json.loads(meta_path.read_text())
    if str(meta.get("checkpoint_kind")) != "final_model" or str(meta.get("after_category_mode", "none")) != "none":
        raise RuntimeError(f"stage A checkpoint metadata is not pure initialization: {meta.get('checkpoint_kind')!r}/{meta.get('after_category_mode')!r}")
    return checkpoint


def find_module_path(model: torch.nn.Module, target: torch.nn.Module) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name
    raise RuntimeError(f"could not find module path for {target!r}")


def block_targets(model: torch.nn.Module, block_idx: int):
    from litebsq.vae_linear import VAELinear
    from rotation.model_utils import get_layers
    layer = get_layers(model)[int(block_idx)]; out = []
    for parent_name, attr_name in BLOCK_ROUTES:
        module = getattr(getattr(layer, parent_name), attr_name)
        if not isinstance(module, VAELinear):
            raise TypeError(f"block {block_idx} {parent_name}.{attr_name} is {type(module).__name__}, expected VAELinear")
        out.append((find_module_path(model, module), module))
    return out


def decoder_modules(module: torch.nn.Module) -> List[torch.nn.Module]:
    packed = getattr(module, "_parallel_stage_decoder", None)
    if isinstance(packed, torch.nn.Module):
        return [packed]
    out: List[torch.nn.Module] = []; seen = set()
    for stage_idx in range(int(getattr(module, "residual_stages", 1))):
        for part_idx in range(int(getattr(module, "parallel_parts", 1))):
            decoder = module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
            if id(decoder) not in seen:
                seen.add(id(decoder)); out.append(decoder)
    return out


def decoder_parameters(model: torch.nn.Module, targets) -> Tuple[List[torch.nn.Parameter], Dict[str, torch.nn.Parameter]]:
    by_id: Dict[int, torch.nn.Parameter] = {}
    for _path, module in targets:
        for decoder in decoder_modules(module):
            for param in decoder.parameters():
                by_id.setdefault(id(param), param)
    name_by_id = {id(param): name for name, param in model.named_parameters()}; named = {}
    for pid, param in by_id.items():
        if pid not in name_by_id:
            raise RuntimeError("decoder parameter is not reachable from model.named_parameters()")
        named[name_by_id[pid]] = param
    return list(named.values()), named


def student_block_output(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, block_idx: int, fp_prefix: torch.Tensor):
    from rotation.model_utils import get_layers
    layer = get_layers(model)[int(block_idx)]; holder: Dict[str, torch.Tensor] = {}
    def replace_input(_module, hook_args, hook_kwargs):
        if hook_args:
            return (fp_prefix,) + tuple(hook_args[1:]), hook_kwargs
        updated = dict(hook_kwargs); updated["hidden_states"] = fp_prefix; return hook_args, updated
    def capture_after(_module, hook_args, _output):
        if hook_args: holder["actual_input"] = hook_args[0]
    pre_handle = layer.register_forward_pre_hook(replace_input, with_kwargs=True); post_handle = layer.register_forward_hook(capture_after)
    try:
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        pre_handle.remove(); post_handle.remove()
    actual = holder.get("actual_input")
    if actual is None: raise RuntimeError("student block pre-hook did not expose hidden input")
    prefix_error = float((actual.float() - fp_prefix.float()).abs().max().item())
    if prefix_error > 1e-5: raise RuntimeError(f"FP-prefix injection mismatch at block {block_idx}: max_abs={prefix_error}")
    return outputs.hidden_states[int(block_idx) + 1], prefix_error


def make_calibration(args: argparse.Namespace, device: torch.device):
    from transformers import AutoTokenizer
    from train_utils.lora_data import build_calibration_input_ids
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    samples = build_calibration_input_ids(args.calib_dataset, tokenizer=tokenizer, nsamples=int(args.calib_samples), seqlen=int(args.seqlen), seed=int(args.seed))
    if not samples: raise RuntimeError("calibration builder returned zero samples")
    input_ids = torch.cat(samples, dim=0).to(device=device); attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def assert_freeze_contract(model: torch.nn.Module, decoder_ids: set[int], score_ids: set[int]) -> None:
    trainable = {id(p): name for name, p in model.named_parameters() if p.requires_grad}; allowed = decoder_ids | score_ids
    unexpected = {name for pid, name in trainable.items() if pid not in allowed}
    if unexpected: raise RuntimeError(f"unexpected trainable parameters: {sorted(unexpected)[:12]}")
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(token in lname for token in ("norm", "bias", "lm_head")) and id(param) not in allowed and param.requires_grad:
            raise RuntimeError(f"frozen contract violated by {name}")


def run_recovery(args: argparse.Namespace, root: Path, checkpoint: Path) -> dict:
    from rotation.model_utils import get_model
    from sparse_bit_tuning.config import SparseBitTuningConfig
    from sparse_bit_tuning.manager import SparseBitTuningManager
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    torch.manual_seed(int(args.seed)); device = torch.device("cuda")
    input_ids, attention_mask = make_calibration(args, device)
    print(f"[stage B] calibration shape={tuple(input_ids.shape)} dataset={args.calib_dataset}", flush=True)
    teacher = get_model(args.model_path); teacher.to(device); teacher.eval()
    with torch.no_grad():
        teacher_outputs = teacher.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, return_dict=True)
    student = load_v6_model_checkpoint(str(checkpoint), base_model_path=args.model_path, map_location="cpu", strict=True); student.to(device); student.eval()
    for param in student.parameters(): param.requires_grad_(False)
    requested_blocks = tuple(int(x.strip()) for x in args.blocks.split(",") if x.strip())
    if len(requested_blocks) != 2 or requested_blocks[1] != requested_blocks[0] + 1: raise ValueError(f"smoke requires exactly two adjacent blocks, got {requested_blocks}")
    records = []; hard_outputs: Dict[int, torch.Tensor] = {}; sidecars: List[Path] = []
    for block_idx in requested_blocks:
        targets = block_targets(student, block_idx); target_paths = [path for path, _module in targets]
        if len(targets) != 7 or {path.rsplit(".", 1)[-1] for path in target_paths} != set(CATEGORIES): raise RuntimeError(f"block {block_idx} target set mismatch: {target_paths}")
        for param in student.parameters(): param.requires_grad_(False)
        target_devices = {path: next(module.parameters()).device for path, module in targets}
        manager = SparseBitTuningManager(root_model=student, targets=targets, target_devices=target_devices, training_seed=int(args.seed) + int(block_idx), config=SparseBitTuningConfig(enabled=True, active_ratio=1.0, optimizer="adamw", bit_lr=2e-5, weight_decay=0.0, round_steps=1), streaming=False)
        for _path, module in targets: module.enable_trainable_sparse_bit_decode_graph(parallel_stage_decode=False)
        manager.initialize_scores();
        if any(int(spec.n_active) != int(spec.n_bits) for spec in manager.bank_specs): raise RuntimeError("smoke requires all-bit score proxies")
        manager.configure_schedule(total_optimizer_steps=1)
        dec_params, named_dec_params = decoder_parameters(student, targets)
        if not dec_params: raise RuntimeError(f"block {block_idx} has no trainable decoder parameters")
        decoder_ids = {id(p) for p in dec_params}; score_ids = manager.score_module.bit_parameter_ids(); assert_freeze_contract(student, decoder_ids, score_ids)
        optimizer = torch.optim.AdamW(dec_params, lr=2e-4, weight_decay=0.0)
        fp_prefix = teacher_outputs.hidden_states[int(block_idx)].detach(); fp_target = teacher_outputs.hidden_states[int(block_idx) + 1].detach()
        optimizer.zero_grad(set_to_none=True); manager.score_module.clear_grads(); student.train()
        student_block, prefix_error = student_block_output(student, input_ids, attention_mask, block_idx, fp_prefix)
        token_mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(student_block); loss = F.mse_loss(student_block.float()[token_mask], fp_target.float()[token_mask])
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite block {block_idx} loss: {loss.item()}")
        loss.backward(); manager.bit_optimizer.validate_gradients()
        if not any(param.grad is not None for param in dec_params): raise RuntimeError(f"block {block_idx} decoder gradients are disconnected")
        optimizer.step(); telemetry = manager.optimizer_step(); manager.final_commit(); student.eval()
        with torch.no_grad(): hard_output, hard_prefix_error = student_block_output(student, input_ids, attention_mask, block_idx, fp_prefix)
        hard_outputs[block_idx] = hard_output.detach().cpu(); packed_snapshot = manager.checkpoint_packed_snapshot(); sidecar = root / f"block_{block_idx}_repacked.pt"
        torch.save({"format": "vaellm_liftquant_smoke_block", "version": 1, "stage_a_checkpoint": str(checkpoint), "block_idx": int(block_idx), "module_paths": target_paths, "packed_banks": packed_snapshot, "decoder_state": {name: param.detach().cpu() for name, param in named_dec_params.items()}, "loss": float(loss.item()), "telemetry": telemetry.__dict__}, sidecar)
        sidecars.append(sidecar); records.append({"block": int(block_idx), "targets": target_paths, "loss": float(loss.item()), "prefix_max_abs": float(max(prefix_error, hard_prefix_error)), "decoder_gradients": int(sum(param.grad is not None for param in dec_params)), "score_chunks": int(len(manager.score_module.score_chunks)), "telemetry": telemetry.__dict__, "sidecar": str(sidecar)})
        manager.detach_runtime(); del optimizer, manager; gc.collect(); torch.cuda.empty_cache()
    del student; gc.collect(); torch.cuda.empty_cache()
    reloaded = load_v6_model_checkpoint(str(checkpoint), base_model_path=args.model_path, map_location="cpu", strict=True); reloaded.to(device); reloaded.eval(); reload_checks = []
    for sidecar in sidecars:
        payload = torch.load(sidecar, map_location="cpu", weights_only=False); block_idx = int(payload["block_idx"]); targets = block_targets(reloaded, block_idx); named = dict(reloaded.named_parameters())
        for name, value in payload["decoder_state"].items():
            if name not in named: raise RuntimeError(f"reloaded model missing decoder parameter {name}")
            named[name].data.copy_(value.to(device=named[name].device, dtype=named[name].dtype))
        target_devices = {path: next(module.parameters()).device for path, module in targets}
        manager = SparseBitTuningManager(root_model=reloaded, targets=targets, target_devices=target_devices, training_seed=int(args.seed) + block_idx, config=SparseBitTuningConfig(enabled=True, active_ratio=1.0, optimizer="adamw", bit_lr=2e-5, round_steps=1), streaming=False)
        manager.restore_checkpoint_packed(payload["packed_banks"]); manager.detach_runtime()
        for _path, module in targets: module.disable_trainable_decode()
        fp_prefix = teacher_outputs.hidden_states[block_idx].detach()
        with torch.no_grad(): output, prefix_error = student_block_output(reloaded, input_ids, attention_mask, block_idx, fp_prefix)
        saved = hard_outputs[block_idx].to(device=output.device, dtype=output.dtype); max_error = float((output - saved).float().abs().max().item())
        if max_error > 1e-4: raise RuntimeError(f"block {block_idx} repack/reload mismatch: max_abs={max_error}")
        reload_checks.append({"block": block_idx, "max_abs_reload_error": max_error, "prefix_max_abs": float(prefix_error)})
    result = {"status": "smoke_pass", "stage_a_checkpoint": str(checkpoint), "blocks": records, "reload_checks": reload_checks, "downstream_eval": "not_run_in_smoke"}
    (root / "smoke_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True)); return result


def main() -> None:
    args = parse_args()
    if args.preflight: preflight(); return
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for the target-model smoke test")
    timestamp = time.strftime("%Y%m%d_%H%M%S"); root = Path(args.output_root or f".result/liftquant_recovery/smoke_{timestamp}"); root.mkdir(parents=True, exist_ok=False)
    checkpoint = run_stage_a(args, root); result = run_recovery(args, root, checkpoint); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
