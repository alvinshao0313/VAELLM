"""One residual-stage VAE fit; the main loss is the only A/B switch."""
from __future__ import annotations

import json
import torch
from transformers import get_scheduler
from litebsq.llm_vae import MultiLayerVAE
from train_utils.cat_train_data import restore_stage_norm
from train_utils.train_args import create_optimizer
from .artifacts import state_digest
from .config import vae_arguments
from .objectives import full_vae_forward
from .output_kernel import output_mse


def initialize_stage(args, stage, count):
    torch.manual_seed(args.seed + stage)
    vae = MultiLayerVAE(vae_arguments(args)).to(args.device).train()
    optimizer = create_optimizer(vae.parameters(), vae.args, args.vae_learning_rate)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=count)
    return vae, optimizer, scheduler


def optimize_stage(blocks, residual_weight, mean, scale, *, name, stage, count, offset, args, next_inputs, log):
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.vae_autocast_dtype == "bf16" and device.type == "cuda" else torch.float32
    blocks = blocks.to(device=device, dtype=dtype)
    target = residual_weight.to(device)
    mean, scale = mean.to(device), scale.to(device)
    vae, optimizer, scheduler = initialize_stage(args, stage, count)
    initial_hash = state_digest(vae)
    for step in range(count):
        inputs = next_inputs(name).detach().to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstructed, auxiliary, _ = full_vae_forward(vae, blocks, chunk_vectors=args.vae_chunk_vectors)
        weight_mse = (reconstructed.float() - blocks.float()).square().mean()
        if args.objective == "linear_output_mse":
            restored = restore_stage_norm(reconstructed.float(), mean=mean, scale=scale)
            main = output_mse(restored, target, inputs, use_triton=True)
        else:
            main = weight_mse
        loss = main * vae.model.l1_weight * vae.model.num_models + auxiliary
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite loss: {name}, stage {stage}, step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        global_step = offset + step + 1
        if step == 0 or global_step % args.log_every == 0 or step + 1 == count:
            item = {
                "module": name, "stage": stage, "stage_step": step + 1, "step": global_step,
                "total_steps": args.steps, "objective": args.objective, "loss": float(loss.detach()),
                "main_loss": float(main.detach()), "auxiliary_loss": float(auxiliary.detach()),
                "normalized_weight_mse": float(weight_mse.detach()), "grad_norm": float(grad_norm),
                "valid_tokens": len(inputs), "sequences": args.batch_size,
                "next_learning_rate": scheduler.get_last_lr()[0],
            }
            line = json.dumps(item)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
        del reconstructed, auxiliary, main, weight_mse, loss, inputs, _
        if args.objective == "linear_output_mse":
            del restored
    del optimizer, scheduler
    vae.eval()
    outputs, bits = [], []
    with torch.no_grad():
        for block in blocks.split(args.vae_chunk_vectors):
            output, code = vae(block, is_train=False)
            outputs.append(output.float().cpu())
            bits.append(code.detach().cpu())
    return vae, torch.cat(outputs), torch.cat(bits), initial_hash
