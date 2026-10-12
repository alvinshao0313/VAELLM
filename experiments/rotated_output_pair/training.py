"""Fresh matched S2 fits: full input-axis output error versus normalized weight MSE."""
from copy import deepcopy
import gc
import json
import math
import time

import torch
from transformers import get_scheduler

from experiments.linear_output.artifacts import state_digest
from experiments.linear_output.objectives import full_vae_forward
from experiments.linear_output.output_kernel import output_mse
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.llm_vae import MultiLayerVAE
from litebsq.protected_channel_quant import decode_protected_channel_weight
from rotation.weight_preprocess import build_weight_rotation
from train_utils.cat_train_data import compute_stage_norm_stats, apply_stage_norm, restore_stage_norm
from train_utils.cat_train_pipeline import _fuse_norm_into_decoder, _fuse_q_scale_into_decoder
from train_utils.train_args import create_optimizer

from .data import ARMS, CHUNK, SEED, STEPS


def complete_weight(rotated, rotation, keep, protected, indices):
    result = rotated.new_zeros((12288, 4096))
    result = result.index_copy(1, keep, rotation(rotated, inverse=True))
    return result.index_copy(1, indices, protected)


def initial_states(args):
    states = []
    for stage in range(2):
        torch.manual_seed(SEED + stage)
        vae = MultiLayerVAE(deepcopy(args))
        require(all(p.dtype == torch.bfloat16 for p in vae.parameters()), 'Actual VAE parameters must be BF16.')
        states.append({k: v.detach().clone() for k, v in vae.state_dict().items()})
    return states


def fit_arm(arm, data, material, args, starts, output, *, deadline, steps=STEPS, diagnostic_steps=()):
    require(arm in ARMS, 'Unknown paired objective.')
    require(steps > 0 and steps % 4 == 0 and all(0 < n <= steps for n in diagnostic_steps), 'Invalid fixed budget/observations.')
    if diagnostic_steps:
        from .diagnostics import physical_sse, reconstruct_readonly
    device = torch.device('cuda:0'); started = time.time()
    meta = material.split_metas[0]
    indices = meta.protected_input_indices.to(device)
    mask = torch.ones(4096, dtype=torch.bool, device=device); mask[indices] = False
    keep = mask.nonzero().flatten()
    rotation = build_weight_rotation(12288, 4032, data['spec']['weight_rotation']).to(device)
    weight = data['weight'].to(device=device, dtype=torch.float32)
    protected = decode_protected_channel_weight(meta.protected_input_qvalues, meta.protected_input_scales,
        quant_format='int8', dtype=torch.bfloat16, device=device).T.float()
    rotated = rotation(weight.index_select(1, keep)).detach()
    residual = rotated.reshape(-1, 1, 32).clone()
    prior = torch.zeros_like(rotated)
    inputs = data['capture']['inputs']['train'].to(device)
    lengths = data['capture']['document_lengths']['train']
    documents = inputs.split(lengths)
    trace = (inputs.index_select(1, keep).double().square().sum() / len(inputs)).item()
    require(math.isfinite(trace) and trace > 0, 'Calibration trace must be positive.')
    payload_bits, payload_decoders, stages = [], [], []
    with (output / (arm + '.jsonl')).open('x') as log:
        for stage in range(2):
            mean, scale = compute_stage_norm_stats(residual)
            blocks = apply_stage_norm(residual, mean=mean, scale=scale).to(torch.bfloat16)
            vae = MultiLayerVAE(deepcopy(args)).to(device).train()
            vae.load_state_dict(starts[stage], strict=True)
            initial_sha = state_digest(vae)
            optimizer = create_optimizer(vae.parameters(), args, args.lr)
            scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=steps)
            generator = torch.Generator().manual_seed(SEED + stage)
            orders = torch.cat([torch.randperm(32, generator=generator).reshape(4, 8) for _ in range(steps // 4)])
            denominator = float(scale.item()) ** 2 * trace
            rows, first_codes = [], None
            fixed_observations, tail_codes = [], None
            initial_parameters = {k: v.detach().clone() for k, v in vae.named_parameters()}
            for step, order in enumerate(orders):
                require(time.time() < deadline, 'Paired fit deadline exceeded; no partial endpoint is successful.')
                x = torch.cat([documents[int(i)] for i in order])
                optimizer.zero_grad(set_to_none=True)
                decoded, auxiliary, codes = full_vae_forward(vae, blocks, chunk_vectors=CHUNK, recompute=False)
                if step == 0:
                    first_codes = codes.detach().cpu().clone()
                if diagnostic_steps and step == steps - 32:
                    tail_codes = codes.detach().cpu().clone()
                normalized_mse = (decoded.float() - blocks.float()).square().mean()
                if arm == 'linear_output_mse':
                    current = restore_stage_norm(decoded.float(), mean=mean, scale=scale).reshape_as(prior)
                    actual = complete_weight(prior + current, rotation, keep, protected, indices)
                    physical_output_mse = output_mse(actual, weight, x, use_triton=True)
                    main = physical_output_mse / denominator
                else:
                    main = normalized_mse
                loss = args.l1_weight * main + auxiliary
                require(bool(torch.isfinite(loss)), 'Nonfinite full-matrix objective.')
                loss.backward()
                gradients = {}
                for group in ('encoder', 'decoder'):
                    values = [p.grad for p in getattr(vae.model, group).parameters()]
                    require(all(v is not None and bool(torch.isfinite(v).all()) for v in values), 'Invalid ' + group + ' gradient.')
                    gradients[group] = math.sqrt(sum(float(v.float().square().sum()) for v in values))
                    require(gradients[group] > 0, 'Zero ' + group + ' gradient.')
                optimizer.step(); scheduler.step()
                if step == 0 or (step + 1) % 32 == 0:
                    row = dict(stage=stage, step=step + 1, loss=float(loss.detach()), main=float(main.detach()),
                        auxiliary=float(auxiliary.detach()), normalized_weight_mse=float(normalized_mse.detach()),
                        valid_tokens=len(x), decoder_encoder_grad_norm=gradients, lr=scheduler.get_last_lr()[0])
                    if arm == 'linear_output_mse':
                        row['physical_output_mse'] = float(physical_output_mse.detach())
                    rows.append(row); log.write(json.dumps(row) + '\n'); log.flush()
                    print(json.dumps({'arm': arm, **row}), flush=True)
                del decoded, auxiliary, codes, normalized_mse, main, loss, x
                if arm == 'linear_output_mse':
                    del current, actual, physical_output_mse
                if step + 1 in diagnostic_steps:
                    observed = reconstruct_readonly(vae, blocks, mean, scale, chunk_vectors=CHUNK)
                    physical = complete_weight(prior + observed.reshape_as(prior), rotation, keep, protected, indices)
                    fixed_observations.append({'after_update': step + 1, 'stage': stage,
                        'fixed_train_fp32_unfused': physical_sse(physical, weight, {'train': inputs})['train']})
                    del observed, physical
            vae.eval(); recon, bits = [], []
            with torch.no_grad():
                for chunk in blocks.split(CHUNK):
                    value, code = vae(chunk, is_train=False)
                    recon.append(value.float()); bits.append(code.cpu())
                reconstruction = restore_stage_norm(torch.cat(recon), mean=mean, scale=scale)
            full_bits = torch.cat(bits)
            update_counts = {group: sum(int((p.detach() != initial_parameters[k]).sum())
                for k, p in vae.named_parameters() if k.startswith('model.' + group + '.')) for group in ('encoder', 'decoder')}
            require(all(update_counts.values()), 'Encoder or decoder has no surviving BF16 updates.')
            stages.append(dict(stage=stage, updates=steps, initial_state_sha256=initial_sha,
                initial_state={k: tensor_record(v) for k, v in starts[stage].items()},
                normalization_mean=float(mean.item()), normalization_std=float(scale.item()),
                trace_H_CC=trace, output_denominator=denominator, token_order=tensor_record(orders),
                source_residual_rms=float(residual.square().mean().sqrt()),
                final_residual_rms=float((residual - reconstruction).square().mean().sqrt()),
                final_code_xor_vs_first_forward=int((full_bits != first_codes).sum()),
                encoder_decoder_bf16_changed_elements=update_counts, trajectory=rows))
            if diagnostic_steps:
                require(tail_codes is not None, 'Missing fixed final32-step code snapshot.')
                stages[-1].update(fixed_train_observations=fixed_observations,
                    final32_net_code_xor=int((full_bits != tail_codes).sum()),
                    final32_from_after_update=steps - 32, final32_to_after_update=steps,
                    final32_boundary='Net XOR between actual codes after updates992 and1024; not cumulative flip events.')
            prior.add_(reconstruction.reshape_as(prior)); residual.sub_(reconstruction)
            decoder = vae.model.decoder.get_sub_decoder(0)
            _fuse_q_scale_into_decoder(decoder, q_scale=1 / math.sqrt(32))
            _fuse_norm_into_decoder(decoder, mean=float(mean.item()), std=float(scale.item()))
            payload_bits.append(full_bits); payload_decoders.append([decoder.cpu()])
            del vae, optimizer, scheduler, blocks
            del recon, bits, reconstruction, first_codes, initial_parameters, tail_codes
            gc.collect(); torch.cuda.empty_cache()
    payload = dict(format='vaellm_group_vae_payload', version=1, target_common_split_metas=material.split_metas,
        parts_per_linear=1, row_parts=1, col_parts=1, residual_stages=2,
        all_stage_bits=payload_bits, all_stage_decoders=payload_decoders, all_stage_codebook_dims=[32, 32],
        all_stage_split_metas=[material.split_metas, material.split_metas],
        protected_channel_quant_format='int8', weight_rotation_specs=[data['spec']['weight_rotation']])
    report = dict(arm=arm, stages=stages, elapsed_seconds=time.time() - started,
        train_matrix_shape=[12288, 4032], all_vectors_per_update=1548288,
        full_matrix_BSQ_calls_per_update=1, precision='BF16 parameters/encoder/decoder; FP32 reconstruction objectives; native FP32 BSQ',
        training_stage_sum_physical_weight=tensor_record(complete_weight(prior, rotation, keep, protected, indices).detach().cpu()))
    if diagnostic_steps:
        physical = complete_weight(prior, rotation, keep, protected, indices)
        report['unfused_fp32_local_fixed_prefix'] = physical_sse(physical, weight, data['capture']['inputs'])
    return payload, report
