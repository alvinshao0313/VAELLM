"""Bounded fixed-code decoder fitting in the true original weight coordinates."""
from __future__ import annotations

import time

from native_pair import (DECODER_PREFIX, deployed_weight, rebuild, require, schema,
                         state_record, tensor_record)
from build_targets import endpoint_metrics, energy
import torch
from torch.nn import functional as F

STEPS, LR, SEED = 500, 1e-5, 31


def stage_norm(values, stages):
    result = torch.zeros(stages, dtype=torch.float64, device=values[0].device)
    for value in values:
        require(value.shape[0] % stages == 0, 'Packed decoder stage axis changed.')
        result += value.detach().double().reshape(stages, -1).square().sum(1)
    return result.sqrt().cpu().tolist()


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('The fixed 15-minute prototype budget expired; no shape/budget reduction or retry.')


def fit_target(spec, source_state, target, pre, post, protected, output, *, deadline, progress):
    device = post.device
    require(device.type == 'cuda' and target.dtype == torch.float32,
            'Fitting requires the assigned CUDA device and complete FP32 original-coordinate target.')
    check_deadline(deadline)
    started = time.perf_counter()
    model = rebuild(spec, source_state, device)
    module = model.projection
    original_schema = schema(module)
    frozen = {key: value for key, value in source_state.items() if not key.startswith(DECODER_PREFIX)}
    require(getattr(module, '_sparse_bit_binding', None) is None,
            'This fixed-code prototype must not bind a code optimizer or encoder.')
    native = deployed_weight(model)
    require(torch.equal(native, post.bfloat16()), 'Fresh G02 deployment failed the zero-intervention exact control.')
    report = dict(initial_native_exact_G02=True, before_deployment=endpoint_metrics(native, target, pre, post, protected))
    del native
    initial = {key: value.detach().clone() for key, value in module._parallel_stage_decoder.named_parameters()}
    require(len(initial) == 6 and all(value.dtype == torch.bfloat16 for value in initial.values()),
            'Expected the actual six finalized BF16 decoder leaves.')
    for value in module._parallel_stage_decoder.parameters():
        value.data = value.data.float()
    module.enable_trainable_decode(parallel_stage_decode=True)
    module._decoder_compute_dtype = torch.bfloat16
    module.training = True
    module._parallel_stage_decoder.eval()
    module._build_parallel_stage_decode_plan()
    parameters = dict(module._parallel_stage_decoder.named_parameters())
    identities = {key: id(value) for key, value in parameters.items()}
    require(all(value.requires_grad and value.dtype == torch.float32 for value in parameters.values()),
            'Only registered FP32 decoder masters may train.')
    require({key for key, value in module.named_parameters() if value.requires_grad} ==
            {DECODER_PREFIX+key for key in parameters}, 'A nondecoder parameter was made trainable.')
    denominator = energy(post)/post.numel()
    require(denominator > 0, 'The shared G02 energy normalizer must be positive.')
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        before_train = module._decode_weight(dtype=torch.bfloat16)
    report['before_master_train_path'] = endpoint_metrics(before_train, target, pre, post, protected)
    report['initial_train_vs_deployment'] = dict(
        error_energy=energy(before_train.float()-post),
        error_energy_over_move=energy(before_train.float()-post)/energy(target-post))
    progress(dict(phase='zero_update_live_vs_deployment',
                  comparison=report['initial_train_vs_deployment'],
                  calibration_direction_survival=report['before_master_train_path']['calibration_direction_survival']))
    require(report['initial_train_vs_deployment']['error_energy_over_move'] < 1,
            'The live initial path displacement reaches or exceeds the target calibration move: '
            f'{report["initial_train_vs_deployment"]}. Pause before fitting; do not attribute this path change to calibration.')
    del before_train
    optimizer = torch.optim.AdamW(parameters.values(), lr=LR, betas=(.9, .95), eps=1e-8, weight_decay=0)
    trajectory, grad_max = [], [0.0]*module.residual_stages
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    for step in range(1, STEPS+1):
        check_deadline(deadline)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr'] = LR*(1-(step-1)/STEPS)
        with torch.autocast('cuda', enabled=False):
            reconstructed = module._decode_weight(dtype=torch.bfloat16)
            require(reconstructed.shape == target.shape and reconstructed.dtype == torch.bfloat16,
                    'The complete native original-coordinate reconstruction geometry changed.')
            raw_mse = F.mse_loss(reconstructed.float(), target)
            loss = raw_mse/denominator
        require(bool(torch.isfinite(loss)), f'Nonfinite loss at fixed update {step}.')
        loss.backward()
        gradients = [value.grad for value in parameters.values()]
        require(all(value is not None and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()) for value in gradients),
                f'Missing, non-FP32 or nonfinite decoder gradient at update {step}.')
        norms = stage_norm(gradients, module.residual_stages)
        grad_max = [max(a,b) for a,b in zip(grad_max, norms, strict=True)]
        optimizer.step()
        torch.cuda.synchronize(device)
        item = dict(step=step, lr=optimizer.param_groups[0]['lr'], raw_mse=float(raw_mse),
                    nmse_loss=float(loss), stage_gradient_l2=norms,
                    step_seconds=time.perf_counter()-tick)
        trajectory.append(item)
        if step <= 2 or step % 100 == 0:
            progress(item)
        del reconstructed, raw_mse, loss
    report['training_seconds'] = time.perf_counter()-training_started
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        master_train = module._decode_weight(dtype=torch.bfloat16)
    report['after_master_train_path'] = endpoint_metrics(master_train, target, pre, post, protected)
    require(energy((master_train.float()-post)[:, protected]) == 0,
            'Training changed a protected input column.')
    del master_train
    moments = sorted({str(state[key].dtype) for state in optimizer.state.values()
                      for key in ('exp_avg', 'exp_avg_sq')})
    require(moments == ['torch.float32'], 'Adam moments must remain FP32.')
    require(all(int(state['step']) == STEPS for state in optimizer.state.values()),
            'Every decoder leaf must receive the fixed 500 updates.')
    master_state = {key: value.detach().cpu().clone() for key, value in module._parallel_stage_decoder.state_dict().items()}
    fp32_delta = stage_norm([value.detach()-initial[key].float() for key,value in parameters.items()], module.residual_stages)
    del optimizer
    for key, value in parameters.items():
        value.grad = None
        value.data = value.data.to(initial[key].dtype)
    module._build_parallel_stage_decode_plan()
    native = deployed_weight(model)
    report['after_BF16_deployment'] = endpoint_metrics(native, target, pre, post, protected)
    require(energy((native.float()-post)[:, protected]) == 0, 'BF16 deployment changed protected columns.')
    final = module.state_dict()
    require(schema(module) == original_schema, 'Final persistent schema/dtype changed.')
    require(all(torch.equal(value, final[key].cpu()) for key,value in frozen.items()),
            'A code bank, protected q/scale, rotation or other frozen field changed.')
    require(identities == {key:id(value) for key,value in module._parallel_stage_decoder.named_parameters()},
            'Optimization replaced registered decoder parameter objects.')
    bf16_delta = stage_norm([value.detach().float()-initial[key].float() for key,value in parameters.items()], module.residual_stages)
    decoder_state = {key: value.detach().cpu().clone()
                     for key,value in module._parallel_stage_decoder.state_dict().items()}
    # This is an isolated module delta payload, not a full native model checkpoint.
    payload = dict(format='geometric_native_single_projection_v1', projection=spec['name'],
                   decoder_master_FP32=master_state, decoder_deployment_BF16=decoder_state,
                   frozen_source=state_record(frozen), schema=original_schema)
    require(not output.exists(), 'Isolated decoder output already exists; refuse overwrite.')
    torch.save(payload, output)
    saved = torch.load(output, map_location='cpu', weights_only=True)
    require(saved['format'] == payload['format'] and saved['frozen_source'] == payload['frozen_source'],
            'Isolated saved module contract changed.')
    require(state_record(saved['decoder_master_FP32']) == state_record(master_state)
            and state_record(saved['decoder_deployment_BF16']) == state_record(decoder_state),
            'Saved FP32 or BF16 decoder tensor values changed.')
    fresh_state = dict(source_state)
    fresh_state.update({DECODER_PREFIX+key:value for key,value in saved['decoder_deployment_BF16'].items()})
    fresh = rebuild(spec, fresh_state, device)
    fresh_native = deployed_weight(fresh)
    require(torch.equal(native, fresh_native), 'Fresh strict native BF16 decode changed after isolated save/reload.')
    report['after_saved_fresh_BF16'] = endpoint_metrics(fresh_native, target, pre, post, protected)
    report.update(status='completed_fixed_500_endpoint', steps=STEPS, trajectory=trajectory,
                  optimizer_moment_dtypes=moments, stage_gradient_l2_max=grad_max,
                  stage_parameter_delta_FP32=fp32_delta, stage_parameter_delta_BF16=bf16_delta,
                  frozen_fields_unchanged=list(frozen), schema_unchanged=True,
                  initial_decoder=state_record(initial), final_decoder=state_record(decoder_state),
                  saved_decoder_path=str(output), fresh_saved_native_BF16_exact=True,
                  weight_scalars_per_update=target.numel(), weight_scalar_exposures=STEPS*target.numel(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  elapsed_seconds=time.perf_counter()-started,
                  settings=dict(steps=STEPS, optimizer='AdamW', lr=LR, betas=[.9,.95], eps=1e-8,
                    weight_decay=0, schedule='linear500', warmup=0, seed=SEED,
                    loss='Full original-coordinate FP32 MSE / shared fixed G02 mean(Wpost^2)',
                    denominator=denominator, sampling='Every complete matrix on every update; no text or sample selection',
                    master_and_moments='FP32', decoder_compute_stage_sum_inverse_rotation='Native BF16',
                    endpoint='Fixed update500, no loss-based checkpoint selection'))
    del model, fresh, native, fresh_native
    return report
