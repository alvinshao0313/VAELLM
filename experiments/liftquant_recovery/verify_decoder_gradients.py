"""Small, permanent full-decoder gradient diagnostic on real checkpoint payloads.

Loads the checkpoint with CPU mmap, selects at most 512 packed rows per block-9
Linear, and never constructs the language model or changes a checkpoint. Outputs
measurements, not a blanket PASS: mixed-precision forward and backward conventions
must be distinguished before interpreting error magnitudes.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from litebsq.autoencoder import Decoder
from litebsq.bitpack import unpack_uint8_tensor_to_bool
from experiments.liftquant_recovery.all_bits import decode_with_proxy


@torch.no_grad()
def difference(actual, reference):
    a, b = actual.detach().double().cpu().reshape(-1), reference.detach().double().cpu().reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    an, bn = a.norm().item(), b.norm().item()
    delta = a - b
    return dict(actual_l2=an, reference_l2=bn, max_abs=delta.abs().max().item(),
                relative_l2=delta.norm().item() / max(bn, 1e-30),
                cosine=max(-1., min(1., torch.dot(a, b).item() / (an * bn))) if an and bn else None,
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                exact_equal=bool(torch.equal(a, b)))


def independent_dense(decoder, scores, dtype, *, packed_rounding):
    """Independent math for this checkpoint's 64 -> LN128 -> SiLU -> 32 decoder.

    packed_rounding=True: reproduce the packed kernel's BF16 multiplicands,
    FP32 accumulation AND FP32 bias, then one final output cast. Straight-through
    rounding for W retains the kernel's FP32 weight-gradient accumulation. The
    resulting score gradient uses the rounded forward W, unlike the historical
    custom backward which uses unrounded FP32 W. This distinction is measured.

    packed_rounding=False: normal dense PyTorch semantics, including BF16 bias
    and GEMM output rounding. This is a different arithmetic path, not an oracle
    for bitwise equality with the native packed kernel.
    """
    p = dict(decoder.named_parameters())
    wi, bi = p['linear_in.linear.weight'], p['linear_in.linear.bias']
    if packed_rounding:
        with torch.autocast('cuda', enabled=False):
            # Explicit rounding STE models the native FP32 master-weight VJP.
            w_forward = wi + (wi.to(dtype).float() - wi).detach()
            h = F.linear(scores.float(), w_forward, bi.float()).to(dtype)
    else:
        h = F.linear(scores.to(dtype), wi.to(dtype), bi.to(dtype))
    norm = decoder.norm_out.norm
    h = F.layer_norm(h, norm.normalized_shape,
                     p['norm_out.norm.weight'].to(h.dtype),
                     p['norm_out.norm.bias'].to(h.dtype), norm.eps).to(dtype)
    h = F.silu(h)
    return F.linear(h, p['linear_out.linear.weight'].to(dtype),
                    p['linear_out.linear.bias'].to(dtype))


def evaluate(template, bits, packed, dtype, cotangent, kind):
    decoder = copy.deepcopy(template).cuda().float().eval().requires_grad_(True)
    scores = bits.float().detach().clone().requires_grad_(True)
    captured = {}
    hook = None
    if kind == 'actual_packed':
        def record_first_output(_module, inputs):
            h = inputs[0]
            if h.requires_grad:
                h.register_hook(lambda gradient: captured.update(first_grad=gradient.detach().clone()))
        hook = decoder.norm_out.register_forward_pre_hook(record_first_output)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        if kind == 'actual_packed':
            output = decode_with_proxy(decoder, packed, scores - 0.5, 1.0, dtype)
        elif kind == 'native_dense':
            output = decoder(scores.to(dtype))
        else:
            output = independent_dense(decoder, scores, dtype,
                                       packed_rounding=kind == 'dense_packed_contract')
    # Fixed cotangent isolates the decoder Jacobian; changing decoder output does
    # not also change the loss gradient as an MSE target would.
    loss = (output.float() * cotangent).sum()
    loss.backward()
    result = dict(output=output.detach().cpu(), scores=scores.grad.detach().cpu(),
                  parameters={name: p.grad.detach().cpu() for name, p in decoder.named_parameters()},
                  objective=loss.item())
    if hook is not None:
        hook.remove()
        g = captured['first_grad'].float()
        w = decoder.linear_in.linear.weight.detach()
        with torch.autocast('cuda', enabled=False):
            old_vjp = F.linear(g, w.T)
            forward_weight_vjp = F.linear(g, w.to(dtype).float().T)
            expected_dw = g.squeeze(1).T @ bits.float().squeeze(1)
            expected_db = g.sum(dim=(0, 1))
        result['local_vjp'] = dict(
            actual_scores_vs_fp32_weight=difference(scores.grad, old_vjp),
            actual_scores_vs_forward_rounded_weight=difference(scores.grad, forward_weight_vjp),
            fp32_vs_rounded_weight=difference(old_vjp, forward_weight_vjp),
            first_weight_vs_dense_reduction=difference(decoder.linear_in.linear.weight.grad, expected_dw),
            first_bias_vs_dense_reduction=difference(decoder.linear_in.linear.bias.grad, expected_db))
    if not all(torch.isfinite(t).all() for t in [result['output'], result['scores'], *result['parameters'].values()]):
        raise FloatingPointError(f'Nonfinite {kind} result')
    del decoder, scores, output, loss
    torch.cuda.empty_cache()
    return result


def compare(a, b):
    return dict(forward=difference(a['output'], b['output']),
                codes=difference(a['scores'], b['scores']),
                parameters={name: difference(a['parameters'][name], b['parameters'][name])
                            for name in a['parameters']})


def inspect_linear(state, spec, rows, seed):
    name = spec['name']
    config = {k: v for k, v in spec['decoders'][0].items() if k != 'param_dtype'}
    expected = dict(in_dim=64, out_dim=32, hidden_dim=128, num_res_blocks=0,
                    norm_type='layer', activation_type='swish', decoder_type='symmetric')
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Independent reference does not cover decoder config: {config}')
    if spec['residual_stages'] != 1 or spec['parallel_parts'] != 1:
        raise ValueError('This diagnostic requires one stage and one part.')
    prefix = name + '._parallel_stage_decoder.'
    payload = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    template = Decoder(**config, num_models=1).float().eval()
    template.load_state_dict(payload, strict=True)
    # The saved first-layer W/b already absorb the signed-code q_scale mapping.
    # Do not call _fuse_q_scale() again.
    bank = state[name + '.vq_weight']
    count = min(rows, len(bank))
    indices = torch.linspace(0, len(bank) - 1, count, dtype=torch.float64).round().long()
    packed = bank.index_select(0, indices).contiguous().cuda()
    logical = (count, 1, config['in_dim'])
    bits = unpack_uint8_tensor_to_bool(packed, logical_shape=logical)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    cotangent = (torch.randn(count, 1, config['out_dim'], generator=generator)
                 / math.sqrt(count * config['out_dim'])).cuda()
    runs, report = {}, dict(name=name, rows=count, logical_shape=logical,
                            source_row_indices=indices.tolist(), decoder_config=config)
    for precision, dtype in [('fp32', torch.float32), ('bf16', torch.bfloat16)]:
        cases = {kind: evaluate(template, bits, packed, dtype, cotangent, kind)
                 for kind in ['actual_packed', 'dense_packed_contract', 'native_dense', 'dense_native_math']}
        report[precision] = dict(
            packed_vs_independent_packed_contract=compare(cases['actual_packed'], cases['dense_packed_contract']),
            packed_vs_native_dense=compare(cases['actual_packed'], cases['native_dense']),
            native_dense_vs_independent_math=compare(cases['native_dense'], cases['dense_native_math']),
            local_vjp=cases['actual_packed']['local_vjp'])
        runs[precision] = cases['actual_packed']
    report['bf16_vs_fp32_actual_packed'] = compare(runs['bf16'], runs['fp32'])
    del packed, bits, cotangent, runs
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rows', default=512, type=int)
    args = parser.parse_args()
    if not 1 <= args.rows <= 512:
        raise ValueError('--rows must be between 1 and 512 to keep this diagnostic small.')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)  # Caller must restrict CUDA_VISIBLE_DEVICES to the approved GPU.
    cap = 2 * 2**30
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., cap / total), device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    meta = json.loads((args.checkpoint / 'checkpoint_meta.json').read_text())
    state = torch.load(args.checkpoint / meta['state_dict_file'], mmap=True,
                       map_location='cpu', weights_only=True)
    specs = [item for item in meta['converted_modules'] if item['name'].startswith('model.layers.9.')]
    if len(specs) != 7:
        raise ValueError(f'Expected seven block-9 compressed linears, found {len(specs)}.')
    report = dict(status='DIAGNOSTIC_COMPLETE_REQUIRES_INTERPRETATION',
                  checkpoint_id=meta['checkpoint_id'], torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), seed=42, rows=args.rows,
                  notes=[
                      'No optimizer, training, checkpoint mutation, full model, or decoded full matrix.',
                      'No permissive tolerance is used to label mixed-precision comparisons as PASS.',
                      'FP32 Triton tl.dot can use TF32 internally; torch TF32 flags affect PyTorch references only.',
                      'The packed-contract reference rounds forward W but retains FP32 master-weight gradients.',
                      'Both FP32-W and rounded-forward-W score VJPs are reported; current recovery uses the latter.',
                      'Native dense BF16 rounds bias and intermediate GEMM differently from the packed kernel.',
                      'This checks decoder Jacobians; it cannot establish the best block optimizer learning rate.'
                  ], linears=[])
    for spec in specs:
        report['linears'].append(inspect_linear(state, spec, args.rows, 42))
        print(f"inspected {spec['name']}", flush=True)
    report['max_memory_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    report['max_memory_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
    if torch.cuda.max_memory_reserved() > cap:
        raise RuntimeError('Allocator reserve exceeded the diagnostic memory budget.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'linears'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
