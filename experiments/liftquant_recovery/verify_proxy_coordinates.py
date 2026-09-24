"""Real-payload validation of calibrated proxies; no language-model training."""
import argparse
import copy
import json
from pathlib import Path
import warnings

import torch

from litebsq.autoencoder import Decoder
from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool
from experiments.liftquant_recovery.all_bits import decode_sample, decode_with_proxy
from experiments.liftquant_recovery.liftquant_optimizer import build_optimizer, advance_learning_rates
from experiments.liftquant_recovery.proxy_coordinates import dense_ste, hard_bits, measure_scale, motion_budget
from experiments.liftquant_recovery.verify_decoder_gradients import independent_dense, difference
from experiments.liftquant_recovery.verify_recovery import check_ste


def check_decoder(state, spec):
    name = spec['name']
    config = {k: v for k, v in spec['decoders'][0].items() if k != 'param_dtype'}
    if (config['decoder_type'], config['in_dim'], config['out_dim'], config['hidden_dim'],
            config['num_res_blocks'], config['norm_type'], config['activation_type']) != (
            'symmetric', 64, 32, 128, 0, 'layer', 'swish'):
        raise ValueError(f'Independent reference does not cover {config}')
    decoder = Decoder(**config, num_models=1).float().cuda().eval().requires_grad_(True)
    prefix = name+'._parallel_stage_decoder.'
    decoder.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                             if key.startswith(prefix)}, strict=True)
    bank = state[name+'.vq_weight']
    logical_shape = tuple(spec['vq_weights'][0]['logical_shape'])
    # Only selected code rows go to the GPU, not the complete code bank.
    indices = torch.linspace(0, len(bank)-1, min(256, len(bank))).round().long()
    packed = bank.index_select(0, indices).contiguous().cuda()
    shape = (len(indices), *logical_shape[1:])
    geometry = measure_scale(packed, shape, lambda p: decode_sample(decoder, p))
    geometry['source_row_indices'] = indices.tolist()
    scale = geometry['scale']
    bits = unpack_uint8_tensor_to_bool(packed, logical_shape=shape)
    p = ((bits.float()-.5)*scale).requires_grad_()
    repacked = pack_bool_tensor_to_uint8(hard_bits(p.detach(), scale), logical_shape=shape)
    torch.testing.assert_close(repacked, packed, rtol=0, atol=0)
    baseline = decode_sample(decoder, packed)
    reference_decoder = copy.deepcopy(decoder)
    rp = p.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = decode_with_proxy(decoder, packed, p, scale, torch.bfloat16)
        reference = independent_dense(reference_decoder, dense_ste(rp, scale), torch.bfloat16,
                                      packed_rounding=True)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    gradient = torch.randn_like(actual).float()/actual.numel()**.5
    (actual.float()*gradient).sum().backward()
    (reference.float()*gradient).sum().backward()
    torch.testing.assert_close(p.grad, rp.grad, rtol=1e-5, atol=1e-7)
    comparisons = {}
    refs = dict(reference_decoder.named_parameters())
    for key, param in decoder.named_parameters():
        torch.testing.assert_close(param.grad, refs[key].grad, rtol=1e-5, atol=1e-7)
        comparisons[key] = difference(param.grad, refs[key].grad)
    lr = min(2e-5, p.detach().std().item()/50)
    geometry['sample_proxy_std'] = p.detach().std().item()
    geometry['lr'] = lr
    geometry['planned_budget'] = motion_budget(lr, 3968, scale/2)
    if geometry['planned_budget']['structurally_unreachable']:
        raise ValueError(f'{name}: measured coordinates still cannot cross under the planned budget.')
    result = dict(name=name, geometry=geometry, unchanged_initial_codes=True,
                  unchanged_initial_decoder_output=True,
                  code_gradient=difference(p.grad, rp.grad), decoder_gradients=comparisons)
    del decoder, reference_decoder, p, rp, actual, reference, packed, bits
    torch.cuda.empty_cache()
    return result


def quantizer_regression(linears):
    # This is a scalar unit test of the quantizer+optimizer, not model recovery.
    cases = [('old_0_1', 1.0, 2e-5)] + [(r['name'], r['geometry']['scale'], r['geometry']['lr']) for r in linears]
    params = [torch.nn.Parameter(torch.tensor(-scale/2)) for _, scale, _ in cases]
    optimizer, schedulers = build_optimizer([dict(params=[p], lr=lr)
                                             for p, (_, _, lr) in zip(params, cases)], 3968)
    first_flip = [None]*len(cases)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for step in range(3968):
            optimizer.zero_grad(set_to_none=True)
            loss = sum((dense_ste(p, scale)-1).square() for p, (_, scale, _) in zip(params, cases))
            loss.backward()
            optimizer.step()
            advance_learning_rates(optimizer, schedulers)
            for i, (p, (_, scale, _)) in enumerate(zip(params, cases)):
                if first_flip[i] is None and hard_bits(p.detach(), scale).item():
                    first_flip[i] = step+1
    if first_flip[0] is not None or any(x is None for x in first_flip[1:]):
        raise AssertionError(f'Quantizer regression failed: {first_flip}')
    return dict(objective='synthetic scalar (hard_bit - 1)^2, initially hard_bit=0',
                meaning='proves optimizer can cross for a known gradient; not evidence of real model quality',
                old_0_1_first_flip=first_flip[0],
                calibrated_first_flip=dict(zip([name for name, _, _ in cases[1:]], first_flip[1:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    meta = json.loads((args.checkpoint/'checkpoint_meta.json').read_text())
    state = torch.load(args.checkpoint/meta['state_dict_file'], map_location='cpu', mmap=True, weights_only=True)
    report = dict(status='RUNNING', checkpoint_id=meta['checkpoint_id'], torch_version=torch.__version__,
                  device=torch.cuda.get_device_name(0), linear_ste=check_ste('cuda'), linears=[])
    for spec in meta['converted_modules']:
        if not any(spec['name'].startswith(f'model.layers.{i}.') for i in (9, 10)):
            continue
        report['linears'].append(check_decoder(state, spec))
        print(f"verified {spec['name']}", flush=True)
        (args.output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    if len(report['linears']) != 14:
        raise ValueError('Expected fourteen real compressed Linear payloads.')
    report['scalar_regression'] = quantizer_regression(report['linears'])
    report.update(status='PASS', max_memory_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    (args.output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'linears'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
