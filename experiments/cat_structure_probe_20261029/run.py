"""Four bounded architecture probes through the existing CAT fitting/export path."""
import argparse
from dataclasses import asdict, replace
import logging
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from experiments.cat_structure_probe_20261029 import common as c
from experiments.cat_biaxial_scale.training import recipe, build_host
from experiments.cat_biaxial_scale.runtime import save_endpoints, load_endpoint
from litebsq.vae_args import resolve_autoencoder_arch_spec
from train_utils.cat_train_pipeline import train_group_vae_payload
from train_utils.utils import configure_deterministic_mode, set_seed


def architecture(arm, steps):
    dim, bits, encoder, decoder, batch = c.ARMS[arm]
    argv, config, training, vae = recipe(str(c.source()['base_model_path']), 'gate_proj', steps, True)
    config = replace(config, codebook_dim=dim, codebook_bits=bits, base_ch=encoder,
        decoder_type='asymmetric', decoder_base_ch=decoder, decoder_num_res_blocks=0, steps=steps)
    from types import SimpleNamespace
    resolved = resolve_autoencoder_arch_spec(SimpleNamespace(**{**vars(vae), **asdict(config)}))
    c.require(resolved.codebook_dim == dim and resolved.codebook_bits == bits
              and resolved.encoder_hidden_dim == encoder and resolved.decoder_hidden_dim == decoder,
              'Actual independent encoder/decoder structure differs from the requested arm.')
    return argv, config, training, vae, batch, resolved


@torch.no_grad()
def observe(prepared, weights, endpoints):
    capture = c.read(prepared/'capture.json')
    c.require(capture['status'] == 'complete_live_prefix_capture' and capture['source_checkpoint_id'] == c.SOURCE_ID,
              'Use captured inputs from the real shared compressed prefix.')
    activations = torch.load(prepared/'activations.pt', map_location='cpu', weights_only=True, mmap=True)
    split = sum(activations['document_lengths'][:64])
    rows = {}
    for name, item in endpoints.items():
        endpoint = load_endpoint(item['path'], device='cuda:0')
        actual = endpoint.decode_weight()
        reference = weights['targets'][name]['reference_weight'].to('cuda:0')
        prefix = name.rsplit('.', 1)[0]
        up = weights['other_MLP'][prefix+'.up_proj.weight'].to('cuda:0')
        down = weights['other_MLP'][prefix+'.down_proj.weight'].to('cuda:0')
        inputs = activations['inputs'][name]
        measurements = {}
        for label, begin, end in (('first_train64', 0, split), ('last_train64', split, len(inputs))):
            sums = torch.zeros(4, dtype=torch.float64, device='cuda:0')
            for start in range(begin, end, 256):
                x = inputs[start:min(start+256, end)].to('cuda:0')
                expected_gate = torch.nn.functional.linear(x, reference)
                actual_gate = torch.nn.functional.linear(x, actual)
                same_up = torch.nn.functional.linear(x, up)
                expected_mlp = torch.nn.functional.linear(torch.nn.functional.silu(expected_gate)*same_up, down)
                actual_mlp = torch.nn.functional.linear(torch.nn.functional.silu(actual_gate)*same_up, down)
                sums += torch.stack(((actual_gate.double()-expected_gate.double()).square().sum(),
                    expected_gate.double().square().sum(), (actual_mlp.double()-expected_mlp.double()).square().sum(),
                    expected_mlp.double().square().sum()))
            c.require(torch.isfinite(sums).all().item() and sums[1] > 0 and sums[3] > 0, 'Invalid real local BF16 readout.')
            measurements[label] = dict(tokens=end-begin, linear_NMSE=float(sums[0]/sums[1]),
                mlp_NMSE=float(sums[2]/sums[3]), raw_sums=sums.cpu().tolist())
        rows[name] = dict(weight_NMSE=float((actual.double()-reference.double()).square().sum()/reference.double().square().sum()),
            observations=measurements)
        del endpoint, actual, reference, up, down
    return dict(layers=rows, source_checkpoint_id=c.SOURCE_ID, data='The same128 predetermined FineWeb train documents; previously seen, not blind heldout.',
        arithmetic='Actual BF16 gate/up Linear, BF16 SiLU/product, BF16 down Linear; FP64 error sums.',
        boundary='Conditional same-prefix linear/MLP diagnostics; not complete downstream accuracy or full-model compressed bpw.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(c.ARMS), required=True)
    parser.add_argument('--steps', type=int, choices=(2, 10000), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES', '').isdigit()
              and Path(sys.executable).resolve() == Path('/home/star/miniconda3/envs/bitvae/bin/python').resolve()
              and not args.output.exists(), 'Use caller-activated bitvae, one assigned GPU and a fresh output.')
    torch.set_num_threads(4)
    configure_deterministic_mode(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.INFO)
    prepared = c.OUTPUT/'prepared'
    record = c.read(prepared/'prepared.json')
    c.require(c.sha(prepared/'weights.pt') == record['weights']['sha256']
              and c.stat(c.SOURCE/c.source()['state_dict_file']) == record['source_weights'], 'Actual shared source changed.')
    capture = c.read(prepared/'capture.json')
    c.require(c.sha(prepared/'activations.pt') == capture['activations']['sha256'], 'Captured real prefix inputs changed.')
    weights = torch.load(prepared/'weights.pt', map_location='cpu', weights_only=True, mmap=True)
    argv, config, training, vae, batch, resolved = architecture(args.arm, args.steps)
    host, refs = build_host(weights['targets'], 'gate_proj')
    plans = {name: value['protected_input_indices'] for name, value in weights['targets'].items()}
    c.require(len(refs) == 3 and all(not ref.transpose for ref in refs), 'All arms use the same three untransposed live gates.')
    source_files = [*c.HERE.glob('*.py'), Path(train_group_vae_payload.__code__.co_filename),
        Path(save_endpoints.__code__.co_filename), ROOT/'litebsq/autoencoder.py', ROOT/'litebsq/bsq.py',
        ROOT/'litebsq/parallel_layers.py', ROOT/'litebsq/vae_args.py', ROOT/'rotation/weight_preprocess.py']
    pins = {str(path): c.sha(path) for path in source_files}
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', arm=args.arm, steps_per_stage=args.steps, stages=2,
        command=[sys.executable, *sys.argv], pid=os.getpid(), started_unix=time.time(),
        source_checkpoint_id=c.SOURCE_ID, prepared_sha256=c.sha(prepared/'prepared.json'),
        capture_sha256=c.sha(prepared/'capture.json'), source_sha256=pins,
        resolved_category=asdict(config), resolved_autoencoder=asdict(resolved), batch_per_linear=batch,
        actual_group_size=3, batch_scalars_per_linear=batch*config.codebook_dim,
        base_native_recipe=argv, implementation='Original CAT M3 pooled BSQ entropy, shared across all four arms.',
        short_only=args.steps == 2)
    c.write(args.output/'report.json', result)
    try:
        set_seed(31)
        payload = train_group_vae_payload(model=host, group_refs=refs, group_tag='structure_'+args.arm,
            runtime_cfg=config, vae_args=vae, training_args=training, train_device='cuda:0', convert_device='cuda:0',
            do_convert=True, batch_size=batch, gpu_resident_data=True, log_every=100 if args.steps>2 else 1,
            eval_every=0, eval_blocks=256, channel_protect_mode='channel', channel_plan=plans,
            channel_rank_metric='channel_weight_abs', channel_axis='input', channel_quant='int8',
            deterministic=True, shuffle_seed=31)
        endpoints = save_endpoints(host, refs, payload, [None]*3, args.output/'endpoints', torch.device('cuda:0'))
        for name, item in endpoints.items():
            endpoint = load_endpoint(item['path'], device='cpu')
            for field in ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales'):
                c.require(torch.equal(getattr(endpoint.codec, field), weights['targets'][name][field]),
                          'Candidate changed the fixed live-source protection state.')
            item.update(file_bytes=Path(item['path']).stat().st_size, sha256=c.sha(item['path']))
            decoder_bits = item['ledger']['components']['decoder']['logical_bits']
            actual_decoder_parameters = resolved.decoder_hidden_dim * (resolved.codebook_bits + resolved.codebook_dim + 3) + resolved.codebook_dim
            c.require(decoder_bits == 2 * actual_decoder_parameters * 16,
                      'Charge the actual two BF16 decoders, including normalization and biases.')
            maximum_decoder_bits = 2 * (256 * (32 + 32 + 3) + 32) * 16
            cap = item['ledger']['logical_bits'] - decoder_bits + maximum_decoder_bits
            item.update(common_linear_cap_bits=cap, unused_linear_cap_bits=cap-item['ledger']['logical_bits'],
                cap_scope='Same fixed2 code rate and fixed protection/rotation, plus the largest declared D-arm decoder; file container bytes reported separately.')
        result.update(endpoints=endpoints, observations=observe(prepared, weights, endpoints),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        c.require(all(c.sha(path) == expected for path, expected in pins.items()), 'Implementation changed during native execution.')
        result.update(status='completed_native_probe', exit_code=0)
    except BaseException:
        result.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        c.write(args.output/'report.json', result)


if __name__ == '__main__':
    main()
