"""Real G09 -> one compact replacement -> single checkpoint -> separate fresh process."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from experiments.adaptive_vector_rvq import common as c
from experiments.adaptive_native_model_20261029 import checkpoint
from experiments.adaptive_native_model_20261029.codec import load_input
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear

PREFIX = 'model.layers.18.mlp.down_proj'
INPUT = Path('/data/star/VAELLM/adaptive_grouped_fit_20261029/grouped/endpoint_18/weighted_dynamic3.avrvqz')
INPUT_SHA = 'd32286371be510ea87e86ed167f4c291a0083f19bea25675334044f14635e2aa'


def event(name, **fields):
    print(json.dumps(dict(event=name, **fields)), flush=True)


def equal_state(actual, expected, context):
    c.require(set(actual) == set(expected), context + ': tensor names changed.')
    for name, tensor in actual.items():
        other = expected[name]
        c.require(tensor.dtype == other.dtype and tensor.shape == other.shape and torch.equal(tensor.cpu(), other.cpu()),
                  context + ': actual tensor changed: ' + name)
    return len(actual)


def fresh(directory, output):
    # Trace the effective weight reads in a distinct Python process, without parent binding imports.
    reads = []
    torch_load = torch.load

    def local_load(filename, *args, **kwargs):
        filename = Path(filename).resolve()
        c.require(filename == (directory / checkpoint.STATE_FILE).resolve(), 'Fresh loader attempted to read external weights.')
        reads.append(str(filename))
        return torch_load(filename, *args, **kwargs)

    torch.load = local_load
    model, meta, loaded, budget = checkpoint.load(directory)
    actual = model.state_dict()
    payload = torch_load(directory / checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    compared = equal_state(actual, payload, 'Separate-process fresh deployment')
    layer = model.get_submodule(PREFIX)
    c.require(all(value.dtype == torch.bfloat16 for value in layer.decoders.parameters()), 'Fresh deploy parameters must remain BF16.')
    c.require(not loaded.missing_keys and not loaded.unexpected_keys and all(not p.requires_grad for p in model.parameters()),
              'Fresh deployment must be strict and initially frozen.')
    c.write(output, dict(status='fresh_strict_verified', pid=os.getpid(), checkpoint_id=meta['checkpoint_id'],
        compared_tensors=compared, deployment_digest=c.digest(actual), requested_weight_files=reads,
        no_external_parent_weight_read=True, native_modules=251, compact_modules=1,
        compact_FP32_training_masters_saved=False, bit_budget=budget,
        FP32_RoPE=all(tensor.dtype == torch.float32 for name, tensor in model.named_buffers() if 'inv_freq' in name)))
    event('fresh_strict_verified', tensors=compared, weight_reads=reads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.fresh:
        fresh(args.fresh.resolve(), args.output.resolve())
        return
    output = args.output.resolve()
    c.require(not output.exists() and not torch.cuda.is_initialized(), 'Use a new independent CPU validation output.')
    output.mkdir(parents=True)
    report = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv], started_unix=time.time(),
                  schema='adaptive_native_model_CPU_validation_v1', boundary='Interface/storage only; historical compact target differs from G09.')
    c.write(output / 'report.json', report)
    try:
        from experiments.g09_recovery_20261029 import bindings as b
        directory, meta, audit = b.source()
        c.require(c.sha(INPUT) == INPUT_SHA, 'Use the actual retained grouped L18 input.')
        report.update(source_checkpoint=str(directory), source_checkpoint_id=meta['checkpoint_id'],
                      compact_input=dict(path=str(INPUT), sha256=INPUT_SHA, bytes=INPUT.stat().st_size))
        event('real_source_verified', path=str(directory), checkpoint_id=meta['checkpoint_id'])
        model, parent_state, parent_loaded = checkpoint.load_parent(directory, meta)
        c.require(not parent_loaded.missing_keys and not parent_loaded.unexpected_keys, 'Parent source must load strict.')
        report['parent_compared_tensors'] = equal_state(model.state_dict(), parent_state, 'Real G09 initial')
        bundle, lengths, spec = load_input(INPUT)
        c.require(spec['geometry']['name'] == PREFIX, 'Use the actual L18 projection identity.')
        layer = CompactRVQLinear(bundle, lengths)
        old_names = {name for name in parent_state if name.startswith(PREFIX + '.')}
        checkpoint.native.set_module_by_name(model, PREFIX, layer)
        rest = {name: value for name, value in model.state_dict().items() if not name.startswith(PREFIX + '.')}
        old_rest = {name: value for name, value in parent_state.items() if not name.startswith(PREFIX + '.')}
        count = equal_state(rest, old_rest, '251 projections and all nonprojection parent state')
        report.update(resolved_spec=spec, removed_parent_target_tensor_names=sorted(old_names),
                      exact_unchanged_nonreplacement_tensors=count, parent_tensor_digest=c.digest(parent_state))
        event('single_replacement_verified', unchanged_tensors=count, removed_parent_tensors=len(old_names), spec=spec)
        saved, budget, serialized = checkpoint.save(model, output / 'model', provenance=dict(
            parent_checkpoint_id=meta['checkpoint_id'], parent_checkpoint_path=str(directory),
            parent_meta_sha256=c.sha(directory / 'checkpoint_meta.json'),
            compact_input=report['compact_input'], historical_target_mismatch=True,
            use='Full-model serializer/consumer validation; not a compression quality experiment'))
        equal_state({name: tensor for name, tensor in serialized.items() if not name.startswith(PREFIX + '.')}, old_rest,
                    'Actually serialized nonreplacement state')
        report.update(saved_checkpoint_id=saved['checkpoint_id'], bit_budget=budget,
                      serialized_deployment_digest=c.digest(serialized), serialized_tensors=len(serialized))
        event('single_complete_model_saved', bit_budget=budget)
        del model, parent_state, rest, old_rest, serialized, layer, bundle
        gc.collect()
        child = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--fresh', str(output / 'model'),
                                '--output', str(output / 'fresh.json')], cwd=ROOT, check=True)
        verified = c.read(output / 'fresh.json')
        c.require(verified['deployment_digest'] == report['serialized_deployment_digest']
                  and verified['compared_tensors'] == report['serialized_tensors'] and child.returncode == 0,
                  'Separate-process serialized/fresh deployment fields differ.')
        report.update(status='CPU_verified_GPU_pending', exit_code=0, fresh=verified,
                      cuda_initialized=torch.cuda.is_initialized(), no_dense_replacement_backup=True,
                      no_base_plus_overlay_payload=True,
                      source_hashes={str(path): c.sha(path) for path in Path(__file__).parent.glob('*.py')})
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'report.json', report)
    event('CPU_verified_GPU_pending', elapsed_seconds=report['elapsed_seconds'], tensors=report['serialized_tensors'])


if __name__ == '__main__':
    main()
