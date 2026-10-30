"""Bounded real-source CPU audit of native -> compact fixed2 import."""
import argparse
from copy import deepcopy
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

from experiments.adaptive_native_model_20261029 import native_prefix as p
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_model_20261029.validate_cpu import equal_state
from experiments.adaptive_vector_rvq import common as c
from litebsq.autoencoder import pack_decoders
from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool
import torch


def event(name, **fields):
    print(json.dumps(dict(event=name, **fields)), flush=True)


def metrics(reference, actual):
    error = actual.double() - reference.double()
    denominator = float(reference.double().square().sum())
    squared = float(error.square().sum())
    return dict(elements=reference.numel(), max_abs=float(error.abs().max()),
                relative_L2=(squared / denominator) ** .5 if denominator else 0.,
                MSE=squared / reference.numel(), changed_elements=int(torch.count_nonzero(reference != actual)))


@torch.no_grad()
def compact_selected(layer, ids, dtype=torch.bfloat16):
    decoders = [deepcopy(decoder).to(dtype=dtype).eval() for decoder in layer.decoders]
    stages = [[] for _ in decoders]
    for start in range(0, len(ids), c.CHUNK):
        selected = ids[start:start + c.CHUNK]
        for index, decoder in enumerate(decoders):
            codes = getattr(layer, f'codes_{index}').index_select(0, selected)
            k = int(layer.resolved['stages'][index]['decoder_spec']['in_dim'])
            bits = unpack_uint8_tensor_to_bool(codes, logical_shape=(len(selected), k))
            stages[index].append(decoder(bits.to(dtype).unsqueeze(1)).squeeze(1))
    values = [torch.cat(stage) for stage in stages]
    summed = torch.zeros_like(values[0])
    for value in values:
        summed = summed + value
    return {**{f'stage{index}': value for index, value in enumerate(values)}, 'sum': summed}


@torch.no_grad()
def numerical(module, layer, ids):
    # These are real selected native code rows. This is the actual CPU fallback
    # packed Decoder operation, not the CUDA packed-u8/fused first-linear path.
    source_bits = []
    for index in range(2):
        spec = module.get_stage_part_vq_spec(index, 0)
        codes = module.get_stage_part_vq_storage(index, 0).index_select(0, ids)
        shape = (len(ids), 1, int(spec['logical_shape'][-1]))
        bits = unpack_uint8_tensor_to_bool(codes, logical_shape=shape)
        c.require(torch.equal(pack_bool_tensor_to_uint8(bits, logical_shape=shape), codes), 'Actual selected code bits did not roundtrip.')
        source_bits.append(bits)
    rows = {}
    compact_bf16 = None
    for dtype in (torch.float32, torch.bfloat16):
        packed = deepcopy(module._parallel_stage_decoder).to(dtype=dtype).eval()
        packed.use_checkpoint = False
        singles = [module.get_stage_part_decoder(index, 0).to(dtype=dtype).eval() for index in range(2)]
        for decoder in singles:
            decoder.use_checkpoint = False
        packed_parts, single_parts = [], [[], []]
        for start in range(0, len(ids), c.CHUNK):
            block = [bits[start:start + c.CHUNK].to(dtype) for bits in source_bits]
            packed_parts.append(packed(torch.cat(block, dim=1)))
            for index, decoder in enumerate(singles):
                single_parts[index].append(decoder(block[index]).squeeze(1))
        packed_values = torch.cat(packed_parts)
        native = dict(stage0=packed_values[:, 0], stage1=packed_values[:, 1], sum=packed_values.sum(dim=1))
        first, second = (torch.cat(parts) for parts in single_parts)
        oracle = dict(stage0=first, stage1=second, sum=(torch.zeros_like(first) + first) + second)
        compact = compact_selected(layer, ids, dtype)
        c.require(all(torch.equal(compact[name], oracle[name]) for name in oracle),
                  'New compact must be exact to the same independently extracted native stage operations.')
        difference = {name: metrics(native[name], oracle[name]) for name in native}
        if dtype == torch.float32:
            c.require(all(torch.allclose(native[name], oracle[name], rtol=2e-5, atol=1e-6)
                          and difference[name]['relative_L2'] <= 1e-4 for name in native),
                      'FP32 packed/extracted operations no longer have the established mathematical correspondence.')
        else:
            compact_bf16 = compact
        rows[str(dtype)] = dict(native_packed_vs_extracted=difference,
            extracted_vs_compact_exact=True, compact_vs_native={name: metrics(native[name], compact[name]) for name in native})
    return dict(selected_rows=len(ids), ids=ids.tolist() if len(ids) <= 3 else dict(first=int(ids[0]), last=int(ids[-1])),
                chunk_rows=[min(c.CHUNK, len(ids) - start) for start in range(0, len(ids), c.CHUNK)],
                actual_operator='Native CPU _parallel_stage_decoder fallback vs extracted independent M1 and compact M1; no CUDA path',
                FP32_tolerance=dict(rtol=2e-5, atol=1e-6, maximum_relative_L2=1e-4), dtypes=rows,
                scope='Only these actual vector rows before inverse rotation/protection; not full-Linear original-coordinate, GPU or downstream equivalence'), c.digest(compact_bf16)


def check_fields(module, original, bundle, lengths, resolved):
    compared = equal_state(module.state_dict(), original, 'Actual single native Linear strict state')
    c.require(bool((lengths == 2).all()) and torch.equal(p.decode_lengths(bundle['length_frame'], len(lengths)), lengths),
              'The actual all2 frame must reconstruct every length exactly.')
    singles = [module.get_stage_part_decoder(index, 0) for index in range(2)]
    repacked = pack_decoders(singles)
    decoder_compared = equal_state(repacked.state_dict(), module._parallel_stage_decoder.state_dict(), 'Extract/repack native decoder')
    for index, (stage, decoder) in enumerate(zip(bundle['stages'], singles, strict=True)):
        equal_state(stage['decoder_state'], decoder.state_dict(), 'Actual extracted current BF16 stage')
        storage = module.get_stage_part_vq_storage(index, 0)
        c.require(torch.equal(stage['codes'], storage[:, 0]) and stage['codes'].untyped_storage().data_ptr() != storage.untyped_storage().data_ptr(),
                  'Import must remove only the singleton code axis and create its own compact storage.')
        c.require(stage['normalization']['original_statistics'] == 'not_persisted_and_not_recoverable'
                  and 'mean' not in stage['normalization'] and 'scale' not in stage['normalization'],
                  'Do not invent original stage normalization statistics.')
    for name in ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales'):
        c.require(torch.equal(bundle['fixed'][name], getattr(module, name)), 'Actual native protection field changed.')
    for axis in ('input', 'output'):
        c.require(torch.equal(bundle['fixed'][f'rotation_{axis}_signs'], getattr(module.weight_rotation, axis + '_signs')),
                  'Actual native rotation sign changed.')
    geometry = resolved['geometry']
    c.require(geometry['transpose'] == module.transpose and geometry['compressed_shape'] ==
              [module.compressed_out_features, module.compressed_in_features]
              and resolved['n_vectors'] * resolved['vector_dim'] == module.compressed_out_features * module.compressed_in_features,
              'All actual compressed coordinates must be covered once, with the same native transpose.')
    return dict(native_fields_exact=compared, repacked_decoder_fields_exact=decoder_compared,
                complete_code_banks_exact=True, complete_code_bank_padding_zero=True,
                complete_BF16_decoder_fields_exact=True, protection_and_rotation_signs_exact=True,
                all2_frame_exact=True, resolved=resolved, source_state_digest=c.digest(original),
                bundle_tensor_digest=c.digest(p.inventory(bundle)),
                extracted_nonpersistent_q_scale_flags=[decoder._q_scale_fused for decoder in singles],
                no_repeated_fusion=True,
                layout=dict(native_transpose=bool(module.transpose), source_matrix_shape=geometry['compressed_shape'],
                    flattened_layout='Wrot.T contiguous' if module.transpose else 'Wrot contiguous',
                    coordinate_mapping='t=v*D+d; output=t%O_c,input=t//O_c' if module.transpose else 't=v*D+d; output=t//I_c,input=t%I_c'))


def fresh(path, output):
    torch_load = torch.load
    reads = []

    def only_prefix(filename, *args, **kwargs):
        c.require(Path(filename).resolve() == path.resolve(), 'The independent imported-prefix consumer attempted an external weight read.')
        reads.append(str(Path(filename).resolve()))
        return torch_load(filename, *args, **kwargs)

    torch.load = only_prefix
    bundle, lengths, resolved = p.load_prefix(path)
    layer = CompactRVQLinear(bundle, lengths)
    values = compact_selected(layer, torch.arange(8193, dtype=torch.long))
    c.write(output, dict(status='independent_native_prefix_fresh_verified', pid=os.getpid(),
        bundle_tensor_digest=c.digest(p.inventory(bundle)), compact_selected_digest=c.digest(values),
        actual_prefix_sha256=c.sha(path), actual_prefix_file_bytes=path.stat().st_size,
        requested_weight_files=reads, no_external_parent_weight_read=True,
        original_length_tensor_persisted=False, cuda_initialized=torch.cuda.is_initialized(),
        ledger=p.ledger(bundle), resolved=resolved))
    event('independent_native_prefix_fresh_verified', pid=os.getpid(), reads=reads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh', type=Path)
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU only; hide all CUDA devices before imports.')
    torch.set_num_threads(8)
    if args.fresh:
        fresh(args.fresh.resolve(), args.output.resolve())
        return
    output = args.output.resolve()
    c.require(not output.exists(), 'Use a new native-prefix validation output.')
    output.mkdir(parents=True)
    report = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv], started_unix=time.time(),
                  complete_model_constructed=False, GPU_used=False, cases={})
    c.write(output / 'report.json', report)
    try:
        from experiments.g09_recovery_20261029 import bindings as b
        directory, meta, audit = b.source()
        state = torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
        report.update(source_checkpoint_id=meta['checkpoint_id'], source_checkpoint=str(directory),
            source_meta_sha256=c.sha(directory / 'checkpoint_meta.json'), source_audit_sha256=c.sha(b.SOURCE_AUDIT),
            source_weight_stat=c.stat(directory / meta['state_dict_file']), only_selected_projection_payloads_touched=True)
        selected_bytes = 0
        for category, expected_k in (('up_proj', 32), ('gate_proj', 34), ('q_proj', 38), ('down_proj', 48)):
            spec = next(item for item in meta['converted_modules'] if item['name'].endswith('.' + category))
            name = spec['name']
            relative = {key[len(name) + 1:]: value for key, value in state.items() if key.startswith(name + '.')}
            before = c.digest(relative)
            selected_bytes += sum(value.numel() * value.element_size() for value in relative.values())
            provenance = dict(checkpoint_id=meta['checkpoint_id'], source_module=name,
                source_meta_sha256=report['source_meta_sha256'], source_weight_stat=report['source_weight_stat'],
                source_relative_state_digest=before, use='Import current native BF16 prefix; no new live target or dynamic routing')
            bundle, lengths, resolved = p.from_native(spec, relative, provenance)
            c.require(all(stage['decoder_spec']['in_dim'] == expected_k for stage in bundle['stages']), 'The actual fixture logical K changed.')
            module = p.load_native(spec, relative)
            layer = CompactRVQLinear(bundle, lengths)
            row = check_fields(module, relative, bundle, lengths, resolved)
            ids = torch.arange(8193, dtype=torch.long) if category == 'gate_proj' else torch.tensor([0, len(lengths) // 2, len(lengths) - 1])
            row['numerical'], compact_digest = numerical(module, layer, ids)
            row['ledger'] = p.ledger(bundle)
            c.require(c.digest(relative) == before, 'A real mmap source field changed during import/validation.')
            if category == 'gate_proj':
                candidate = output / 'gate.native_prefix.pt'
                row['actual_file'] = dict(path=str(candidate), **p.save_prefix(bundle, candidate), sha256=c.sha(candidate))
                row['compact_selected_digest'] = compact_digest
            report['cases'][category] = row
            event('real_native_prefix_case_verified', category=category, K=expected_k, N=len(lengths), numerical_rows=len(ids))
            c.write(output / 'report.json', report)
            del bundle, lengths, module, layer, relative
            gc.collect()
        gate = report['cases']['gate_proj']
        subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--fresh', str(output / 'gate.native_prefix.pt'),
                        '--output', str(output / 'fresh.json')], cwd=ROOT, check=True)
        verified = c.read(output / 'fresh.json')
        c.require(verified['bundle_tensor_digest'] == gate['bundle_tensor_digest']
                  and verified['compact_selected_digest'] == gate['compact_selected_digest']
                  and verified['no_external_parent_weight_read'] and not verified['cuda_initialized'],
                  'Fresh single-prefix fields or actual selected-row compact outputs differ.')
        report.update(status='native_prefix_CPU_verified', exit_code=0, independent_fresh=verified,
            selected_source_tensor_bytes=selected_bytes, cuda_initialized=torch.cuda.is_initialized(),
            source_sha256={str(path): c.sha(path) for path in (Path(p.__file__), Path(__file__),
                Path(__file__).with_name('spec.py'), Path(__file__).with_name('consumer.py'), Path(__file__).with_name('codec.py'))},
            boundary='Full source fields/banks/padding/geometry exact for four actual projections; numerical ranges are 8193 gate rows and 3 rows per other K, before inverse rotation. No full-Linear, GPU, downstream or new-target quality claim.',
            future_comparison='Both arms must share this compact fixed2 consumer and recompute stage3 residual, prefix errors and sparse residual on its actual deployed prefix; do not reuse native residual or compensate conversion deltas.')
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'report.json', report)
    event(report['status'], seconds=report['elapsed_seconds'])


if __name__ == '__main__':
    main()
