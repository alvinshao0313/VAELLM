"""One CPU-only zlib6 lossless probe on existing packed length metadata."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import zlib

import torch

from . import codec
from . import common as c

MAGIC = b'AVRQLZ1\0'
HEADER = struct.Struct('<8sBQQI')
CODEC_ZLIB = 1


def probe(raw, lengths, name, output, original_elements):
    """Measure one compression/decompression and verify the actual saved frame."""
    c.require(len(raw) == (len(lengths) + 3) // 4, 'Use the existing tight two-bit stream.')
    started = time.perf_counter_ns()
    compressed = zlib.compress(raw, level=6)
    compress_seconds = (time.perf_counter_ns() - started) / 1e9
    checksum = zlib.crc32(raw)
    frame = HEADER.pack(MAGIC, CODEC_ZLIB, len(lengths), len(raw), checksum) + compressed
    path = output / (name + '.lengths.zlib')
    with path.open('xb') as stream:
        stream.write(frame)
    saved = path.read_bytes()
    magic, codec_id, n, raw_bytes, expected_crc = HEADER.unpack(saved[:HEADER.size])
    c.require(magic == MAGIC and codec_id == CODEC_ZLIB and n == len(lengths)
        and raw_bytes == len(raw), 'Saved independent length frame header changed.')
    started = time.perf_counter_ns()
    restored = zlib.decompress(saved[HEADER.size:])
    decompress_seconds = (time.perf_counter_ns() - started) / 1e9
    c.require(restored == raw and len(restored) == raw_bytes and zlib.crc32(restored) == expected_crc,
        'The zlib round trip must preserve every packed byte and checksum.')
    packed = torch.frombuffer(bytearray(restored), dtype=torch.uint8)
    c.require(torch.equal(codec.unpack_lengths(packed, n), lengths), 'A route/length changed after decompression.')
    net_saved = len(raw) - len(saved)
    return dict(n_vectors=n, length_counts=[int((lengths == stage).sum()) for stage in range(1, 5)],
        original_packed_bytes=len(raw), original_packed_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_zlib_bytes=len(compressed), frame_header_bytes=HEADER.size,
        compressed_total_actual_bytes=path.stat().st_size, net_saved_bytes=net_saved,
        raw_length_bits_per_original_linear_weight=8 * len(raw) / original_elements,
        saved_bits_per_original_linear_weight=8 * net_saved / original_elements,
        compression_seconds=compress_seconds, decompression_seconds=decompress_seconds,
        compression_calls=1, decompression_calls=1, zlib_level=6,
        zlib_stream_header_and_adler32_already_in_compressed_bytes=True,
        byte_exact_roundtrip=True, length_and_route_exact=True,
        artifact=dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=c.sha(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicitly hide CUDA for this CPU-only probe.')
    c.require(not args.output_dir.exists(), 'Never overwrite a prior diagnostic.')
    torch.set_num_threads(4)
    root, output = args.root_dir.resolve(), args.output_dir.resolve()
    prepared_path = root / 'prepared/prepared.json'
    prepared = c.check_prepared(prepared_path.parent)
    layer = prepared['layers']['0']
    n = layer['vector_count']
    original_elements = math.prod(layer['geometry']['original_shape'])
    c.require(n == 1540096 and c.DIM == 32 and original_elements == 50331648,
        'Use the complete real L0 D32 geometry and original-linear denominator.')
    formal_path = root / 'formal_l0/layer_0/adaptive.avrvq'
    fixed2_path = formal_path.with_name('fixed2.avrvq')
    formal_report_path = formal_path.with_name('report.json')
    ideal_report_path = root / 'ideal_bound/report.json'
    formal, fixed2 = codec.load(formal_path), codec.load(fixed2_path)
    formal_report, ideal_report = c.read(formal_report_path), c.read(ideal_report_path)
    c.require(formal['mode'] == 'adaptive' and formal['n_vectors'] == n
        and formal['geometry'] == layer['geometry'] and c.digest(formal['fixed']) == layer['fixed_digest']
        and formal_report['freshdecode']['adaptive']['exact'], 'Use the completed actual formal adaptive codec.')
    formal_ledger, fixed2_ledger = codec.ledger(formal), codec.ledger(fixed2)
    c.require(fixed2_ledger['total_bytes'] == fixed2_path.stat().st_size == 19607424
        and formal_ledger['count_bytes'] == 385024, 'Actual complete fixed2/length accounting changed.')
    ideal_path = Path(ideal_report['length_artifact']['path']).resolve()
    c.require(ideal_report['status'] == 'completed' and ideal_report['target_digest'] == layer['target_digest']
        and c.sha(ideal_path) == ideal_report['length_artifact']['sha256']
        and c.sha(fixed2_path) == ideal_report['source_sha256'][str(fixed2_path)],
        'Use the completed source-bound ideal allocation artifact.')
    paths = [Path(__file__), c.HERE / 'common.py', c.HERE / 'codec.py', c.HERE / 'ideal_bound.py',
        prepared_path, formal_path, fixed2_path, formal_report_path, ideal_report_path, ideal_path]
    sources = {str(path.resolve()): c.sha(path) for path in paths}
    frozen_digest = c.digest(codec.tensor_inventory(formal))
    started = time.time()
    output.mkdir(parents=True)
    report = dict(status='running', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        command=[sys.executable, *sys.argv], started_unix=started, source_sha256=sources,
        used_cuda=False, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cpu_threads=4,
        target_digest=layer['target_digest'], n_vectors=n, vector_dimension=c.DIM,
        original_linear_elements=original_elements, fixed2_actual_bytes=fixed2_path.stat().st_size,
        frozen_formal_tensor_digest=frozen_digest,
        codec=dict(name='Python stdlib zlib', level=6, runtime_version=zlib.ZLIB_RUNTIME_VERSION,
            frame_schema='AVRQLZ1: tightly packed little-lane (length-1) two-bit labels; no reordered vectors',
            frame_struct=HEADER.format, header_bytes=HEADER.size,
            header_fields_bytes=dict(magic_and_version=8, codec_id=1, n_vectors=8, raw_bytes=8, crc32=4),
            existing_model_header_retained_bytes=c.HEADER_BYTES),
        scope='Length metadata only. Existing stage codes/decoders/geometry/routes and model files remain unchanged. No reconstruction/accuracy improvement or production integration is evaluated.',
        cases={})
    c.write(output / 'report.json', report)
    try:
        sizes = codec.lengths(formal)
        raw = formal['lengths'].numpy().tobytes()
        item = probe(raw, sizes, 'formal_geometric_l0', output, original_elements)
        item.update(source_kind='actual completed formal geometric allocation',
            source=str(formal_path), existing_complete_model_actual_bytes=formal_path.stat().st_size,
            existing_complete_ledger=formal_ledger,
            accounting_only_replacement_complete_bytes=formal_ledger['total_bytes'] - len(raw)
                + item['compressed_total_actual_bytes'],
            rewritten_model_created=False)
        report['cases']['formal_geometric_l0'] = item

        raw = ideal_path.read_bytes()
        lengths = codec.unpack_lengths(torch.frombuffer(bytearray(raw), dtype=torch.uint8), n)
        c.require([int((lengths == stage).sum()) for stage in range(1, 5)]
            == ideal_report['solution']['length_counts'], 'Persisted ideal length distribution changed.')
        item = probe(raw, lengths, 'ideal_bound_l0', output, original_elements)
        item.update(source_kind='optimistic mathematical ideal allocation; not deployable quantized weights',
            source=str(ideal_path), deployment_model=False,
            original_charged_relaxation_budget=ideal_report['budget'],
            accounting_only_replacement_relaxation_bytes=ideal_report['budget']['actual_ideal_complete_bytes']
                - len(raw) + item['compressed_total_actual_bytes'])
        report['cases']['ideal_bound_l0'] = item

        # Observe once; an unfinished active experiment is neither waited for nor altered.
        active_path = root / 'active_threshold/stage2_lengths.bin'
        active_report_path = active_path.with_name('report.json')
        observed = c.read(active_report_path) if active_report_path.exists() else None
        if active_path.exists() and observed and observed.get('artifacts', {}).get('lengths'):
            binding = observed['artifacts']['lengths']
            c.require(active_path.resolve() == Path(binding['path']).resolve()
                and c.sha(active_path) == binding['sha256'], 'Actual uniform-threshold length identity changed.')
            raw = active_path.read_bytes()
            lengths = codec.unpack_lengths(torch.frombuffer(bytearray(raw), dtype=torch.uint8), n)
            sources[str(active_path)] = c.sha(active_path)
            item = probe(raw, lengths, 'active_threshold_stage2_l0', output, original_elements)
            item.update(source_kind='actual saved strict-uniform-tau stage2 allocation; partial model',
                source=str(active_path), tau_MSE=observed['tau_MSE'], tau_normalized=observed['tau_normalized'],
                deployment_model=False, later_stages_trained=observed['later_stages_trained'],
                original_partial_budget=observed.get('budget'), observed_report_sha256=c.sha(active_report_path))
            report['cases']['active_threshold_stage2_l0'] = item
        else:
            report['skipped_active_threshold'] = dict(checked_unix=time.time(),
                length_file_exists=active_path.exists(), observed_status=observed.get('status') if observed else None,
                reason='No published source-bound actual uniform-tau length artifact exists at this one-time observation; no lengths were invented or recomputed and no active work was modified.')
        c.require(c.digest(codec.tensor_inventory(formal)) == frozen_digest,
            'The actual formal codes, decoder, fixed payload or length routes changed in memory.')
        for filename, expected in sources.items():
            c.require(c.sha(filename) == expected, 'A bound source or input changed: ' + filename)
        report.update(status='completed', exit_code=0, source_unchanged=True,
            formal_state_and_route_unchanged=True, source_sha256=sources,
            all_byte_roundtrips_exact=True, production_codec_modified=False)
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - started)
        c.write(output / 'report.json', report)
    print({name: {key: row[key] for key in ('original_packed_bytes', 'compressed_total_actual_bytes',
        'net_saved_bytes', 'saved_bits_per_original_linear_weight', 'length_counts')}
        for name, row in report['cases'].items()}, flush=True)


if __name__ == '__main__':
    main()
