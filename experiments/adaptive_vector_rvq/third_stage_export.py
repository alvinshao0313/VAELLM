"""Specific L0 three-stage artifact: actual selected codes and one measured zlib length frame."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import zlib

import torch

from . import codec, common as c
from .length_codec_probe import HEADER, MAGIC as LENGTH_MAGIC, CODEC_ZLIB
from .observe import compare
from .third_stage import vector_errors
from experiments.night_sequence_20261026.queue_wait import free_gpu

MAGIC = b'AVRQL01\0'


def inventory(bundle):
    tensors = {f'fixed/{name}': value for name, value in bundle['fixed'].items()}
    for index, stage in enumerate(bundle['stages']):
        tensors[f'stage{index}/codes'] = stage['codes']
        tensors.update({f'stage{index}/decoder/{name}': value for name, value in stage['decoder_state'].items()})
    tensors['length_frame'] = bundle['length_frame']
    return tensors


def save(bundle, path):
    tensors = inventory(bundle)
    table, offset = {}, c.HEADER_BYTES
    for name, value in tensors.items():
        size = c.compact(value)
        table[name] = dict(dtype=str(value.dtype), shape=list(value.shape), offset=offset, bytes=size)
        offset += size
    metadata = dict(schema='adaptive_vector_rvq_zlib_l0_three_stage_v1', n_vectors=bundle['n_vectors'],
        geometry=bundle['geometry'], tensors=table,
        stages=[{key: stage[key] for key in ('decoder_spec', 'normalization')} for stage in bundle['stages']],
        length_frame='AVRQLZ1; 29-byte header and zlib6; original tight two-bit (length-1) routes',
        accumulation='Persisted BF16 decoder outputs and stage-ordered BF16 additions; inactive stages skipped')
    header = json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode('utf-8')
    c.require(len(header) + 12 <= c.HEADER_BYTES, 'The charged fixed header is too small.')
    with path.open('xb') as stream:
        stream.write(struct.pack('<8sI', MAGIC, len(header)))
        stream.write(header)
        stream.write(bytes(c.HEADER_BYTES - 12 - len(header)))
        for value in tensors.values():
            stream.write(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    c.require(path.stat().st_size == offset, 'Actual compact file differs from its complete state ledger.')
    return offset


def load(path):
    with path.open('rb') as stream:
        magic, size = struct.unpack('<8sI', stream.read(12))
        c.require(magic == MAGIC and size <= c.HEADER_BYTES - 12, 'Invalid specific compact codec header.')
        meta = json.loads(stream.read(size))
        c.require(meta['schema'] == 'adaptive_vector_rvq_zlib_l0_three_stage_v1'
            and len(meta['stages']) == 3 and not any(stream.read(c.HEADER_BYTES - 12 - size)),
            'Invalid specific schema/stage count/header padding.')
        tensors, offset = {}, c.HEADER_BYTES
        for name, item in meta['tensors'].items():
            c.require(item['offset'] == offset and item['dtype'] in codec.DTYPES, 'Invalid compact tensor table.')
            raw = bytearray(stream.read(item['bytes']))
            c.require(len(raw) == item['bytes'], 'Truncated actual state payload.')
            dtype = codec.DTYPES[item['dtype']]
            value = torch.frombuffer(raw, dtype=dtype).clone().reshape(item['shape']) if raw else torch.empty(item['shape'], dtype=dtype)
            c.require(c.compact(value) == item['bytes'], 'Tensor size/shape changed.')
            tensors[name], offset = value, offset + item['bytes']
        c.require(not stream.read(1) and offset == path.stat().st_size, 'Uncharged trailing bytes.')
    stages = []
    for index, record in enumerate(meta['stages']):
        prefix = f'stage{index}/decoder/'
        stages.append({**record, 'codes': tensors[f'stage{index}/codes'],
            'decoder_state': {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}})
    bundle = dict(n_vectors=meta['n_vectors'], geometry=meta['geometry'], stages=stages,
        fixed={name[len('fixed/'):]: value for name, value in tensors.items() if name.startswith('fixed/')},
        length_frame=tensors['length_frame'])
    frame = bundle['length_frame'].numpy().tobytes()
    magic, kind, n, raw_size, checksum = HEADER.unpack(frame[:HEADER.size])
    c.require(magic == LENGTH_MAGIC and kind == CODEC_ZLIB and n == bundle['n_vectors'], 'Length frame identity changed.')
    raw = zlib.decompress(frame[HEADER.size:])
    c.require(len(raw) == raw_size and zlib.crc32(raw) == checksum, 'Length frame size/checksum changed.')
    lengths = codec.unpack_lengths(torch.frombuffer(bytearray(raw), dtype=torch.uint8), n)
    c.require(int(lengths.max()) <= 3, 'The actual artifact cannot reference an absent fourth stage.')
    for index, stage in enumerate(stages, start=1):
        c.require(tuple(stage['codes'].shape) == (int((lengths >= index).sum()), 6), 'Actual selected stage row count changed.')
    return bundle, lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root_dir.resolve(), args.output_dir.resolve()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and not output.exists(), 'Use idle assigned GPU0 and a new validation output.')
    resource = free_gpu(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = c.check_prepared(root / 'prepared')
    layer = prepared['layers']['0']
    report_path = root / 'entropy_stage3/report.json'
    parent = c.read(report_path)
    c.require(parent['status'] == 'completed' and parent['requires_compact_file_validation'], 'Validate the actual locally improved completed candidate.')
    fixed2_path = root / 'formal_l0/layer_0/fixed2.avrvq'
    stage_path = Path(parent['stage3_artifact']['path'])
    frame_path = Path(parent['actual_three_stage_allocation']['chosen']['length_frame']['artifact']['path'])
    errors_path = Path(parent['errors_artifact']['path'])
    c.require(c.sha(stage_path) == parent['stage3_artifact']['sha256']
        and c.sha(errors_path) == parent['errors_artifact']['sha256']
        and c.sha(frame_path) == parent['actual_three_stage_allocation']['chosen']['length_frame']['artifact']['sha256'],
        'Use the completed real stage3/error/length artifacts.')
    source = codec.load(fixed2_path)
    banks = [*source['stages'], torch.load(stage_path, map_location='cpu', weights_only=True)]
    frame = frame_path.read_bytes()
    raw = zlib.decompress(frame[HEADER.size:])
    lengths = codec.unpack_lengths(torch.frombuffer(bytearray(raw), dtype=torch.uint8), layer['vector_count'])
    stages = [{**stage, 'codes': stage['codes'].index_select(0, torch.nonzero(lengths >= index, as_tuple=False).flatten()).contiguous()}
        for index, stage in enumerate(banks, start=1)]
    bundle = dict(n_vectors=layer['vector_count'], geometry=source['geometry'], fixed=source['fixed'], stages=stages,
        length_frame=torch.frombuffer(bytearray(frame), dtype=torch.uint8).clone())
    paths = [Path(__file__), c.HERE / 'codec.py', c.HERE / 'common.py', c.HERE / 'third_stage.py',
        c.HERE / 'length_codec_probe.py', c.HERE / 'observe.py', report_path, fixed2_path, stage_path, frame_path, errors_path]
    output.mkdir(parents=True)
    record = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv], started_unix=time.time(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]), resource=resource,
        source_sha256={str(path.resolve()): c.sha(path) for path in paths}, actual_decoder_stages=3,
        fourth_decoder_written=False, production_loader_modified=False)
    c.write(output / 'report.json', record)
    try:
        path = output / 'candidate_three_stage.avrvqz'
        actual_bytes = save(bundle, path)
        c.require(actual_bytes == parent['complete_accounting']['actual_state_accounting_bytes'], 'Real file and existing three-stage complete accounting differ.')
        fresh, loaded_lengths = load(path)
        c.require(c.digest(inventory(fresh)) == c.digest(inventory(bundle)) and torch.equal(loaded_lengths, lengths),
            'Fresh actual fields or routing changed.')
        vectors = torch.zeros((layer['vector_count'], c.DIM), dtype=torch.bfloat16)
        for index, stage in enumerate(fresh['stages'], start=1):
            codec.add_stage(vectors, stage, torch.nonzero(loaded_lengths >= index, as_tuple=False).flatten(), 'cuda:0')
        data = torch.load(layer['path'], map_location='cpu', weights_only=True, mmap=True)
        errors = torch.load(errors_path, map_location='cpu', weights_only=True)
        expected = torch.stack([errors[f'e{index}_MSE'] for index in range(1, 4)], dim=1)
        expected = expected.gather(1, (lengths.long() - 1)[:, None]).flatten()
        c.require(torch.equal(vector_errors(data['vectors'], vectors), expected), 'Fresh codec numerical prefix/error contract changed.')
        metrics = compare(fresh, vectors, prepared['activations'], 'cuda:0')
        for key, value in metrics.items():
            if isinstance(value, float):
                c.require(abs(value - parent['actual_candidate_metrics'][key]) < 1e-12, 'Fresh original-coordinate/local-output metric changed: ' + key)
        for filename, expected_sha in record['source_sha256'].items():
            c.require(c.sha(filename) == expected_sha, 'Bound source/input changed: ' + filename)
        record.update(status='completed', exit_code=0, source_unchanged=True, complete_file_written=True,
            actual_file=dict(path=str(path), bytes=actual_bytes, sha256=c.sha(path)),
            all_persistent_fields_exact=True, lengths_and_route_exact=True, actual_prefix_errors_exact=True,
            original_coordinate_and_local_output_metrics=metrics,
            fourth_decoder_reservation_only_bytes=21312,
            conservative_budget_accounting_bytes=actual_bytes + 21312,
            complete_storage_budget_passed=actual_bytes <= 19607424,
            boundary='One actual L0 three-stage artifact and local readout; no full-model/downstream improvement or public codec integration claimed.')
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        c.write(output / 'report.json', record)
    print(dict(status=record['status'], actual_file=record['actual_file'], metrics=metrics), flush=True)


if __name__ == '__main__':
    main()
