"""Strict fixed-prefix2->3 positive-gain routing with measured zlib6 frame costs."""
import zlib

from . import bindings as b
from . import artifact
from experiments.adaptive_native_model_20261029.codec import decode_lengths
from experiments.adaptive_vector_rvq.codec import pack_lengths
from experiments.adaptive_vector_rvq.length_codec_probe import HEADER, MAGIC, CODEC_ZLIB


def frame(lengths):
    b.c.require(lengths.device.type == 'cpu' and lengths.dtype == b.torch.uint8
        and lengths.ndim == 1 and len(lengths) > 0 and bool(((lengths == 2) | (lengths == 3)).all()),
        'This route may only preserve2 or add3; stage1 exits are forbidden.')
    raw = pack_lengths(lengths).numpy().tobytes()
    data = HEADER.pack(MAGIC, CODEC_ZLIB, len(lengths), len(raw), zlib.crc32(raw)) + zlib.compress(raw, level=6)
    value = b.torch.frombuffer(bytearray(data), dtype=b.torch.uint8).clone()
    b.c.require(b.torch.equal(decode_lengths(value, len(lengths)), lengths), 'Actual2bit zlib6 frame must roundtrip.')
    return value


def materialize(bundle, lengths):
    b.c.require(len(bundle['stages']) == 3 and len(lengths) == bundle['n_vectors'], 'Use actual three full banks.')
    original = decode_lengths(bundle['length_frame'], bundle['n_vectors'])
    b.c.require(bool((original == 3).all()), 'Routing needs full third-stage codes before assignment.')
    frame_value = frame(lengths)
    stages = [{**stage, 'codes': stage['codes'].clone().contiguous()} for stage in bundle['stages'][:2]]
    selected = b.torch.nonzero(lengths == 3).flatten()
    stages.append({**bundle['stages'][2], 'codes': bundle['stages'][2]['codes'].index_select(0, selected).contiguous()})
    candidate = {**bundle, 'stages': stages, 'length_frame': frame_value}
    artifact.measure(candidate)
    return candidate


@b.torch.no_grad()
def allocate(errors, bundle, cap_bytes):
    b.c.require(errors.device.type == 'cpu' and errors.dtype == b.torch.float64
        and errors.shape == (bundle['n_vectors'], 2) and bool(b.torch.isfinite(errors).all())
        and bool((errors >= 0).all()), 'Use actual weighted FP64 [e2,e3] with this full-bank layout.')
    b.c.require(isinstance(cap_bytes, int) and cap_bytes > 0, 'Use an integer full deployed-payload cap.')
    n = bundle['n_vectors']
    gains = errors[:, 0] - errors[:, 1]
    positive = b.torch.nonzero(gains > 0).flatten()
    # nonzero is in original index order; stable sort makes ties prefer that order.
    ranked = positive[b.torch.argsort(gains[positive], descending=True, stable=True)]
    seed_lengths = b.torch.full((n,), 2, dtype=b.torch.uint8)
    seed = artifact.measure(materialize(bundle, seed_lengths))
    b.c.require(seed['payload_tensor_bytes'] <= cap_bytes, 'The cap cannot hold all original two-stage fields plus the third decoder and all2 frame.')
    row_bytes = (int(bundle['stages'][2]['decoder_spec']['in_dim']) + 7) // 8
    fixed = seed['payload_tensor_bytes'] - seed['length_frame_bytes']
    # zlib compressBound for raw two-bit lengths, plus the real29B frame header.
    raw = (n + 3) // 4
    reserved = HEADER.size + raw + (raw >> 12) + (raw >> 14) + (raw >> 25) + 13
    rows = min(len(ranked), max(0, (cap_bytes - fixed - reserved) // row_bytes))
    seen, rounds, chosen = set(), [], None
    for index in range(1, 5):
        seen.add(rows)
        lengths = seed_lengths.clone()
        lengths[ranked[:rows]] = 3
        candidate = materialize(bundle, lengths)
        measured = artifact.measure(candidate)
        next_rows = min(len(ranked), max(0, (cap_bytes - fixed - measured['length_frame_bytes']) // row_bytes))
        record = dict(round=index, selected_rows=rows, next_rows=next_rows,
            payload_tensor_bytes=measured['payload_tensor_bytes'], length_frame_bytes=measured['length_frame_bytes'],
            feasible=measured['payload_tensor_bytes'] <= cap_bytes)
        rounds.append(record)
        if record['feasible']:
            chosen = lengths, candidate, measured, index
        if next_rows == rows or next_rows in seen:
            break
        rows = next_rows
    b.c.require(chosen is not None, 'No feasible actual2->3 payload within the four declared frame recalculations.')
    lengths, candidate, measured, chosen_round = chosen
    b.c.require(measured['stage_active_rows'][:2] == [n, n]
        and b.torch.equal(candidate['stages'][0]['codes'], bundle['stages'][0]['codes'])
        and b.torch.equal(candidate['stages'][1]['codes'], bundle['stages'][1]['codes']), 'Never trim either original full bank.')
    return lengths, candidate, dict(cap_bytes=cap_bytes, maximum_rounds=4, rounds=rounds,
        chosen_round=chosen_round, chosen=measured, positive_gain_rows=len(positive),
        selected_rows=int((lengths == 3).sum()), strict_positive_gain=True, original_prefix_rows_preserved=True,
        selection='Strictly positive e2-e3, stable descending gain/original-index tie; last feasible frame round, never quality-selected',
        cap_scope='All actual deployed tensors; metadata and serialization overhead reported separately')
