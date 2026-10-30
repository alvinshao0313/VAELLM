"""Prefix lambda allocation with actual per-stage code and serializer costs."""
import math
import zlib

import torch

from experiments.adaptive_vector_rvq.codec import pack_lengths
from experiments.adaptive_vector_rvq.common import compact, require
from experiments.adaptive_vector_rvq.length_codec_probe import HEADER, MAGIC, CODEC_ZLIB
from .codec import decode_lengths
from .native_prefix import inventory
from .spec import resolve


def _bits(stage_bits, stages):
    require(isinstance(stage_bits, (tuple, list)) and len(stage_bits) == stages
            and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in stage_bits),
            'Explicit positive logical K for each actual prefix stage is required.')
    return torch.tensor(stage_bits, dtype=torch.int64)


@torch.no_grad()
def allocate(errors, max_code_bytes, stage_bits):
    """Minimize error + lambda * cumulative packed bytes, with shorter ties.

    Unequal K here is a cost-math interface, not evidence for fitting or deploying
    unequal-K banks. Given equal K this is the existing fixed-row lambda solve.
    """
    require(errors.device.type == 'cpu' and errors.dtype == torch.float64 and errors.ndim == 2
            and errors.shape[0] > 0 and 1 <= errors.shape[1] <= 3
            and bool(torch.isfinite(errors).all()) and bool((errors >= 0).all()),
            'Use finite nonnegative CPU FP64 errors for one to three actual ordered prefixes.')
    n, stages = errors.shape
    bits = _bits(stage_bits, stages)
    row_bytes = (bits + 7) // 8
    packed_costs = row_bytes.cumsum(0)
    logical_costs = bits.cumsum(0)
    require(isinstance(max_code_bytes, int) and not isinstance(max_code_bytes, bool)
            and max_code_bytes >= n * int(row_bytes[0]), 'The code budget must hold stage1 for every vector.')
    costs = packed_costs.double()
    best_dual = -math.inf

    def evaluate(value):
        nonlocal best_dual
        minima, selected = (errors + value * costs).min(dim=1)
        used = int(packed_costs[selected].sum())
        primal = float(errors.gather(1, selected[:, None]).sum())
        best_dual = max(best_dual, float(minima.sum()) - value * max_code_bytes)
        return selected, used, primal

    selected, used, primal = evaluate(0.)
    iterations, value = 0, 0.
    if used > max_code_bytes:
        high = max(0., float(((errors[:, :1] - errors[:, 1:]) / (costs[1:] - costs[0])).max()))
        high += max(1., abs(high)) * 16 * torch.finfo(torch.float64).eps
        _, used, _ = evaluate(high)
        require(used == n * int(row_bytes[0]), 'Upper lambda must select stage1 for every vector.')
        low = 0.
        for _ in range(60):
            middle = (low + high) / 2
            _, used, _ = evaluate(middle)
            if used > max_code_bytes:
                low = middle
            else:
                high = middle
        value, iterations = high, 60
        selected, used, primal = evaluate(value)
    require(used <= max_code_bytes and primal + n * 1e-12 >= best_dual, 'Packed-code feasibility/weak duality failed.')
    lengths = (selected + 1).to(torch.uint8)
    logical = int(logical_costs[selected].sum())
    return lengths, dict(lambda_value=value, max_code_bytes=max_code_bytes, code_bytes=used,
        code_rows=int(lengths.long().sum()), logical_code_bits=logical, packed_code_bits=used * 8,
        code_padding_bits=used * 8 - logical, stage_bits=list(stage_bits), stage_row_bytes=row_bytes.tolist(),
        NMSE=primal / n, fixed_byte_budget_dual_NMSE=best_dual / n, bisection_iterations=iterations)


def _frame(lengths):
    raw = pack_lengths(lengths).numpy().tobytes()
    payload = HEADER.pack(MAGIC, CODEC_ZLIB, len(lengths), len(raw), zlib.crc32(raw)) + zlib.compress(raw, level=6)
    frame = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    require(torch.equal(decode_lengths(frame, len(lengths)), lengths), 'The actual length frame must roundtrip exactly.')
    return frame


def materialize(bundle, lengths):
    """Select real full-bank rows; keep every decoder and every fixed field."""
    require(1 <= len(bundle['stages']) <= 3, 'Only one to three materialized stages are supported.')
    n = int(bundle['n_vectors'])
    original_lengths = decode_lengths(bundle['length_frame'], n)
    resolve(bundle, original_lengths)
    require(bool((original_lengths == len(bundle['stages'])).all()),
            'Allocation needs complete actual banks, not previously truncated routes.')
    require(lengths.device.type == 'cpu' and lengths.dtype == torch.uint8 and tuple(lengths.shape) == (n,)
            and bool(((lengths >= 1) & (lengths <= len(bundle['stages']))).all()), 'Invalid real prefix lengths.')
    # No decoder/normalization/protection/sign is changed or inferred here.
    candidate = {name: value for name, value in bundle.items() if name not in ('stages', 'length_frame')}
    candidate['stages'] = [{**stage, 'codes': stage['codes'].index_select(0,
        torch.nonzero(lengths >= index + 1, as_tuple=False).flatten()).contiguous()}
        for index, stage in enumerate(bundle['stages'])]
    candidate['length_frame'] = _frame(lengths)
    resolve(candidate, lengths)
    return candidate


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)


def _measure(candidate, lengths, serialize, persistent_fields):
    require(isinstance(persistent_fields, dict) and all(isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in persistent_fields.items()), 'Pass all extra persisted tensor fields explicitly by name.')
    tensors = inventory(candidate)
    require(not set(tensors) & set(persistent_fields), 'Extra persisted fields must have unique inventory names.')
    tensors.update(persistent_fields)
    require({id(value) for value in _tensor_leaves(candidate)} <= {id(value) for value in tensors.values()},
            'List any additional persisted tensor in persistent_fields; do not hide tensor costs in metadata.')
    sizes = {name: compact(value) for name, value in tensors.items()}
    pointers = [value.untyped_storage().data_ptr() for value in tensors.values() if value.numel()]
    require(len(set(pointers)) == len(pointers), 'Persist compact independent tensors; shared storages invalidate the byte ledger.')
    # The serializer must include candidate and persistent_fields without omission;
    # its actual complete bytes, including metadata and any container, own the cap.
    serialized = serialize(candidate, persistent_fields)
    require(isinstance(serialized, (bytes, bytearray, memoryview)), 'Serializer must return actual complete serialized bytes, not a size estimate.')
    actual = len(serialized)
    payload_bytes = sum(sizes.values())
    require(actual >= payload_bytes, 'Serializer omitted/deduplicated/compressed payload fields; this compact raw-tensor ledger does not support it.')
    stage_bits = [int(stage['decoder_spec']['in_dim']) for stage in candidate['stages']]
    counts = [int((lengths >= index + 1).sum()) for index in range(len(stage_bits))]
    codes = sum(sizes[f'stage{index}/codes'] for index in range(len(stage_bits)))
    logical = sum(k * count for k, count in zip(stage_bits, counts, strict=True))
    frame_bytes = sizes['length_frame']
    return dict(actual_complete_bytes=actual, payload_tensor_bytes=payload_bytes,
        serialization_and_metadata_bytes=actual - payload_bytes, tensor_bytes=sizes,
        code_bytes=codes, logical_code_bits=logical, packed_code_bits=codes * 8,
        code_padding_bits=codes * 8 - logical, stage_bits=stage_bits,
        stage_row_bytes=[(k + 7) // 8 for k in stage_bits], stage_active_rows=counts,
        decoder_bytes=sum(value for name, value in sizes.items() if '/decoder/' in name),
        fixed_bytes=sum(value for name, value in sizes.items() if name.startswith('fixed/')),
        extra_persistent_tensor_bytes=sum(sizes[name] for name in persistent_fields),
        length_frame_bytes=frame_bytes, length_frame_header_bytes=HEADER.size,
        raw_two_bit_length_bytes=(len(lengths) + 3) // 4,
        noncode_nonframe_bytes=actual - codes - frame_bytes)


@torch.no_grad()
def iterate(errors, bundle, cap_bytes, serialize, *, persistent_fields):
    """At most four lambda/real-zlib/real-serialization rounds; last feasible wins.

    serialize(candidate, persistent_fields) returns the real complete bytes for the
    caller's serializer and metadata. Extra current/future persistent tensor fields
    are explicit; metadata must be serialized too. This is not a global compressed-
    cost optimum. The all1 seed only measures container cost and is not a fallback.
    """
    require(isinstance(cap_bytes, int) and not isinstance(cap_bytes, bool) and cap_bytes > 0,
            'Provide the actual positive complete-file budget in bytes.')
    require(callable(serialize), 'Provide the actual complete serializer.')
    require(1 <= len(bundle['stages']) <= 3, 'Only one to three materialized stages are supported.')
    n = int(bundle['n_vectors'])
    require(errors.ndim == 2 and errors.shape == (n, len(bundle['stages'])), 'Prefix errors and actual bundle stage order/N disagree.')
    stage_bits = [int(stage['decoder_spec']['in_dim']) for stage in bundle['stages']]
    # Validate the score/cost contract even if the complete-file budget is too small.
    minimum_code_bytes = n * ((stage_bits[0] + 7) // 8)
    allocate(errors, minimum_code_bytes, stage_bits)
    seed_lengths = torch.ones(n, dtype=torch.uint8)
    seed = _measure(materialize(bundle, seed_lengths), seed_lengths, serialize, persistent_fields)
    current = cap_bytes - seed['noncode_nonframe_bytes'] - seed['raw_two_bit_length_bytes']
    # A raw-length estimate may exceed its compressed frame. Never call that an
    # infeasible real budget when the measured all1 file actually fits.
    if current < minimum_code_bytes and seed['actual_complete_bytes'] <= cap_bytes:
        current = cap_bytes - seed['noncode_nonframe_bytes'] - seed['length_frame_bytes']
    current = min(current, n * sum((k + 7) // 8 for k in stage_bits))
    rounds, seen, chosen, chosen_lengths, chosen_bundle = [], set(), None, None, None
    stop = 'four_round_limit'
    for index in range(1, 5):
        if current < minimum_code_bytes:
            stop = 'code_budget_below_stage1'
            break
        seen.add(current)
        lengths, solution = allocate(errors, current, stage_bits)
        candidate = materialize(bundle, lengths)
        measured = _measure(candidate, lengths, serialize, persistent_fields)
        require(measured['code_bytes'] == solution['code_bytes']
                and measured['logical_code_bits'] == solution['logical_code_bits'], 'Actual selected codes disagree with lambda cost.')
        next_budget = cap_bytes - measured['noncode_nonframe_bytes'] - measured['length_frame_bytes']
        next_budget = min(next_budget, n * sum((k + 7) // 8 for k in stage_bits))
        row = dict(round=index, max_code_bytes=current, solution=solution, measured=measured,
            actual_complete_bytes=measured['actual_complete_bytes'], cap_bytes=cap_bytes,
            feasible=measured['actual_complete_bytes'] <= cap_bytes,
            unused_bytes=cap_bytes - measured['actual_complete_bytes'], next_max_code_bytes=next_budget)
        rounds.append(row)
        if row['feasible']:
            chosen, chosen_lengths, chosen_bundle = row, lengths, candidate
        if next_budget == current or next_budget in seen:
            stop = 'byte_budget_unchanged' if next_budget == current else 'byte_budget_repeated'
            break
        current = next_budget
    require(chosen is not None, 'No feasible candidate found by the declared four-round real-serialization allocation.')
    return chosen_lengths, chosen_bundle, dict(rounds=rounds, chosen=chosen, chosen_round=chosen['round'],
        stop_reason=stop, maximum_rounds=4, cap_bytes=cap_bytes, seed_measurement=seed,
        actual_decoder_stages=len(bundle['stages']), reserved_unmaterialized_decoders=0,
        selection='Last actual-complete-file-feasible round; never select by quality.',
        boundary='Lambda solves a fixed-packed-code-byte separable objective. Remeasured nonseparable length/container costs are not globally optimized.')
