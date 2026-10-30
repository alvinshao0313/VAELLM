"""Bounded real-zlib allocation charging exactly the three actual decoder stages."""
from __future__ import annotations

from pathlib import Path

from experiments.adaptive_vector_rvq import codec, common as c
from experiments.adaptive_vector_rvq.entropy_budget import allocate
from experiments.adaptive_vector_rvq.length_codec_probe import probe


def iterate(errors, budget, cap, output):
    output = Path(output)
    output.mkdir()
    n = len(errors)
    base = c.HEADER_BYTES + budget['fixed_bytes'] + 3 * budget['decoder_bytes']
    current = (cap - base - (n + 3) // 4) // 6
    rounds, seen, chosen, chosen_lengths = [], set(), None, None
    for index in range(1, 5):
        seen.add(current)
        lengths, solution = allocate(errors, current)
        frame = probe(codec.pack_lengths(lengths).numpy().tobytes(), lengths,
            f'round_{index:02d}', output, 50331648)
        total = base + 6 * solution['code_rows'] + frame['compressed_total_actual_bytes']
        next_rows = (cap - base - frame['compressed_total_actual_bytes']) // 6
        row = dict(round=index, max_code_rows=current, solution=solution, length_frame=frame,
            actual_complete_bytes=total, cap_bytes=cap, feasible=total <= cap,
            unused_bytes=cap - total, next_max_code_rows=next_rows)
        rounds.append(row)
        if row['feasible']:
            chosen, chosen_lengths = row, lengths
        if next_rows == current or next_rows in seen:
            stop = 'row_budget_unchanged' if next_rows == current else 'row_budget_repeated'
            break
        current = next_rows
    else:
        stop = 'four_round_limit'
    c.require(chosen is not None, 'No feasible candidate under the predeclared four-round allocation.')
    return chosen_lengths, dict(rounds=rounds, chosen=chosen, chosen_round=chosen['round'],
        stop_reason=stop, maximum_rounds=4, actual_decoder_stages=3, reserved_unmaterialized_decoders=0,
        base_bytes=base, cap_bytes=cap,
        selection='Last complete-file-feasible round; never select by quality.',
        boundary='Lambda solves only the fixed-code-row separable objective; the bounded remeasured-zlib loop is not a global compressed-cost optimum.')
