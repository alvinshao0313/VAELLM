"""Fixed-grid plain INT4 scale search against the deployed BF16 protection."""
from __future__ import annotations

import torch

from experiments.protected_int4_probe.codec import Payload, decode_output_scale, pack_int4, require

RATIOS = tuple(1 - i / 32 for i in range(17))


def search_scales(reference, original, *, row_chunk=512):
    require(reference.device.type == 'cpu' and reference.dtype == torch.bfloat16
            and reference.ndim == 2 and bool(torch.isfinite(reference).all()),
            'Reference must be the actual finite CPU BF16 matrix.')
    require(original.bits == 4 and original.signs is None and original.shape == tuple(reference.shape)
            and isinstance(row_chunk, int) and row_chunk > 0, 'Require the original plain INT4 payload.')
    values, scales = original.values.clone(), original.scales.clone()
    choices = torch.zeros(reference.shape[0], dtype=torch.int64)
    baseline_sse, final_sse = [], []
    for start in range(0, reference.shape[0], row_chunk):
        stop = min(start + row_chunk, reference.shape[0])
        target = reference[start:stop].float()
        shape = tuple(target.shape)
        baseline = Payload(values[start:stop], scales[start:stop], shape, 4)
        best_sse = (decode_output_scale(baseline).double() - target.double()).square().sum(1)
        baseline_sse.append(best_sse.clone())
        maximum = target.abs().amax(1)
        for index, ratio in enumerate(RATIOS[1:], start=1):
            candidate_scales = torch.where(maximum > 0, maximum * ratio / 7,
                                           torch.ones_like(maximum)).to(torch.bfloat16).contiguous()
            require(bool(torch.isfinite(candidate_scales).all()) and bool((candidate_scales > 0).all()),
                    'Candidate BF16 scales must remain finite and positive.')
            qvalues = torch.round(target / candidate_scales.float().unsqueeze(1)).clamp(-8, 7).to(torch.int8)
            candidate = Payload(pack_int4(qvalues), candidate_scales, shape, 4)
            error = (decode_output_scale(candidate).double() - target.double()).square().sum(1)
            improve = error < best_sse  # Strict comparison preserves the original on ties.
            values[start:stop][improve] = candidate.values[improve]
            scales[start:stop][improve] = candidate.scales[improve]
            choices[start:stop][improve] = index
            best_sse = torch.where(improve, error, best_sse)
        replay = Payload(values[start:stop], scales[start:stop], shape, 4)
        replay_sse = (decode_output_scale(replay).double() - target.double()).square().sum(1)
        require(torch.equal(replay_sse, best_sse) and bool((replay_sse <= baseline_sse[-1]).all()),
                'Actual packed-to-BF16 replay must not worsen any output row.')
        kept = choices[start:stop] == 0
        require(torch.equal(values[start:stop][kept], original.values[start:stop][kept])
                and torch.equal(scales[start:stop][kept], original.scales[start:stop][kept]),
                'Rows retaining the original candidate must preserve its exact bytes.')
        final_sse.append(replay_sse)
    return (Payload(values, scales, original.shape, 4), choices,
            torch.cat(baseline_sse), torch.cat(final_sse))
