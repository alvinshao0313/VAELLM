"""Greedy acceptance of pre-existing weight-neighbor groups on native MLP SSE."""
from __future__ import annotations

import time
import torch

from experiments.gate_student_context_probe.core import Objective, ready, records
from experiments.joint_reconstruction_probe.core import require
from experiments.rotation_factor_probe.capture import tensor_record

BANKS = ('vq_weight', 'vq_weight_s1')
GROUPS, VECTORS_PER_GROUP = 384, 4032


def group_inventory(choices):
    require(choices.dtype == torch.uint8 and choices.shape == (GROUPS*VECTORS_PER_GROUP,)
            and int(choices.max()) <= 64, 'Require the original one-pass D32/S2 choices.')
    rows=[]
    for group in range(GROUPS):
        chunk=choices[group*VECTORS_PER_GROUP:(group+1)*VECTORS_PER_GROUP]
        counts=[int(((chunk>stage*32)&(chunk<=(stage+1)*32)).sum()) for stage in range(2)]
        rows.append({'group':group,'output_start':group*32,'vector_start':group*VECTORS_PER_GROUP,
                     'stage_xor':counts,'proposed_bit_flips':sum(counts)})
    require(sum(r['proposed_bit_flips'] for r in rows)==int((choices>0).sum()), 'Group partition lost choices.')
    return rows


def apply_choices(bank, choices, stage):
    indices=((choices>stage*32)&(choices<=(stage+1)*32)).nonzero().flatten()
    bits=choices[indices].long()-1-stage*32
    bank[indices,0,bits//8] ^= (1 << (bits%8)).to(torch.uint8)


@torch.no_grad()
def search(layer, archive, cache, choices, *, deadline, observer):
    device=next(layer.parameters()).device
    require(device.type=='cuda' and not layer.training and all(not p.requires_grad for p in layer.parameters()),
            'Only one frozen native gate is required; no optimizer or gradients.')
    ready(layer)
    require(layer._parallel_stage_restore_identity and layer.weight_rotation.block_size==32,
            'The fixed contiguous group layout requires this original block32 gate.')
    initial_state=records(layer)
    initial_banks=[layer.get_stage_part_vq_storage(s,0).clone() for s in range(2)]
    inventory=group_inventory(choices)
    choices=choices.to(device)
    objective=Objective(archive,cache,device).eval()
    require(objective.inputs.shape==(4289,4096), 'Use all original train32 valid tokens.')
    torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    weight=layer._decode_weight(dtype=torch.bfloat16)
    target=objective.reference
    initial_sse=float((objective.prediction(weight).float()-target.float()).double().square().sum())
    current_sse=initial_sse
    target_sse=float(target.double().square().sum())
    accepted=torch.zeros(GROUPS,dtype=torch.bool,device=device)
    trajectory=[]
    for item in inventory:
        require(time.time()<deadline, 'The assigned wall budget expired; do not retry or extend automatically.')
        step_started=time.perf_counter()
        group,start=item['group'],item['vector_start']
        end=start+VECTORS_PER_GROUP
        chunk=choices[start:end]
        bank_slices=[layer.get_stage_part_vq_storage(s,0)[start:end] for s in range(2)]
        # Earlier accepted groups cannot overlap an as-yet unvisited original group.
        require(all(torch.equal(bank_slices[s],initial_banks[s][start:end]) for s in range(2)),
                'A previous group changed an unvisited original code block.')
        before_sse=current_sse
        keep=False
        try:
            for stage in range(2): apply_choices(bank_slices[stage],chunk,stage)
            ready(layer)
            trial_weight=layer._decode_weight(dtype=torch.bfloat16)
            trial_sse=float((objective.prediction(trial_weight).float()-target.float()).double().square().sum())
            require(torch.isfinite(torch.tensor(trial_sse)), 'Nonfinite native trial SSE.')
            keep=trial_sse<before_sse  # Strict same-metric improvement, no proxy or tolerance.
            if keep:
                accepted[group]=True
                current_sse=trial_sse
            row={**item,'before_sse':before_sse,'trial_sse':trial_sse,'trial_delta':trial_sse-before_sse,
                 'accepted':keep,'after_sse':current_sse,'accepted_bit_flips':item['proposed_bit_flips'] if keep else 0,
                 'seconds':time.perf_counter()-step_started}
        finally:
            if not keep:
                for stage in range(2): bank_slices[stage].copy_(initial_banks[stage][start:end])
                ready(layer)
        trajectory.append(row)
        observer(row)
        del trial_weight
    effective_choices=torch.where(accepted.repeat_interleave(VECTORS_PER_GROUP),choices,torch.zeros_like(choices))
    final_state=records(layer)
    require(all(final_state[k]==v for k,v in initial_state.items() if k not in BANKS),
            'A decoder, protection, rotation, or other noncode tensor changed.')
    stage_xor=[]
    for stage in range(2):
        expected=initial_banks[stage].clone()
        apply_choices(expected,effective_choices,stage)
        require(torch.equal(expected,layer.get_stage_part_vq_storage(stage,0)),
                'Saved accepted mask does not exactly reproduce a final native bank.')
        stage_xor.append(int(((effective_choices>stage*32)&(effective_choices<=(stage+1)*32)).sum()))
    ready(layer)
    final_sse=float((objective.prediction(layer._decode_weight(dtype=torch.bfloat16)).float()-target.float()).double().square().sum())
    require(final_sse==current_sse and current_sse<=initial_sse
            and sum(stage_xor)==sum(r['accepted_bit_flips'] for r in trajectory),
            'Restoration, final native loss or accepted bit inventory changed.')
    require(all(r['after_sse']<=r['before_sse'] and (r['trial_sse']<r['before_sse'])==r['accepted'] for r in trajectory),
            'A non-improving group was accepted.')
    masks={'accepted_groups':accepted.cpu(),'effective_choices':effective_choices.cpu()}
    report={'status':'completed','initial_sse':initial_sse,'final_sse':final_sse,'target_sse':target_sse,
        'initial_nmse':initial_sse/target_sse,'final_nmse':final_sse/target_sse,
        'accepted_groups':int(accepted.sum()),'rejected_groups':GROUPS-int(accepted.sum()),
        'stage_xor':stage_xor,'accepted_bit_flips':sum(stage_xor),'trajectory':trajectory,
        'initial_state':initial_state,'final_state':final_state,'noncode_frozen_exact':True,
        'accepted_mask_replay_exact':True,'final_native_sse_repeated_exact':True,
        'train_input':tensor_record(objective.inputs),'target':tensor_record(target),
        'elapsed_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
        'boundary':'One pass through384 predetermined output blocks, each proposal restricted to existing weight-search winners; train32 only. This is not an exhaustive output-optimal bit search or a production integration.'}
    return masks,report
