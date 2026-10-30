"""One actual Adam proposal and bounded deployment-based amplitude calibration."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time

from native_pair import (DECODER_PREFIX, HERE, OUTPUT_ROOT, PROJECTIONS, NativePair,
                         deployed_weight, rebuild, require, schema, sha256, state_record,
                         write_json)
from build_targets import build_targets, endpoint_metrics, energy
import torch
from torch.nn import functional as F

T0 = 1e-5
CASES = ((PROJECTIONS[0], 'space'), (PROJECTIONS[0], 'shrink'), (PROJECTIONS[2], 'space'))


def make_master(model):
    module = model.projection
    for value in module._parallel_stage_decoder.parameters():
        value.data = value.data.float()
    module.enable_trainable_decode(parallel_stage_decode=True)
    module._decoder_compute_dtype = torch.bfloat16
    module.training = True
    module._parallel_stage_decoder.eval()
    module._build_parallel_stage_decode_plan()
    parameters = dict(module._parallel_stage_decoder.named_parameters())
    require(len(parameters) == 6 and all(p.dtype == torch.float32 and p.requires_grad for p in parameters.values()),
            'The actual six registered FP32 decoder leaves must define the single proposal.')
    require({name for name,p in module.named_parameters() if p.requires_grad} ==
            {DECODER_PREFIX+name for name in parameters}, 'A nondecoder parameter became trainable.')
    return parameters


@torch.no_grad()
def evaluate_candidate(spec, source_state, initial, proposal_delta, scale, target, pre, post, protected):
    # Every trial reloads source frozen fields and initializes fresh parameter leaves.
    model = rebuild(spec, source_state, post.device)
    module = model.projection
    original_schema = schema(module)
    parameters = make_master(model)
    for name,p in parameters.items():
        p.copy_(initial[name]+scale*proposal_delta[name])
    master_state = {name:p.detach().cpu().clone() for name,p in parameters.items()}
    with torch.autocast('cuda', enabled=False):
        live = module._decode_weight(dtype=torch.bfloat16)
    row = dict(scale=float(scale), learning_rate=float(scale*T0),
               master_live=endpoint_metrics(live,target,pre,post,protected),
               decoder_master=state_record(master_state))
    del live
    for p in parameters.values():
        p.data = p.data.bfloat16()
    module._build_parallel_stage_decode_plan()
    deployed = deployed_weight(model)
    row['BF16_deployment'] = endpoint_metrics(deployed,target,pre,post,protected)
    state = module.state_dict()
    require(schema(module) == original_schema, 'Candidate persistent schema/dtype changed.')
    frozen = [name for name in source_state if not name.startswith(DECODER_PREFIX)]
    require(all(torch.equal(state[name].cpu(),source_state[name]) for name in frozen),
            'Candidate changed codes, protection, rotation or another frozen field.')
    require(row['BF16_deployment']['protected_move_energy'] == 0,
            'Candidate deployment changed protected input columns.')
    changed = {name:int(torch.count_nonzero(state[DECODER_PREFIX+name].cpu()!=source_state[DECODER_PREFIX+name]))
               for name in parameters}
    row.update(decoder_BF16_changed_elements=changed,
               decoder_BF16_changed_elements_total=sum(changed.values()),
               all_decoder_changes_erased_by_BF16=sum(changed.values()) == 0,
               decoder_deployment=state_record({name:state[DECODER_PREFIX+name] for name in parameters}),
               frozen_fields_exact=True, schema_unchanged=True, optimizer_steps_for_candidate=0,
               initialization='Fresh strict real G02, then D0 + scale*(one actual first Adam proposal minus D0)')
    if scale == 0:
        require(torch.equal(deployed,post.bfloat16()), 'Zero-amplitude native deployment is not exact G02.')
        row['zero_amplitude_exact_G02'] = True
    return deployed.float(), row


def run_case(spec, source_state, target, pre, post, protected):
    model = rebuild(spec,source_state,post.device)
    module = model.projection
    require(torch.equal(deployed_weight(model),post.bfloat16()), 'The single-proposal initial native deployment is not G02.')
    parameters = make_master(model)
    initial = {name:p.detach().clone() for name,p in parameters.items()}
    denominator = energy(post)/post.numel()
    with torch.autocast('cuda',enabled=False):
        live = module._decode_weight(dtype=torch.bfloat16)
        raw_mse = F.mse_loss(live.float(),target)
        loss = raw_mse/denominator
    initial_path = dict(master_live=endpoint_metrics(live.detach(),target,pre,post,protected),
                        error_energy_over_calibration_move=energy(live.detach().float()-post)/energy(target-post),
                        source_deployment_exact_G02=True)
    require(bool(torch.isfinite(loss)), 'The single-proposal loss is nonfinite.')
    loss.backward()
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all())
                for p in parameters.values()), 'The actual single-proposal decoder gradient is missing or invalid.')
    gradient = {name:p.grad.detach().clone() for name,p in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(),lr=T0,betas=(.9,.95),eps=1e-8,weight_decay=0)
    optimizer.step()
    require(all(int(s['step']) == 1 for s in optimizer.state.values()), 'The Adam clock must advance exactly once.')
    require(all(s['exp_avg'].dtype == s['exp_avg_sq'].dtype == torch.float32 for s in optimizer.state.values()),
            'The actual first Adam proposal must retain FP32 moments.')
    proposed = {name:p.detach().clone() for name,p in parameters.items()}
    proposal_delta = {name:proposed[name]-initial[name] for name in initial}
    report = dict(initial_path=initial_path, initial_master=state_record(initial),
                  actual_first_Adam_proposal=state_record(proposed),
                  actual_proposal_delta=state_record(proposal_delta), gradient=state_record(gradient),
                  optimizer_steps_total=1, backward_calls_total=1,
                  proposal_definition='Retain the actual PyTorch AdamW first FP32 parameter displacement D1-D0; all amplitudes scale this one displacement',
                  fixed_normalizer=denominator, raw_mse=float(raw_mse.detach()), nmse_loss=float(loss.detach()),
                  controls={}, bounded_trials=[])
    del optimizer,live,raw_mse,loss,model,module,parameters,gradient,proposed
    zero,report['controls']['zero'] = evaluate_candidate(spec,source_state,initial,proposal_delta,0.,target,pre,post,protected)
    trial_t0,report['controls']['t0'] = evaluate_candidate(spec,source_state,initial,proposal_delta,1.,target,pre,post,protected)
    error = target-post
    displacement = trial_t0-post
    d2 = energy(displacement)
    dot = float((error.double()*displacement.double()).sum())
    report['amplitude_nomination'] = dict(t0=T0, deployed_displacement_energy=d2,
        target_error_dot_deployed_displacement=dot,
        formula='a=clip(<T-Wdeploy(D0),Wdeploy(D1)-Wdeploy(D0)> / ||Wdeploy(D1)-Wdeploy(D0)||^2,0,1)')
    del zero,trial_t0,error,displacement
    if d2 == 0 or dot <= 0:
        report.update(status='failed_zero_or_non_descent_deployment_direction',
                      failure_reason='The one actual BF16 deployment displacement is zero or has nonpositive target inner product',
                      accepted_scale=None)
        return report
    alpha = min(max(dot/d2,0.),1.)
    report['amplitude_nomination']['a'] = alpha
    baseline_error = report['controls']['zero']['BF16_deployment']['error_energy']
    for halvings in range(5):
        scale = alpha*(.5**halvings)
        candidate,row = evaluate_candidate(spec,source_state,initial,proposal_delta,scale,target,pre,post,protected)
        row['halvings_after_nomination'] = halvings
        row['strict_deployment_error_improvement'] = row['BF16_deployment']['error_energy'] < baseline_error
        report['bounded_trials'].append(row)
        del candidate
        if row['strict_deployment_error_improvement']:
            report.update(status='one_direction_amplitude_improved_deployment_error', accepted_scale=scale,
                          accepted_learning_rate=scale*T0, accepted_halvings=halvings,
                          selected_by='First actual BF16 deployment improvement under the prespecified nomination/backtracking rule')
            break
    else:
        erased = all(row['all_decoder_changes_erased_by_BF16'] for row in report['bounded_trials'])
        report.update(status='failed_all_candidates_erased_by_BF16' if erased else 'failed_bounded_amplitudes_no_deployment_improvement',
                      accepted_scale=None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT) and output != OUTPUT_ROOT and not output.exists(),
            'Use a new independent amplitude-probe output directory in the approved root.')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,'Exactly one root-assigned GPU is required.')
    def terminated(signum, frame):
        raise TimeoutError('The external bounded amplitude-probe deadline expired.')
    signal.signal(signal.SIGTERM,terminated)
    torch.set_num_threads(4)
    torch.manual_seed(31)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.perf_counter()
    pair = NativePair()
    output.mkdir(parents=True)
    report = dict(status='running',pid=os.getpid(),argv=sys.argv,started_unix=time.time(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),source=pair.manifest(),
        code_sha256={name:sha256(HERE/name) for name in ('step_calibration.py','native_pair.py','build_targets.py')},
        settings=dict(optimizer='AdamW',t0=T0,betas=[.9,.95],eps=1e-8,weight_decay=0,
                      complete_weight_matrix=True,loss='Original-coordinate FP32 MSE / fixed G02 mean(Wpost^2)',
                      FP32_master_and_moments=True,BF16_native_compute_and_deployment=True,
                      optimizer_steps_per_case=1,maximum_halvings=4,case_order=[list(x) for x in CASES]),
        scope='Amplitude control of one already-observed Adam overshoot; no new optimization direction, 500-step fit or capability claim',cases={})
    write_json(output/'report.json',report)
    try:
        for name in (PROJECTIONS[0],PROJECTIONS[2]):
            spec,states = pair.projection(name)
            models = [rebuild(spec,state,torch.device('cuda:0')) for state in states]
            weights = [deployed_weight(model) for model in models]
            protected = states[1]['protected_input_indices'].to('cuda:0')
            pre,post,targets,geometry = build_targets(*weights,protected)
            del models,weights
            for case_name,label in CASES:
                if case_name != name:
                    continue
                key = name+':'+label
                print('case_start='+key,flush=True)
                result = run_case(spec,states[1],targets[label],pre,post,protected)
                result['geometry'] = geometry
                report['cases'][key] = result
                write_json(output/'report.json',report)
                chosen = next((x for x in result['bounded_trials'] if x['strict_deployment_error_improvement']),None)
                print(json.dumps(dict(case=key,status=result['status'],nomination=result['amplitude_nomination'],
                    t0_error=result['controls']['t0']['BF16_deployment']['error_energy_over_calibration_move'],
                    accepted_scale=result['accepted_scale'],
                    accepted_error=chosen['BF16_deployment']['error_energy_over_calibration_move'] if chosen else None,
                    trials=[dict(scale=x['scale'],error=x['BF16_deployment']['error_energy_over_calibration_move'],
                                 erased=x['all_decoder_changes_erased_by_BF16']) for x in result['bounded_trials']]),allow_nan=False),flush=True)
                gc.collect()
            del pre,post,targets
            torch.cuda.empty_cache()
        pair.check_source_stats()
        report['status'] = 'completed_three_case_single_direction_amplitude_probe'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        write_json(output/'report.json',report)
        print(f'amplitude_status={report["status"]} elapsed_seconds={report["elapsed_seconds"]:.3f}',flush=True)


if __name__ == '__main__':
    main()
