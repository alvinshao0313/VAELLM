"""Read-only real-state/geometry contract checks before the assigned GPU run."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import time

from native_pair import (HERE, OUTPUT_ROOT, PROJECTIONS, NativePair, deployed_weight,
                         read_json, rebuild, require, sha256, state_record, write_json)
from build_targets import build_targets, endpoint_metrics
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT) and output != OUTPUT_ROOT,
            'CPU evidence must use the independent approved output root.')
    require(not output.exists(), 'CPU evidence directory already exists.')
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    pair = NativePair()
    report = dict(status='checking', source=pair.manifest(), projections={},
                  code_sha256={path.name:sha256(path) for path in HERE.glob('*.py')})
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    for name in PROJECTIONS:
        spec,states = pair.projection(name)
        models = [rebuild(spec,state,torch.device('cpu')) for state in states]
        require(all(state_record(model.projection.state_dict()) == state_record(state)
                    for model,state in zip(models,states,strict=True)),
                'Fresh strict CPU state reconstruction changed the real persistent values.')
        report['projections'][name] = dict(shape=[spec['out_features'],spec['in_features']],
            transpose=spec['transpose'], state_fields=len(states[0]),
            native_strict_state_exact=True, frozen_parent_child_exact=True,
            state_tensor_bytes=sum(value.numel()*value.element_size() for value in states[0].values()))
        if name == PROJECTIONS[0]:
            weights = [deployed_weight(model) for model in models]
            pre,post,targets,geometry = build_targets(*weights,states[1]['protected_input_indices'])
            previous = read_json(HERE/'cpu_diagnostic.json')
            require(geometry['rank'] == previous['rank'], 'The existing real-K exact SVD rank changed.')
            require(abs(geometry['beta']-previous['retained_norm_shrink_beta']) < 1e-7,
                    'The original per-projection retained-update shrink definition changed.')
            fresh = rebuild(spec,states[1],torch.device('cpu'))
            require(torch.equal(deployed_weight(fresh),weights[1]),
                    'The unmodified G02 deployment did not reconstruct exactly.')
            metrics = {label:endpoint_metrics(weights[1],target,pre,post,states[1]['protected_input_indices'])
                       for label,target in targets.items()}
            require(abs(metrics['space']['error_energy_over_calibration_move']-1) < 1e-7,
                    'No-intervention calibration-relative error must be one.')
            require(metrics['space']['calibration_direction_survival'] == 0,
                    'No-intervention calibration direction survival must be zero.')
            require(geometry['targets']['space']['protected_unreachable_energy'] > 0,
                    'The real SPACE target must retain the known nonzero protected-column lower bound.')
            require(geometry['targets']['shrink']['protected_unreachable_energy'] == 0,
                    'The equal-retained-norm shrink control must keep protected columns unchanged.')
            report['real_K_geometry'] = geometry
            report['real_K_zero_intervention'] = metrics
        del models
    pair.check_source_stats()
    require(not torch.cuda.is_initialized(), 'CPU validation must not initialize CUDA.')
    report.update(status='passed_cpu_real_state_and_geometry_only', cuda_initialized=False,
                  elapsed_seconds=time.perf_counter()-started,
                  limitation='CUDA decode, decoder backward, fixed500 fitting and GPU save/reload are not covered by CPU evidence')
    write_json(output/'report.json',report)
    print(f'cpu_status={report["status"]} elapsed_seconds={report["elapsed_seconds"]:.3f}',flush=True)


if __name__ == '__main__':
    main()
