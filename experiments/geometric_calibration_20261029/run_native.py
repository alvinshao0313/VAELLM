"""Three preselected complete projections, two fixed-budget native fit targets."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time

from native_pair import (HERE, OUTPUT_ROOT, PROJECTIONS, NativePair, deployed_weight,
                         rebuild, require, sha256, state_record, write_json)
from build_targets import build_targets, endpoint_metrics
from fit_native import STEPS, check_deadline, fit_target
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def terminated(signum, frame):
        raise TimeoutError('The external 15-minute runtime limit terminated this bounded prototype.')
    signal.signal(signal.SIGTERM, terminated)
    output = args.output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT) and output != OUTPUT_ROOT,
            'Use a new independent run directory below the approved geometric output root.')
    require(not output.exists(), 'Run directory exists; refuse overwriting or automatic resume.')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Exactly one root-assigned visible CUDA device is required.')
    started = time.monotonic()
    deadline = started+15*60
    torch.set_num_threads(4)
    torch.manual_seed(31)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    pair = NativePair()
    output.mkdir(parents=True)
    report = dict(status='running', pid=os.getpid(), argv=sys.argv, started_unix=time.time(),
                  process_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split(')',1)[1].split()[19]),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(0),
                  python=sys.executable, torch_version=torch.__version__, deadline_seconds=900,
                  code_sha256={path.name:sha256(path) for path in HERE.glob('*.py')},
                  source=pair.manifest(), cases={},
                  scope='Representation prototype only; no whole-model task/capability/bpw claim')
    write_json(output/'report.json', report)
    try:
        for name in PROJECTIONS:
            require(time.monotonic() < deadline, 'The fixed prototype deadline expired before the next projection.')
            spec, states = pair.projection(name)
            print(f'projection_start={name}', flush=True)
            models = [rebuild(spec,state,device) for state in states]
            weights = [deployed_weight(model) for model in models]
            protected = states[1]['protected_input_indices'].to(device)
            pre, post, targets, geometry = build_targets(*weights, protected)
            check_deadline(deadline)
            case = dict(geometry=geometry, source_projection_state=[state_record(s) for s in states], arms={},
                        source_deployment='Audited snapshot VAELinear native CUDA BF16 deployment decode',
                        original_controls={label:{'G01':endpoint_metrics(pre,target,pre,post,protected),
                                                  'G02':endpoint_metrics(post,target,pre,post,protected)}
                                           for label,target in targets.items()},
                        dense_controls={label:endpoint_metrics(target.bfloat16(),target,pre,post,protected)
                                        for label,target in targets.items()})
            report['cases'][name] = case
            del models, weights
            gc.collect()
            for label,target in targets.items():
                def progress(item):
                    if 'step' in item:
                        print(f'{name} {label} step={item["step"]}/{STEPS} nmse={item["nmse_loss"]:.9g} seconds={item["step_seconds"]:.4f}', flush=True)
                    else:
                        print(f'{name} {label} {json.dumps(item, allow_nan=False)}',flush=True)
                case['arms'][label] = fit_target(spec,states[1],target,pre,post,protected,
                    output/f'{name}.{label}.decoder.pt',deadline=deadline,progress=progress)
                check_deadline(deadline)
                write_json(output/'report.json',report)
                gc.collect()
            left,right = case['arms'].values()
            require(left['initial_decoder'] == right['initial_decoder']
                    and left['weight_scalar_exposures'] == right['weight_scalar_exposures']
                    and left['settings'] == right['settings'],
                    'Paired decoder initialization, scalar exposure or optimizer/loss budget differs.')
            case['paired_recipe_exact'] = True
            del pre,post,targets
            torch.cuda.empty_cache()
        pair.check_source_stats()
        check_deadline(deadline)
        report['status'] = 'completed_representation_prototype'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(output/'report.json',report)
        print(f'prototype_status={report["status"]} elapsed_seconds={report["elapsed_seconds"]:.3f}',flush=True)


if __name__ == '__main__':
    main()
