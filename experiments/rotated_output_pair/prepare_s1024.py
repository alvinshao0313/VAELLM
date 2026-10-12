"""Fresh1024-per-stage pair; same old data/initialization, fixed read-only observations."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from .data import OUTPUT as CONTROL, R2, ROOT, assets, check, native_inputs, read
from .diagnostics import verify_readonly
from .training import initial_states
from .run import run  # Pin the unchanged native persistence/full-model observation chain.

OUTPUT = R2 / 'qwen_l18_rotated_output_pair_s1024'


def prepare():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not OUTPUT.exists(),
            'CPU-only preparation once, fresh1024 output.')
    torch.set_num_threads(4)
    old = read(CONTROL / 'launch.prepared.json')
    recovery = CONTROL / 'observation_recovery'
    completed, report = read(recovery / 'launch.json'), read(recovery / 'report.json')
    require(completed['status'] == 'completed' and completed['exit_code'] == 0
            and report['status'] == 'completed_observation_recovery'
            and all((recovery / (n + '.exit')).read_text().strip() == '0' for n in ('run', 'outer')),
            'Require actual recovered old256 observations; do not rewrite its failed first run.')
    require(all(not Path('/proc/' + str(pid)).exists() for pid in (851313, 851318, 860781, 860787)),
            'Old helper consumers must have terminated.')
    archive = CONTROL / 'source_before_s1024'
    history = read(archive / 'archive.json')
    previous = read(recovery / 'launch.prepared.json')
    for path, row in history['source_files'].items():
        require(previous['source_sha256'][path] == sha256(Path(row['archived_path'])) == row['sha256'],
                'Old implementation archive differs from executed recovery pins.')
    check_hashes({p: h for p, h in previous['source_sha256'].items() if p not in history['source_files']})
    check_hashes(previous['input_sha256'])
    data = assets(); material, args, argv = native_inputs(data, steps=1024)
    expected_argv = list(old['native_argument_recipe'])
    at = expected_argv.index('--vae_steps') + 1
    require(expected_argv[at] == 'default=256', 'Unexpected old native training horizon.')
    expected_argv[at] = 'default=1024'
    require(argv == expected_argv and vars(args) == old['resolved_vae_args'],
            'Only the native horizon may differ; VAE geometry/optimizer/loss unchanged.')
    starts = initial_states(args)
    require([{k: tensor_record(v) for k, v in s.items()} for s in starts] == old['initial_states']
            and data['spec'] == old['native_spec'], 'Fresh initialization/spec changed.')
    require({d: {k: tensor_record(v) for k, v in b.items()} for d, b in data['batches'].items()}
            == old['token_batches'], 'Old development data changed.')
    require({k: tensor_record(getattr(material.split_metas[0], k)) for k in old['common_fresh_hf_protection']}
            == old['common_fresh_hf_protection'], 'Fresh HF protection changed.')
    checks = verify_readonly(args)
    record = deepcopy(old)
    record.update(output_dir=str(OUTPUT), status='prepared_unbound', gpu=None,
        native_argument_recipe=argv, steps_per_stage=1024, diagnostic_steps=[256, 512, 1024],
        timeout_seconds=1500, termination_grace_seconds=20, cpu_checks=checks,
        historical_control=str(CONTROL), initial_states=old['initial_states'],
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.rotated_output_pair.run',
            '--launch-record', str(OUTPUT / 'launch.json')])
    record['protocol'].update(
        training='Fresh seed31/32, 1024 updates per stage with linear1024 schedule; all1548288 vectors and one global BSQ per update, same8/32 document permutations. No resume of old256 endpoints.',
        observations='After each stage updates256/512/1024, read-only native eval reconstruction and fixed4289-token full-input SSE in unfused FP32. Final unfused FP32 versus actual native BF16/group8 SSE on all three old capture domains. Modes/RNG/state versions/gradients must remain unchanged.',
        codes='Per stage net code XOR after update992 versus1024. Not cumulative flips; first-to-final XOR and actual BF16 parameter changes also retained.',
        auxiliary_gradients='Separate main/auxiliary gradients intentionally omitted to keep the read-only path small; existing total encoder/decoder gradient norms retained. Auxiliary dominance remains unresolved.',
        comparison='Budget and linear schedule horizon both256→1024; per-arm stage2 residual follows its own stage1. Fixed observations are not endpoint selection. Only final1024-per-stage endpoints receive existing fixed development KL/NLL; no full8/LongBench/acceptance claim.',
        resource_estimate='Old256 fits88.4129+98.4582sec, observation54.5589sec; 4x fit extrapolation747.485sec plus observations/loading. Allow1500+20sec, one80GB GPU, prior observation peak70554192896 allocated/71064092672 reserved bytes, two local checkpoints <1GiB.')
    record['input_sha256'] = {str(p.resolve()): sha256(p) for p in data['inputs']}
    evidence = [CONTROL / 'launch.prepared.json', CONTROL / 'launch.json', archive / 'archive.json',
        recovery / 'launch.prepared.json', recovery / 'launch.json', recovery / 'report.json',
        recovery / 'run.exit', recovery / 'outer.exit']
    evidence += [CONTROL / 'run' / (arm + '.json') for arm in old['arms']]
    evidence += [Path(x['archived_path']) for x in history['source_files'].values()]
    record['input_sha256'].update({str(p): sha256(p) for p in evidence})
    record['source_sha256'] = {p: sha256(Path(p)) for p in previous['source_sha256']}
    for module in tuple(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and Path(p).is_absolute() and Path(p).is_file() and Path(p).suffix == '.py' and Path(p).resolve().is_relative_to(ROOT):
            p = Path(p).resolve(); record['source_sha256'][str(p)] = sha256(p)
    for p in Path(__file__).parent.glob('*.py'):
        ast.parse(p.read_text()); record['source_sha256'][str(p.resolve())] = sha256(p)
    record['protected_stats'] = data['stats']
    record['cuda_initialized'] = torch.cuda.is_initialized()
    require(not record['cuda_initialized'], 'CPU preparation initialized CUDA.')
    check(record); OUTPUT.mkdir(parents=True); write_json(OUTPUT / 'launch.prepared.json', record)
    print(json.dumps({'prepared': str(OUTPUT / 'launch.prepared.json'), 'sha256': sha256(OUTPUT / 'launch.prepared.json'),
        'sources': len(record['source_sha256']), 'inputs': len(record['input_sha256']), 'stats': len(record['protected_stats']),
        'cpu_checks': checks, 'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    prepare()
