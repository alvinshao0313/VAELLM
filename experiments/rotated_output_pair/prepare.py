"""CPU-only preparation for the two fresh rotated full-matrix fits."""
import ast
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import sha256, source_pins, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from .checks import verify
from .data import ARMS, CHUNK, OUTPUT, R2, ROOT, SEED, SOURCE, SOURCE_ID, STEPS, TARGET, assets, check, native_inputs, read
from .observe import evaluate, persist  # Include runtime-only native dependencies in the source closure.
from .training import initial_states


def prepare():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not OUTPUT.exists(),
            'Prepare once from the actual public source with CUDA hidden and a fresh output directory.')
    torch.set_num_threads(4)
    data = assets(); material, args, argv = native_inputs(data)
    starts = initial_states(args); cpu = verify(args)
    tail = read(R2 / 'gate_tail_objective_probe_r1/launch.prepared.json')
    sources = source_pins()
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_absolute() and Path(path).is_file() and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(ROOT):
            path = Path(path).resolve(); sources[str(path)] = sha256(path)
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path.resolve())] = sha256(path)
    inputs = {str(p.resolve()): sha256(p) for p in data['inputs']}
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protection = {k: tensor_record(getattr(material.split_metas[0], k)) for k in fields}
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(OUTPUT),
        'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID, 'target': TARGET,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': data['stats'],
        'source_native_policy': tail['data']['source_data']['model_policy'],
        'source_bit_budget': tail['source_bit_budget'], 'native_spec': data['spec'],
        'common_fresh_hf_protection': protection,
        'initial_states': [{k: tensor_record(v) for k, v in state.items()} for state in starts],
        'token_batches': {d: {k: tensor_record(v) for k, v in b.items()} for d, b in data['batches'].items()},
        'native_argument_recipe': argv, 'resolved_vae_args': vars(args), 'arms': list(ARMS),
        'steps_per_stage': STEPS, 'stages': 2, 'seed': SEED, 'chunk_vectors': CHUNK,
        'cpu_checks': cpu, 'timeout_seconds': 3600, 'termination_grace_seconds': 20,
        'environment': {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'},
        'protocol': {
            'loss': 'physical full output MSE / (stage_std^2 * trace(H_CC)); H_CC=E_train[X_C.T X_C], uncentered fixed4289-token trace. Native BSQ auxiliary unchanged.',
            'protection': 'Both arms regenerate identical native INT8 q/scales from original HF at fixed f92764 input indices. Full output error includes fixed protected error and all cross terms.',
            'normalization': 'Isotropic H_CC=I and zero protected error reduce algebraically to normalized WMSE. Nonzero protection retains its constant/cross terms. Actual BF16 input and deployment rounding are reported boundaries, not exact algebra claims.',
            'weight_control': 'Decoded BF16 values and normalized BF16 VAE input cast to FP32 for WMSE, as existing full_vae_forward diagnostic. Output target is original HF physical BF16 weight cast FP32.',
            'fresh': 'No historical encoder is retained. Both arms use the same real fresh stage seed31/32 initial states. Stage2 uses its own native stage1 residual; no shared residual substitution.',
            'training': 'Every update includes all1548288 vectors, M1, one global BSQ call;65536 chunks only encoder/decoder memory. 256 updates per stage with full linear256 schedule;8 of old32 train documents per output update, same seeded permutations.',
            'cat_boundary': 'Global BSQ population and M1/full-matrix updates differ from original CAT sampled blocks; this pair is not a formal CAT reconstruction-only single-variable comparison.',
            'evaluation': 'Native local13 v6 save/reload, f9273410 frozen complement, BF16 group8 full-model fixed train32 and old Alpaca/FineWeb32 top100-mass KL/NLL. Local train/held/replay full-input SSE; no endpoint selection.',
            'limitations': 'Already-used development data, no unseen-test claim; no full8, LongBench, convergence, or final quality/acceptance claim.',
            'resource_estimate': 'Unmeasured full-matrix fit10-20GiB; native full-model prewarm previously about66GiB. Sequential phases on one80GB GPU, estimated15-30min with3600+20 hard cap; reserve1GiB output for two native local checkpoints and evidence.'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.rotated_output_pair.run', '--launch-record', str(OUTPUT / 'launch.json')],
        'cuda_initialized': torch.cuda.is_initialized()}
    require(not record['cuda_initialized'], 'CPU preparation initialized CUDA.')
    check(record)
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / 'launch.prepared.json', record)
    print(json.dumps({'prepared': str(OUTPUT / 'launch.prepared.json'), 'sha256': sha256(OUTPUT / 'launch.prepared.json'),
          'sources': len(sources), 'inputs': len(inputs), 'stats': len(data['stats']), 'cpu_checks': cpu}), flush=True)


if __name__ == '__main__':
    prepare()
