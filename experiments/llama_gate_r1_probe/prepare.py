"""CPU native parsing, small decoder geometry and disjoint fixed-data preparation."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from e2e_common.data import build_tokenizer
from experiments.llama_gate_r1_probe.common import (
    CONTROL, DOMAINS, HOLD, OUTPUT, PREFIX, PUBLIC, ROOT, SOURCE, SOURCE_ID, STEPS, TARGET,
    _load_state, actual_configuration, arguments, check, configuration, read, require,
    sha256, state_inventory, write_json)
from experiments.longbench_v1.common import file_stat
from experiments.rotation_factor_probe.capture import tensor_record
from litebsq.autoencoder import Decoder
from train_utils.distill_data import build_distill_dataset


def prepare():
    require(ROOT != PUBLIC and Path.cwd() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and not OUTPUT.exists(), 'Use actual S, hidden CUDA and a new output.')
    torch.set_num_threads(4)
    original = read(CONTROL); meta = read(SOURCE / 'checkpoint_meta.json')
    audit_path = SOURCE.parent / 'audit.json'; audit = read(audit_path)
    review_path = SOURCE.parent / 'independent_review.json'; review = read(review_path)
    require(meta['checkpoint_id'] == original['source_checkpoint_id'] == audit['checkpoint_id'] == SOURCE_ID
            and audit['status'] == 'verified' and review['status'] == 'independent_cpu_verified', 'Source proof is incomplete.')
    spec = next(x for x in meta['converted_modules'] if x['name'] == TARGET)
    require(spec['transpose'] and spec['residual_stages'] == 2 and spec['parallel_parts'] == 1
            and spec['weight_rotation']['block_size'] == 32 and spec['protected_input_indices']['shape'] == [64]
            and all((x['in_dim'], x['out_dim'], x['hidden_dim'], x['num_res_blocks'], x['norm_type'],
                     x['activation_type'], x['decoder_type']) == (34, 32, 128, 0, 'layer', 'swish', 'symmetric')
                    for x in spec['stage_decoders']), 'Source is not the exact authorized L16 gate geometry.')
    state = _load_state(SOURCE)
    r0 = state_inventory(state)
    small = Decoder(in_dim=34, out_dim=32, hidden_dim=128, num_res_blocks=1, norm_type='layer',
                    activation_type='swish', decoder_type='symmetric', use_checkpoint=True, num_models=2).bfloat16()
    added = {PREFIX + k: v for k, v in small.state_dict().items() if k.startswith('blocks.0.')}
    r1 = state_inventory({**state, **added})
    require(r0['state_tensors'] == 2979 and r0['frozen_tensors'] == r1['frozen_tensors'] == 2973
            and len(r0['selected']) == 6 and len(r1['selected']) == 10 and r0['banks'] == r1['banks'] == 448
            and r1['selected_elements'] - r0['selected_elements'] == 33536,
            'Native constructed R1 leaves do not match the source packed inventory.')
    for key, value in small.state_dict().items():
        if not key.startswith('blocks.0.'):
            require(list(value.shape) == r0['selected'][PREFIX + key]['shape'], 'Existing native decoder shape changed.')
    stats_path = Path(original['data']['canonical_statistics']); stats = read(stats_path)
    rows = stats['models']['llama']['sampler_first8000']['rows'][:4 * STEPS]
    batches = stats['models']['llama']['sampler_first8000']['batches'][:STEPS]
    hold_record = read(HOLD / 'launch.prepared.json')
    archive = torch.load(HOLD / 'token_batches.pt', weights_only=True, map_location='cpu')
    held = {k: archive[k] for k in DOMAINS}
    require({d: {k: tensor_record(v) for k, v in b.items()} for d, b in held.items()}
            == {d: hold_record['data']['token_batches'][d] for d in DOMAINS}, 'Existing fixed observations drifted.')
    require(not set(x['row_index'] for x in rows) & set(range(700032, 700064)), 'Training and fixed FineWeb rows overlap.')
    sources, inputs = {}, {}
    for raw, digest in original['source_sha256'].items():
        path = Path(raw)
        mapped = path if path.is_relative_to(ROOT) else ROOT / path.relative_to(PUBLIC) if path.is_relative_to(PUBLIC) else path
        chosen = mapped if mapped.is_file() else path
        require(sha256(chosen) == digest, 'Original runtime differs: ' + str(chosen))
        (sources if mapped.is_file() else inputs)[str(chosen)] = digest
    pending, seen = list(Path(__file__).parent.glob('*.py')), set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path); sources[str(path)] = sha256(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else
                     [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                local = ROOT / (name.replace('.', '/') + '.py')
                if local.is_file():
                    pending.append(local)
    for path in Path(__file__).parent.glob('*.py'):
        require(sha256(path) == sha256(PUBLIC / path.relative_to(ROOT)), 'New public/S source differs.')
    evidence = [CONTROL, SOURCE / 'checkpoint_meta.json', SOURCE / 'config.json', audit_path, review_path,
                SOURCE.parent / 'full8_summary.json', Path(original['data']['manifest']), stats_path,
                HOLD / 'launch.prepared.json', HOLD / 'token_batches.pt', HOLD / 'independent_review.json',
                PUBLIC / 'experiments/gate_tail_objective_probe/core.py', PUBLIC / 'experiments/gate_final_kl_accept_probe/core.py']
    inputs.update({str(p): sha256(p) for p in evidence})
    environment = deepcopy(original['environment']); environment['PYTHONPATH'] = str(ROOT)
    record = {'schema': 'llama_single_gate_identity_r1_prefix128_v1', 'status': 'prepared_unbound',
        'gpu': None, 'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(OUTPUT), 'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
        'source_weight_stat': file_stat(SOURCE / 'pytorch_model.bin'), 'source_bit_budget': audit['bit_budget'],
        'target_spec': spec, 'inventories': {'r0': r0, 'r1': r1}, 'hf_source': original['hf_source'],
        'model_policy': hold_record['model_policy'], 'input_data_stat': original['input_data_stat'],
        'source_sha256': sources, 'input_evidence_sha256': inputs, 'environment': environment,
        'expansion': {'added_parameters': 33536, 'added_bits': 536576, 'added_state_tensors': 4,
            'identity': 'Existing native ResnetBlock1D LayerNorm+Swish with linear1 weight/bias zero and norm gamma1/beta0.',
            'gpu_function_error': 'pending; require native BF16 relative L2 <=0.002, record exact and cached/packed differences',
            'not_lora': 'No factorized low-rank matrix or external VAELinear branch; one internal residual block per stage.'},
        'data': {**deepcopy(original['data']), 'training_rows': rows, 'training_batches': batches,
            'holdout_source': str(HOLD / 'token_batches.pt'), 'holdout_domains': list(DOMAINS),
            'holdout_boundary': 'Reused development observations, disjoint from this128-step training prefix only. '
                'No historical unseen claim, no benchmark examples, no hyperparameter/step selection from these64 documents.'},
        'schedule': {'max_steps': 2000, 'warmup_steps': 20, 'actual_stop_step': STEPS, 'save_steps': STEPS,
            'arms': ['r0', 'r1'], 'teacher_and_initial_observation_before_training': True,
            'final_native_save_and_fresh_reload': True, 'full8_or_longbench': False},
        'resource_budget_seconds': 5400, 'termination_grace_seconds': 20,
        'estimate': {'single_gpu_peak_gib': [50, 75], 'host_peak_gib': [30, 60], 'disk_bytes': 32_000_000_000,
            'elapsed_seconds': [1500, 4200], 'boundary': 'Unmeasured R1. Frozen223 dense caches plus BF16 teacher and B4x1024 '
                'activations dominate; R1 retains its real backward. Two native128 arms run sequentially, each exits before scoring.',
            'disk_free_at_prepare': shutil.disk_usage(OUTPUT.parent).free},
        'scientific_boundary': 'Codec capacity plus residual parametrization; same original parameters/codes/function initialization. '
            'R1 costs536576 more logical bits and extra FLOPs. Short development KL/NLL is not downstream quality proof.'}
    record['training_argv'] = {arm: arguments(record, arm) for arm in ('r0', 'r1')}
    configs = {arm: configuration(record, arm) for arm in ('r0', 'r1')}
    record['cpu_configuration'] = {arm: actual_configuration(*values) for arm, values in configs.items()}
    expected = deepcopy(original['training_argv'])
    for key in ('--student_checkpoint_dir', '--run_root_dir', '--target_layers', '--target_modules', '--save_steps'):
        expected[expected.index(key) + 1] = record['training_argv']['r0'][expected.index(key) + 1]
    require(expected == record['training_argv']['r0'], 'Unexpected difference from the actual same-source LM control.')
    tokenizer = build_tokenizer(record['hf_source']['path'], None)
    dataset = build_distill_dataset(configs['r0'][0].data, tokenizer).train_dataset
    hold_prefixes = [tuple(b['input_ids'][i, :int(b['attention_mask'][i].sum()) - 1].tolist())
                     for b in held.values() for i in range(32)]
    for row in rows:
        sample = dataset[row['row_index']]
        ids = sample['input_ids']
        require(tensor_record(ids)['sha256'] == row['input_ids_sha256'], 'Native CPU training row differs from canonical proof.')
        payload = tuple(ids[:-1].tolist())
        require(all(payload[:len(h)] != h and h[:len(payload)] != payload for h in hold_prefixes),
                'A held document is identical to or prefix-overlaps a training document; do not replace it heuristically.')
    record['data']['native512_training_vs64_observation_prefix_disjoint'] = True
    record['cpu_checks'] = {'status': 'verified', 'cuda_initialized': torch.cuda.is_initialized(),
        'native_parser_both_arms': True, 'native_small_decoder_shapes': True, 'actual_source_mmap_inventory': True,
        'actual512_training_examples_encoded': True, 'gpu_migration_backward_and_reload': 'not_executed'}
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record)
    OUTPUT.mkdir()
    torch.save(held, OUTPUT / 'holdout.pt')
    record['input_evidence_sha256'][str(OUTPUT / 'holdout.pt')] = sha256(OUTPUT / 'holdout.pt')
    record['argv'] = [sys.executable, '-u', '-m', 'experiments.llama_gate_r1_probe.run', '--launch-record', str(OUTPUT / 'launch.json')]
    write_json(OUTPUT / 'launch.prepared.json', record)
    print({'path': str(OUTPUT / 'launch.prepared.json'), 'sha256': sha256(OUTPUT / 'launch.prepared.json'),
           'source_pins': len(sources), 'input_pins': len(record['input_evidence_sha256']), 'cpu': 'verified', 'gpu': 'unbound'}, flush=True)


if __name__ == '__main__':
    prepare()
