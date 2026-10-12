"""CPU preparation only. A parent-approved bound launch is required for any GPU work."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

from experiments.llama_column_feedback_probe.cpu_checks import run_checks
from experiments.llama_column_feedback_probe.data import build_data
from experiments.llama_column_feedback_probe.model import BANKS, SOURCE_ID, TARGET
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.checkpoint_bpw import audit_state_dict

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
SOURCE = R2 / 'llama_fineweb1600_down_d16_combo/final_model'
CONTROL = R2 / 'general_lm_llama_d16_head_s2000/launch.prepared.json'
FINEWEB = ROOT / 'data/fineweb_edu_10bt_20261005/train.jsonl'


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for stat in record['protected_stats']:
        check_stat(stat)


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU preparation requires CUDA explicitly hidden.')
    require(not output.exists(), 'Use a fresh output; never overwrite another preparation.')
    require(shutil.disk_usage(output.parent).free >= 2_000_000_000, 'Reserve 2GB for probe records and packed endpoints.')
    torch.set_num_threads(4)
    control = read(CONTROL)
    require(control['source_checkpoint_id'] == SOURCE_ID and Path(control['source_checkpoint']) == SOURCE,
            'Use the pinned original e767, not a recovery endpoint.')
    check_hashes(control['source_sha256']); check_hashes(control['hf_source']['metadata_sha256'])
    check_stat(control['source_weight_stat'])
    for row in control['hf_source']['weight_files']:
        check_stat(row)
    meta_path = SOURCE / 'checkpoint_meta.json'; meta = read(meta_path)
    require(meta['checkpoint_id'] == SOURCE_ID and meta['checkpoint_kind'] == 'final_model'
            and meta['state_dict_file'] == 'pytorch_model.bin', 'Require unchanged stable e767 source metadata.')
    spec = next(value for value in meta['converted_modules'] if value['name'] == TARGET)
    require(spec['transpose'] and spec['codebook_dim'] == 16 and spec['stage_codebook_dims'] == [16, 16]
            and spec['residual_stages'] == 2 and spec['parallel_parts'] == 1
            and (spec['in_features'], spec['out_features'], spec['compressed_in_features']) == (14336, 4096, 14080)
            and spec['protected_channel_quant_format'] == 'int8'
            and spec['protected_input_indices']['shape'] == [256]
            and spec['weight_rotation']['type'] == 'two_sided_hadamard' and spec['weight_rotation']['block_size'] == 32
            and all((d['in_dim'], d['out_dim'], d['hidden_dim'], d['num_res_blocks'], d['decoder_type'], d['param_dtype'])
                    == (24, 16, 128, 0, 'symmetric', 'bfloat16') for d in spec['stage_decoders']),
            'The selected L16 down is not the approved original transposed D16/K24/S2 geometry.')
    state = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    local = {key[len(TARGET) + 1:]: tensor_record(value) for key, value in state.items() if key.startswith(TARGET + '.')}
    require(all(local[key]['shape'] == [14080 * 256, 1, 3] and local[key]['dtype'] == 'torch.uint8' for key in BANKS),
            'Actual packed source banks differ from their transposed logical shape.')
    budget = audit_state_dict(meta, state, max_bpw=2.5, max_lora_rank=0)
    require(budget['budget_passed'] and budget['user_linear_bits'] == 17095180288
            and budget['original_linear_weights'] == 6979321856 and len(state) == 2979,
            'Actual source storage budget or full-state inventory differs.')
    hf_path = meta['base_model_path']; cfg = AutoConfig.from_pretrained(hf_path, local_files_only=True)
    require(cfg.model_type == 'llama' and cfg.num_hidden_layers == 32 and cfg.hidden_act == 'silu',
            'Use native Llama-3.1-8B-Instruct configuration.')
    tokenizer = AutoTokenizer.from_pretrained(hf_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    batches, data = build_data(tokenizer, FINEWEB)
    checks = run_checks()
    sources = deepcopy(control['source_sha256'])
    dependencies = ('experiments/qwen3_w25/block_ldlq.py', 'experiments/qwen3_w25/block_ldlq_model.py',
        'experiments/qwen3_w25/calibration.py', 'experiments/linear_output/calibration.py',
        'experiments/gate_tail_objective_probe/core.py', 'experiments/gate_student_context_evaluation/deploy.py',
        'experiments/rotation_factor_probe/capture.py', 'experiments/longbench_v1/common.py',
        'rotation/model_utils.py', 'rotation/weight_preprocess.py', 'train_utils/distill_loss_core.py',
        'litebsq/fused_multistage_decoder.py', 'litebsq/bitpack.py', 'litebsq/vae_linear.py',
        'train_utils/checkpoint_bpw.py', 'train_utils/checkpoint_v6.py', 'tools/cat_eval.py')
    for path in [*(ROOT / name for name in dependencies), *sorted(Path(__file__).parent.glob('*.py'))]:
        ast.parse(path.read_text()); digest = sha256(path)
        require(str(path) not in sources or sources[str(path)] == digest, 'A reused source pin changed: ' + str(path))
        sources[str(path)] = digest
    inputs = {str(CONTROL): sha256(CONTROL), str(meta_path): sha256(meta_path),
              **control['hf_source']['metadata_sha256']}
    for path in (FINEWEB.parent / 'manifest.json', FINEWEB.parent / 'canonical_statistics.json'):
        inputs[str(path)] = sha256(path)
    stats = [control['source_weight_stat'], *control['hf_source']['weight_files'],
             file_stat(FINEWEB), file_stat(data['alpaca']['source_arrow'])]
    record = {'schema': 'llama_transposed_column_feedback_probe_v1', 'status': 'prepared_unbound',
        'gpu': None, 'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(output), 'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
        'hf_source': control['hf_source'], 'target': TARGET, 'target_spec': spec,
        'expected_target_state': local, 'source_serialized_tensors': len(state), 'source_bit_budget': budget,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': stats,
        'model_policy': {'model_type': 'llama', 'attn_implementation': 'sdpa', 'rope_theta': cfg.rope_theta,
                         'rope_scaling': cfg.rope_scaling, 'hidden_act': cfg.hidden_act},
        'data': data, 'cpu_checks': checks, 'environment': deepcopy(control['environment']),
        'resource_budget_seconds': 3600, 'timeout_seconds': 3600, 'termination_grace_seconds': 20,
        'command': ['timeout', '--signal=TERM', '--kill-after=20s', '3600s', sys.executable, '-u', '-m',
                    'experiments.llama_column_feedback_probe.run', '--launch', str(output / 'launch.json')],
        'resource_estimate': {'gpus': 1, 'gpu_peak_gib_estimate': 60, 'host_peak_gib_estimate': 48,
                             'wall_seconds_estimate': [600, 1800], 'disk_bytes_upper_estimate': 2_000_000_000,
                             'basis': 'Estimate only: one native model resident at a time; no optimizer or full export. '
                                      'FP64 full14336 Hessian and14080 factors plus three finite-neighbor endpoints. '
                                      '880 groups share candidate decoding; 192 teacher docs and four student endpoint passes.'},
        'protocol': {'arms': ['original', 'off', 'delayed16', 'per_column'], 'damping_ratio': .01,
            'feedback_order': 'descending rotated input column', 'candidate_generation_group_columns': 16,
            'delayed16': 'Intentionally no within-group feedback; first feedback occurs after all16 columns selected.',
            'per_column': 'Immediate within-group feedback; equivalent lazy batching only outside the group.',
            'off': 'No feedback; shares the same fixed-protection compensated target as all arms.',
            'candidates': 'Original plus24 single-bit flips from each of two stages; same49 pool and one visit per vector.',
            'bank_index': 'input_column*256+output_group; D16 groups partition the rotated output axis.',
            'target': 'T=U[W_HF,C+(W_HF,P-Q_P)H_PC(H_CC+lambda I)^-1]V^T; Q_P is actual deployed BF16 INT8 decode.',
            'hessian': 'Training student-prefix uncentered E[x x^T], lambda=.01 mean(diag(H_CC)); Arot=V A V^T.',
            'factor': 'Arot=L diag(d)L^T; immediate t_j=T_j+sum(k>j)(T_k-Q_k)L_kj.',
            'candidate_metric': 'd_j*sum over16 output coordinates of squared FP64 error after native BF16 stage sum.',
            'observations': 'Native BF16 local SSE/NMSE plus complete-model top100+tail-mass KL and text NLL, '
                            'token weighted and per document, on calibration and both pre-fixed holdouts.',
            'success_boundary': 'Local proxy reduction alone is not success. No eight-task/LongBench claim; '
                                'a native full-model short probe is still required after these CPU checks.',
            'frozen': 'All decoder, protection, rotation, norm, head and other projection tensors; only L16 down two packed banks.',
            'papers': ['https://arxiv.org/html/2402.15319v2', 'https://aclanthology.org/2024.emnlp-main.467.pdf'],
            'previous_negative': 'Qwen4b6 all36 nontranspose down,49-neighbor BlockLDL: every local dev improves, '
                                 'full8 64.14388457 ->63.83972757/63.83909699. Not this matched transposed granularity test.'}}
    del state
    require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA.')
    check(record)
    output.mkdir(parents=True)
    torch.save(batches, output / 'token_batches.pt')
    record['input_sha256'][str(output / 'token_batches.pt')] = sha256(output / 'token_batches.pt')
    write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'path': str(output / 'launch.prepared.json'),
                      'sha256': sha256(output / 'launch.prepared.json'), 'cpu_checks': checks,
                      'source_pins': len(sources), 'input_pins': len(record['input_sha256'])}, ensure_ascii=False), flush=True)


def prepare_repair(output, previous):
    """Reuse the already verified CPU formulas and exact token archive after the CPU-audit API fix."""
    output, previous = Path(output).resolve(), Path(previous).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and not output.exists(), 'CPU repair preparation requires hidden CUDA and a fresh output.')
    original = read(previous); prior_dir = previous.parent
    failure = read(prior_dir / 'report.json')
    require(original['status'] == 'prepared_unbound' and original['source_checkpoint_id'] == SOURCE_ID
            and failure['status'] == 'failed'
            and failure['error'] == 'ValueError: Expected a CPU strided tensor: model.embed_tokens.weight.'
            and (prior_dir / 'run.exit').read_text().strip() == (prior_dir / 'outer.exit').read_text().strip() == '1',
            'Only the completed, identified CPU-auditor API failure is covered by this repair.')
    changed = {str(Path(__file__).resolve()), str(Path(__file__).with_name('core.py').resolve())}
    check_hashes({path: digest for path, digest in original['source_sha256'].items() if path not in changed})
    check_hashes(original['input_sha256'])
    for stat in original['protected_stats']:
        check_stat(stat)
    for path in changed:
        archived = prior_dir / 'failed_source' / Path(path).name
        require(sha256(archived) == original['source_sha256'][path], 'The failed source was not retained exactly.')
        ast.parse(Path(path).read_text())
    record = deepcopy(original)
    record.update(output_dir=str(output), gpu=None, status='prepared_unbound', queued=False, automatic_retry=False)
    record['command'][-1] = str(output / 'launch.json')
    record['source_sha256'].update({path: sha256(path) for path in changed})
    for path in [previous, prior_dir / 'report.json', prior_dir / 'run.log', prior_dir / 'run.exit',
                 prior_dir / 'outer.exit', *(prior_dir / 'failed_source' / Path(path).name for path in changed)]:
        record['input_sha256'][str(path)] = sha256(path)
    record['repair'] = {'previous_prepared': str(previous), 'previous_prepared_sha256': sha256(previous),
        'change': 'Audit original CPU source mmap plus each actual two-bank CPU endpoint; persist observations before final audit.',
        'same_algorithm_data_and_fresh_e767': True, 'new_cpu_formula_tests_run': False,
        'reused_cpu_checks': original['cpu_checks'], 'same_native_hard_limit_seconds': original['timeout_seconds'],
        'previous_score_completed_seconds': 125.05710124969482,
        'previous_gpu_nvidia_mib_observation': 70791,
        'previous_peak_boundary': 'nvidia process total includes reserved memory; exact allocator peaks were not persisted before the API failure.'}
    record['resource_estimate'].update(gpu_peak_gib_estimate=75, wall_seconds_estimate=[130, 300])
    record['resource_estimate']['basis'] += ' Repaired-run estimate incorporates first actual125s scoring and70791MiB nvidia observation; allocator peaks remain to be measured.'
    require(shutil.disk_usage(output.parent).free >= 2_000_000_000, 'Reserve 2GB for this separate probe output.')
    check(record)
    output.mkdir(parents=True)
    shutil.copyfile(prior_dir / 'token_batches.pt', output / 'token_batches.pt')
    token_digest = sha256(output / 'token_batches.pt')
    require(token_digest == original['input_sha256'][str(prior_dir / 'token_batches.pt')],
            'The repair must reuse byte-identical fixed token batches, without re-sampling or re-tokenizing.')
    record['input_sha256'][str(output / 'token_batches.pt')] = token_digest
    require(not torch.cuda.is_initialized(), 'Repair preparation unexpectedly initialized CUDA.')
    write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'path': str(output / 'launch.prepared.json'),
                      'sha256': sha256(output / 'launch.prepared.json'), 'token_batches_sha256': token_digest,
                      'source_pins': len(record['source_sha256']), 'input_pins': len(record['input_sha256']),
                      'new_cpu_formula_tests_run': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-prepared', type=Path)
    args = parser.parse_args()
    if args.reuse_prepared is None:
        prepare(args.output_dir)
    else:
        prepare_repair(args.output_dir, args.reuse_prepared)
