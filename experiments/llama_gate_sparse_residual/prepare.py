"""CPU preparation for native sparse residuals on the retained complete e767."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import read
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import (check_hashes, check_stat, file_stat,
    sha256, source_pins, write_json)
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_train_residual_protection import RESIDUAL_SPARSE_RANK_METRICS
from train_utils.checkpoint_v6 import load_v6_meta

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'
BASE = RESULT / 'llama_fineweb1600_down_d16_combo/final_model'
HF = Path('/home/star/models/meta-llama/Llama-3.1-8B-Instruct')
BASE_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
BASE_BITS = 17095180288
DENOMINATOR = 6979321856
PROBE = 'model.layers.16.mlp.gate_proj'
RECIPE = {'rank_metric': 'sparse_residual_abs', 'top_p': .0075, 'min_abs': 0.,
          'codec': 'blocked_quantized', 'index_bits': 8, 'value_bits': 8,
          'block_shape': [256, 256]}
DECODE_POLICY = 'Temporary single-VAELinear copy with exact BF16 decoder values promoted to FP32, then native _decode_base_reconstruction(dtype=FP32). Original complete model parameters are never cast or replaced; deployed cache/local forward remains BF16.'


def check(record):
    _require(Path(record['cwd']) == ROOT and record['base_dir'] == str(BASE)
             and record['base_checkpoint_id'] == BASE_ID and record['recipe'] == RECIPE
             and record['decode_policy'] == DECODE_POLICY,
             'The fixed native sparse recipe or full e767 source changed.')
    expected = [PROBE] if record['scope'] == 'l16' else [f'model.layers.{i}.mlp.gate_proj' for i in range(32)]
    _require(record['scope'] in ('l16', 'all32') and record['targets'] == expected,
             'Use the explicit L16 validation or all32 formal scope.')
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)


def prepare(scope, output):
    output = Path(output).resolve()
    _require(Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only preparation in the public repository.')
    previous = read(BASE.parent / 'launch.json'); review = read(BASE.parent / 'review.json')
    meta = load_v6_meta(str(BASE)); assembly = read(BASE / 'assembly_report.json')
    budget = assembly['bit_budget']
    _require(previous['status'] == 'completed' and previous['exit_code'] == 0
             and review['status'] == 'verified' and meta['checkpoint_id'] == BASE_ID
             and budget['user_linear_bits'] == BASE_BITS and budget['original_linear_weights'] == DENOMINATOR
             and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared'] == 2979
             and len(meta['compressed_targets']) == 224 and budget['budget_passed'],
             'Require the complete independently reviewed e767 baseline.')
    specs = {item['name']: item for item in meta['converted_modules']}
    targets = [PROBE] if scope == 'l16' else [f'model.layers.{i}.mlp.gate_proj' for i in range(32)]
    _require(RECIPE['rank_metric'] in RESIDUAL_SPARSE_RANK_METRICS, 'Illegal native rank metric.')
    for name in targets:
        item = specs[name]
        _require(item['transpose'] is False and (item['out_features'], item['in_features']) == (14336, 4096)
                 and item['protected_input_indices']['shape'] == [64]
                 and item['stage_codebook_dims'] == [32, 32]
                 and item['weight_rotation']['type'] == 'two_sided_hadamard'
                 and not any(item.get(key) for key in ('sparse_residual_active_block_ids',
                    'sparse_residual_values', 'protected_residual_indices', 'low_rank_a', 'low_rank_b')),
                 'Expected original rotated input64 gate geometry without another residual: ' + name)
    paths = [BASE.parent / name for name in ('launch.json', 'review.json', 'full8_summary.json', 'mmlu.json', 'seven.json')]
    paths += [BASE / name for name in ('checkpoint_meta.json', 'assembly_report.json', 'config.json')]
    paths += [HF / name for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json',
                                      'tokenizer_config.json', 'special_tokens_map.json') if (HF / name).exists()]
    validation = None
    if scope == 'all32':
        validation = RESULT / 'llama_gate_sparse075/validate_l16_v2'
        completed = read(validation / 'launch.json'); audited = read(validation / 'audit.json')
        _require(completed['status'] == 'completed' and completed['exit_code'] == 0
                 and audited['status'] == 'verified' and audited['scope'] == 'l16'
                 and audited['original_frozen_tensors'] == 2979
                 and audited['fresh_native_bf16_local_forward']['matches_saved_probe']
                 and int((validation / 'outer.exit').read_text()) == 0,
                 'Formal preparation requires successful real L16 save/fresh/BF16 validation.')
        paths += [validation / name for name in ('launch.json', 'audit.json', 'build_report.json', 'outer.exit')]
    control, _ = score([BASE.parent / (name + '.json') for name in ('mmlu', 'seven')],
                       previous['architecture'], BASE)
    _require(control['checkpoint_id'] == BASE_ID and control['mean_percent'] == 62.12036318745571,
             'Baseline full eight-task results changed.')
    sources = source_pins()
    extra = ['experiments/full_rht_gate_pair/common.py', 'experiments/full_rht_gate_pair/compare.py',
             'experiments/full_rht_gate_pair/audit.py', 'experiments/qwen3_w25/assemble_categories.py',
             'experiments/w25_eval/acceptance.py']
    sources.update({str(ROOT / p): sha256(ROOT / p) for p in extra})
    sources.update({str(p): sha256(p) for p in (ROOT / 'experiments/llama_gate_sparse_residual').glob('*.py')})
    index = read(HF / 'model.safetensors.index.json')['weight_map']
    protected = [file_stat(BASE / meta['state_dict_file'])]
    protected += [file_stat(HF / name) for name in sorted(set(index.values()))]
    nnz = math.ceil(14336 * 4096 * RECIPE['top_p']); max_blocks = 56 * 16
    max_bytes = len(targets) * (3 * nnz + 14 * max_blocks + 4)
    record = dict(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        cwd=str(ROOT), python_executable=sys.executable, output_dir=str(output), scope=scope,
        base_dir=str(BASE), base_checkpoint_id=BASE_ID, hf_reference=str(HF), targets=targets,
        recipe=RECIPE, decode_policy=DECODE_POLICY, architecture=previous['architecture'], base_budget=budget,
        source_sha256=sources, input_sha256={str(p): sha256(p) for p in paths},
        protected_stats=protected, environment=previous['environment'], cuda_initialized=False,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_gate_sparse_residual.run',
                       '--launch-record', str(output / 'launch.json')],
        validation_source=str(validation) if validation else None,
        timeout_seconds=1200 if scope == 'l16' else None, termination_grace_seconds=20,
        budget_estimate={'maximum_nnz_per_gate': nnz, 'maximum_active_blocks_per_gate': max_blocks,
            'maximum_payload_bytes': max_bytes, 'maximum_user_linear_bits': BASE_BITS + 8 * max_bytes,
            'maximum_user_linear_bpw': (BASE_BITS + 8 * max_bytes) / DENOMINATOR,
            'codec_layout': 'Two uint8 local coordinates + one uint8 value per entry; uint16 block IDs, int32 pointers, FP32 scale and zero point. Per gate bytes=3N+14A+4.'},
        residual_target='HF BF16 tensor viewed in FP32 minus native FP32 base decode using a temporary FP32 decoder copy from exact BF16 values. Original2979 tensors remain frozen; deployed cache/local forward remains BF16.',
        comparison_boundary='Adds sparse corrections to complete e767 without training, dense replacement or code/decoder/protection/rotation changes. Absolute residual ranking is not historical input_act_weighted_abs. Reverting selected coefficients toward original HF may cancel useful category or FineWeb recovery; local weight/output error is not downstream quality.',
        validation_boundary='L16 constructs, saves and fresh-strict reloads the complete model, with native BF16 local forward before/after/save-reload. No hybrid/dense-restored quality evaluation. Only all32 runs complete MMLU64/seven32.',
        resource_estimate={'l16_hard_timeout_seconds': 1200, 'termination_grace_seconds': 20,
            'baseline_full8_seconds': sum(previous['stages'][key]['finished_unix'] - previous['stages'][key]['started_unix'] for key in ('mmlu', 'seven')),
            'all32_runtime': 'Unmeasured CPU selection over32x58.7M entries; estimate after actual L16 timing. Not queued or promised this GPU window.',
            'persistent_model_extra_bytes_upper_bound': max_bytes},
        experience='Reuse native blocked residual schema and FP32 reconstruction on a disposable decoder copy; preserve complete e767 and all2979 tensors. L16v1 failed before payload construction because native FP32 fused decode rejects BF16 linear_in weights; the standalone native residual entry only moves active modules to a device and has the same missing dtype adaptation. No sparse quality result exists; old1% index8/value8 exceeds2.5bpw when added to e767.')
    check(record); _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(paths),
           'maximum_bpw': record['budget_estimate']['maximum_user_linear_bpw'],
           'prepared_sha256': sha256(output / 'launch.prepared.json'), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('l16', 'all32'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); prepare(args.scope, args.output_dir)
