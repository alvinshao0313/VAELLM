"""CPU preparation: retained D1200 plus the unchanged e767 sparse payload."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import read
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, source_pins, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.checkpoint_v6 import load_v6_meta

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'
BASE = RESULT / 'general_lm_llama_d16_combo1600_s2000/step_1200/final_model'
PATCH = RESULT / 'llama_gate_sparse075/all32/final_model'
E767 = RESULT / 'llama_fineweb1600_down_d16_combo/final_model'
HF = Path('/home/star/models/meta-llama/Llama-3.1-8B-Instruct')
BASE_ID = 'c74df41b-81d2-4176-a7a0-862b96b0ed6b'
PATCH_ID = '646d95f4-cafe-4eb6-8270-71c4324432d4'
E767_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
TARGETS = [f'model.layers.{i}.mlp.gate_proj' for i in range(32)]
SCALARS = ('format', 'index_bits', 'value_bits', 'block_rows', 'block_cols')
EXPECTED_BITS = 17436621312
DENOMINATOR = 6979321856


def check(record):
    _require(record['cwd'] == str(ROOT) and record['base_dir'] == str(BASE)
             and record['patch_dir'] == str(PATCH) and record['base_checkpoint_id'] == BASE_ID
             and record['patch_checkpoint_id'] == PATCH_ID and record['residual_source_checkpoint_id'] == E767_ID
             and record['targets'] == TARGETS and record['operation'] == 'copy_frozen_native_sparse_payload_only',
             'The fixed D1200 source or unchanged sparse donor differs.')
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)


def prepare(output):
    output = Path(output).resolve()
    _require(Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only preparation in the public repository.')
    base_meta, patch_meta = [load_v6_meta(str(p)) for p in (BASE, PATCH)]
    base_review = read(BASE.parent / 'independent_review.json')
    patch_review = read(PATCH.parent / 'independent_review.json')
    patch_launch = read(PATCH.parent / 'launch.json')
    base_audit, patch_audit = [read(p.parent / 'audit.json') for p in (BASE, PATCH)]
    _require(base_meta['checkpoint_id'] == base_review['checkpoint_id'] == BASE_ID
             and patch_meta['checkpoint_id'] == patch_review['checkpoint_id'] == PATCH_ID
             and base_review['status'] == 'verified' and patch_review['status'] == 'independent_cpu_verified'
             and base_review['source_checkpoint_id'] == patch_review['source_checkpoint_id'] == E767_ID
             and base_review['global_step'] == 1200
             and base_review['reused_native_audit']['frozen_bytes_unchanged']
             and base_review['reused_native_audit']['selected_decoder_tensor_count'] == 1344
             and patch_launch['status'] == 'completed' and patch_launch['exit_code'] == 0
             and patch_audit['original_frozen_tensors'] == 2979 and patch_audit['added_tensors'] == 192
             and patch_audit['bit_budget']['user_linear_bits'] == EXPECTED_BITS,
             'Require independently reviewed D1200 and the completed native sparse donor.')
    specs = [{item['name']: item for item in meta['converted_modules']} for meta in (base_meta, patch_meta)]
    _require(set(specs[0]) == set(specs[1]) and len(specs[0]) == 224, 'Projection inventories differ.')
    for name, original in specs[0].items():
        donor = specs[1][name]
        strip = lambda item: {k: v for k, v in item.items() if not k.startswith('sparse_residual_')}
        _require(strip(original) == strip(donor) and not original.get('sparse_residual_qvalues')
                 and not original.get('sparse_residual_values'), 'Non-sparse geometry differs: ' + name)
        _require(bool(donor.get('sparse_residual_qvalues')) == (name in TARGETS), 'Sparse donor scope differs.')
        if name in TARGETS:
            _require([donor['sparse_residual_' + k] for k in SCALARS]
                     == ['blocked_quantized', 8, 8, 256, 256], 'Native sparse scalar schema differs.')
    architecture = patch_launch['architecture']
    controls = {}
    for label, checkpoint in (('d1200', BASE), ('e767_sparse', PATCH), ('e767', E767)):
        controls[label], _ = score([checkpoint.parent / (name + '.json') for name in ('mmlu', 'seven')],
                                  architecture, checkpoint)
    _require(controls['d1200']['mean_percent'] == 62.34449959532272
             and controls['e767_sparse']['mean_percent'] == 62.18048855913933,
             'Existing complete controls differ.')
    paths = [p.parent / name for p in (BASE, PATCH) for name in ('independent_review.json', 'audit.json')]
    paths += [PATCH.parent / name for name in ('launch.json', 'build_report.json', 'outer.exit')]
    paths += [p / name for p in (BASE, PATCH, E767) for name in ('checkpoint_meta.json', 'config.json')]
    paths += [p.parent / (name + '.json') for p in (BASE, PATCH, E767) for name in ('mmlu', 'seven')]
    paths += [HF / name for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                                     'special_tokens_map.json') if (HF / name).exists()]
    sources = source_pins()
    for folder in ('llama_decoder_sparse_combo', 'llama_gate_sparse_residual'):
        sources.update({str(p): sha256(p) for p in (ROOT / 'experiments' / folder).glob('*.py')})
    extra = ('experiments/full_rht_gate_pair/common.py', 'experiments/full_rht_gate_pair/compare.py',
             'experiments/full_rht_gate_pair/audit.py', 'experiments/qwen3_w25/assemble_categories.py',
             'experiments/w25_eval/acceptance.py')
    sources.update({str(ROOT / p): sha256(ROOT / p) for p in extra})
    record = dict(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        cwd=str(ROOT), python_executable=sys.executable, output_dir=str(output),
        base_dir=str(BASE), base_checkpoint_id=BASE_ID, patch_dir=str(PATCH), patch_checkpoint_id=PATCH_ID,
        residual_source_checkpoint_id=E767_ID, targets=TARGETS,
        operation='copy_frozen_native_sparse_payload_only', architecture=architecture,
        base_budget=base_audit['bit_budget'], expected_budget=patch_audit['bit_budget'],
        source_sha256=sources, input_sha256={str(p): sha256(p) for p in paths},
        protected_stats=[file_stat(p / m['state_dict_file']) for p, m in ((BASE, base_meta), (PATCH, patch_meta))],
        environment=patch_launch['environment'], cuda_initialized=False,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_decoder_sparse_combo.run',
                       '--launch-record', str(output / 'launch.json')],
        timeout_seconds=900, termination_grace_seconds=20,
        expected_inventory={'base_frozen': 2979, 'added_frozen_sparse': 192, 'saved': 3171,
                            'trainable_parameters': 0, 'copied_payload_bytes': 42680128},
        controls=controls,
        residual_target=patch_launch['residual_target'], decode_policy=patch_launch['decode_policy'],
        comparison_boundary='D1200 + unchanged P_e767, where P_e767 was constructed from HF minus e767 native FP32 decode. No HF-minus-D1200 recomputation, decoder replacement, optimizer step or joint adaptation. Primary comparison D1200; two e767 controls describe interaction, not additive gains.',
        resource_estimate={'expected_chain_seconds': '450-550, estimated from all32 build85.67/verify20.61/full8 406.32 seconds and D1200 export/full8 473.54 seconds; no residual reconstruction here.',
            'hard_timeout_seconds': 900, 'termination_grace_seconds': 20,
            'persistent_checkpoint_bytes': (PATCH / patch_meta['state_dict_file']).stat().st_size,
            'gpu_peak_reference': 'Reuse native full8 BF16 group8; one gate local probe only. No teacher or optimizer.'},
        experience='Prior decoder gains and residual gains are not additive. Sparse patch alone improves e767 only0.060125pp and reaches2.498326bpw. Prefer this bounded existing-endpoint composition before any fresh200/400 recovery; no claim of long-context or nine-item improvement.')
    check(record); _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(paths),
           'stat_pins': len(record['protected_stats']), 'expected_bpw': EXPECTED_BITS / DENOMINATOR,
           'prepared_sha256': sha256(output / 'launch.prepared.json'), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
