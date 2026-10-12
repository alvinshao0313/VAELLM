"""CPU-only preparation of an unchanged native CAT prefix and read-only hooks."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import ROOT, ORIGINAL, RESULT, check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.llama_bsq_observation.observe import require
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload


def prepare():
    require(Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Use CPU-only S cwd/PYTHONPATH.')
    control = RESULT / 'full_rht_gate_pair/llama_block32'
    template = read(control / 'launch.json')
    require(template['status'] == 'completed' and template['exit_code'] == 0, 'Require the completed native control.')
    output = RESULT / 'llama_bsq_observation/native_prefix128'
    require(not output.exists(), 'Use a fresh diagnostic directory.')
    argv = list(template['training_argv'][2:])
    argv[argv.index('--output_dir') + 1] = str(output / 'native')
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'native')
    require(normalized == expected and resolved['gate_proj'].steps == 10000
            and cat.batch_size == 8192 and cat.linear_group_size == 32
            and vae.lfq_weight == 2.5 and vae.commitment_loss_weight == .25
            and vae.entropy_loss_weight == .01 and vae.inv_temperature == 100., 'Only output may differ from native control.')
    sources = dict(template['source_sha256'])
    for root in (ROOT, ORIGINAL):
        for path in (root / 'experiments/llama_bsq_observation').glob('*.py'):
            sources[str(path)] = sha256(path)
    record = {key: copy.deepcopy(template[key]) for key in ('cwd', 'environment', 'snapshot_import_paths')}
    record.update(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        output_dir=str(output), python_executable=sys.executable, training_argv=[sys.executable, 'tools/cat_train.py', *argv],
        normalized_expected=normalized, source_sha256=sources,
        input_sha256={str(control / name): sha256(control / name) for name in ('launch.json', 'audit.json', 'full8_summary.json')},
        protected_stats=[item for item in template['protected_stats'] if 'remaining_prefix_block32_s500' not in item['path']],
        timeout_seconds=1200, termination_grace_seconds=20,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_bsq_observation.train',
                       '--launch-record', str(output / 'launch.json')],
        observed_steps=[1, 16, 64, 128], native_optimizer_updates=128,
        scope='Unchanged current native CAT first-stage prefix only; no candidate BSQ setting, no saved weights or evaluation.',
        resource_estimate={'disk_bytes_reserved': 2 * 1024**3, 'maximum_seconds': 1200,
            'boundary': 'Full original HF and Alpaca512x256 calibration retained. Four observation steps add native BSQ replay and five gradient reads; peak GPU memory must be measured.'},
        cuda_initialized=False)
    check(record)
    output.mkdir(parents=True)
    write_json(output / 'launch.prepared.json', record)
    write_json(output / 'launch.json', record)
    print({'status': record['status'], 'sha256': sha256(output / 'launch.prepared.json'),
           'source_count': len(sources), 'input_count': len(record['input_sha256']), 'stat_count': len(record['protected_stats'])}, flush=True)


if __name__ == '__main__':
    prepare()
