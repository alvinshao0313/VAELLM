"""Bind the actual FineWeb1600 decoder endpoint to the preserved D16 down donor."""
from __future__ import annotations

import argparse
import ast
import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.compare import score
from experiments.llama_fineweb_down_combo.prepare import DONOR, DONOR_ID, ROOT, R2, SOURCE_ID, file_stat
from experiments.llama_fineweb_down_combo.run import check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.assemble_categories import validate_sources
from tools.cat_eval import build_parser

BASE = R2 / 'general_lm_recovery_s2000/llama/step_1600'
BASE_ID = 'dbde405a-328e-4bd2-84d3-f18ad60e3049'
CONTROL = R2 / 'llama_fineweb800_down_d16_combo'
CONTROL_ID = 'e985bada-3960-4a5a-82af-4652ef94ca6c'


def prepare(output):
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and Path.cwd() == ROOT and not output.exists(), 'Use one new CPU-only preparation directory.')
    record = copy.deepcopy(read(CONTROL / 'launch.prepared.json'))
    parent = read(BASE.parent / 'launch.json')
    audit, review, terminal = (read(BASE / name) for name in ('audit.json', 'independent_review.json', 'step_export.json'))
    export = read(BASE / 'final_model/export_record.json')
    control_review, control_launch, control_union = (read(CONTROL / name) for name in
        ('review.json', 'launch.json', 'final_model/assembly_report.json'))
    donor_audit = read(DONOR / 'audit.json')
    _require(audit['status'] == review['status'] == 'verified' and terminal['status'] == 'evaluated'
             and export['status'] == 'exported' and all(x['checkpoint_id'] == BASE_ID for x in (audit, review, terminal, export))
             and audit['source_checkpoint_id'] == parent['source_checkpoint_id'] == SOURCE_ID
             and audit['selected_global_step'] == export['source_global_step'] == terminal['global_step'] == 1600
             and audit['selected_decoder_tensor_count'] == 1344 and audit['frozen_tensors_compared'] == 1635
             and audit['train_mode'] == 'decoder' and audit['rank_zero'] and audit['non_decoder_non_code_bytes_unchanged']
             and audit['frozen_bytes_unchanged'] and audit['changed_code_bits'] == 0
             and audit['fresh_strict_reload_state_exact'] and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 2979
             and terminal['audit_sha256'] == sha256(BASE / 'audit.json'), 'Require the completed same3577 FineWeb1600 endpoint.')
    _require(control_review['status'] == control_union['status'] == donor_audit['status'] == 'verified'
             and control_launch['status'] == 'completed' and control_launch['exit_code'] == 0
             and all(x['checkpoint_id'] == CONTROL_ID for x in (control_review, control_launch, control_union))
             and control_union['sources']['donor_checkpoint_id'] == donor_audit['checkpoint_id'] == DONOR_ID
             and control_union['sources']['categories'] == ['down_proj']
             and control_union['serialized_tensors_compared'] == control_union['reloaded_tensors_compared'] == 2979,
             'Require the measured800+D16 primary control and the same retained donor.')
    evidence = [BASE / name for name in ('audit.json', 'independent_review.json', 'step_export.json', 'full8_summary.json', 'final_model/export_record.json')]
    evidence += [CONTROL / name for name in ('launch.prepared.json', 'launch.json', 'review.json', 'full8_summary.json', 'final_model/assembly_report.json')]
    for directory, stages in ((BASE, ('export', 'audit', 'mmlu', 'seven')), (CONTROL, ('assembly', 'mmlu', 'seven', 'dispatcher', 'outer'))):
        for stage in stages:
            path = directory / (stage + '.exit')
            _require(path.read_text().strip() == '0', 'Incomplete actual source stage: ' + str(path)); evidence.append(path)
    bm, dm, names, architecture, _, runtime, differences = validate_sources(BASE / 'final_model', DONOR / 'final_model', ['down_proj'])
    _require(bm['checkpoint_id'] == BASE_ID and dm['checkpoint_id'] == DONOR_ID
             and architecture == record['architecture'] and architecture['model_type'] == 'llama'
             and names == record['selected_targets'] and len(names) == 32
             and differences == record['selected_spec_differences'], 'Require the same32 D16 target geometries and native model architecture.')
    bits = sum((donor_audit if name in names else audit)['bit_budget']['per_target'][name]['budget_bits']
               for name in architecture['projection_shapes'])
    _require(bits == record['expected_user_linear_bits'] == 17095180288
             and bits / architecture['original_linear_weights'] < 2.5, 'Actual source inventories must retain the D16 integer storage budget.')
    controls, raw_controls = {}, []
    for label, directory, checkpoint in (
            ('fineweb800_down_d16', CONTROL, CONTROL / 'final_model'), ('fineweb1600', BASE, BASE / 'final_model'),
            ('down_d16', DONOR, DONOR / 'final_model'),
            ('original3577', Path(parent['control_result_paths'][0]).parent, Path(parent['source_checkpoint']))):
        paths = [directory / (name + '.json') for name in ('mmlu', 'seven')]
        value, raw = score(paths, architecture, checkpoint); raw_controls.append(raw)
        controls[label] = {'checkpoint_dir': str(checkpoint), 'checkpoint_id': value['checkpoint_id'],
            'result_paths': list(map(str, paths)), 'mean_percent': value['mean_percent'], 'mmlu_correct': value['integer_counts']['mmlu']['correct']}
        evidence += paths + [checkpoint / name for name in ('checkpoint_meta.json', 'config.json')]
    for raw in raw_controls[1:]:
        for current, old in zip(raw, raw_controls[0], strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == old[key], 'Full8 control protocol differs: ' + key)
    old_base = record['stage_commands'][0]['argv'][record['stage_commands'][0]['argv'].index('--base') + 1]
    for row in record['stage_commands']:
        row['argv'] = [str(value).replace(str(CONTROL), str(output)).replace(old_base, str(BASE / 'final_model')) for value in row['argv']]
        row['argv'][0] = sys.executable
        if row['name'] != 'assembly':
            args = build_parser().parse_args(row['argv'][2:])
            _require(args.num_fewshot == 0 and str(args.lm_batch_size) == ('64' if row['name'] == 'mmlu' else '32'), 'Native parser changed.')
    sources = record['source_sha256']
    for path, digest in parent['source_sha256'].items():
        _require(path not in sources or sources[path] == digest, 'Source versions conflict: ' + path); sources[path] = digest
    path = Path(__file__).resolve(); ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    record['input_sha256'].update({str(p): sha256(p) for p in evidence})
    hf = Path(architecture['base_model_path'])
    record.update(schema='llama_fineweb1600_down_d16_combo_v1', status='prepared_unbound', gpu=None,
        output_dir=str(output), python_executable=sys.executable, base_checkpoint_id=BASE_ID, controls=controls,
        primary_control='fineweb800_down_d16', runtime_config_compatibility=runtime,
        protected_stats=[file_stat(p / 'pytorch_model.bin') for p in
            (BASE / 'final_model', DONOR / 'final_model', CONTROL / 'final_model', Path(parent['source_checkpoint']))]
            + [file_stat(p) for p in hf.glob('*.safetensors')],
        frozen_backbone_evidence='FineWeb1600 native audit freezes1635 nondecoder tensors versus3577; retained D16 donor preserves3577 backbone. Native assembler must verify the complete2979 union and fresh save/reload.',
        scientific_difference='Same retained32 D16 down as e985; only the other192 FineWeb decoder projections advance800 to1600 along the same2000-step run. Original3577 backbone/norm/head/codes remain frozen. No new fitting or additive-gain claim.',
        validation_reuse='Reuse the completed800+D16 native assembly2979/full8, retained D16 proof and actual1600 independent export/audit/full8. No GPU smoke or rotating trainer-state dependency.',
        resource_budget_seconds=1200, termination_grace_seconds=20,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_fineweb_down_combo.run', '--launch-record', str(output / 'launch.json')],
        cuda_initialized=torch.cuda.is_initialized())
    check(record); _require(not record['cuda_initialized'], 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print({'status': record['status'], 'sources': len(sources), 'inputs': len(record['input_sha256']),
           'launch_sha256': sha256(output / 'launch.json'), 'bits_projected': bits, 'controls': controls}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=R2 / 'llama_fineweb1600_down_d16_combo')
    prepare(parser.parse_args().output.resolve())
