"""CPU-only preparation of the full2000 single-variable full-KL comparison."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.general_lm_full_kl_probe.prepare import CONTROL, R2, ROOT
from experiments.general_lm_full_kl_recovery.train import check_short_proof, configuration, training_arguments
from experiments.general_lm_recovery.train import SAVE_STEPS
from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from train_utils.runtime_snapshot import to_runtime_jsonable

SHORT = R2 / 'general_lm_full_kl_probe'


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and os.environ.get('ACCELERATE_GRADIENT_ACCUMULATION_STEPS', '1') == '1', 'CPU preparation only.')
    free = shutil.disk_usage(output.parent).free
    require(not output.exists() and free >= 30_000_000_000, 'Use a fresh output and reserve five complete exports.')
    short, control = read(SHORT / 'launch.json'), read(CONTROL / 'launch.json')
    check_inputs(short); check_inputs(control)
    require(short['status'] == 'verified' and short['exit_code'] == 0
            and all((SHORT / name).read_text().strip() == '0' for name in ('training.exit', 'run.exit', 'outer.exit')),
            'The real full-KL four-update resource/export diagnostic must finish first.')
    for key in ('source_checkpoint', 'source_checkpoint_id', 'source_weight_stat', 'architecture',
                'expected_inventory', 'hf_source', 'input_data_stat', 'data', 'environment'):
        require(short[key] == control[key], 'Full-KL short and current mass control differ: ' + key)
    require(read(short['control_normalized'])['canonical_config'] == control['cpu_configuration']['canonical_config'],
            'Immutable actual mass-control configuration differs from its prepared science.')
    record = {key: deepcopy(control[key]) for key in ('source_checkpoint', 'source_checkpoint_id', 'source_weight_stat',
        'source_bit_budget', 'architecture', 'expected_inventory', 'hf_source', 'input_data_stat', 'data', 'environment',
        'control_result_paths', 'control_mean_percent', 'protocol')}
    sources, inputs = deepcopy(short['source_sha256']), deepcopy(short['input_evidence_sha256'])
    for path in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    evidence = [SHORT / name for name in ('launch.json', 'training.json', 'verification.json', 'review.json',
        'step_0004/audit.json', 'step_0004/final_model/export_record.json')]
    evidence += [Path(read(SHORT / 'training.json')['training_result']['run_output_dir']) / 'normalized_e2e_runtime_args.json']
    for path in evidence:
        inputs[str(path)] = sha256(path)
    record.update(schema='general_lm_full_kl_2000_v1', status='prepared_unbound', kind='formal_s2000',
        gpu=None, export_gpu=None, queued=False, automatic_retry=False,
        cwd=str(ROOT), python_executable=sys.executable, output_dir=str(output),
        source_sha256=sources, input_evidence_sha256=inputs, control_normalized=short['control_normalized'],
        full_kl_short_proof={key:str(SHORT/name) for key,name in [('verification','verification.json'),
            ('review','review.json'),('training','training.json')]},
        validation_reuse={'full_kl_resource_and_independent_export':str(SHORT/'review.json'),
            'every400_full8_callback':'Unchanged DecoderStepCallback already used by the matching mass-control run.',
            'boundary':'Four B4x1024 updates validate the resource path; no long-term stability or quality inference.'},
        schedule={'max_steps':2000,'warmup_steps':20,'scheduler':'cosine','save_steps':400,
                  'actual_stop_step':2000,'full8_steps':SAVE_STEPS},
        resource_budget_seconds=64800, termination_grace_seconds=20,
        readiness={'cpu':'pending','gpu':'not_scheduled'},
        comparison_plan={'primary':'Same optimizer step versus the existing FineWeb decoder-only top100-mass trajectory.',
            'mass_control_dir':str(CONTROL),'steps':SAVE_STEPS,
            'candidate_summary_paths':[str(output/f'step_{step:04d}/full8_summary.json') for step in SAVE_STEPS],
            'mass_summary_paths':[str(CONTROL/f'step_{step:04d}/full8_summary.json') for step in SAVE_STEPS],
            'secondary':'Original f927 full8; existing callback reports this source comparison.',
            'live_control_paths_are_not_pinned':'Future same-step summaries are consumed only after their actual completion.',
            'quality_boundary':'No MMLU65/69 cutoff; full8 cannot establish the nine-item LongBench goal.'},
        estimate={'training_hours':[10,15],'outer_budget_hours':18,'disk_free_bytes':free,
            'short_training_peak_allocated_bytes':read(SHORT/'verification.json')['training_peak_allocated_bytes'],
            'short_training_peak_reserved_bytes':read(SHORT/'verification.json')['training_peak_reserved_bytes'],
            'timing_boundary':'Only four updates measured at approximately16s/step; full2000 and five full8 pauses remain unmeasured.'},
        experience={'reference':'docs/experience/lessons/kd_and_data.md',
            'difference':'Only loss_type kl_top_mass -> kl; full vocabulary constrains tail-internal probability errors.',
            'hypothesis_not_result':'Existing top100 tail mass is constrained; whether the omitted tail-conditional KL affects quality is untested.',
            'fresh_source':'Use f927 and seed0; never continue the four-update diagnostic.'})
    record['training_argv'] = training_arguments(record)
    record['argv'] = [sys.executable,'-u','-m','experiments.general_lm_full_kl_recovery.train',
                      '--launch_record',str(output/'launch.json')]
    cfg, hf, args, names, architecture = configuration(record)
    check_short_proof(record)
    record['cpu_configuration'] = {'canonical_config':to_runtime_jsonable(cfg), 'hf_args':to_runtime_jsonable(hf),
        'training_args':to_runtime_jsonable(args),'only_canonical_differences':['run_root_dir','loss.loss_type'],
        'all_projection_count':len(names),'model_type':architecture['model_type']}
    checkpoint = output / 'NATIVE_RUN/trainer_state/checkpoint-400'
    record['stage_templates'] = [{'name':name,'argv':argv,'allowed_exit_codes':allowed} for name,argv,allowed
        in stage_commands(checkpoint=checkpoint,source_dir=Path(record['source_checkpoint']),output_dir=output/'step_0400')]
    require([row['name'] for row in record['stage_templates']] == ['export','audit','mmlu','seven'], 'Unexpected full-eight stages.')
    check_inputs(record)
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['cuda_initialized'] = False; record['readiness']['cpu'] = 'passed'
    output.mkdir(parents=True); write_json(output/'launch.json',record)
    print({'launch':str(output/'launch.json'),'sha256':sha256(output/'launch.json'),
           'sources':len(sources),'inputs':len(inputs),'cuda_initialized':False},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=R2/'general_lm_full_kl_s2000')
    prepare(parser.parse_args().output_dir)
