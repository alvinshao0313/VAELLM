"""Prepare32 only after the real C0 terminal model and audit exist."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time

from . import bindings as b


def data_contract(dataset, recovery):
    groups = dataset['groups']
    b.require(len(groups) == 32 and [row['group_index'] for row in groups] == b.GROUPS,
              'Keep every original group0..31 exactly once.')
    counts = Counter(key for row in groups for key in row['case_ids'])
    unique, presentations, prompt_tokens, max_prompt = Counter(), Counter(), Counter(), Counter()
    for case in dataset['training_cases']:
        domain, key = case['domain'], case['case_id']
        b.require(case['source']['official_split'] == 'train'
            and counts[key] == (1 if domain == 'mmlu' else 2), 'Only original train exposures may be used.')
        unique[domain] += 1
        presentations[domain] += counts[key]
        ids = case.get('prompt_input_ids', case['input_ids'])
        prompt_tokens[domain] += counts[key] * len(ids)
        max_prompt[domain] = max(max_prompt[domain], len(ids))
    b.require(dict(unique) == dict(mmlu=1024, qasper=32, hotpotqa=96)
        and sum('target_log_probs' in row for row in recovery.values()) == 1024
        and all('target_log_probs' not in row for row in recovery.values() if row['case']['domain'] != 'mmlu'),
        'Keep only once-mixed F-M targets; all QA targets are newly computed online.')
    return dict(unique=dict(unique), presentations=dict(presentations), prompt_tokens=dict(prompt_tokens),
        maximum_prompt_tokens=dict(max_prompt), group_indices=b.GROUPS,
        comparison='Equal document presentations and32 updates; online generated supervision is measured, not assumed equal.')


def prepare(checkpoint_dir, pipeline_report, output):
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized(), 'Prepare in CPU-only bitvae.')
    b.require(not output.exists(), 'Never overwrite a preparation or experiment.')
    b.configure()
    parent, parent_meta = b.source(checkpoint_dir, pipeline_report)
    proof = b.verified_implementation()
    started = time.monotonic()
    b.torch.set_num_threads(4)
    model, meta, strict, budget = b.checkpoint.load(parent['checkpoint_dir'])
    b.require(meta['checkpoint_id'] == parent['checkpoint_id'] and not strict.missing_keys
        and not strict.unexpected_keys and budget['budget_passed'], 'Actual C0 strict load failed.')
    modules = b.mature.selected_modules(model)
    active = b.lifecycle.decoder_parameters(modules)
    layout = b.optimizer.derive_layout(model, meta)
    b.require(layout == proof['record']['matrix_layout']
        and len(active) == proof['record']['trainable_tensors']
        and sum(p.numel() for p in active.values()) == proof['record']['trainable_elements'],
        'New geometry is outside the already verified full-model fixed2 Muon path.')
    identity = b.inputs.teacher_identity(model.state_dict(), list(modules))
    dataset, recovery, cache, evidence = b.inputs.load_inputs()
    exposure = data_contract(dataset, recovery)
    record = dict(source_checkpoint_id=meta['checkpoint_id'], groups=dataset['groups'], train_ids=dataset['train_ids'],
        decoder_parameters={name:b.core.tensor_record(value) for name,value in active.items()},
        trainable_tensors=len(active), trainable_elements=sum(p.numel() for p in active.values()),
        matrix_layout=layout, validated_frozen_state=b.lifecycle.frozen_proof(model, set(active)),
        teacher_identity=identity, model_config=dict(model={'hf_path':str(b.HF)}, protocol=identity['protocol']))
    del active, modules, model, recovery
    gc.collect()
    bound = b.code_bindings()
    for path in (Path(parent['checkpoint_dir'])/'adaptive_model_meta.json', Path(parent['pipeline_report']),
                 b.inputs.HOT/'dataset_prepared.json', b.inputs.HOT/'teacher_targets.json'):
        bound[str(path.resolve())] = b.binding(path)
    output.mkdir(parents=True)
    result = dict(schema='fresh_C0_Muon32_prepared_v1', status='verified_actual_C0_source_and_reused_GPU_contract',
        source_binding=parent, source_budget=budget, source_geometry=dict(native_modules=parent_meta['native_modules'],
            compact_modules=parent_meta['compact_modules']), record=record, contract=b.CONTRACT, group_indices=b.GROUPS,
        data=exposure, teacher_cache=cache, M_target_evidence=evidence,
        GPU_evidence={key:value for key,value in proof.items() if key != 'record'}, bindings=bound,
        CUDA_initialized=b.torch.cuda.is_initialized(), elapsed_seconds=time.monotonic()-started)
    b.require(not result['CUDA_initialized'], 'CPU preparation unexpectedly initialized CUDA.')
    b.core.write(output/'prepared.json', result)
    print(json.dumps(dict(status=result['status'], source_checkpoint_id=meta['checkpoint_id'], output=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--pipeline-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.checkpoint, args.pipeline_report, args.output.resolve())


if __name__ == '__main__':
    main()
