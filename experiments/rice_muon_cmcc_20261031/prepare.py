"""Derive actual1728 D leaves/1008 matrices and every Rice frozen field from real C1."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

from . import bindings as b
from experiments.fresh_muon_cmcc_20261030.prepare import data_contract


def prepare(checkpoint_dir, pipeline_report, output):
    output = Path(output).resolve()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized(), 'Prepare in CPU-only bitvae.')
    b.require(not output.exists(), 'Never overwrite a Rice Muon preparation or experiment.')
    b.configure()
    parent, parent_meta = b.source(checkpoint_dir, pipeline_report)
    proof = b.verified_implementation()
    started = time.monotonic()
    b.torch.set_num_threads(4)
    model, meta, strict, budget = b.checkpoint.load(parent['checkpoint_dir'])
    b.require(meta == parent_meta and not strict.missing_keys and not strict.unexpected_keys
        and budget == parent['bit_budget'] and budget['budget_passed'], 'Actual C1 strict source load or budget identity failed.')
    modules = b.mature.selected_modules(model)
    active = b.lifecycle.decoder_parameters(modules)
    layout = b.optimizer.derive_layout(model, meta)
    frozen, rice = b.lifecycle.frozen_proof(model, set(active)), b.rice_proof(model)
    b.require(layout == proof['record']['matrix_layout'] and len(active) == proof['record']['trainable_tensors'] == 1728
        and sum(p.numel() for p in active.values()) == proof['record']['trainable_elements']
        and sum(row['stages'] for row in layout.values()) == 1008
        and not any('.sparse_' in name for name in active),
        'Derive the real all-D layout and match only the already verified decoder geometry; Rice fields must remain frozen.')
    identity = b.inputs.teacher_identity(model.state_dict(), list(modules))
    dataset, recovery, cache, evidence = b.inputs.load_inputs()
    exposure = data_contract(dataset, recovery)
    record = dict(source_checkpoint_id=meta['checkpoint_id'], groups=dataset['groups'], train_ids=dataset['train_ids'],
        decoder_parameters={name:b.core.tensor_record(value) for name,value in active.items()},
        trainable_tensors=len(active), trainable_elements=sum(p.numel() for p in active.values()),
        matrix_layout=layout, actual_independent_Muon_matrices=sum(row['stages'] for row in layout.values()),
        validated_frozen_state=frozen, validated_Rice_state=rice, teacher_identity=identity,
        model_config=dict(model={'hf_path':str(b.HF)}, protocol=identity['protocol']))
    del active, modules, model, recovery
    gc.collect()
    bound = b.code_bindings()
    for path in (Path(parent['checkpoint_dir']) / 'adaptive_model_meta.json', Path(parent['pipeline_report']),
        b.inputs.HOT / 'dataset_prepared.json', b.inputs.HOT / 'teacher_targets.json'):
        bound[str(path.resolve())] = b.binding(path)
    output.mkdir(parents=True)
    result = dict(schema='fresh_C1_Rice_Muon_prepared_v1', status='verified_actual_fresh_Rice_source_Muon_CPU_prepared',
        source_binding=parent, source_budget=budget, source_geometry=dict(native_modules=parent_meta['native_modules'],
            compact_modules=parent_meta['compact_modules']), record=record, contract=b.CONTRACT, group_indices=b.GROUPS,
        data=exposure, teacher_cache=cache, M_target_evidence=evidence,
        reused_fixed2_GPU_evidence={key:value for key,value in proof.items() if key != 'record'}, bindings=bound,
        CUDA_initialized=b.torch.cuda.is_initialized(), elapsed_seconds=time.monotonic() - started,
        boundary='Actual C1 CPU source/layout/frozen preparation. The new Rice Muon GPU lifecycle2 remains required before original-source32.')
    b.require(not result['CUDA_initialized'], 'CPU preparation unexpectedly initialized CUDA.')
    b.core.write(output / 'prepared.json', result)
    print(json.dumps(dict(status=result['status'], source_checkpoint_id=meta['checkpoint_id'], output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--pipeline-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.checkpoint, args.pipeline_report, args.output)
