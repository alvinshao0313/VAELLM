"""Prepare H96/Muon32 only from a complete and independently verified BR36."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

from . import recovery_bindings as b
from experiments.fresh_muon_cmcc_20261030.prepare import data_contract


def prepare(checkpoint_dir, br_report, output):
    output = Path(output).resolve()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized(), 'Prepare in CPU-only bitvae.')
    b.require(not output.exists(), 'Never overwrite a BR->E preparation or execution.')
    b.configure()
    parent, parent_meta = b.source(checkpoint_dir, br_report)
    proof = b.verified_implementation()
    started = time.monotonic()
    b.torch.set_num_threads(4)
    model, meta, strict, budget = b.checkpoint.load(parent['checkpoint_dir'])
    b.require(meta == parent_meta and not strict.missing_keys and not strict.unexpected_keys
        and budget == parent['bit_budget'] and budget['budget_passed'], 'Actual BR strict load or storage identity failed.')
    modules = b.mature.selected_modules(model)
    active = b.lifecycle.decoder_parameters(modules)
    layout = b.optimizer.derive_layout(model, meta)
    frozen = b.lifecycle.frozen_proof(model, set(active))
    b.require(layout == proof['record']['matrix_layout']
        and len(active) == proof['record']['trainable_tensors']
        and sum(p.numel() for p in active.values()) == proof['record']['trainable_elements']
        and sum(row['stages'] for row in layout.values())
            == sum(row['stages'] for row in proof['record']['matrix_layout'].values())
        and frozen['tensors'] == proof['record']['validated_frozen_state']['tensors'],
        'Actual BR geometry/frozen inventory must match the verified mixed Muon path; no assumed old leaf count.')
    identity = b.inputs.teacher_identity(model.state_dict(), list(modules))
    b.require(identity['nonprojection_tensors_compared'] == 147,
        'The actual BR nonprojection backbone must remain identical to the independently loaded HF teacher.')
    dataset, recovery, cache, evidence = b.inputs.load_inputs()
    exposure = data_contract(dataset, recovery)
    record = dict(source_checkpoint_id=meta['checkpoint_id'], groups=dataset['groups'], train_ids=dataset['train_ids'],
        decoder_parameters={name:b.core.tensor_record(value) for name,value in active.items()},
        trainable_tensors=len(active), trainable_elements=sum(p.numel() for p in active.values()),
        matrix_layout=layout, validated_frozen_state=frozen, teacher_identity=identity,
        model_config=dict(model={'hf_path':str(b.HF)}, protocol=identity['protocol']))
    del active, modules, model, recovery
    gc.collect()
    bound = b.code_bindings()
    for path in (Path(parent['checkpoint_dir'])/'adaptive_model_meta.json', Path(parent['pipeline_report']),
                 b.inputs.HOT/'dataset_prepared.json', b.inputs.HOT/'teacher_targets.json'):
        bound[str(path.resolve())] = b.binding(path)
    output.mkdir(parents=True)
    result = dict(schema='BR_E32_prepared_v1', status='verified_actual_BR36_source_and_reused_Muon_GPU_contract',
        source_binding=parent, source_C0_binding=parent['source_C0_binding'], BR_record=parent['BR_record'],
        source_budget=budget, source_geometry=dict(native_modules=parent_meta['native_modules'],
            compact_modules=parent_meta['compact_modules']), record=record, contract=b.CONTRACT, group_indices=b.GROUPS,
        data=exposure, teacher_cache=cache, M_target_evidence=evidence,
        GPU_evidence={key:value for key,value in proof.items() if key != 'record'}, bindings=bound,
        CUDA_initialized=b.torch.cuda.is_initialized(), elapsed_seconds=time.monotonic()-started)
    b.require(not result['CUDA_initialized'], 'CPU BR->E preparation unexpectedly initialized CUDA.')
    b.core.write(output/'prepared.json', result)
    print(json.dumps(dict(status=result['status'], source_BR=meta['checkpoint_id'], output=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--br-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.checkpoint, args.br_report, args.output)


if __name__ == '__main__':
    main()
