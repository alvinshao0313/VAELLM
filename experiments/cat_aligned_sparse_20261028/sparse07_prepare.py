"""CPU materials without a future manifest, or actual06-bound abs005 preparation."""
import argparse
import ast
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
import sparse07_bindings as b


def code_bindings():
    pending = [value for value in sys.modules.values() if isinstance(value, ModuleType)]
    visited, paths = set(), set(b.HERE.glob('*.py')) | set(b.HERE.glob('*.sh'))
    paths.update(b.NIGHT.glob('sparse*.py'))
    paths.update((b.c.RICE, b.PLAN_SOURCE, *b.RICE_MULTI.glob('*.py')))
    paths.add(b.SNAPSHOT / 'experiments/qwen3_w25/step_export_callback.py')
    while pending:
        value = pending.pop()
        path = getattr(value, '__file__', None)
        if id(value) in visited or not path or not Path(path).is_absolute() or not Path(path).resolve().is_relative_to(b.ROOT):
            continue
        visited.add(id(value))
        paths.add(Path(path).resolve())
        pending.extend(item for item in vars(value).values() if isinstance(item, ModuleType))
    api = b.helpers()
    paths.update(api.source_paths())
    return [b.binding(path) for path in sorted(paths)]


def reference_evidence():
    from experiments.longbench_v1.protocol import model_policy
    proofs = {}
    for profile in ('C', 'G'):
        root = b.REFERENCE / profile
        audit, gpu, evaluation = (b.read(root / name) for name in ('audit.json', 'gpu_validation.json', 'full9/evaluation.json'))
        prepared = b.read(root / 'prepared.json')
        build = b.read(root / 'build.json')
        config = b.read(root / 'full9/config.json')
        canonical_policy = model_policy(config['model']['hf_path'])
        b.require(canonical_policy == config['model_policy'], 'Saved real raw-profile config differs from canonical inference policy.')
        b.require(audit['status'] == gpu['status'] == 'verified' and evaluation['status'] == 'complete_full9'
            and audit['checkpoint_id'] == gpu['checkpoint_id'] == evaluation['checkpoint_id']
            and audit['source_checkpoint_id'] == gpu['source_checkpoint_id'] == build['source_checkpoint_id']
            and prepared['recipe'] == b.RECIPE and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 4287
            and audit['base_frozen_tensors_compared'] == 3423 and audit['rice_tensors_compared'] == 864 and audit['source_banks_compared'] == 504
            and audit['actual_native_save'] and audit['independent_fresh_strict'] and audit['bit_budget']['budget_passed']
            and gpu['actual_whole_model_forwards'] == 1 and gpu['actual_MLP_Flinear_hooks'] == gpu['actual_native_sparse_caches_compared'] == 108
            and gpu['all_protected_columns_exact'] and gpu['persistent_state_unchanged']
            and len(build['modules']) == 108 and all(row['nnz'] == 251659 and row['coordinate_and_BF16_patch_exact']
                and row['protected_columns_exact'] for row in build['modules'].values()), 'Completed raw profile evidence differs.')
        for item in prepared['bindings']:
            if Path(item['path']).is_relative_to(b.NIGHT):
                b.require(b.binding(item['path']) == item, 'Mature sparse source changed since actual raw-profile verification.')
        proofs[profile] = dict(reference_only_not_new06=True,
            bindings=[b.binding(root / name) for name in ('prepared.json', 'build.json', 'audit.json', 'gpu_validation.json',
                'full9/config.json', 'full9/evaluation.json', 'full9/nine_task_acceptance.json')], verified_base3423_Rice864_joint4287=True,
            source_banks=504, consumer_MLPs=108, new_source_reconstruction_still_required=True,
            saved_native_load_audit_has_model_policy='model_policy' in gpu['native_load_audit'],
            canonical_policy_matches_saved_real_config=True, canonical_policy=canonical_policy,
            new_wrapper_actual_model_policy_validation_pending=True)
    old_nine = b.RICE_MULTI.parent / 'evaluation_staged/longbench/merged/nine_task_acceptance.json'
    nine = b.read(old_nine)
    b.require(nine['quality_target_evaluated'] and nine['protocol']['longbench_tasks'] == 21
        and nine['protocol']['longbench_samples_per_model'] == 4750 and nine['protocol']['max_linear_bpw'] == 2.6,
        'Actual historical Rice multi-partition full9 evidence missing.')
    return dict(raw_profiles=proofs, actual_Rice_multi_partition_nine=b.binding(old_nine),
        old_0075_bundle_retired_not_a_candidate_source=True)


def materials():
    import sparse07_worker
    import sparse07_evaluate
    import predict07
    import summary07
    import sparse07_plan
    context = b.c.native_context()
    _, auxiliary = b.c.aux32()
    index_path = b.HF / 'model.safetensors.index.json'
    donors = b.read(index_path)['weight_map']
    old = b.read(b.PARENT.parent / 'selected_cpu_materials.json')
    geometry = {}
    for profile in ('C', 'G'):
        reference = old['source_parent_feasibility']['parents'][profile]
        directory = Path(reference['source_checkpoint'])
        b.require(b.binding(directory / 'checkpoint_meta.json') == reference['source_metadata'], 'Historical09 metadata changed.')
        meta = b.read(directory / 'checkpoint_meta.json')
        names = sorted(row['name'] for row in meta['converted_modules'] if '.mlp.' in row['name'])
        shapes = {row['name']: [row['out_features'], row['in_features']] for row in meta['converted_modules'] if row['name'] in names}
        b.require(len(names) == 108 and all(math.prod(shape) == 50331648 and math.ceil(math.prod(shape)*.005) == 251659
            and name+'.weight' in donors for name, shape in shapes.items()), 'Existing profile cannot cover the fixed108 recipe.')
        geometry[profile] = dict(kind='09_geometry_only_not_new06', metadata=reference['source_metadata'], shapes=shapes,
            expected_support_each=251659, expected_total_support=108*251659, actual_new_Rice_lengths_unknown=True)
    groups = sparse07_plan.task_groups(['2', '3'])
    original_plan = b.ROOT / 'experiments/cat_aligned_evaluation_20261028/evaluate.py'
    definitions = lambda path: {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)}
    old_definitions, current_definitions = definitions(original_plan), definitions(b.HERE / 'sparse07_plan.py')
    for name in ('gpu_list', 'task_groups', 'check_resources'):
        b.require(old_definitions[name] == current_definitions[name], 'Copied canonical planning function differs: ' + name)
    rejected = None
    try:
        b.source()
    except ValueError as error:
        b.require('not independently audited yet' in str(error), 'Unexpected actual06 prerequisite failure.')
        rejected = str(error)
    syntax = []
    for path in sorted(b.HERE.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        syntax.append(str(path))
    for path in sorted(b.HERE.glob('*.sh')):
        subprocess.run(['bash', '-n', str(path)], check=True)
        syntax.append(str(path))
    b.require(tuple(inspect.signature(b.builder.build).parameters) == ('prepared_path', 'artifact')
        and tuple(inspect.signature(b.builder.export).parameters) == ('prepared_path', 'artifact')
        and callable(b.bundle.load_cpu_bundle) and callable(b.runtime.load_model)
        and callable(b.evaluator.collect_results), 'Mature explicit-source interfaces changed.')
    return dict(schema='CG_aligned_sparse07_source_independent_materials_v1', status='materials_verified_student_unbound',
        actual06_source_bound=False, future_manifest_or_UUID_created=False, new_support_or_values_constructed=False,
        new_budget_measured=False, GPU_forward=0, CUDA_initialized=b.torch.cuda.is_initialized(), contract=b.CONTRACT,
        references=reference_evidence(), historical_profile_geometry=geometry, native_context=context,
        auxiliary32=auxiliary, HF_index=b.binding(index_path), unready_source_rejected=rejected,
        two_card_partition=groups, partition_expected_samples=[sum(sparse07_plan.EXPECTED_COUNTS[t] for t in group) for group in groups],
        canonical_group_resource_functions_AST_exact=True, original_plan_source=b.binding(original_plan),
        syntax_checked=syntax, bindings=code_bindings())


def prepare():
    directory, meta, audit, audit_path = b.source()
    b.require(not b.OUTPUT.exists() and not b.ARTIFACT.exists(), 'One fresh actual06 source preparation; never overwrite.')
    state = b.selected.load_state(directory, meta)
    from train_utils.checkpoint_bpw import audit_state_dict
    budget = audit_state_dict(meta, state, max_bpw=2.6, max_lora_rank=0)
    banks = [key for key, value in state.items() if '.vq_weight' in key and value.dtype == b.torch.uint8]
    b.require(len(state) == 3423 and len(banks) == 504 and budget['budget_passed'] and not budget['contract_violations']
        and not budget['unclassified_target_keys'] and all(budget[key] == audit['bit_budget'][key] for key in b.selected.BUDGET_FIELDS)
        and Path(meta['base_model_path']).resolve() == b.HF.resolve(), 'Actual06 identity/storage/HF differs from its verified parent audit.')
    names = sorted(row['name'] for row in meta['converted_modules'] if '.mlp.' in row['name'])
    b.require(len(names) == 108 and all(state[name+'.protected_input_indices'].numel() in (64, 256) for name in names),
        'Use actual complete108 input-protected MLPs.')
    index_path = b.HF / 'model.safetensors.index.json'
    donors = b.read(index_path)['weight_map']
    hf_donors = {name: dict(key=name+'.weight', shard=donors[name+'.weight']) for name in names}
    _, auxiliary = b.c.aux32()
    bindings = code_bindings()
    for path in (audit_path, directory / 'checkpoint_meta.json', directory / 'config.json', index_path,
            b.PARENT / '06_B32/prepared.json', b.PARENT / '06_B32/formal/report.json'):
        bindings.append(b.binding(path))
    record = dict(schema='CG_aligned06_abs005_Rice7_prepared_v1', status='CPU_prepared_GPU_not_started', profile=b.PROFILE,
        source_dir=str(directory), source_checkpoint_id=meta['checkpoint_id'], source_native_audit=b.binding(audit_path),
        artifact_root=str(b.ARTIFACT), source_bit_budget=budget, source_tensors=3423, recipe=b.RECIPE, contract=b.CONTRACT,
        names=names, HF_donors=hf_donors, auxiliary32=auxiliary, bindings=bindings,
        weight_stats=[b.c.stamp(directory / meta['state_dict_file']),
            *(b.c.stamp(b.HF / name) for name in sorted({row['shard'] for row in hf_donors.values()}))],
        native_context=b.c.native_context(), actual06_source_bound=True, new_support_or_values_constructed=False,
        pending_validation=['actual06 BF16 caches and HF-minus-native support/values', 'actual Rice lengths/full ledger',
            'new4287 native save/fresh strict', 'one independent real108-cache auxiliary32 consumer', 'full8 and two-part full21/nine'],
        training=False, external_LoRA=False, created_unix=time.time())
    b.write(b.OUTPUT / 'prepared.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materials-only', action='store_true')
    args = parser.parse_args()
    b.c.cpu_only()
    b.torch.set_num_threads(4)
    started = time.monotonic()
    if args.materials_only:
        path = b.PARENT.parent / 'sparse07_cpu_materials.json'
        b.require(not path.exists(), 'Never overwrite sparse07 material evidence.')
        result = materials()
        result['elapsed_seconds'] = time.monotonic()-started
        b.write(path, result)
    else:
        result = prepare()
        path = b.OUTPUT / 'prepared.json'
    b.require(not b.torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    print(json.dumps(dict(status=result['status'], path=str(path), elapsed_seconds=time.monotonic()-started,
        CUDA_initialized=False, actual06_source_bound=result.get('actual06_source_bound'))), flush=True)


if __name__ == '__main__':
    main()
