"""Source-independent CPU materials, or real04/05-bound preparation after audited parents exist."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
import selected_bindings as b
from selected_state import terminal_steps


def implementation_bindings():
    pending = [value for value in sys.modules.values() if isinstance(value, ModuleType)]
    visited, paths = set(), set(b.HERE.glob('*.py')) | set(b.HERE.glob('*.sh'))
    while pending:
        value = pending.pop()
        path = getattr(value, '__file__', None)
        if id(value) in visited or not path or not Path(path).is_absolute() or not Path(path).resolve().is_relative_to(b.ROOT):
            continue
        visited.add(id(value))
        paths.add(Path(path).resolve())
        pending.extend(item for item in vars(value).values() if isinstance(item, ModuleType))
    paths.update(b.PROTECTION.glob('*.py'))
    paths.update(b.PROTECTION.glob('*.sh'))
    paths.update((b.R / 'qwen_c1_eta_retention128_20261020/pv_runtime.py',
        b.R / 'qwen_j32_selected_mlp_d32_20261021/mlpd_runtime.py',
        b.R / 'qwen_j32_selected_mlp_d32_20261021/mlpd_native.py',
        b.R / 'qwen_c1_selected_down_head128_20261017/head_runtime.py'))
    return {str(path): b.binding(path) for path in sorted(paths)}


def kernel_equivalence():
    result = {}
    for old, new in (('pv_capture.py', 'selected_capture.py'), ('pv_projection.py', 'selected_projection.py')):
        definitions = lambda path: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        b.require(definitions(b.PV / old) == definitions(b.HERE / new), 'Copied mathematical definitions differ: ' + new)
        result[new] = dict(original=b.binding(b.PV / old), definitions_AST_exact=True, only_module_binding_changed=True)
    return result


def metadata_groups(meta, selected):
    specs = meta['converted_modules']
    # Native prewarm groups decoder/shape signatures; each peer applies its own rotation afterward.
    fields = ('in_features', 'out_features', 'compressed_in_features', 'compressed_out_features', 'transpose',
        'parallel_parts', 'residual_stages', 'stage_codebook_dims', 'stage_decoders', 'protected_input_indices')
    groups = []
    for name in selected:
        item = next(spec for spec in specs if spec['name'] == name)
        category = name.rsplit('.', 1)[-1]
        peers = [spec['name'] for spec in specs if spec['name'].rsplit('.', 1)[-1] == category
            and all(spec[key] == item[key] for key in fields)]
        position = peers.index(name)
        first = int(name.split('.')[2])//8*8
        chunk = peers[position//8*8:position//8*8+8]
        b.require(chunk == [f'model.layers.{i}.mlp.{category}' for i in range(first, first+8)],
            'Existing metadata cannot cover the expected native signature group8.')
        groups.append(chunk)
    return groups


def parent_feasibility():
    original03 = b.read(b.PARENT.parent / 'protection_cpu_materials.json')
    b.require(original03['status'] == 'materials_verified_student_unbound' and not original03['student_source_bound'],
        'Only source-independent03 evidence may be reused here.')
    for item in original03['bindings'].values():
        if Path(item['path']).is_relative_to(b.PROTECTION):
            b.require(b.binding(item['path']) == item, 'Frozen03 implementation changed.')
    sources = {profile: Path(item['source_checkpoint'])
        for profile, item in original03['source_parent_feasibility']['parents'].items()}
    sources['old_C1'] = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_scale_qat_scope_merge_20261015/C1/final_model')
    parents = {}
    for label, directory in sources.items():
        meta = b.read(directory / 'checkpoint_meta.json')
        if label in ('C', 'G'):
            old = original03['source_parent_feasibility']['parents'][label]
            b.require(old['source_metadata'] == b.binding(directory / 'checkpoint_meta.json'), 'Changed09 metadata since verified03 feasibility.')
            b.mature.check_stat(old['source_weight_stat'])
        state = b.load_state(directory, meta)
        scopes = {stage: b.scope(state, meta, stage) for stage in b.STAGES}
        parents[label] = dict(source_kind='historical_geometry_only_not_new04_or05', source_checkpoint=str(directory),
            source_checkpoint_id=meta['checkpoint_id'], source_metadata=b.binding(directory / 'checkpoint_meta.json'),
            source_weight_stat=b.mature.file_stat(directory / meta['state_dict_file']), scopes=scopes,
            metadata_expected_group8={stage: metadata_groups(meta, scope['selected']) for stage, scope in scopes.items()})
        del state
    b.require(parents['C']['scopes']['J32']['geometry'] == parents['G']['scopes']['J32']['geometry']
        == parents['old_C1']['scopes']['J32']['geometry'], 'L18 projection geometry is not transferable through this fixed path.')
    return dict(status='CPU_historical_geometry_only', actual04_or05_source_bound=False,
        GPU_bridge_validated=False, parents=parents, original03_materials=b.binding(b.PARENT.parent / 'protection_cpu_materials.json'))


def cpu_checks(materials):
    # Replay/EOS and once-mixed targets use exactly the frozen03 implementation and its actual CPU evidence.
    groups = materials['groups']['q64']
    b.require([g['group_index'] for g in groups] == list(range(32)) and len({key for g in groups for key in g['case_ids']}) == 1280
        and all(len(g['case_ids']) == 40 and all(g['weights'][key] == (.5/32 if i < 32 else .5/8)
            for i, key in enumerate(g['case_ids'])) for g in groups), 'Original one-pass N32 group/domain weights changed.')
    numeric = b.proxy.second_step_numeric_check()
    actual_eta = b.proxy.AccumulatingAdam((2, 3), 'cpu', b.ETA)
    b.require(actual_eta.optimizer.param_groups[0]['lr'] == .001 and actual_eta.steps == 0
        and actual_eta.parameter.dtype == b.torch.float32, 'New U eta wiring changed.')
    accepted, rejected = [], []
    for phase, actual, planned in (('short2', 2, 2), ('formal', 32, 32)):
        terminal_steps(phase, actual, planned)
        accepted.append([phase, actual, planned])
    for values in (('formal', 2, 32), ('formal', 32, 128), ('formal', 128, 128), ('short2', 1, 2)):
        try:
            terminal_steps(*values)
        except ValueError:
            rejected.append(list(values))
        else:
            raise ValueError('False32/128 terminal was accepted.')
    unready = {}
    for stage in b.STAGES:
        try:
            b.source(stage)
        except ValueError as error:
            b.require('not independently audited yet' in str(error), 'Unexpected source prerequisite failure.')
            unready[stage] = str(error)
        else:
            unready[stage] = False
    syntax = []
    for path in sorted(b.HERE.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        syntax.append(str(path))
    for path in sorted(b.HERE.glob('*.sh')):
        subprocess.run(['bash', '-n', str(path)], check=True)
        syntax.append(str(path))
    return dict(N32_one_pass_exact=True, first_pass_exposure=materials['first_pass_exposure']['q64'],
        M_targets_mixed_once=3072, QA_TF_targets_used=False, old_R_probabilities_loaded=False,
        original_proxy_two_step_numeric=numeric, U_eta=.001, shared_gW_contributions=16,
        honest_endpoint_accepts=accepted, false_endpoint_rejected=rejected, unready_sources_rejected=unready,
        copied_kernel_equivalence=kernel_equivalence(), syntax_checked=syntax,
        CUDA_initialized=b.torch.cuda.is_initialized(), student_FW_BW=0, new_R_probabilities_generated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materials-only', action='store_true')
    parser.add_argument('--stage', choices=tuple(b.STAGES), default='J32')
    args = parser.parse_args()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not b.torch.cuda.is_initialized(), 'CPU preparation in caller-activated bitvae with CUDA hidden.')
    b.torch.set_num_threads(4)
    started = time.monotonic()
    student = None if args.materials_only else b.source(args.stage)
    if student is not None:
        b.require(not (b.stage_dir(args.stage) / 'prepared.json').exists(), 'Never overwrite actual selected preparation.')
    _, archive, materials = b.inputs.load_materials()
    import selected_run
    import selected_export
    import selected_cache
    b.require(all(Path(value.__file__).resolve().parent == b.HERE for value in (selected_run, selected_export, selected_cache)),
        'Wrong actual selected entry imported.')
    validation = cpu_checks(materials)
    common = dict(contract=b.CONTRACT, stages=b.STAGES, groups=materials['groups']['q64'], materials=materials,
        CPU_validation=validation, bindings=implementation_bindings())
    if args.materials_only:
        destination = b.PARENT.parent / 'selected_cpu_materials.json'
        b.require(not destination.exists(), 'Never overwrite selected material evidence.')
        result = dict(schema='CG_selected_source_independent_materials_v1', status='materials_verified_student_unbound',
            student_source_bound=False, new_R_cache_generated=False, GPU_bridge_validated=False,
            source_parent_feasibility=parent_feasibility(), **common)
    else:
        directory, meta, audit, audit_path = student
        state = b.load_state(directory, meta)
        scope = b.scope(state, meta, args.stage)
        names = list(b.mature.hot.student_projection_names(meta['base_model_path'], 'all'))
        teacher = b.teacher_identity(state, meta, names, directory)
        r_directory, r_meta, _, r_audit_path = b.source04()
        R_targets, R_record = b.retention.load_native(archive, materials, directory=r_directory, meta=r_meta,
            source_audit=b.binding(r_audit_path), cache_dir=b.R_CACHE, profile=b.PROFILE, schema=b.R_SCHEMA)
        r_state = state if args.stage == 'J32' else b.load_state(r_directory, r_meta)
        r_proof = dict(tensors=len(r_state), sha256=b.core.mapping_digest({key: b.core.tensor_record(v) for key, v in r_state.items()}))
        b.require(r_proof == R_record['source_initial_state'], 'R cache differs from complete actual04 initial state.')
        result = dict(schema='CG_aligned_selected_prepared_v1', status='CPU_prepared_GPU_unvalidated', profile=b.PROFILE, stage=args.stage,
            source_checkpoint=str(directory), source_checkpoint_id=meta['checkpoint_id'], source_audit=b.binding(audit_path),
            source_weight_stat=b.mature.file_stat(directory / meta['state_dict_file']), source_bit_budget=audit['bit_budget'],
            source_initial_state=dict(tensors=len(state), sha256=b.core.mapping_digest({key: b.core.tensor_record(v) for key, v in state.items()})),
            R_source_checkpoint=str(r_directory), R_source_checkpoint_id=r_meta['checkpoint_id'], R_source_audit=b.binding(r_audit_path),
            R_source_initial_state=r_proof, R_cache=b.binding(b.R_CACHE / 'cache.json'), scope=scope,
            model_config=b.model_config(directory, meta), teacher_identity=teacher, teacher_projection_names=names, **common)
        destination = b.stage_dir(args.stage) / 'prepared.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        del R_targets
    b.require(not b.torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    result['elapsed_seconds'] = time.monotonic()-started
    b.core.write(destination, result)
    print(json.dumps(dict(status=result['status'], path=str(destination), elapsed_seconds=time.monotonic()-started,
        CUDA_initialized=False, actual04_or05_source_bound=not args.materials_only)), flush=True)


if __name__ == '__main__':
    main()
