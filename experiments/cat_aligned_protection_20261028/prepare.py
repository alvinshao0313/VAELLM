"""CPU source-independent materials or actual02/R-bound preparation, never a future UUID."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
import bindings as b
import data
from controller import effective_scales, scale_rows, signed_coordinates, q_rows
from rollout import replay_case


def cpu_checks(recovery, materials):
    tokenizer = b.core.build_tokenizer(b.teacher_recipe()['model']['hf_path'])
    protocol = b.teacher_recipe()['protocol']
    eos = protocol['eos_token_id']
    seen, replays, rows = set(), 0, []
    for group in materials['groups']['q64']:
        for key in group['case_ids'][32:]:
            case = recovery[key]['case']
            b.require(key not in seen and 'target_log_probs' not in recovery[key] and case['source']['official_split'] == 'train',
                'QA must occur once per pass and have no offline target in recovery.')
            seen.add(key)
            limit = b.LIMITS[case['domain']]
            options = b.qa.generation_options(tokenizer, case['domain'], len(case['prompt_input_ids']), limit, protocol=protocol)
            for tokens in ([eos[0]], [0]*(limit-1)+[eos[0]], [0]*limit):
                result = replay_case(case, dict(output_token_ids=tokens), eos)
                b.require(b.torch.equal(result['input_ids'], b.torch.cat((case['prompt_input_ids'], b.torch.tensor(tokens[:-1]))))
                    and result['target_ids'].tolist() == tokens
                    and result['answer_query_positions'].tolist() == list(range(len(case['prompt_input_ids'])-1,
                        len(case['prompt_input_ids'])+len(tokens)-1)), 'Complete prefix/EOS/causal positions changed.')
                replays += 1
            rows.append(dict(case_id=key, domain=case['domain'], prompt_tokens=len(case['prompt_input_ids']), generation_options=options))
    # CPU arithmetic boundary checks, not a claim of actual-source native/GPU execution.
    source = b.torch.tensor([1.003, .005], dtype=b.torch.bfloat16)
    delta = b.torch.tensor([.002, -.004], dtype=b.torch.float32, requires_grad=True)
    q = b.torch.tensor([[127, -128, 1], [3, -7, 19]], dtype=b.torch.int8).view(b.torch.uint8)
    expected_scale = (source.float()*delta.exp()).bfloat16()
    expected_rows = (q.view(b.torch.int8).float()*expected_scale.float().unsqueeze(1)).bfloat16()
    b.require(b.torch.equal(effective_scales(source, delta), expected_scale)
        and b.torch.equal(scale_rows(q, source, delta, dtype=b.torch.bfloat16), expected_rows), 'Scale must round before rows multiply.')
    master = b.torch.tensor([-129., -127.5, -126.5, -.5, .5, 1.5, 126.5, 127.5], requires_grad=True)
    hard = signed_coordinates(master)
    b.require(hard.tolist() == [-128., -128., -126., 0., 0., 2., 126., 127.], 'Clamp/ties-even INT8 mathematics changed.')
    hard.sum().backward()
    b.require(master.grad.tolist() == [0., 1., 1., 1., 1., 1., 1., 0.], 'Clamped STE derivative changed.')
    signed = b.torch.arange(-128, 128, dtype=b.torch.float32).reshape(256, 1)
    scales = b.torch.ones(256, dtype=b.torch.bfloat16)
    native = signed_coordinates(signed).to(b.torch.int8).view(b.torch.uint8)
    b.require(b.torch.equal(native.view(b.torch.int8).float(), signed)
        and b.torch.equal(q_rows(signed, scales, dtype=b.torch.bfloat16), signed.bfloat16()), 'Signed-byte native q serialization changed.')
    syntax = []
    for path in sorted(b.HERE.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path)); syntax.append(str(path))
    for path in sorted(b.HERE.glob('*.sh')):
        subprocess.run(['bash', '-n', str(path)], check=True); syntax.append(str(path))
    try:
        b.source()
    except ValueError as error:
        b.require('not independently audited yet' in str(error), 'Unexpected source binding rejection.')
        unready = True
    else:
        unready = False
    return dict(QA_documents=len(rows), EOS_limit_complete_prefix_replays=replays,
        QA_generation_options=rows, M_targets_mixed_once=3072, original_N32_slots_exact=True,
        R_original_inputs_only_896=True, old_R_probabilities_loaded=False, source_not_ready_rejected=unready,
        scale_round_before_multiply=True, q_clamp_ties_even_STE_and_uint8=True, syntax_checked=syntax,
        CPU_arithmetic_only=True, CUDA_initialized=b.torch.cuda.is_initialized(), GPU_forward_backward=0)


def parent_feasibility():
    """Read existing09 only to check geometry; never bind a future02 student."""
    root = b.R / 'qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027'
    parents, inventories = {}, {}
    for profile in ('C', 'G'):
        audit_path = root / profile / '09_joint1200/native_audit.json'
        audit = b.read(audit_path)
        directory = Path(audit['checkpoint_dir'])
        meta = b.read(directory / 'checkpoint_meta.json')
        b.require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id']
            and audit['fresh_strict_reload_state_exact'] and audit['bit_budget']['budget_passed'],
            'Existing09 feasibility source is not verified.')
        prepared01 = b.PARENT.parent / profile / '01_qk_fineweb2000/prepared.json'
        parent_binding = b.read(prepared01)
        b.require(parent_binding['source_checkpoint'] == str(directory)
            and parent_binding['source_checkpoint_id'] == meta['checkpoint_id']
            and parent_binding['input_evidence_sha256'][str(audit_path)] == b.sha256(audit_path)
            and parent_binding['input_evidence_sha256'][str(directory / 'checkpoint_meta.json')]
                == b.sha256(directory / 'checkpoint_meta.json'), 'Existing09 differs from actual01 preparation binding.')
        b.check_stat(parent_binding['source_weight_stat'])
        state = b.load_state(directory, meta)
        names = list(b.hot.student_projection_names(meta['base_model_path'], 'all'))
        arms = {}
        for arm, choice in b.ARMS.items():
            decoder, protection, frozen = b.inventory(state, names, choice['kind'])
            arms[arm] = dict(decoder_tensors=len(decoder), decoder_elements=sum(v['elements'] for v in decoder.values()),
                protection_tensors=len(protection), protection_elements=sum(v['elements'] for v in protection.values()),
                frozen=frozen, inventory_sha256=b.core.mapping_digest(dict(decoder=decoder, protection=protection)))
            inventories[profile, arm] = {**decoder, **protection}
        teacher = b.teacher_identity(state, meta, names, directory)
        b.require(teacher['nonprojection_tensors_compared'] == 147, 'Existing09 full-F backbone differs.')
        parents[profile] = dict(source_checkpoint=str(directory), source_checkpoint_id=meta['checkpoint_id'],
            source_kind='09_parent_geometry_only_not_actual02_endpoint', source_audit=b.binding(audit_path),
            source_01_preparation=b.binding(prepared01),
            source_metadata=b.binding(directory / 'checkpoint_meta.json'),
            source_weight_stat=b.file_stat(directory / meta['state_dict_file']), arms=arms,
            teacher_identity=teacher, geometry={item['name']: [d['in_dim'] for d in item['stage_decoders']]
                for item in meta['converted_modules']})
        del state
    coverage = {}
    for arm in b.ARMS:
        c, g = inventories['C', arm], inventories['G', arm]
        coverage[arm] = set(c) == set(g) and all(len(g[key]['shape']) == len(item['shape'])
            and all(a >= v for a, v in zip(g[key]['shape'], item['shape'], strict=True)) for key, item in c.items())
        b.require(coverage[arm], 'G09 geometry does not cover C09; actual02 must still be checked independently.')
    return dict(status='CPU_verified09_geometry_only', actual02_source_bound=False,
        GPU_bridge_validated=False, parents=parents, G09_shapes_cover_C09=coverage)


def implementation_bindings():
    # Follow only actual project module imports; never walk torch's installed module tree.
    pending = [b, data]
    pending += [value for value in sys.modules.values() if isinstance(value, ModuleType)]
    visited, paths = set(), set(b.HERE.glob('*.py')) | set(b.HERE.glob('*.sh'))
    while pending:
        value = pending.pop()
        path = getattr(value, '__file__', None)
        if id(value) in visited or not path or not Path(path).is_absolute() or not Path(path).resolve().is_relative_to(b.ROOT):
            continue
        visited.add(id(value)); paths.add(Path(path).resolve())
        pending.extend(item for item in vars(value).values() if isinstance(item, ModuleType))
    for path in (data.N32 / 'prepared.json', data.COVERAGE / 'training_prepared.json',
            data.COVERAGE / 'target_binding.json', data.SHARED / 'retention_prepared.json', data.SHARED / 'recovery_prepared.json'):
        paths.add(path)
    return {str(path): b.binding(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materials-only', action='store_true')
    args = parser.parse_args()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not b.torch.cuda.is_initialized(), 'CPU-only preparation in caller-activated bitvae.')
    b.torch.set_num_threads(4)
    started = time.monotonic()
    student = None if args.materials_only else b.source()
    if not args.materials_only:
        b.require(not (b.OUTPUT / 'prepared.json').exists(), 'Never replace actual protection preparation.')
    recovery, archive, materials = data.load_materials()
    # Import every new entry before pinning the complete loaded implementation chain.
    import run
    import export
    import assemble
    import cache
    entries = (run, export, assemble, cache)
    b.require(all(Path(module.__file__).resolve().parent == b.HERE for module in entries),
        'Imported another experiment entry instead of the actual protection entry.')
    validation = cpu_checks(recovery, materials)
    common = dict(materials=materials, CPU_validation=validation, contract=b.CONTRACT, arms=b.ARMS,
        bindings=implementation_bindings(), elapsed_seconds=time.monotonic()-started)
    if args.materials_only:
        destination = b.PARENT.parent / 'protection_cpu_materials.json'
        b.require(not destination.exists(), 'Never overwrite material evidence.')
        result = dict(schema='CG_protection_source_independent_materials_v1', status='materials_verified_student_unbound',
            student_source_bound=False, new_R_cache_generated=False, GPU_bridge_validated=False,
            source_parent_feasibility=parent_feasibility(), **common)
    else:
        directory, meta, audit = student
        state = b.load_state(directory, meta)
        names = list(b.hot.student_projection_names(meta['base_model_path'], 'all'))
        teacher = b.teacher_identity(state, meta, names, directory)
        b.require(teacher['nonprojection_tensors_compared'] == 147, 'Full-F restoration carrier backbone differs.')
        from retention_cache import load, CACHE
        source_targets, r_cache = load(archive, materials)
        # The cached complete native initial teacher must equal the actual source, not merely its UUID.
        source_proof = dict(tensors=len(state), sha256=b.core.mapping_digest({key: b.core.tensor_record(v) for key, v in state.items()}))
        b.require(r_cache['source_initial_state'] == source_proof, 'R cache does not bind complete actual02 initial source state.')
        records = {}
        config = b.model_config(directory, meta)
        for arm, choice in b.ARMS.items():
            decoder, protection, frozen = b.inventory(state, names, choice['kind'])
            records[arm] = dict(source_checkpoint_id=meta['checkpoint_id'], model_config=config, student_scope='all',
                student_projection_names=names, decoder_parameters=decoder, trainable_elements=sum(v['elements'] for v in decoder.values()),
                protection_parameters=protection, protection_elements=sum(v['elements'] for v in protection.values()),
                protection_kind=choice['kind'], validated_frozen_state=frozen, teacher_kind='full7',
                teacher_projection_names=names, teacher_identity=teacher, retention_lambda=.5, mmlu_execution_batch_size=32)
        result = dict(schema='CG_aligned_protection_prepared_v1', status='CPU_prepared_GPU_unvalidated', profile=b.PROFILE,
            source_checkpoint=str(directory), source_checkpoint_id=meta['checkpoint_id'], source_audit=b.binding(b.SOURCE_AUDIT),
            source_weight_stat=b.file_stat(directory / meta['state_dict_file']), source_initial_state=source_proof,
            source_geometry={item['name']: item for item in meta['converted_modules']}, source_bit_budget=audit['bit_budget'],
            model_config=config, teacher_identity=teacher, R_cache=b.binding(CACHE / 'cache.json'), records=records, **common)
        destination = b.OUTPUT / 'prepared.json'
        b.OUTPUT.mkdir(parents=True, exist_ok=True)
        del source_targets
    b.require(not b.torch.cuda.is_initialized(), 'Preparation initialized CUDA.')
    result['elapsed_seconds'] = time.monotonic()-started
    b.core.write(destination, result)
    print(json.dumps(dict(status=result['status'], path=str(destination), profile=b.PROFILE,
        QA_documents=validation['QA_documents'], replays=validation['EOS_limit_complete_prefix_replays'],
        GPU_bridge_validated=False, elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
