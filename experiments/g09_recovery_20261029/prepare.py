"""CPU-only G09 identity, exact teacher backbone, inputs, and existing GPU evidence."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import bindings as b


def main():
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not b.torch.cuda.is_initialized(), 'Prepare only on CPU in caller-activated bitvae.')
    b.require(not (b.OUTPUT / 'prepared.json').exists() and not (b.OUTPUT / 'formal32').exists(),
        'Do not overwrite a prepared/running direct OPD task.')
    b.torch.set_num_threads(4)
    started = time.monotonic()
    directory, meta, audit = b.source()
    original_prepare = b.module(b.OPD / 'prepare.py', '_g09_unchanged_OPD_material_checks')
    # Its pure materials helper also reads the two established representative groups.
    # It only needs these indices for CPU input checks, not a new GPU short run.
    original_steps = b.STEPS
    try:
        b.STEPS = {**original_steps, 'short2': [2, 14]}
        dataset, recovery, cache, evidence, rows, short, cpu = original_prepare.materials()
    finally:
        b.STEPS = original_steps
    state = b.load_state(directory, meta)
    names = list(b.hot.student_projection_names(meta['base_model_path'], 'all'))
    decoder, frozen = b.hot.state_inventory(state, names)
    from train_utils.checkpoint_bpw import audit_state_dict
    budget = audit_state_dict(meta, state, max_bpw=2.6, max_lora_rank=0)
    b.require(len(state) == 3423 and len(names) == 252 and len(decoder) == 1512
        and frozen['tensors'] == 1911 and budget['budget_passed']
        and all(budget[key] == audit['bit_budget'][key] for key in b.BUDGET_FIELDS),
        'G09 actual decoder inventory/frozen complement/native storage differs.')
    teacher = b.teacher_identity(state, meta, names, directory)
    b.require(teacher['nonprojection_tensors_compared'] == 147, 'Actual G09 full-F teacher backbone differs.')
    gpu_evidence = b.reused_gpu_evidence(meta)
    record = dict(source_checkpoint_id=b.SOURCE_ID, model_config=b.model_config(directory, meta),
        student_scope='all', student_projection_names=names, teacher_projection_names=names,
        decoder_parameters=decoder, trainable_elements=sum(item['elements'] for item in decoder.values()),
        validated_frozen_state=frozen, teacher_kind='full7', teacher_identity=teacher,
        groups=dataset['groups'], train_ids=dataset['train_ids'], retention_lambda=0., steps=32,
        mmlu_execution_batch_size=32)
    # Import the actual reused functions before pinning to catch import routing errors on CPU.
    import run
    import export
    b.require(Path(run.runtime.fit.__file__).resolve() == b.OPD / 'fit.py'
        and Path(run.runtime.rollout.__file__).resolve() == b.OPD / 'rollout.py'
        and run.runtime.core is b.core and run.runtime.fit.STEPS == b.STEPS,
        'Native OPD implementation did not bind to the new source module.')
    syntax = []
    for path in b.HERE.glob('*.py'):
        if path.name != 'evaluate_g02.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path)); syntax.append(str(path))
    subprocess.run(['bash', '-n', str(b.HERE / 'run.sh')], check=True)
    old = b.read(b.REFERENCE / 'prepared.json')
    files = [path for path in b.HERE.glob('*.py') if path.name != 'evaluate_g02.py'] + [b.HERE / 'run.sh']
    files += list(b.OPD.glob('*.py')) + [b.PROTECTION / 'bindings.py', b.REFERENCE / 'prepared.json']
    files += [Path(item['path']) for item in old['bindings'].values()]
    files += [Path(teacher['configuration_runtime_fields']['shared_HF_config']['path']),
        Path(b.ROOT / 'experiments/qwen3_w25/assemble_runtime_categories.py')]
    bound = {str(path.resolve()): b.binding(path) for path in files}
    record['dataset_binding'] = b.read(b.aligned.reference.HOT / 'dataset_prepared.json')['dataset_binding']
    result = dict(schema='G09_direct_online_OPD32_prepared_v1', status='CPU_prepared_reused_GPU_evidence',
        profile='G', source_checkpoint=str(directory), source_checkpoint_id=b.SOURCE_ID,
        source_audit=b.binding(b.SOURCE_AUDIT), source_weight_stat=b.file_stat(directory / meta['state_dict_file']),
        source_bit_budget=budget, source_geometry={item['name']: item for item in meta['converted_modules']},
        actual_target_contract=b.CONTRACT, decoder_parameters=decoder, record=record,
        dataset_binding=record['dataset_binding'], data_contract=dataset['evidence'], teacher_cache=cache,
        unchanged_M_target_evidence=evidence, QA_generation_bindings=rows,
        CPU_validation={**cpu, 'source_actual_storage_audited': True, 'fullF_nonprojection_tensors_exact': 147,
            'frozen_tensors': 1911, 'syntax_checked': syntax},
        GPU_evidence_reuse=gpu_evidence, phases=b.STEPS, bindings=bound,
        optimizer=dict(name='AdamW', seed=137, learning_rate=1e-6, betas=[.9,.95], eps=1e-8,
            weight_decay=0, clip_norm=1.5, schedule='1e-6*(1-(step-1)/500)', fresh=True),
        elapsed_seconds=time.monotonic()-started)
    b.require(not b.torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    b.OUTPUT.mkdir(parents=True, exist_ok=True)
    b.core.write(b.OUTPUT / 'prepared.json', result)
    print(json.dumps(dict(status=result['status'], source_checkpoint_id=b.SOURCE_ID,
        bit_budget=budget['user_linear_bpw'], CPU_validation=result['CPU_validation'],
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
