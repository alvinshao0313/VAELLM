"""Check real pending-C1 identity, local readers, H96 data and inherited math evidence."""
import argparse
import ast
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b, evaluate, fresh, prepare, run, sequence


def execute(output):
    output = Path(output).resolve()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized()
        and not output.exists(), 'Use CPU-only bitvae and a new evidence directory.')
    output.mkdir(parents=True)
    started = time.monotonic()
    record = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv])
    try:
        b.configure()
        parent = sequence.current_parent()
        proof = b.verified_implementation()
        b.require(proof['record']['trainable_tensors'] == 1728
            and sum(row['stages'] for row in proof['record']['matrix_layout'].values()) == 1008
            and b.mature.STEPS == b.PHASES == dict(short2=[0, 1], formal32=list(range(32)))
            and b.mature.LIMITS == dict(qasper=128, hotpotqa=32), 'Do not change inherited math, full group schedule or rollout limits.')
        dataset, recovery, cache, evidence = b.inputs.load_inputs()
        exposure = prepare.data_contract(dataset, recovery)
        b.require(dataset['groups'] == proof['record']['groups'] and dataset['train_ids'] == proof['record']['train_ids'],
            'Use exactly the same actual H96 cases/group order as the verified mixed Muon path.')
        current_meta_path = b.SOURCE / '04_gate_proj/round_base/adaptive_model_meta.json'
        current_meta = b.read(current_meta_path)
        compact = current_meta['compact_modules']
        inventory = evaluate.b.model_inventory(b.HF)
        b.require(len(inventory['projection_shapes']) == 252 and len(compact) == 36
            and {item['name'] for item in compact} == {f'model.layers.{i}.mlp.gate_proj' for i in range(36)}
            and all(item['spec']['consumer'] == 'fixed2_rice7' and item['spec']['implicit_length'] == 2
                and item['spec']['sparse_fields'] == list(b.c1.s.sparse.FIELDS)
                and len(item['spec']['stages']) == 2
                and item['spec']['geometry']['original_shape'] == inventory['projection_shapes'][item['name']]
                for item in compact), 'Inspect the real pending parent Rice metadata and local HF252 geometry, without loading its weights.')
        rejected = {}
        for name, checkpoint_dir, report in (
            ('actual_pending_fresh_C1', b.SOURCE / 'final_model', b.SOURCE / 'report.json'),
            ('old_C0', b.ARTIFACTS / 'fresh_fixed2_cmcc_20261030/single_20261030_v3_C0/final_model',
                b.ARTIFACTS / 'fresh_fixed2_cmcc_20261030/single_20261030_v3_C0/report.json'),
            ('engineering_short_cannot_be_parent', sequence.TARGET / 'short2/final_model', sequence.TARGET / 'short2/report.json')):
            try:
                b.source(checkpoint_dir, report)
            except (ValueError, FileNotFoundError) as error:
                rejected[name] = str(error)
            else:
                raise ValueError('Ineligible actual source accepted: ' + name)
        try:
            sequence.terminal_parent(parent)
        except ValueError as error:
            pending_controller_rejected = str(error)
        else:
            raise ValueError('An active C1 controller was accepted as successful complete-nine.')
        evaluate.setup(b.SUPPORT)
        dispatch = (evaluate.b.source, evaluate.b.source_pins, evaluate.b.command)
        b.configure()
        try:
            b.source(b.SOURCE / 'final_model', b.SOURCE / 'report.json')
        except ValueError:
            pass
        else:
            raise ValueError('Pending C1 unexpectedly became a real final during this CPU validation.')
        b.require(dispatch == (evaluate.b.source, evaluate.b.source_pins, evaluate.b.command)
            and evaluate.b.source is evaluate.source
            and evaluate.b.BASE.resolve(strict=True) == b.HF.resolve(strict=True)
            and evaluate.b.command('predict')[4] == 'experiments.rice_muon_cmcc_20261031.evaluate',
            'Repeated actual source/configure calls must preserve local HF and Rice-Muon dispatch.')
        rejected_headers = []
        for header in (
            dict(schema='fresh_C1_Rice_Muon_execution_v1', phase='short2', steps=2, validation_only=True,
                status='Rice_Muon_training_export_complete_fresh_pending'),
            dict(schema='fresh_C0_Muon32_execution_v1', phase='formal32', steps=32, validation_only=False,
                status='formal32_training_export_complete_fresh_pending'),
            dict(schema='fresh_C1_Rice_Muon_execution_v1', phase='formal32', validation_only=False, status='running'),
        ):
            try:
                evaluate.validate_header(header)
            except ValueError as error:
                rejected_headers.append(dict(header=header, error=str(error)))
            else:
                raise ValueError('An ineligible final evaluation header was accepted.')
        namespaces = {name:list(sys.modules[name].__path__) for name in ('train_utils', 'litebsq', 'rotation', 'e2e_common')}
        modules = {name:str(Path(module.__file__).resolve()) for name,module in list(sys.modules.items())
            if name.split('.')[0] in namespaces and getattr(module, '__file__', None)}
        b.require(all(paths == [str(b.c1.s.common.s.SNAPSHOT / name)] for name,paths in namespaces.items())
            and all(Path(path).is_relative_to(b.c1.s.common.s.SNAPSHOT) for path in modules.values()),
            'All complete run/fresh/evaluation/helper imports must remain in selected N.')
        syntax = []
        for path in b.HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            syntax.append(str(path))
        protected = b.read(b.EVIDENCE / 'protected_pins_before.json')
        for path,item in protected.items():
            b.require(b.executor.sha(path) == item['plan_sha256'], 'An active immutable plan changed.')
            pins = b.read(path)['source_sha256']
            b.require(len(pins) == item['source_count'], 'An active immutable pin set changed.')
            b.executor.check_pins(pins)
        record.update(status='verified_pending_fresh_C1_Rice_Muon_protocol_inputs_and_reader_CPU', exit_code=0,
            actual_pending_parent=parent, real_pending_Rice_metadata=b.binding(current_meta_path),
            current_metadata_only_not_final=True, actual_compact_Rice_modules=len(compact), local_HF_projection_count=252,
            data=exposure, M_target_evidence=evidence, teacher_cache=cache,
            reused_fixed2_GPU_evidence={key:value for key,value in proof.items() if key != 'record'},
            reused_fixed2_trainable_leaves=1728, reused_fixed2_independent_Muon_matrices=1008,
            actual_new_final_layout_not_yet_derived=True, actual_source_rejections=rejected,
            active_controller_not_successful_terminal_rejected=pending_controller_rejected,
            invalid_evaluation_headers_rejected=rejected_headers, source_calls_preserve_dispatch=True,
            namespace_paths=namespaces, actual_runtime_module_sha256={path:b.executor.sha(path) for path in sorted(set(modules.values()))},
            syntax_checked=syntax, protected_active_source_pins=protected, source_bindings=b.code_bindings(),
            original_schedule_steps=500, formal_group_indices=b.GROUPS, engineering_group_indices=[0, 1],
            CUDA_initialized=b.torch.cuda.is_initialized(), actual_model_weights_loaded=False,
            boundary='Real active C1 process/plan, current36Rice metadata, unchanged H96 inputs and inherited fixed2 GPU evidence only. '
                'No future final or Rice Muon GPU success is fabricated. Complete C1/full9, actual-final layout/frozen preparation, '
                'new whole-Rice two-update lifecycle/fresh, then independent original-C1 formal32/fresh/full9 remain required.')
        b.require(not record['CUDA_initialized'], 'CPU verification unexpectedly initialized CUDA.')
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        b.core.write(output / 'validation.json', record)
    print(dict(status=record['status'], output=str(output)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    execute(parser.parse_args().output)
