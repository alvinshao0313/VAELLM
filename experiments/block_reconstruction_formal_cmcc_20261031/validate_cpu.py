"""Real H96/core-evidence checks and actual ineligible-source rejection only."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b
from . import export, source, training


def rejected(call):
    try:
        call()
    except (ValueError, FileNotFoundError) as error:
        return dict(rejected=True, reason=str(error))
    raise ValueError('An actual ineligible source was incorrectly accepted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    b.sequence.cpu_environment()
    b.require(not b.torch.cuda.is_initialized() and not args.output.exists(), 'Use a new CPU-only validation output.')
    args.output.mkdir(parents=True)
    began = time.monotonic()
    result = dict(schema='BR36_H96_32_CPU_validation_v1', status='running', argv=sys.argv, python=sys.executable)
    try:
        proof = b.verified_short()
        pool, evidence = b.data.load()
        actual = b.data.summary(pool)
        b.require(len(pool.entries) == 160 and len(pool.groups) == 32
            and [group.group_index for group in pool.groups] == list(b.GROUPS)
            and actual['M32_entries'] == 32 and actual['QA_B1_entries'] == 128
            and actual['unique_total_valid_tokens'] == 708845
            and actual['group_presented_valid_tokens'] == 1053156
            and actual == evidence['full'] and b.CONTRACT['total_updates'] == 36*32,
            'Keep the complete actual H96 identities, original input shapes and group presentations.')
        tree = ast.parse(Path(training.__file__).read_text(), filename=training.__file__)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        helpers = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Name) and node.func.value.id == 'fit'}
        b.require({'projection_parameters', 'entry_objective', 'check_group', 'gradient_check',
            'parameter_motion', 'optimizer_check', 'frozen_proof'}.issubset(helpers)
            and 'fit_layer' not in helpers and 'evaluate_layer' not in helpers and b.fit.GROUPS == (2, 10),
            'The formal traversal must reuse immutable numerical primitives without short-loop/schema or global mutation.')
        adam = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'AdamW')
        actual_adam = {keyword.arg: ast.literal_eval(keyword.value) for keyword in adam.keywords}
        b.require(actual_adam == dict(lr=1e-6, betas=(.9, .95), eps=1e-8, weight_decay=0., foreach=False),
                  'The explicit formal optimizer deviated from the one fixed candidate.')
        short_meta = b.read(b.SHORT/'final_model'/b.checkpoint.META_FILE)
        actual_D = export.decoder_names(short_meta)
        b.require(len(actual_D) == 1728 and len(short_meta['tensor_table'])-len(actual_D) == 1947,
                  'The real tested complete mixed geometry must derive the expected D/frozen inventory.')
        current = b.ARTIFACTS/'fresh_fixed2_cmcc_20261030/single_20261030_v3_C0'
        reject = dict(
            actual_short_as_C0=rejected(lambda: b.bind_C0(b.SHORT/'final_model', b.SHORT/'report.json')),
            actual_native_B_as_C0=rejected(lambda: b.bind_C0(b.core.SOURCE_B, b.SHORT/'report.json')),
            actual_short_as_formal_BR=rejected(lambda: source.source(b.SHORT/'final_model', b.SHORT/'report.json')),
            unfinished_actual_C0=rejected(lambda: b.bind_C0(current/'final_model', current/'report.json')))
        syntax = []
        for path in b.HERE.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
            syntax.append(str(path))
        b.sequence.check_pins(proof['numerical_core_sha256'])
        b.require(not b.torch.cuda.is_initialized(), 'CPU validation must never initialize CUDA.')
        result.update(status='verified_actual_inputs_core_evidence_and_rejections', exit_code=0,
            full_data=actual, data=evidence, reused_short=proof, source_rejections=reject,
            reused_numerical_helpers=sorted(helpers), explicit_Adam=actual_adam,
            actual_learning_rates=[1e-6*(1-(step-1)/500) for step in range(1, 33)],
            tested_geometry_D_fields=len(actual_D), tested_geometry_frozen_fields=len(short_meta['tensor_table'])-len(actual_D),
            tested_geometry_sha256=export.geometry(short_meta), syntax_checked=syntax, source_sha256=b.code_bindings(),
            short_GROUPS_unchanged=list(b.fit.GROUPS), CUDA_initialized=False, actual_C0_model_loaded=False,
            formal_model_or_success_fixture_fabricated=False, formal_output_prepared=False,
            GPU_training_or_evaluation_started=False,
            scope='Actual train inputs, immutable reused GPU evidence, formal loop syntax/contract and real source refusal only; no formal quality or full36 GPU proof.')
    except BaseException:
        result.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-began
        b.write(args.output/'validation.json', result)
    print(json.dumps(dict(status=result['status'], report=str(args.output/'validation.json'))), flush=True)


if __name__ == '__main__':
    main()
