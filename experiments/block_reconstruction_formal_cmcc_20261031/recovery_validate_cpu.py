"""Verify real H96/proofs and reject unavailable or prototype BR recovery sources."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import recovery_bindings as b
from . import recovery_evaluate, recovery_fresh, recovery_prepare, recovery_run
from . import bindings as formal_BR


def rejected(call, message):
    try:
        call()
    except (ValueError, FileNotFoundError) as error:
        return str(error)
    raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized()
        and not args.output.exists(), 'Use caller bitvae, hidden CUDA and a new CPU evidence directory.')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', argv=sys.argv, python=sys.executable)
    try:
        b.torch.set_num_threads(4)
        b.configure()
        proof = b.verified_implementation()
        BR_proof = formal_BR.verified_short()
        dataset, recovery, cache, evidence = b.inputs.load_inputs()
        exposure = recovery_prepare.data_contract(dataset, recovery)
        b.require(b.mature.STEPS == {'formal32':list(range(32))}
            and b.mature.LIMITS == {'qasper':128, 'hotpotqa':32}
            and b.fit is b.direct.fit and b.optimizer is b.direct.optimizer
            and b.lifecycle is b.direct.lifecycle and b.inputs is b.direct.inputs,
            'Reuse the original mathematical helpers, QA limits and complete32-group schedule.')
        geometry = dict(decoder_leaves=proof['record']['trainable_tensors'],
            decoder_elements=proof['record']['trainable_elements'],
            independent_matrices=sum(row['stages'] for row in proof['record']['matrix_layout'].values()),
            frozen_fields=proof['record']['validated_frozen_state']['tensors'])
        b.require(geometry == dict(decoder_leaves=1728, decoder_elements=5336064,
            independent_matrices=1008, frozen_fields=1947), 'Actual proved fixed2 geometry differs.')
        short = formal_BR.SHORT
        short_report = b.read(short/'report.json')
        b.require(short_report['schema'] == 'BR_adjacent_blocks_short2_v1'
            and short_report['validation_only'] is True, 'Test rejection against the actual preserved BR prototype report.')
        short_reason = rejected(lambda:b.source(short/'final_model', short/'report.json'),
            'The real adjacent-block prototype was incorrectly accepted as a formal BR recovery parent.')
        b.require('adjacent-block prototype' in short_reason,
            'The prototype must be rejected by its actual source contract, not only by the already cleaned weights.')
        missing = formal_BR.OUTPUT/'not_created_formal_BR36'
        missing_reason = rejected(lambda:b.source(missing/'final_model', missing/'report.json'),
            'An unavailable formal BR source was silently replaced.')
        evaluation_reason = rejected(lambda:recovery_evaluate.source(short/'final_model', short/'report.json'),
            'The complete-nine adapter accepted a prototype without actual BR->E32.')
        syntax = []
        for path in sorted(b.HERE.glob('recovery_*.py')):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            syntax.append(str(path))
        b.require(not b.torch.cuda.is_initialized(), 'CPU BR->E verification initialized CUDA.')
        result.update(status='verified_BR_E32_protocol_data_proofs_and_source_rejection', exit_code=0,
            data=exposure, proved_geometry=geometry,
            reused_Muon_GPU={key:value for key,value in proof.items() if key != 'record'}, reused_BR_GPU=BR_proof,
            teacher_cache=cache, M_target_evidence=evidence, source_bindings=b.code_bindings(),
            actual_BR_prototype_rejected=dict(report=b.binding(short/'report.json'), reason=short_reason),
            unavailable_formal_BR_rejected=dict(path=str(missing), reason=missing_reason),
            ineligible_full9_source_rejected=evaluation_reason, syntax_checked=syntax,
            unchanged_mature_helpers=True, original_schedule_steps=500,
            actual_learning_rates=[1e-6*(1-(step-1)/500) for step in range(1,33)],
            required_runtime_protocol=dict(fresh_first_gradient_kappa=True,
                all_QA_rollouts_and_teacher_targets_recomputed=True),
            CUDA_initialized=False,
            boundary='No complete C0/BR parent currently loaded or fabricated; no GPU, training, export, fresh or evaluation executed. Actual complete BR source preparation remains mandatory.')
    except BaseException:
        result.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        b.core.write(args.output/'validation.json', result)
    print(json.dumps(dict(status=result['status'], geometry=geometry, data=exposure)), flush=True)


if __name__ == '__main__':
    main()
