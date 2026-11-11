"""Validate two independent full selected k prefixes without changing shared code."""
from .execution import configure_deterministic_backward

EXECUTION = configure_deterministic_backward()

import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from .probe import describe, parameters, read, sha
from experiments.cat_auto_protocol_cmcc_20261104.native_builder import (
    RecoveryExecutionPolicy, make_native_recovery_trainer,
)
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.sc_cat_live_cmcc_20261106.live_io import load_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-result', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = dict(status='running', phase='technical_validation', purpose='full_selected_k_prefix_reproducibility',
        execution=EXECUTION, started_epoch=started, selected_step=250, schedule_max=500,
        original_result=dict(path=str(Path(args.original_result).resolve()), sha256=sha(args.original_result)),
        implementation={str(path.resolve()): sha(path) for path in Path(__file__).parent.glob('*.py')})
    run = None
    try:
        torch.set_num_threads(4)
        original = read(args.original_result)
        for path, expected in original['actual_run']['provenance']['source_pins'].items():
            if sha(path) != expected:
                raise RuntimeError('Original recovery consumer changed: ' + path)
        inventory, prepared, sources, tokenizer = load_inputs(original, with_sources=True)
        run = make_native_recovery_trainer(round_base_dir=original['live_raw']['round_base_dir'],
            inventory=inventory, category_id='k_proj', settings=RecoverySettings(**original['settings']),
            prepared_data=prepared, sources=sources, tokenizer=tokenizer, output_dir=output / 'train',
            execution_policy=RecoveryExecutionPolicy(**original['actual_run']['provenance']['execution_policy']),
            phase='technical_validation', replay_selected_step=250)
        report['initial'] = describe(parameters(run.trainer.model))
        report['actual_run'] = run.train()
        if report['actual_run']['actual_data']['rows'] != original['actual_run']['actual_data']['rows'][:250]:
            raise RuntimeError('The real original250 input prefix changed.')
        report['final'] = describe(parameters(run.trainer.model))
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report.update(status='completed', exit_code=0, input_prefix_verified=True)
    except Exception:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - started
        (output / 'prefix.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        if run is not None:
            run.close()
    print(json.dumps({key: report[key] for key in ('status', 'elapsed_seconds', 'input_prefix_verified')}), flush=True)


if __name__ == '__main__':
    main()
