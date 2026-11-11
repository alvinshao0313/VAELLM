"""Bind this C0 learning study without weakening the production storage checks."""
from pathlib import Path

from experiments.mlp_learnability_cmcc_20261030 import bindings as training
from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as shared

HERE = Path(__file__).resolve().parent


def snapshot(directory, report_path, *, validation=False, updates=None):
    from .source import source
    binding, meta = source(directory, report_path)
    b.require(binding['validation_only'] is validation
        and binding['planned_updates'] == (2 if validation else 128)
        and binding['actual_updates'] in ((1, 2) if validation else (64, 128)),
        'Require a genuine snapshot from this fixed two-update validation or 128-update study.')
    if updates is not None:
        b.require(binding['actual_updates'] == updates, 'This readout belongs to another update boundary.')
    arm, budget = binding['arm'], binding['bit_budget']
    b.require(arm in ('decoder', 'dense') and binding['diagnostic_overbudget'] == (arm == 'dense'),
        'The parameterization and diagnostic storage status differ.')
    if arm == 'decoder':
        b.require(len(meta['native_modules']) == 216 and len(meta['compact_modules']) == 36
            and not meta['pending_dense_modules'] and budget['budget_passed']
            and budget['final_budget_applicable'], 'The decoder arm must retain its complete low-bit inventory.')
    else:
        b.require(len(meta['native_modules']) == 144 and not meta['compact_modules']
            and len(meta['pending_dense_modules']) == 108
            and all('.self_attn.' in item['name'] for item in meta['native_modules'])
            and all('.mlp.' in item['name'] for item in meta['pending_dense_modules'])
            and not budget['budget_passed'] and not budget['final_budget_applicable'],
            'Free-dense is the complete 108-MLP overbudget diagnostic, never a low-bit product.')
    return binding, meta


def annotations(binding):
    return dict(arm=binding['arm'], actual_updates=binding['actual_updates'],
        planned_updates=binding['planned_updates'], validation_only=binding['validation_only'],
        diagnostic_overbudget=binding['diagnostic_overbudget'],
        source_C0_checkpoint_id=binding['source_C0_binding']['checkpoint_id'],
        checkpoint_selection=False)


def setup(support):
    shared.setup(support)
    original_pins = b.source_pins

    def source_pins(binding):
        paths = set(HERE.glob('*.py')) | set(training.HERE.glob('*.py'))
        return {**original_pins(binding), **{str(path.resolve()): b.sha256(path) for path in paths}}

    def source(directory, report_path):
        return snapshot(directory, report_path)

    def command(component, *arguments):
        return [str(b.PYTHON), '-B', '-u', '-m',
            'experiments.mlp_learnability_formal_cmcc_20261101.evaluate',
            '--support', str(support), '--component', component, *arguments]

    b.source, b.source_pins, b.command = source, source_pins, command

