"""Bind the actual fresh C0; reuse the unchanged mixed Muon implementation."""
from copy import deepcopy
import importlib
from pathlib import Path

# Close the category-aware runtime imports before loading legacy data helpers.
from experiments.fresh_fixed2_cmcc_20261030 import evaluate as c0
# These are namespace packages. Legacy helpers prepend S to sys.path; explicitly
# close their child-module search to the already selected complete N runtime.
for _package_name in ('train_utils', 'litebsq', 'rotation', 'e2e_common'):
    _package = importlib.import_module(_package_name)
    _package.__path__ = [str(c0.s.common.s.SNAPSHOT / _package_name)]
from experiments.muon_compact_cmcc_20261030 import bindings as mature
from experiments.muon_compact_cmcc_20261030 import fit, inputs, lifecycle, optimizer
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as shared
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO

HERE = Path(__file__).resolve().parent
SUPPORT = ARTIFACTS / 'adaptive_class_cmcc_20261030/evaluation_support'
PROOF = ARTIFACTS / 'muon_compact_cmcc_20261030'
HF = mature.HF
torch, core, checkpoint = mature.torch, mature.core, mature.checkpoint
read, require, binding = mature.read, mature.require, mature.binding
GROUPS = list(range(32))
CONTRACT = {**deepcopy(mature.CONTRACT),
    'student_source': 'The actual completed fresh C0 fixed2 endpoint; fresh optimizer, no short weights or donor conversion.',
    'budget': 'H96 M1024 once/Q32 twice/H96 twice, original500 linear schedule prefix32, seed137/R0; full9 decides quality.',
    'initialization': 'Actual source BF16 D -> FP32 masters; recompute first-gradient kappa and every student/QA teacher trajectory.'}


def configure():
    c0.s.common.s.enable_snapshot()
    shared.setup(SUPPORT)
    # Only the group-index schedule is rebound in this isolated process. The
    # validated objective, optimizer, rollout and export primitives stay intact.
    mature.STEPS = {'formal32': GROUPS}


def source(checkpoint_dir, report_path):
    checkpoint_dir, report_path = Path(checkpoint_dir).resolve(), Path(report_path).resolve()
    require(checkpoint_dir.is_dir() and report_path.is_file(),
            'The actual completed C0 and pipeline report must exist; no historical/short substitution.')
    return c0.source(checkpoint_dir, report_path)


def verified_implementation():
    prepared_path = PROOF / 'prepared.json'
    prepared = read(prepared_path)
    report_path = PROOF / 'short2/report.json'
    audit_path = PROOF / 'short2/fresh_audit.json'
    report, audit = read(report_path), read(audit_path)
    require((PROOF/'short2/run.exit').read_text().strip() == '0'
        and (PROOF/'short2.exit').read_text().strip() == '0'
        and audit['status'] == 'verified' and audit['full_state_exact']
        and audit['same_complete_input_BF16_logits_exact'] and audit['finite']
        and audit['training_report'] == binding(report_path)
        and audit['prepared'] == binding(prepared_path)
        and report['training']['steps'] == 2
        and audit['native_modules'] == 216 and audit['compact_modules'] == 36 and audit['pending_dense'] == 0,
        'Reuse only the completed actual mixed fixed2 Muon GPU lifecycle evidence.')
    helpers = (mature.HERE / name for name in ('bindings.py', 'fit.py', 'inputs.py', 'lifecycle.py',
                                              'optimizer.py', 'rollout.py', 'motion.py'))
    files = {}
    for path in helpers:
        item = binding(path)
        require(item == prepared['bindings'][str(path.resolve())], 'The reused Muon helper changed: '+str(path))
        files[str(path.resolve())] = item
    return dict(prepared=binding(prepared_path), training=binding(report_path), fresh=binding(audit_path),
        helper_bindings=files, record=prepared['record'], scope='Existing full8B two-update fixed2 lifecycle; not32-step quality evidence.')


def read_prepared(path):
    path = Path(path).resolve()
    value = read(path)
    require(value['schema'] == 'fresh_C0_Muon32_prepared_v1'
        and value['status'] == 'verified_actual_C0_source_and_reused_GPU_contract'
        and value['contract'] == CONTRACT and value['group_indices'] == GROUPS,
        'Use this explicit actual-source Muon32 preparation.')
    for item in value['bindings'].values():
        require(binding(item['path']) == item, 'Prepared source/input changed: '+item['path'])
    stat = value['source_binding']['checkpoint_weight_stat']
    from experiments.longbench_v1.common import check_stat
    check_stat(stat)
    require(value['record']['source_checkpoint_id'] == value['source_binding']['checkpoint_id'], 'Source UUID changed.')
    return value


def code_bindings():
    paths = set(HERE.glob('*.py'))
    paths.update(mature.HERE / name for name in ('bindings.py', 'fit.py', 'inputs.py', 'lifecycle.py',
                                               'optimizer.py', 'rollout.py', 'motion.py'))
    paths.update((Path(c0.__file__), Path(shared.__file__), SUPPORT/'prepared.json'))
    return {str(path.resolve()): binding(path) for path in paths}
