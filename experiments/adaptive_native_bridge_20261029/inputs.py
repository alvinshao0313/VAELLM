"""The two existing auxiliary-train M32 groups and their once-mixed targets."""
import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
OPD = ROOT / 'experiments/cat_aligned_opd_20261028'


def load_groups():
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(OPD))
        spec = importlib.util.spec_from_file_location('_adaptive_bridge_M_bindings', OPD / 'bindings.py')
        backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend)
    finally:
        sys.path[:] = previous
    dataset, recovery, _, evidence = backend.load_inputs()
    groups = []
    for index in (0, 1):
        keys = dataset['groups'][index]['case_ids'][:32]
        cases = [recovery[key]['case'] for key in keys]
        target = torch.cat([recovery[key]['target_log_probs'] for key in keys])
        backend.require(len(cases) == 32 and all(case['domain'] == 'mmlu' for case in cases)
                        and target.shape == (32, 4) and target.dtype == torch.float32,
                        'Use complete original auxiliary-train M32 groups and once-mixed targets.')
        groups.append((cases, target))
    teacher = backend.teacher_recipe()
    backend.qa.check_config(teacher)
    return backend, groups, dict(actual_target_contract=backend.CONTRACT,
        actual_target_evidence=evidence, group_indices=[0, 1],
        case_ids=[[case['case_id'] for case in cases] for cases, _ in groups],
        teacher_identity=teacher['model'], target='Original .5F+.5gold exactly once; original .5/32 document weight.')
