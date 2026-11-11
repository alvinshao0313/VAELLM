"""Actual B endpoint, fixed HOT inputs, and unchanged mature objective helpers."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
R = ROOT / 'result/qwen3_8b_mmlu65_20261004'
sys.dont_write_bytecode = True
from experiments.adaptive_native_model_20261029 import checkpoint
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.cmcc_runtime_20261030.paths import resolve_legacy_path

SOURCE = Path('/data1/cmcc/VAELLM/artifacts/cat_structure_full_20261029/B/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/final_model')
OUTPUT = Path('/data1/cmcc/VAELLM/artifacts/muon_compact_cmcc_20261030')
HF = Path('/data1/cmcc/models/Qwen/Qwen3-8B')
STEPS = {'short2': [0, 1]}


def module(path, name):
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(Path(path).parent))
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    finally:
        sys.path[:] = previous


aligned = module(ROOT / 'experiments/cat_aligned_opd_20261028/bindings.py', '_compact_aligned_objective')
core, qa, torch = aligned.core, aligned.qa, aligned.torch
base, mmlu, hot = aligned.base, aligned.mmlu, aligned.hot
read, require, sha256 = core.read, core.require, core.sha256
LIMITS = deepcopy(aligned.LIMITS)
CONTRACT = {**deepcopy(aligned.CONTRACT),
    'student_source': 'Real completed B, gate-only fixed2 compact conversion; preserve216 native and non-gate state',
    'rollout_cache': '216 native group8;36 compact own no_grad BF16 caches; clear both before FP32 master training',
    'optimizer': 'Original NS5 .95 Nesterov FP32 momentum/BF16 NS/.2sqrtmax; explicit native-stage and compact-leaf matrices; affine Adam',
    'kappa': 'Fresh first actual jointly clipped gradient FP32 candidate norm match, fixed afterwards',
    'budget': 'Original500 linear prefix, group0/1 only, M32+QA8 per update, seed137/R0; no quality conclusion'}


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def selected_modules(model, record=None):
    groups = checkpoint.inventories(model)
    require(not groups['pending_dense'], 'Muon endpoint must have no pending dense projection.')
    return {name: model.get_submodule(name) for group in ('native', 'compact') for name in groups[group]}
