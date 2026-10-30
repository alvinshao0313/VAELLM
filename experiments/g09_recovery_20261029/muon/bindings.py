"""Same verified G09/OPD inputs; isolated stage-Muon optimizer and lifecycle."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIRECT = HERE.parent
sys.dont_write_bytecode = True


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


direct = module(DIRECT / 'bindings.py', '_G09_Muon_verified_direct_binding')
core, qa, torch = direct.core, direct.qa, direct.torch
fit_core, engine, base, mmlu, hot = direct.fit_core, direct.engine, direct.base, direct.mmlu, direct.hot
read, require, sha256 = direct.read, direct.require, direct.sha256
binding, load_state = direct.binding, direct.load_state
file_stat = direct.file_stat
source, model_config, teacher_identity = direct.source, direct.model_config, direct.teacher_identity
selected_modules, load_inputs = direct.selected_modules, direct.load_inputs
SNAPSHOT, PYTHON, R, OPD = direct.SNAPSHOT, direct.PYTHON, direct.R, direct.OPD
PROFILE, SOURCE_ID, SOURCE_AUDIT = direct.PROFILE, direct.SOURCE_ID, direct.SOURCE_AUDIT
LIMITS, BUDGET_FIELDS = direct.LIMITS, direct.BUDGET_FIELDS
OUTPUT = Path('/data/star/VAELLM/g09_recovery_20261029/muon_opd32')
STEPS = {'short2': [0, 1], 'formal32': list(range(32))}
MATH_SOURCE = R / 'qwen_optimizer_pair4_20261014/optimizer_math.py'
CONTRACT = {**deepcopy(direct.CONTRACT),
    'student_source': 'Verified G09 complete native endpoint, fresh Muon OPD32 directly; QK2000 omitted',
    'initialization': 'Same G09 BF16 initial D -> FP32 masters; fresh stage-Muon/auxiliary Adam, seed137',
    'data_and_optimizer': 'Same M1024 Q32 H96 train/full-F once .5/.5 M; online QA; R0; original500 prefix; jointclip1.5',
    'Muon': 'Metadata-derived independent stage matrices; FP32 momentum .95/Nesterov; BF16 NS5(3.4445,-4.7750,2.0315); FP32 .2sqrt(max(m,n)); bias/norm Adam(.9,.95)/eps1e-8/WD0',
    'kappa': 'Fresh first actual clipped joint gradient; unrounded FP32 Adam/Muon matrix-candidate norm ratio; fixed thereafter; not decoded BF16 step equality'}


def read_prepared(path=None):
    path = OUTPUT / 'prepared.json' if path is None else Path(path).resolve()
    require(path == OUTPUT / 'prepared.json', 'Use only this Muon prepared path.')
    value = read(path)
    parent, parent_record = direct.read_prepared()
    directory, meta, _ = source()
    require(value['schema'] == 'G09_Muon_OPD_prepared_v1' and value['source_checkpoint_id'] == SOURCE_ID
        and value['source_checkpoint'] == str(directory) and value['contract'] == CONTRACT
        and value['phases'] == STEPS and value['direct_prepared'] == binding(direct.OUTPUT / 'prepared.json')
        and value['record']['model_config'] == parent_record['model_config']
        and value['record']['groups'] == parent_record['groups']
        and value['record']['validated_frozen_state'] == parent_record['validated_frozen_state'],
        'Use only the same-G09 isolated Muon recipe.')
    for item in value['bindings'].values():
        require(binding(item['path']) == item, 'Changed Muon binding: '+item['path'])
    return value, value['record']
