"""Same G09 Muon lifecycle; only the second Hotpot exposure uses new train cases."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MUON = ROOT / 'experiments/g09_recovery_20261029/muon'
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


parent = module(MUON / 'bindings.py', '_hotpot192_verified_Muon_binding')
data_contract = parent.direct.aligned.reference.data
for _name in ('core', 'qa', 'torch', 'fit_core', 'engine', 'base', 'mmlu', 'hot',
        'read', 'require', 'sha256', 'binding', 'load_state', 'file_stat', 'source',
        'model_config', 'teacher_identity', 'selected_modules', 'SNAPSHOT', 'PYTHON',
        'R', 'OPD', 'PROFILE', 'SOURCE_ID', 'SOURCE_AUDIT', 'LIMITS', 'BUDGET_FIELDS'):
    globals()[_name] = getattr(parent, _name)

OUTPUT = Path('/data/star/VAELLM/g09_hotpot_coverage_20261030')
POOL = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_hot_qa512_s128_20261013/dataset.pt')
POOL_SHA256 = 'b954e547def4756033cd43e0286375e04d3112584d892ff6074740a588a9b9ba'
POOL_MANIFEST = R / 'qwen_hot_qa512_s128_20261013/dataset_prepared.json'
TRAIN_MANIFEST = ROOT / 'data/hotpotqa_train_20261011/manifest.json'
STEPS = {'formal32': list(range(32))}
CONTRACT = {**deepcopy(parent.CONTRACT),
    'data_and_optimizer': 'M1024 once, Qasper32 twice, Hotpot192 once; M.5/QA.5 R0; fresh Muon seed137; original500 prefix32',
    'Hotpot_coverage': 'Only 0-based groups20..31 Hotpot8 replaced by existing QA512 hash-pool H[96:192]; groups0..19 and every M/Q value unchanged.',
    'qa_group_index': 'Original domain-slot index modulo16 retained by mature fit; not a claim of 16 actual unique QA groups.',
    'exposure': 'Same document presentations and32 updates; Hotpot prompt tokens277934 versus control278818 (-884); actual OPD generated supervision logged, not assumed equal.',
    'selection': 'Existing train hash order sha256(causal_output_fit/137/+case_id), tie case_id; no test-based selection or new filtering.'}


def load_inputs():
    """The actual runtime consumes the CPU-verified explicit dataset; no QA TF target."""
    prepared, _ = read_prepared()
    item = prepared['dataset_binding']
    require(binding(item['path']) == item, 'Changed explicit Hotpot192 dataset.')
    dataset = torch.load(item['path'], map_location='cpu', weights_only=False)
    _, old, cache, evidence = parent.load_inputs()
    recovery = {}
    for case in dataset['training_cases']:
        key = case['case_id']
        recovery[key] = {'case': case}
        if case['domain'] == 'mmlu':
            data_contract.require_same(case, old[key]['case'])
            recovery[key]['target_log_probs'] = old[key]['target_log_probs']
    require(len(recovery) == 1248 and sum('target_log_probs' in value for value in recovery.values()) == 1024
        and all('target_log_probs' not in value for value in recovery.values() if value['case']['domain'] != 'mmlu'),
        'Only the unchanged once-mixed full-F M1024 targets may be reused.')
    return dataset, recovery, cache, dict(original_M_evidence=evidence,
        explicit_Hotpot192_dataset=item, QA_targets='Fresh current native-greedy trajectories and full-F online targets only.')


def read_prepared(path=None):
    path = OUTPUT / 'prepared.json' if path is None else Path(path).resolve()
    require(path == OUTPUT / 'prepared.json', 'Only the isolated Hotpot192 prepared path is supported.')
    value = read(path)
    old, old_record = parent.read_prepared()
    directory, _, _ = source()
    require(value['schema'] == 'G09_Muon_OPD_prepared_v1'
        and value['experiment_schema'] == 'G09_Hotpot192_coverage_v1'
        and value['status'] == 'CPU_verified_reused_Muon_GPU_evidence'
        and value['source_checkpoint_id'] == SOURCE_ID and value['source_checkpoint'] == str(directory)
        and value['contract'] == CONTRACT and value['phases'] == STEPS
        and value['control_prepared'] == binding(parent.OUTPUT / 'prepared.json')
        and value['record']['model_config'] == old_record['model_config']
        and value['record']['matrix_layout'] == old_record['matrix_layout']
        and value['record']['validated_frozen_state'] == old_record['validated_frozen_state'],
        'Changed isolated same-G09/H192-only coverage contract.')
    for item in value['bindings'].values():
        require(binding(item['path']) == item, 'Changed coverage source/input: ' + item['path'])
    require(binding(value['dataset_binding']['path']) == value['dataset_binding'], 'Changed H192 dataset.')
    return value, value['record']
