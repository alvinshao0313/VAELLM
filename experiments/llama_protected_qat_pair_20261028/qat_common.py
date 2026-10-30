"""Independent same-a1 Llama D-only/protected-INT8-q matched64 experiment."""
from pathlib import Path
import importlib
import sys

CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parents[1]
R = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
PREVIOUS = R / 'llama_sparse005_adam_muon32_pair_20261025'
sys.path.insert(0, str(PREVIOUS))
previous = importlib.import_module('pair_common')
retention = importlib.import_module('pair_retention')
mature_rollout = importlib.import_module('pair_rollout')
HERE = R / CODE.name
DATA = Path('/data/star/VAELLM/qwen3_8b_mmlu65_20261004') / CODE.name
SOURCE, SOURCE_ID, HF, PYTHON = previous.SOURCE, previous.SOURCE_ID, previous.HF, previous.PYTHON
S, ENVIRONMENT = previous.S, previous.ENVIRONMENT
protocol = previous.protocol
native, training, objective = previous.native, previous.training, previous.objective
torch, require, read, write = previous.torch, previous.require, previous.read, previous.write
sha, digest, tensor_record, state_proof = previous.sha, previous.digest, previous.tensor_record, previous.state_proof
binding, check_binding, check_runtime = previous.binding, previous.check_binding, previous.check_runtime
tick, policy, load_student, load_teacher, inputs = previous.tick, previous.policy, previous.load_student, previous.load_teacher, previous.inputs
TASKS, LIMITS, VOCAB, SEED, SHORT_GROUPS = previous.TASKS, previous.LIMITS, previous.VOCAB, previous.SEED, previous.SHORT_GROUPS
MAX_BPW, BPW, STEPS = previous.MAX_BPW, previous.BPW, 64
ARMS = ('D', 'DQ')


def mutable(record, arm):
    require(arm in ARMS, 'Use the declared same-source D/DQ pair.')
    return set(record['decoder_parameters']) | (set(record['q_parameters']) if arm == 'DQ' else set())


def prepared():
    record = read(HERE / 'prepared.json')
    require(record['schema'] == 'llama_protected_qat_matched64_v1'
        and record['source_checkpoint_id'] == SOURCE_ID and record['steps'] == STEPS
        and record['group_indices'] == list(range(32)) * 2 and record['seed'] == SEED,
        'Require fresh same-a1 matched64 preparation.')
    for path, expected in record['code_sha256'].items():
        require(sha(path) == expected, 'Changed bound implementation: ' + path)
    for item in record['bindings'].values(): check_binding(item)
    return record
