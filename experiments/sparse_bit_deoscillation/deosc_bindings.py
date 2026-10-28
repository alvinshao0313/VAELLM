"""Use the current bestabs native loader and the existing complete train trajectory."""
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_abs005_rank8_subspace_train32_pair_20261025'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PREVIOUS))
import subspace32_bindings as previous
import subspace32_data as data

t, core = previous.t, previous.core
require, read, write, binding = previous.require, previous.read, previous.write, previous.binding
TARGET = 'model.layers.18.mlp.down_proj'
SOURCE, SOURCE_ID = previous.SOURCE, previous.SOURCE_ID
ARMS = ('baseline', 'risk', 'matched', 'half_lr', 'freeze_codes')
COMMON_STEPS, FINAL_STEPS, WINDOW = 128, 256, 64
BIT_LR, DECODER_LR = 4e-5, 1e-6
THRESHOLD, DISTANCE_BINS, MATCH_SEED = 4., 64, 137


def inputs():
    prior = read(PREVIOUS / 'prepared.json')
    training, evidence = data.load()
    require(evidence == prior['data_evidence'], 'The existing complete32 train groups or cached teachers changed.')
    record = {key: prior[key] for key in ('validated_source_state', 'source_bit_budget',
                                         'source_weight_stat', 'group_case_ids')}
    require(SOURCE_ID == 'e01b09b7-a85d-4534-a5d2-5ab3b91312e6', 'Only the current bestabs source is supported.')
    return training, record, evidence


def own_bindings():
    return [binding(HERE / name) for name in ('deosc_bindings.py', 'controller.py', 'deosc_runtime.py', 'run.py')]


def memory():
    return {name + '_bytes': int(getattr(t.cuda, name)()) for name in
            ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved')}
