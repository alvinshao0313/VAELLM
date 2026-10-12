"""Fixed all-attention target-source pair; existing native CAT and INT4 simulation."""
from pathlib import Path
import json

from experiments.longbench_v1.common import check_hashes, check_stat
from experiments.qwen3_w25.assemble_categories import _require

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = Path('/home/star/VAELLM')
RESULT = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
BITS = dict(zip(CATEGORIES, (40, 40, 40, 56)))
NAMES = tuple(f'model.layers.{i}.self_attn.{category}' for category in CATEGORIES for i in range(36))
ARMS = ('hf', 'decoded')
PROTECTION = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    _require(ROOT != PUBLIC and Path(record['cwd']) == ROOT
             and record['schema'] == 'qwen_attention_decoded_target_pair_v1'
             and record['arm'] in ARMS and record['target_kind'] == record['arm']
             and record['base_checkpoint_id'] == SOURCE_ID and record['codebook_bits'] == BITS
             and record['categories'] == list(CATEGORIES) and record['steps_per_stage'] == 10000
             and not record['queued'] and not record['automatic_retry'], 'Unexpected isolated target pair contract.')
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)


def paired_recipe(record):
    from copy import deepcopy
    value = deepcopy(record['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        _require(value[section]['output_dir'] == str(Path(record['output_dir']) / 'native'), 'Wrong native output mirror.')
        value[section]['output_dir'] = '<paired-output>'
    return value
