"""Fixed fresh gate/up allocation pair in the recovered e767 Llama context."""
from copy import deepcopy
from pathlib import Path

from experiments.full_rht_gate_pair.common import ROOT, ORIGINAL, RESULT, SHARDS, check, eval_argv, read
from experiments.qwen3_w25.assemble_categories import _require

CATEGORIES = ['gate_proj', 'up_proj']
BITS = {'control': {'gate_proj': 32, 'up_proj': 32},
        'candidate': {'gate_proj': 40, 'up_proj': 24}}
BASE = RESULT / 'llama_fineweb1600_down_d16_combo/final_model'
BASE_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
TOTAL_BITS = 17095180288
DENOMINATOR = 6979321856


def paired_recipe(record):
    value = deepcopy(record['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        _require(value[section]['output_dir'] == str(Path(record['output_dir']) / 'donor'),
                 'Unexpected paired output path.')
        value[section]['output_dir'] = '<arm>'
    for section in ('canonical_config', 'vae_args'):
        _require(value[section]['codebook_bits'] == {'default': 32, 'by_category': record['codebook_bits']},
                 'Unexpected category codebook table.')
        value[section]['codebook_bits'] = '<paired-K>'
    for category in CATEGORIES:
        _require(value['resolved_category_runtime'][category]['codebook_bits'] == record['codebook_bits'][category],
                 'Resolved K differs from this arm.')
        value['resolved_category_runtime'][category]['codebook_bits'] = '<paired-K>'
    return value
