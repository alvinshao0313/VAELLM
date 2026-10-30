"""Bind the selected raw B and the common live after_o without copying weights."""
import importlib.util
from pathlib import Path

import torch

from experiments.adaptive_native_model_20261029 import checkpoint
from experiments.adaptive_native_model_20261029.archive_cat_gate_prefixes import BASE, BASE_ID, load_archive
from experiments.adaptive_native_model_20261029.native_prefix import from_native
from experiments.adaptive_vector_rvq import common as c

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('/data/star/VAELLM/adaptive_class_20261030')
ARCHIVE = Path('/data/star/VAELLM/adaptive_native_model_20261029/cat_gate_prefixes/B')
TOKENS = ROOT / 'result/qwen_cat_biaxial_20261022/prepared/tokens.pt'
TOKEN_MANIFEST = TOKENS.with_name('manifest.json')
RICE_PATH = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_sparse005_lossless_index_rice7_20261024/rice7.py'
spec = importlib.util.spec_from_file_location('_adaptive_class_verified_rice7', RICE_PATH)
rice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rice)


def raw_inputs():
    gate, specs, manifest = load_archive(ARCHIVE)
    c.require(manifest['source_raw']['checkpoint_id'] == '7c0b1199-b9ff-46af-a1b7-c2f5b7dabf7f'
              and manifest['common_after_o']['checkpoint_id'] == BASE_ID and manifest['after_o_complement_exact'],
              'Use selected B raw2 and its actual common live after_o.')
    meta = c.read(BASE / 'checkpoint_meta.json')
    c.require(meta['checkpoint_id'] == BASE_ID, 'The common live parent changed.')
    live = torch.load(BASE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    return gate, specs, manifest, live


def prefix(gate, spec):
    name = spec['name']
    state = {key[len(name) + 1:]: value for key, value in gate.items() if key.startswith(name + '.')}
    bundle, lengths, resolved = from_native(spec, state, dict(raw_B=str(ARCHIVE),
        source_raw_checkpoint_id='7c0b1199-b9ff-46af-a1b7-c2f5b7dabf7f', live_after_o=BASE_ID))
    c.require(resolved['vector_dim'] == 64 and all(row['decoder_spec']['in_dim'] == 64 for row in resolved['stages']),
              'The selected whole-gate comparison is actual D64/K64.')
    return bundle, lengths, resolved


def code_identity():
    here = Path(__file__).parent
    core = Path(checkpoint.__file__).parent
    paths = [*(here / name for name in ('__init__.py', 'sources.py', 'sparse.py', 'artifact.py', 'fit.py', 'run_fit.py', 'capture.py')),
        ROOT / 'experiments/adaptive_grouped_fit_20261029/grouped.py', RICE_PATH,
        *(core / name for name in ('__init__.py', 'checkpoint.py', 'objective.py', 'allocation.py', 'native_prefix.py', 'codec.py', 'spec.py', 'consumer.py'))]
    return {str(path.resolve()): c.sha(path) for path in paths}
