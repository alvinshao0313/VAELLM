"""Fixed same-parent M36 CAT structure comparison; public execution stays frozen."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = Path('/data/star/VAELLM/cat_structure_full_20261029')
SOURCE = Path('/data/star/VAELLM/qwen_recompression_20261026/C/initial_after_o')
SOURCE_ID = 'b63bfe79-3889-40c6-bf66-1d002893f874'
OLD_PREPARATION = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_recompression_20261026/prepare.json'
SNAPSHOT = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation/source_snapshot'
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
SCALARS_PER_LINEAR_UPDATE = 262144
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT))

spec = importlib.util.spec_from_file_location('_structure_existing_fork_tools', ROOT / 'experiments/cat_recompression/fork_core.py')
fork = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fork)
read, write, sha, require = fork.read, fork.write, fork.sha, fork.require


def source_binding():
    previous = read(OLD_PREPARATION)
    row = previous['profiles']['C']
    meta = fork.load_v6_meta(str(SOURCE))
    require(previous['status'] == 'cpu_verified' and Path(row['boundary']) == SOURCE
        and meta['checkpoint_id'] == row['checkpoint_id'] == SOURCE_ID
        and meta['checkpoint_kind'] == 'category_boundary'
        and meta['completed_categories'] == list(CATEGORIES[:4])
        and len(meta['compressed_targets']) == 144 and len(meta['pending_dense_targets']) == 108
        and meta['lora_config'] is None, 'Require the actual finalized C after_o parent.')
    require(sha(SOURCE / 'checkpoint_meta.json') == row['metadata_sha256']
        and sha(SOURCE / 'cat_runtime_state.pt') == row['runtime_sha256'], 'Parent metadata/runtime changed.')
    fork.check_stable_stat(row['weight_stat'])
    for path, expected in previous['snapshot_source_sha256'].items():
        require(sha(path) == expected, 'Frozen native source changed: ' + path)
    return previous, row, meta


def implementation_pins():
    paths = set(HERE.glob('*.py')) | set(HERE.glob('*.sh'))
    paths.add(ROOT / 'experiments/cat_recompression/fork_core.py')
    return {str(path): sha(path) for path in sorted(paths)}


def check_pins(pins):
    for path, expected in pins.items():
        require(sha(path) == expected, 'Bound implementation/input changed: ' + path)
