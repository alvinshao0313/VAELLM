"""Locate migrated historical inputs without rewriting their recorded identity."""
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path('/data1/cmcc/VAELLM/artifacts')
DATA = Path('/data1/cmcc/VAELLM/data')
MODELS = Path('/data1/cmcc/models')
PYTHON = Path('/home/cmcc/miniconda3/envs/bitvae/bin/python')
TOPIC = REPO / 'result/qwen3_8b_mmlu65_20261004'
NATIVE_RUNTIME = TOPIC / 'qwen_native_category_rotation/source_snapshot'
SPARSE_RUNTIME = TOPIC / 'rotation_factor_probe/source_snapshot_20261005_01'
EVALUATION = REPO / 'result/prepare_eval_20261030'


def resolve_legacy_path(value):
    """Translate a known source root only; never alter JSON, hashes or file stats."""
    path = Path(value)
    if path == Path('/home/star/miniconda3/envs/bitvae/bin/python'):
        return PYTHON
    for source, target in (
        (Path('/data/star/VAELLM'), ARTIFACTS),
        (Path('/home/star/VAELLM'), REPO),
        (Path('/home/star/models'), MODELS),
    ):
        if path.is_relative_to(source):
            return target / path.relative_to(source)
    if path.is_relative_to('/home/star') or path.is_relative_to('/data/star'):
        raise ValueError('No migration mapping is established for: ' + str(path))
    return path
