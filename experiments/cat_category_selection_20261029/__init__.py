"""Read-only A gate250/500 deployment comparison; current CAT runs stay frozen."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation/source_snapshot'
sys.path.insert(0, str(RUNTIME))
from rotation import hadamard_utils as _hadamard

if Path(_hadamard.__file__).resolve() != RUNTIME / 'rotation/hadamard_utils.py':
    raise RuntimeError('Import the explicit N runtime before any conflicting snapshot.')
