"""Use the category-aware N runtime without modifying public or snapshot code."""
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parents[2] / 'result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation/source_snapshot'
if not RUNTIME.is_dir():
    raise FileNotFoundError('This isolated prototype requires the category-aware native code snapshot: ' + str(RUNTIME))
sys.path.insert(0, str(RUNTIME))

# This is an exact runtime dependency, not an alternate-factor fallback or a
# mutation of the shared public implementation. A prior incompatible import fails.
from rotation import hadamard_utils as _hadamard

if Path(_hadamard.__file__).resolve() != RUNTIME / 'rotation/hadamard_utils.py':
    raise RuntimeError('Import adaptive_native_model_20261029 before a conflicting rotation runtime.')
