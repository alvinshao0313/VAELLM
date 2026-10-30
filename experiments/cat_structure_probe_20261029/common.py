"""Shared identity and four fixed CAT architecture candidates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = Path('/data/star/VAELLM/qwen_recompression_20261026/C/initial_after_o')
SOURCE_ID = 'b63bfe79-3889-40c6-bf66-1d002893f874'
SNAPSHOT = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation/source_snapshot'
TOKENS = ROOT / 'result/qwen_cat_biaxial_20261022/prepared/tokens.pt'
TOKEN_MANIFEST = TOKENS.with_name('manifest.json')
OUTPUT = Path('/data/star/VAELLM/cat_structure_probe_20261029')
LAYERS = (0, 18, 35)
ARMS = {'A': (32, 32, 128, 128, 8192), 'B': (64, 64, 128, 128, 4096),
        'C': (32, 32, 256, 128, 8192), 'D': (32, 32, 128, 256, 8192)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stat(path):
    path = Path(path).resolve()
    value = path.stat()
    return dict(path=str(path), bytes=value.st_size, inode=value.st_ino, mtime_ns=value.st_mtime_ns)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def source():
    meta = read(SOURCE/'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID and meta['checkpoint_kind'] == 'category_boundary'
            and len(meta['compressed_targets']) == 144 and len(meta['pending_dense_targets']) == 108,
            'Use the actual completed attention prefix and live dense MLP boundary.')
    return meta
