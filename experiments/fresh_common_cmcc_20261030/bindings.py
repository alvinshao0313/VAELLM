"""New-machine bindings for the unchanged fresh recipe and native N runtime."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FRESH = ROOT / 'experiments/cat_structure_fresh_20261029'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FRESH))

import fresh_support as s
import recipe
import validate_cpu as original_cpu
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, MODELS, PYTHON

s.PYTHON = PYTHON
HF = MODELS / 'Qwen/Qwen3-8B'
_spec = importlib.util.spec_from_file_location('_unchanged_fresh_controller', FRESH / 'run.py')
_controller = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_controller)
gpu_available = _controller.gpu_available


def source_pins():
    pins = s.source_pins()
    for path in (*HERE.glob('*.py'), ROOT / 'experiments/cmcc_runtime_20261030/paths.py'):
        pins[str(path)] = s.sha(path)
    return pins


def boundary_event(checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    meta = s.read(checkpoint / 'checkpoint_meta.json')
    return dict(path=str(checkpoint), checkpoint_id=meta['checkpoint_id'], kind=meta['checkpoint_kind'],
        completed_categories=meta['completed_categories'], metadata_sha256=s.sha(checkpoint / 'checkpoint_meta.json'),
        runtime_sha256=s.sha(checkpoint / 'cat_runtime_state.pt'), weight_stat=s.stat(checkpoint / meta['state_dict_file']))


def validate_raw2_meta(meta, base):
    s.require(meta['checkpoint_kind'] == 'round_base'
        and meta['completed_categories'] == list(s.CATEGORIES[:4])
        and meta['compression_categories'] == list(s.CATEGORIES)
        and meta['converted_module_count'] == 180
        and set(meta['compressed_targets']) == s.projection_names(s.CATEGORIES[:5])
        and set(meta['pending_dense_targets']) == s.projection_names(s.CATEGORIES[5:])
        and meta['lora_config'] is None and not meta['skip_targets']
        and Path(meta['base_model_path']).resolve() == Path(base).resolve()
        and meta['extra_meta']['distill_stage'] is None,
        'Require the canonical complete B gate raw2, saved before its recovery.')
    history = meta['extra_meta']['distill_stage_history']
    s.require([item['category'] for item in history] == list(s.CATEGORIES[:4])
        and all(item['did_train'] and item['mode'] == 'remaining_lora_prefix_decoder' for item in history),
        'Raw2 must inherit all four genuine prefix recoveries.')
    gates = [item for item in meta['converted_modules'] if item['name'].endswith('.gate_proj')]
    s.require(len(gates) == 36 and all(item['residual_stages'] == 2
        and item['stage_codebook_dims'] == [64, 64]
        and all(stage['in_dim'] == 64 and stage['out_dim'] == 64
            and stage['decoder_type'] == 'asymmetric' for stage in item['stage_decoders']) for item in gates),
        'The common raw2 must be all 36 actual B asymmetric D64/K64/S2 gates.')
