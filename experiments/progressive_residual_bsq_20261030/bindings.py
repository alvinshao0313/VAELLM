"""Isolated bindings for BF16-origin, fixed four-stage native initialization."""
from pathlib import Path
import importlib.util
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import experiments

_path = ROOT / 'experiments/cat_structure_fresh_20261029/fresh_support.py'
_spec = importlib.util.spec_from_file_location('_progressive_fresh_support', _path)
s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s)
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, DATA, EVALUATION, MODELS, PYTHON

s.PYTHON = PYTHON
require = s.require
HF = MODELS / 'Qwen/Qwen3-8B'
BASELINE = ARTIFACTS / 'cmcc_baselines_20261030/qwen3_8b'
SAMPLE = ROOT / 'result/qwen3_8b_mmlu65_20261004/lm_eval_sample_artifacts_20261010'
MANIFEST = EVALUATION / 'longbench/manifest.json'
CATEGORIES = s.CATEGORIES
OUTPUT_ROOT = ARTIFACTS / 'progressive_residual_bsq_20261030'


def source_pins():
    paths = {*HERE.glob('*.py'), *HERE.glob('*.sh'), *s.SNAPSHOT.rglob('*.py'), _path,
        ROOT / 'experiments/cmcc_runtime_20261030/paths.py',
        ROOT / 'experiments/cmcc_runtime_20261030/sequence.py',
        ROOT / 'experiments/fresh_muon_cmcc_20261030/after_control.py',
        ROOT / 'experiments/cat_aligned_evaluation_20261028/evaluate.py',
        ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_aux_retention32_batch_20261011/evaluate.py',
        ROOT / 'result/qwen3_8b_mmlu65_20261004/llama_fullqk_qkonly2000_longbench21/bind_terminal.py',
        ROOT / 'result/qwen3_8b_mmlu65_20261004/llama_fullqk_qkonly2000_longbench21/merge_full21.py',
        SAMPLE / 'cat_eval.py', SAMPLE / 'eval_utils_with_samples.py', SAMPLE / 'sample_artifacts.py'}
    for directory in ('experiments/longbench_v1', 'experiments/w25_eval'):
        paths.update((ROOT / directory).rglob('*.py'))
    return {str(path): s.sha(path) for path in sorted(paths)}


def environment(gpu):
    value = s.worker_environment(gpu)
    value['PYTHONPATH'] = str(ROOT) + ':' + str(s.SNAPSHOT)
    value['MKL_NUM_THREADS'] = '4'
    value['TRANSFORMERS_OFFLINE'] = '1'
    return value


def process_event(pid):
    return dict(pid=pid, start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate_meta(meta, category=None):
    completed = CATEGORIES if category is None else CATEGORIES[:CATEGORIES.index(category) + 1]
    s.require(meta['format'] == 'vaellm_model_checkpoint_v6'
        and meta['completed_categories'] == list(completed)
        and meta['compression_categories'] == list(CATEGORIES)
        and set(meta['compressed_targets']) == s.projection_names(completed)
        and set(meta['pending_dense_targets']) == s.projection_names(CATEGORIES[len(completed):])
        and meta['converted_module_count'] == 36 * len(completed)
        and meta['lora_config'] is None and not meta['skip_targets']
        and Path(meta['base_model_path']).resolve() == HF.resolve()
        and meta['extra_meta']['distill_stage'] is None
        and not meta['extra_meta']['distill_stage_history'], 'Wrong BF16-origin no-recovery cumulative boundary.')
    for item in meta['converted_modules']:
        cat = item['name'].rsplit('.', 1)[1]
        dim = 64 if cat == 'gate_proj' else 32
        s.require(item['residual_stages'] == 4 and item['stage_codebook_dims'] == [dim] * 4
            and all(stage['in_dim'] == dim and stage['out_dim'] == dim for stage in item['stage_decoders']),
            'Every converted projection must retain all four K=D stages.')


def boundary(directory):
    directory = Path(directory).resolve(strict=True)
    meta = s.read(directory / 'checkpoint_meta.json')
    validate_meta(meta, meta['extra_meta']['category'] if meta['checkpoint_kind'] == 'category_boundary' else None)
    return dict(path=str(directory), checkpoint_id=meta['checkpoint_id'],
        metadata_sha256=s.sha(directory / 'checkpoint_meta.json'),
        runtime_sha256=s.sha(directory / 'cat_runtime_state.pt'),
        weight_stat=s.stat(directory / meta['state_dict_file']))


def check_environment(gpu):
    s.require(Path(sys.executable).resolve() == PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu), 'Use caller-activated bitvae and its assigned GPU.')


s.enable_snapshot()
