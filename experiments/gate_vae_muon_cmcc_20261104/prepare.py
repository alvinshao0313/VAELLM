"""Verify the original gate source, resolved recipe and new optimizer ownership."""
import ast
from copy import deepcopy
import gc
import os
from pathlib import Path
import sys
import time

from . import bindings as b


def main():
    from experiments.cat_geometry_cohort_cmcc_20261102.prepare import load
    from transformers import AutoTokenizer
    from train_utils.checkpoint_v6 import save_v6_full_checkpoint
    from train_utils.cat_train_runtime import load_cat_resume_distill_progress
    from litebsq.llm_vae import MultiLayerVAE
    from .optimizer import derive_layout, matrices

    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and Path(sys.executable).resolve() == b.PYTHON.resolve() and not b.OUTPUT.exists()
        and not (b.RESULT / 'prepared.json').exists(), 'Use activated CPU bitvae and new gate output roots.')
    b.torch.set_num_threads(8)
    started, binding = time.time(), b.base.sources()
    b.RESULT.mkdir(parents=True, exist_ok=True)
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    parent = Path(binding['after_o']['path'])
    meta = b.read(parent / 'checkpoint_meta.json')
    runtime = b.torch.load(parent / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    cache = runtime['activation_runtime']['cache']
    b.require(cache['dataset'] == 'alpaca=1' and cache['nsamples'] == len(cache['input_ids']) == 512
        and cache['seqlen'] == 256 and len(runtime['activation_runtime']['stats_by_linear']) == 252,
        'Reuse the genuine original train512x256 calibration and all live statistics.')
    recipes = {}
    for short in (False, True):
        phase = 'short' if short else 'formal'
        argv, parsed, identity, normalized = b.recipe('muon', b.OUTPUT / 'muon' / phase / 'native', short=short)
        recipes[phase] = dict(native_argv=argv, runtime_identity=identity, normalized=normalized)
    with b.torch.random.fork_rng(devices=[]):
        b.torch.manual_seed(31)
        vae = MultiLayerVAE(b.stage_args(parsed))
        named, layout, affine = derive_layout(vae)
        owner_checks = []
        for name, spec in layout.items():
            actual = matrices(named[name], spec)
            b.require(actual.shape == tuple(spec['matrix_shape']) and actual.untyped_storage().data_ptr()
                == named[name].untyped_storage().data_ptr() and actual[0].numel() == 8192,
                'Gate owner conversion must preserve the original packed storage and independent matrix axis.')
            owner_checks.append(dict(name=name, layout=spec, shared_original_storage=True, matrix_elements=8192))
        geometry = dict(parameter_elements=sum(p.numel() for p in named.values()), matrix_elements=1179648,
            auxiliary_elements=32256, leaves=12, matrices=144, owner_checks=owner_checks, auxiliary_names=affine,
            boundary='CPU layout fixture only; GPU stages calibrate their own real first gradients. No historical control digest exists.')
        del vae, named
    model, actual_meta, loaded = load(parent)
    b.require(actual_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys, 'Strict native after-o loading failed.')
    state = b.torch.load(parent / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    digest = b.base.fork.state_digest(state)
    b.require(b.base.fork.state_digest(model.state_dict()) == digest, 'Native parent loading changed a persistent value.')
    del state
    b.OUTPUT.mkdir(parents=True)
    (b.OUTPUT / 'muon').mkdir()
    fork_runtime = deepcopy(runtime)
    fork_runtime['runtime_identity'] = recipes['short']['runtime_identity']
    extra = deepcopy(meta['extra_meta'])
    extra['experiment_fork'] = dict(schema='gate_VAE_Muon_functional_short_v1', source_binding=binding,
        formal_parent_eligible=False, inherited_global_rng=False, contract=b.CONTRACT)
    boundary = b.OUTPUT / 'muon/short_initial_after_o'
    save_v6_full_checkpoint(model, str(boundary), checkpoint_kind='category_boundary',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
        skip_targets=meta['skip_targets'], legacy_original_only_sources=meta['legacy_original_only_sources'],
        train_mode=meta['train_mode'], after_category_mode=meta['after_category_mode'], norm_train_mode=meta['norm_train_mode'],
        lm_head_train_mode=meta['lm_head_train_mode'], lora_config=None, resolved_learning_rates=meta['resolved_learning_rates'],
        completed_categories=meta['completed_categories'], compression_categories=b.CATEGORIES,
        target_layers=meta['target_layers'], target_modules=meta['target_modules'], finalized_status=meta['finalized_status'],
        base_model_path=str(b.HF), tokenizer=AutoTokenizer.from_pretrained(b.HF, local_files_only=True),
        cat_runtime_state=fork_runtime, extra_meta=extra)
    saved = b.read(boundary / 'checkpoint_meta.json')
    state = b.torch.load(boundary / saved['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    b.require(b.base.fork.state_digest(state) == digest
        and load_cat_resume_distill_progress(str(boundary)) == load_cat_resume_distill_progress(str(parent)),
        'The functional short fork changed the parent full state or recovery history.')
    b.base.fork.equal_nested(b.torch.load(boundary / 'cat_runtime_state.pt', map_location='cpu', weights_only=True), fork_runtime)
    recipes['short']['fork'] = b.base.event(boundary)
    recipes['formal']['fork'] = binding['after_o']
    control = b.read(b.CONTROL_FRESH)
    control_m, control_r = b.read(b.CONTROL_ROOT / 'mmlu.json'), b.read(b.CONTROL_ROOT / 'readout.json')
    b.require(control['status'] == 'verified_native_full_state_strict' and control['checkpoint_id'] == b.CONTRACT['control']
        and control['non_gate_after_o_complement_exact'] and control['non_gate_tensor_count'] == 2091
        and control_m['status'] == 'completed_full_mmlu' and control_r['status'] == 'completed_deployment_readout'
        and control_m['checkpoint_id'] == control_r['checkpoint_id'] == control['checkpoint_id'],
        'Reuse the completed genuine control MMLU/readout/fresh; only seven short tasks are missing.')
    record = dict(status='cpu_verified', exit_code=0, source_binding=binding, source_state=digest,
        source_sha256=b.pins(), arms=dict(muon=recipes), contract=b.CONTRACT, optimizer_geometry=geometry,
        shared_control_evidence={str(path): b.sha(path) for path in
            (b.CONTROL_FRESH, b.CONTROL_ROOT / 'mmlu.json', b.CONTROL_ROOT / 'readout.json')},
        calibration=dict(dataset=cache['dataset'], samples=512, seqlen=256, seed=cache['seed'], split='train'),
        formal_direct_original_parent=True, CUDA_initialized=b.torch.cuda.is_initialized(),
        started_unix=started, elapsed_seconds=time.time()-started)
    b.require(not b.torch.cuda.is_initialized() and b.base.sources() == binding, 'CPU preparation changed the source or initialized CUDA.')
    b.base.check_pins(record['source_sha256'])
    b.write(b.RESULT / 'prepared.json', record)
    print(dict(status=record['status'], elapsed_seconds=record['elapsed_seconds'], geometry=geometry), flush=True)


if __name__ == '__main__':
    main()
