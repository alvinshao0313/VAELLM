"""Real CPU native forks and independent full-state strict readback."""
import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bindings as b


def load(directory):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    from rotation.model_utils import temporarily_disable_torch_weight_init
    from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    with no_init_weights(), temporarily_disable_torch_weight_init():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=b.torch.bfloat16, attn_implementation='eager')
    model.seqlen = 2048
    return load_v6_full_checkpoint_into_model(model, str(directory), map_location='cpu', strict=True,
        expected_kind=b.read(directory / 'checkpoint_meta.json')['checkpoint_kind'])


def verify(directory, expected, output, arm):
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Strict readback is CPU only.')
    model, meta, result = load(directory)
    digest = b.fork.state_digest(model.state_dict())
    b.require(not result.missing_keys and not result.unexpected_keys and digest == expected
        and not b.torch.cuda.is_initialized(), 'Independent full native strict readback failed.')
    record = dict(status='verified_native_full_state_strict', exit_code=0, checkpoint_id=meta['checkpoint_id'],
        checkpoint_dir=str(directory), full_state=digest, persistent_fields=digest['tensors'],
        CUDA_initialized=False, expected_state_equal=True)
    if meta['checkpoint_kind'] == 'round_base':
        b.validate_raw2(meta, arm)
        parent = Path(b.sources()['after_o']['path'])
        parent_meta = b.read(parent / 'checkpoint_meta.json')
        parent_state = b.torch.load(parent / parent_meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
        state = model.state_dict()
        prefixes = tuple(f'model.layers.{index}.mlp.gate_proj.' for index in range(36))
        actual = {name: value for name,value in state.items() if not name.startswith(prefixes)}
        reference = {name: value for name,value in parent_state.items() if not name.startswith(prefixes)}
        b.require(len(actual) == len(reference) == 2091 and actual.keys() == reference.keys(),
            'Native raw2 non-gate complement field set changed.')
        for name, value in actual.items():
            source = reference[name]
            b.require(value.dtype == source.dtype and value.shape == source.shape and b.torch.equal(value, source),
                'Native raw2 changed a shared-parent non-gate field: ' + name)
        record.update(non_gate_after_o_complement_exact=True, non_gate_tensor_count=2091,
            non_gate_tensor_digest=b.fork.state_digest(actual), after_o=b.event(parent))
        from .ledger import measure
        record['partial_storage'] = measure(meta, state)
        record['partial_storage']['serialized_file_bytes'] = (directory / meta['state_dict_file']).stat().st_size
    b.write(output, record)
    return record


def prepare():
    from transformers import AutoTokenizer
    from train_utils.checkpoint_v6 import save_v6_full_checkpoint
    from train_utils.cat_train_runtime import load_cat_resume_distill_progress
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not b.OUTPUT.exists(), 'Use CPU bitvae and a new artifact root.')
    from .validate_cpu import validate
    geometry = validate()
    started, binding = time.time(), b.sources()
    parent = Path(binding['after_o']['path'])
    meta = b.read(parent / 'checkpoint_meta.json')
    runtime = b.torch.load(parent / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    cache = runtime['activation_runtime']['cache']
    b.require(cache['nsamples'] == len(cache['input_ids']) == 512 and cache['seqlen'] == 256
        and len(runtime['activation_runtime']['stats_by_linear']) == 252, 'Require the existing genuine full train calibration.')
    model, loaded_meta, loaded = load(parent)
    b.require(loaded_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys,
        'Real common after_o failed strict native loading.')
    raw = b.torch.load(parent / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    digest = b.fork.state_digest(raw)
    b.require(b.fork.state_digest(model.state_dict()) == digest, 'Parent strict load changed values.')
    del raw
    progress = load_cat_resume_distill_progress(str(parent))
    tokenizer = AutoTokenizer.from_pretrained(b.HF, local_files_only=True)
    record = dict(status='preparing', schema='cat_gate_geometry_cpu_v1', source_binding=binding,
        source_state=digest, source_sha256=b.pins(), arms={}, geometry_cpu=geometry, started_unix=started,
        reused_shape_and_native_path_evidence=str(b.COMMON / 'boundary_verification.json'),
        new_short_required=True,
        reason='New E256 capacity and D128/K128 vector geometry each require native M36, formal batch, S2 two-update short validation.')
    b.OUTPUT.mkdir(parents=True)
    for arm in b.ARMS:
        record['arms'][arm] = {}
        for short in (False, True):
            phase = 'short' if short else 'formal'
            argv, _, identity, normalized = b.recipe(arm, b.OUTPUT / arm / phase / 'native', short=short)
            changed = deepcopy(runtime['runtime_identity'])
            settings = b.ARMS[arm]
            changed['resolved_category_runtime']['gate_proj'].update(codebook_dim=settings['dim'],
                codebook_bits=settings['bits'], base_ch=settings['encoder'], steps=2 if short else settings['steps'])
            changed['experiment_runtime'] = identity['experiment_runtime']
            b.require(changed == identity, 'Fork identity changed outside the approved shape/capacity factor.')
            fork_runtime = deepcopy(runtime)
            fork_runtime['runtime_identity'] = identity
            extra = deepcopy(meta['extra_meta'])
            extra['experiment_fork'] = dict(schema='cat_gate_geometry_fork_v1', arm=arm, phase=phase,
                source_binding=binding, source_state=digest, settings=settings, inherited_global_rng=False,
                purpose='Fresh native gate fit; unchanged actual parent state/history, no gate recovery or live suffix.')
            boundary = b.OUTPUT / arm / ('short_initial_after_o' if short else 'initial_after_o')
            save_v6_full_checkpoint(model, str(boundary), checkpoint_kind='category_boundary',
                compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
                skip_targets=meta['skip_targets'], legacy_original_only_sources=meta['legacy_original_only_sources'],
                train_mode=meta['train_mode'], after_category_mode=meta['after_category_mode'],
                norm_train_mode=meta['norm_train_mode'], lm_head_train_mode=meta['lm_head_train_mode'], lora_config=None,
                resolved_learning_rates=meta['resolved_learning_rates'], completed_categories=meta['completed_categories'],
                compression_categories=b.CATEGORIES, target_layers=meta['target_layers'], target_modules=meta['target_modules'],
                finalized_status=meta['finalized_status'], base_model_path=str(b.HF), tokenizer=tokenizer,
                cat_runtime_state=fork_runtime, extra_meta=extra)
            saved_meta = b.read(boundary / 'checkpoint_meta.json')
            state = b.torch.load(boundary / saved_meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
            b.require(b.fork.state_digest(state) == digest and load_cat_resume_distill_progress(str(boundary)) == progress,
                'Saved fork changed inherited full state or history.')
            b.fork.equal_nested(b.torch.load(boundary / 'cat_runtime_state.pt', map_location='cpu', weights_only=True), fork_runtime)
            record['arms'][arm][phase] = dict(native_argv=argv, runtime_identity=identity, normalized=normalized,
                fork=b.event(boundary), full_state_equal=True, recovery_history_equal=True, settings=settings)
            b.write(b.RESULT / 'prepared.json', record)
            del state, fork_runtime
    del model, runtime
    gc.collect()
    # Every actual fork was compared in full; topology is unchanged before fitting.
    subprocess.run([str(b.PYTHON), '-B', '-u', '-m', 'experiments.cat_geometry_cohort_cmcc_20261102.prepare',
        '--verify', str(b.OUTPUT / 'encoder256/initial_after_o'), '--expected', str(b.RESULT / 'prepared.json'),
        '--output', str(b.RESULT / 'fork_fresh.json'), '--arm', 'encoder256'],
        cwd=b.REPO, check=True, stdin=subprocess.DEVNULL)
    control = b.read(b.CONTROL_EVIDENCE / 'control_fresh.json')
    b.require(control['status'] == 'verified_native_full_state_strict'
        and control['checkpoint_id'] == binding['control']['checkpoint_id']
        and control['non_gate_after_o_complement_exact'] and control['non_gate_tensor_count'] == 2091,
        'Require the completed shared control strict evidence.')
    paths = [b.CONTROL_EVIDENCE / 'control_fresh.json']
    control_root = b.ARTIFACTS / 'cat_hyperparam_cohort_cmcc_20261102/control'
    for name, status in (('mmlu.json', 'completed_full_mmlu'), ('readout.json', 'completed_deployment_readout')):
        actual = b.read(control_root / name)
        b.require(actual['status'] == status and actual['checkpoint_id'] == control['checkpoint_id'],
            'Require the existing completed same-node control evaluation.')
        paths.append(control_root / name)
    record['shared_control_evidence'] = {str(path): b.sha(path) for path in paths}
    b.require(b.sources() == binding and not b.torch.cuda.is_initialized(), 'Parent changed or CPU initialized CUDA.')
    b.check_pins(record['source_sha256'])
    record.update(status='cpu_verified', exit_code=0, elapsed_seconds=time.time()-started,
        representative_fresh=str(b.RESULT / 'fork_fresh.json'))
    b.write(b.RESULT / 'prepared.json', record)
    print(dict(status=record['status'], elapsed_seconds=record['elapsed_seconds']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arm', choices=tuple(b.ARMS))
    args = parser.parse_args()
    b.torch.set_num_threads(8)
    if args.verify:
        report = b.read(args.expected)
        verify(args.verify, report.get('source_state', report.get('state')), args.output, args.arm)
    else:
        prepare()


if __name__ == '__main__':
    main()
