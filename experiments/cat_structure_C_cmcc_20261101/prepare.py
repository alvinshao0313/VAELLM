"""Strict CPU fork of the real original after_o; historical records stay intact."""
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
    from rotation.model_utils import temporarily_disable_torch_weight_init
    from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    with temporarily_disable_torch_weight_init():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=b.torch.bfloat16, attn_implementation='eager')
    model.seqlen = 2048
    return load_v6_full_checkpoint_into_model(model, str(directory), map_location='cpu', strict=True,
        expected_kind=b.read(directory / 'checkpoint_meta.json')['checkpoint_kind'])


def verify(directory, expected, output):
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Fresh strict readback uses CPU.')
    model, meta, result = load(directory)
    digest = b.fork.state_digest(model.state_dict())
    b.require(not result.missing_keys and not result.unexpected_keys and digest == expected
        and not b.torch.cuda.is_initialized(), 'Independent native full-state strict readback failed.')
    record = dict(status='verified_native_full_state_strict', exit_code=0, checkpoint_id=meta['checkpoint_id'],
        checkpoint_dir=str(directory), persistent_fields=digest['tensors'], full_state=digest,
        CUDA_initialized=False, expected_state_equal=True)
    b.write(output, record)
    return record


def prepare():
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.OUTPUT.exists(),
        'Use CPU bitvae and a new C artifact directory.')
    from transformers import AutoTokenizer
    from train_utils.checkpoint_v6 import save_v6_full_checkpoint
    from train_utils.cat_runtime_state_v6 import validate_cat_runtime_identity
    from train_utils.cat_train_runtime import load_cat_resume_distill_progress
    started = time.monotonic()
    source = b.source_binding()
    previous, meta = b.read(b.BLUEPRINT), b.read(b.SOURCE / 'checkpoint_meta.json')
    runtime = b.torch.load(b.SOURCE / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    cache = runtime['activation_runtime']['cache']
    b.require(cache['nsamples'] == len(cache['input_ids']) == 512 and cache['seqlen'] == 256
        and all(value.numel() == 256 for value in cache['input_ids'])
        and len(runtime['activation_runtime']['stats_by_linear']) == 252,
        'Require the real full512x256 parent calibration and all252 statistics.')
    argv, _, identity, normalized = b.recipe(b.OUTPUT / 'formal/native')
    allowed = ('base_ch', 'decoder_type', 'decoder_base_ch', 'decoder_num_res_blocks')
    expected, changes = deepcopy(runtime['runtime_identity']), {}
    for field in allowed:
        before = expected['resolved_category_runtime']['gate_proj'][field]
        after = identity['resolved_category_runtime']['gate_proj'][field]
        if before != after:
            changes['gate_proj.' + field] = dict(source=before, fork=after)
            expected['resolved_category_runtime']['gate_proj'][field] = after
    expected['experiment_runtime'] = b.policy()
    b.require(expected == identity and changes['gate_proj.base_ch'] == dict(source=128, fork=256),
        'The fork changed something outside the explicit future gate architecture.')
    try:
        validate_cat_runtime_identity(runtime['runtime_identity'], identity)
    except ValueError:
        strict_rejected = True
    else:
        raise ValueError('Original exact resume must reject the changed future architecture.')
    # Negative actual-identity tests do not construct artificial models or data.
    for category, field in (('q_proj', 'base_ch'), ('up_proj', 'codebook_bits')):
        trial = deepcopy(identity)
        trial['resolved_category_runtime'][category][field] += 1
        b.require(trial != expected, 'Unrelated identity mutation escaped exact comparison.')
    model, loaded_meta, loaded = b.fork.load_cpu_boundary(b.SOURCE)
    b.require(loaded_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys,
        'The actual original parent did not load strictly.')
    b.fork.check_model(model, meta)
    digest = b.fork.state_digest(model.state_dict())
    b.require(digest == source['expected_original_state'], 'Actual parent state differs from original A/B evidence.')
    progress = load_cat_resume_distill_progress(str(b.SOURCE))
    fork_runtime = b.rebind(runtime)
    fork_runtime['runtime_identity'] = identity
    b.require(b.rebind(runtime['runtime_identity']) == runtime['runtime_identity'], 'Path mapping changed mathematical identity.')
    extra = deepcopy(meta['extra_meta'])
    extra['experiment_fork'] = dict(schema='original_C_actual_fork_v1', source_binding=source,
        training_contract=b.CONTRACT, allowed_changes=changes, inherited_global_rng=False,
        inherited_state=digest, purpose='Original C E256/D128; native M36 continuation, not exact old architecture resume.')
    b.OUTPUT.mkdir(parents=True)
    boundary = b.OUTPUT / 'initial_after_o'
    saved = save_v6_full_checkpoint(model, str(boundary), checkpoint_kind='category_boundary',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
        skip_targets=meta['skip_targets'], legacy_original_only_sources=meta['legacy_original_only_sources'],
        train_mode=meta['train_mode'], after_category_mode=meta['after_category_mode'],
        norm_train_mode=meta['norm_train_mode'], lm_head_train_mode=meta['lm_head_train_mode'], lora_config=None,
        resolved_learning_rates=meta['resolved_learning_rates'], completed_categories=meta['completed_categories'],
        compression_categories=b.CATEGORIES, target_layers=meta['target_layers'], target_modules=meta['target_modules'],
        finalized_status=meta['finalized_status'], base_model_path=str(b.HF),
        tokenizer=AutoTokenizer.from_pretrained(b.HF, local_files_only=True), cat_runtime_state=fork_runtime, extra_meta=extra)
    b.require(saved['checkpoint_id'] != b.SOURCE_ID
        and load_cat_resume_distill_progress(str(boundary)) == progress, 'Fork changed the inherited recovery history.')
    b.fork.equal_nested(b.torch.load(boundary / 'cat_runtime_state.pt', map_location='cpu', weights_only=True), fork_runtime)
    del model, runtime, fork_runtime
    gc.collect()
    record = dict(schema='original_C_actual_fork_preparation_v1', status='preparing', source_binding=source,
        originalA_control=source['originalA_control'], training_contract=b.CONTRACT, source_state=digest,
        native_argv=argv, runtime_identity=identity, normalized=normalized, allowed_changes=changes,
        original_exact_resume_rejected=strict_rejected, fork=b.event(boundary), source_sha256=b.pins())
    b.write(b.RESULT / 'prepared.json', record)
    subprocess.run([str(b.PYTHON), '-B', '-u', '-m', 'experiments.cat_structure_C_cmcc_20261101.prepare',
        '--verify', str(boundary), '--expected', str(b.RESULT / 'prepared.json'),
        '--output', str(b.RESULT / 'fork_fresh.json')], cwd=b.REPO, check=True, stdin=subprocess.DEVNULL)
    b.require(b.source_binding() == source, 'Actual parent changed while preparing the fork.')
    b.check_pins(record['source_sha256'])
    record.update(status='cpu_verified', exit_code=0, elapsed_seconds=time.monotonic()-started,
        fork_fresh=dict(path=str(b.RESULT / 'fork_fresh.json'), sha256=b.sha(b.RESULT / 'fork_fresh.json')))
    b.write(b.RESULT / 'prepared.json', record)
    print(dict(status=record['status'], elapsed_seconds=record['elapsed_seconds']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    b.torch.set_num_threads(8)
    if args.verify:
        value = b.read(args.expected)
        verify(args.verify, value.get('source_state', value.get('state')), args.output)
    else:
        prepare()


if __name__ == '__main__':
    main()
