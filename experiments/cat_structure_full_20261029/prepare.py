"""Fork two actual after_o states on CPU; preserve all parent tensors and data."""
import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import common as c
import recipe
from transformers import AutoTokenizer
from train_utils.checkpoint_v6 import load_v6_cat_runtime_state, load_v6_meta, save_v6_full_checkpoint
from train_utils.cat_train_runtime import load_cat_resume_distill_progress


def verify(report_path, arm):
    record = c.read(report_path)
    row = record['arms'][arm]
    model, meta, loaded = c.fork.load_cpu_boundary(row['boundary'])
    c.require(not loaded.missing_keys and not loaded.unexpected_keys
        and meta['checkpoint_id'] == row['checkpoint_id'], 'Independent strict fork reload failed.')
    c.fork.check_model(model, meta)
    c.require(c.fork.state_digest(model.state_dict()) == record['source_state'], 'Fresh state differs from the real parent.')
    c.require(not c.fork.torch.cuda.is_initialized(), 'CPU fork reload initialized CUDA.')
    print({'fresh_strict_arm': arm, 'state': record['source_state']}, flush=True)


def prepare():
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path(sys.executable).resolve() == c.PYTHON.resolve(),
        'Use CPU-only caller-activated bitvae.')
    report_path = c.OUTPUT / 'prepared.json'
    c.require(not report_path.exists() and not any((c.OUTPUT / arm).exists() for arm in recipe.ARMS),
        'Never overwrite a preparation or an existing structure-full branch.')
    c.OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = dict(schema='cat_structure_full_fork_v1', status='preparing', arms={},
                  command=[sys.executable, *sys.argv], source=str(c.SOURCE), source_checkpoint_id=c.SOURCE_ID)
    try:
        previous, source_row, meta = c.source_binding()
        runtime = load_v6_cat_runtime_state(str(c.SOURCE), map_location='cpu')
        progress = load_cat_resume_distill_progress(str(c.SOURCE))
        c.require(list(progress.completed_categories) == list(c.CATEGORIES[:4]) and progress.lora_round_idx == 4,
            'Original category/recovery position changed.')
        record.update(source_metadata_sha256=source_row['metadata_sha256'], source_runtime_sha256=source_row['runtime_sha256'],
            source_weight_stat=source_row['weight_stat'], source_state=source_row['state'],
            snapshot_source_sha256=previous['snapshot_source_sha256'], implementation_sha256=c.implementation_pins(),
            parent_preparation=str(c.OLD_PREPARATION), parent_preparation_sha256=c.sha(c.OLD_PREPARATION))
        plans = {}
        for arm in recipe.ARMS:
            argv = recipe.training_argv(arm, source_row)
            cat, hf, training, vae, cfgs, identity, normalized = recipe.parse_recipe(arm, argv, runtime)
            changes = recipe.validate_fork(runtime['runtime_identity'], identity, arm)
            plans[arm] = (argv, identity, normalized, changes)
        model, loaded_meta, loaded = c.fork.load_cpu_boundary(c.SOURCE)
        c.require(not loaded.missing_keys and not loaded.unexpected_keys and loaded_meta == meta, 'Original parent did not load strict.')
        c.fork.check_model(model, meta)
        c.require(c.fork.state_digest(model.state_dict()) == record['source_state'], 'Actual parent state changed.')
        tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
        for arm, (argv, identity, normalized, changes) in plans.items():
            boundary = c.OUTPUT / arm / 'initial_after_o'
            fork_runtime = {**runtime, 'runtime_identity': identity}
            extra = deepcopy(meta['extra_meta'])
            extra['experiment_fork'] = dict(source_dir=str(c.SOURCE), source_checkpoint_id=c.SOURCE_ID,
                purpose='New full36 future-gate architecture; preserve actual144 attention and all live MLP tensors/data; not old exact-resume.',
                allowed_changes=changes, experiment_runtime=recipe.policy(arm), inherited_global_rng=False,
                inherited_history=True, inherited_state=record['source_state'])
            print({'event': 'save_independent_fork', 'arm': arm, 'path': str(boundary)}, flush=True)
            saved = save_v6_full_checkpoint(model, str(boundary), checkpoint_kind='category_boundary',
                compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
                skip_targets=meta['skip_targets'], legacy_original_only_sources=meta['legacy_original_only_sources'],
                train_mode=meta['train_mode'], after_category_mode=meta['after_category_mode'],
                norm_train_mode=meta['norm_train_mode'], lm_head_train_mode=meta['lm_head_train_mode'], lora_config=None,
                resolved_learning_rates=meta['resolved_learning_rates'], completed_categories=meta['completed_categories'],
                compression_categories=c.CATEGORIES, target_layers=meta['target_layers'], target_modules=meta['target_modules'],
                finalized_status=meta['finalized_status'], base_model_path=meta['base_model_path'], tokenizer=tokenizer,
                cat_runtime_state=fork_runtime, extra_meta=extra)
            actual_meta = load_v6_meta(str(boundary))
            c.require(saved['checkpoint_id'] != c.SOURCE_ID, 'Fork must have a new UUID.')
            for field in ('converted_modules', 'compressed_targets', 'pending_dense_targets', 'finalized_status'):
                c.require(actual_meta[field] == meta[field], 'Fork changed actual parent topology: ' + field)
            state = c.fork.torch.load(boundary / actual_meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
            c.require(c.fork.state_digest(state) == record['source_state'], 'Saved fork state differs from its parent.')
            del state
            c.fork.equal_nested(load_v6_cat_runtime_state(str(boundary), map_location='cpu'), fork_runtime)
            c.require(load_cat_resume_distill_progress(str(boundary)) == progress, 'Fork changed actual recovery history.')
            c.require(c.read(boundary / 'config.json') == c.read(c.SOURCE / 'config.json'), 'Fork changed saved model configuration.')
            record['arms'][arm] = dict(boundary=str(boundary), checkpoint_id=saved['checkpoint_id'], allowed_changes=changes,
                native_argv=argv, runtime_identity=identity, normalized=normalized,
                metadata_sha256=c.sha(boundary / 'checkpoint_meta.json'), runtime_sha256=c.sha(boundary / 'cat_runtime_state.pt'),
                weight_stat=c.fork.stat(boundary / actual_meta['state_dict_file']))
            c.write(report_path, record)
        del model
        gc.collect()
        subprocess.run([str(c.PYTHON), '-B', str(Path(__file__)), '--verify', 'B'], check=True, stdin=subprocess.DEVNULL)
        c.source_binding(); c.check_pins(record['implementation_sha256'])
        c.require(not c.fork.torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
        record.update(status='cpu_verified', exit_code=0, independent_fresh_strict_arm='B', CUDA_initialized=False)
    except BaseException as error:
        record.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        c.write(report_path, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', choices=tuple(recipe.ARMS))
    args = parser.parse_args()
    verify(c.OUTPUT / 'prepared.json', args.verify) if args.verify else prepare()
