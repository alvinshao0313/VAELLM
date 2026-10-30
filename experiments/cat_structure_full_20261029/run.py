"""Run one fixed native CAT branch with an explicit per-category scalar budget."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import time
import traceback

import common as c
import recipe


class ShortComplete(Exception):
    pass


def require_gpu(gpu):
    import subprocess
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    devices = {int(row[0]): (row[1].strip(), int(row[2])) for line in text.splitlines() if (row := line.split(','))}
    uuid, used = devices[gpu]
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    c.require(used <= 256 and not any(line.split(',')[0].strip() == uuid for line in active.splitlines()),
        'Assigned GPU is occupied; do not wait, reserve or retry.')
    return dict(physical_gpu=gpu, uuid=uuid, used_MiB=used, checked_unix=time.time())


def execute(arm, phase, gpu, short_round_base=None):
    c.require(Path(sys.executable).resolve() == c.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu), 'Use caller-activated bitvae and exactly the assigned GPU.')
    c.require(phase == 'formal' or arm == 'B', 'Only B needs the isolated short2 check.')
    c.require(short_round_base is None or (phase == 'short2' and arm == 'B'), 'A round_base source is only for the isolated B fresh-recovery check.')
    prepared = c.read(c.OUTPUT / 'prepared.json')
    c.require(prepared['status'] == 'cpu_verified', 'Preparation must have completed its full actual-source checks.')
    c.check_pins(prepared['implementation_sha256']); c.check_pins(prepared['snapshot_source_sha256'])
    row = prepared['arms'][arm]
    boundary = Path(row['boundary'])
    c.require(c.sha(boundary / 'checkpoint_meta.json') == row['metadata_sha256']
        and c.sha(boundary / 'cat_runtime_state.pt') == row['runtime_sha256'], 'Fresh branch identity changed.')
    c.fork.check_stable_stat(row['weight_stat'])
    if phase == 'formal':
        short = c.read(c.OUTPUT / 'short2_validation.json')
        c.require(short['status'] == 'verified' and short['actual_steps'] == 2
            and short['preparation_sha256'] == c.sha(c.OUTPUT / 'prepared.json'), 'B formal-shape short evidence is missing or stale.')
    output = c.OUTPUT / arm
    report_path = output / ('short2_recovery.json' if short_round_base is not None else
                            'run.json' if phase == 'formal' else 'short2.json')
    c.require(not report_path.exists(), 'Never overwrite an existing execution record.')
    resources = require_gpu(gpu)
    from train_utils import cat_train_pipeline as native
    from train_utils import cat_after_category_common as recovery
    from train_utils.checkpoint_v6 import load_v6_meta
    from short_check import RecoveryCheck
    c.require(Path(native.__file__).resolve().is_relative_to(c.SNAPSHOT), 'Execution escaped the fixed original snapshot.')
    argv = list(row['native_argv'])
    if phase == 'short2':
        recipe.replace_option(argv, '--output_dir', output / ('short2_recovery_native' if short_round_base is not None else 'short2_native'))
    runtime = c.fork.load_v6_cat_runtime_state(str(boundary), map_location='cpu')
    cat, hf, training, vae, cfgs, identity, normalized = recipe.parse_recipe(arm, argv, runtime)
    c.require(identity == row['runtime_identity'], 'Actual native runtime differs from this new branch contract.')
    record = dict(status='running', arm=arm, phase=phase, command=[sys.executable, *sys.argv], native_argv=argv,
        source_checkpoint=str(boundary), source_checkpoint_id=row['checkpoint_id'],
        preparation_sha256=c.sha(c.OUTPUT / 'prepared.json'), experiment_runtime=recipe.policy(arm),
        resources=resources, pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        started_unix=time.time(), actual_groups=[], saved_boundaries=[], short_recovery={})
    record['expected_runtime_identity'] = identity
    c.write(report_path, record)
    original_fit = native.train_group_vae_payload
    original_identity = native.build_cat_cross_category_runtime_identity
    original_save = native.save_cat_v6_full_checkpoint
    original_trainer = recovery.CanonicalCatSFTTrainer

    def identity_with_policy(**kwargs):
        return recipe.add_experiment_identity(original_identity(**kwargs), arm)

    def fit_with_scalar_exposure(**kwargs):
        category = kwargs['runtime_cfg'].category
        refs = kwargs['group_refs']
        c.require(category in c.CATEGORIES[4:] and len(refs) == 36
            and {ref.name for ref in refs} == {f'model.layers.{i}.mlp.{category}' for i in range(36)},
            'Every actual category must fit all36 live same-branch matrices together.')
        batch = c.SCALARS_PER_LINEAR_UPDATE // kwargs['runtime_cfg'].codebook_dim
        c.require(batch == recipe.policy(arm)['category_vector_batches'][category]
            and kwargs['shuffle_seed'] == recipe.policy(arm)['category_shuffle_seeds'][category], 'Actual batch/shuffle violates the effective category contract.')
        kwargs['batch_size'] = batch
        if phase == 'short2':
            c.require(category == 'gate_proj', 'Validation must stop before any up/down fitting.')
            kwargs['runtime_cfg'] = replace(kwargs['runtime_cfg'], steps=2)
        record['actual_groups'].append(dict(category=category, linears=36, vector_dim=kwargs['runtime_cfg'].codebook_dim,
            batch_vectors_per_linear=batch, scalar_exposure_per_linear_update=batch*kwargs['runtime_cfg'].codebook_dim,
            steps_per_stage=kwargs['runtime_cfg'].steps, residual_stages=kwargs['runtime_cfg'].residual_stages,
            shuffle_seed=kwargs['shuffle_seed'], source='Actual live dense matrices in this branch; no donor or preencoded probe codes.'))
        c.write(report_path, record)
        return original_fit(**kwargs)

    def save_and_record(*positional, **kwargs):
        if phase == 'short2' and kwargs.get('checkpoint_kind') == 'category_boundary':
            evidence = record['short_recovery']
            c.require(evidence.get('actual_steps') == 2, 'Short recovery did not complete two actual updates.')
            annotation = dict(validation_only=True, actual_updates=2, schedule_max_steps=500, formal_parent_eligible=False)
            kwargs['distill_stage_meta'] = {**kwargs['distill_stage_meta'], **annotation}
            history = list(kwargs['distill_stage_history'])
            history[-1] = {**history[-1], **annotation}
            kwargs['distill_stage_history'] = history
        result = original_save(*positional, **kwargs)
        if kwargs.get('checkpoint_kind') in ('category_boundary', 'final_model'):
            path = Path(result['output_dir'])
            meta = load_v6_meta(str(path))
            event = dict(path=str(path), checkpoint_id=meta['checkpoint_id'], kind=meta['checkpoint_kind'],
                completed_categories=meta['completed_categories'], metadata_sha256=c.sha(path / 'checkpoint_meta.json'),
                runtime_sha256=c.sha(path / 'cat_runtime_state.pt'), weight_stat=c.fork.stat(path / meta['state_dict_file']))
            if phase == 'short2':
                event['state'] = c.fork.state_digest(positional[0].state_dict())
            record['saved_boundaries'].append(event)
            c.write(report_path, record)
            if phase == 'short2' and meta['checkpoint_kind'] == 'category_boundary':
                raise ShortComplete()
        return result

    class ShortTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            kwargs['callbacks'] = [*kwargs.get('callbacks', []), RecoveryCheck(record['short_recovery'])]
            super().__init__(*args, **kwargs)

    try:
        native.build_cat_cross_category_runtime_identity = identity_with_policy
        native.train_group_vae_payload = fit_with_scalar_exposure
        native.save_cat_v6_full_checkpoint = save_and_record
        if phase == 'short2':
            recovery.CanonicalCatSFTTrainer = ShortTrainer
        if short_round_base is not None:
            from short_recovery import recover
            recover(short_round_base, native=native, cat=cat, hf=hf, training=training, vae=vae,
                    record=record, save=save_and_record)
        else:
            native.run_cat_train(cat_args=cat, hf_args=hf, training_args=training, vae_args=vae)
        c.require(phase == 'formal' and record['saved_boundaries'][-1]['kind'] == 'final_model', 'Missing actual final native endpoint.')
        record.update(status='completed', exit_code=0)
    except ShortComplete:
        record.update(status='short2_complete_not_formal', exit_code=0, formal_parent_eligible=False)
    except BaseException as error:
        record.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        native.build_cat_cross_category_runtime_identity = original_identity
        native.train_group_vae_payload = original_fit
        native.save_cat_v6_full_checkpoint = original_save
        recovery.CanonicalCatSFTTrainer = original_trainer
        record['elapsed_seconds'] = time.time() - record['started_unix']
        c.write(report_path, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(recipe.ARMS), required=True)
    parser.add_argument('--phase', choices=('short2', 'formal'), required=True)
    parser.add_argument('--gpu', type=int, choices=(3, 4), required=True)
    parser.add_argument('--short-round-base', type=Path, help='Actual pre-update B validation round_base; fresh recovery only, no Trainer exact-resume claim.')
    args = parser.parse_args()
    execute(args.arm, args.phase, args.gpu, args.short_round_base)
