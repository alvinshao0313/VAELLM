"""Native fresh attention prefix or its own canonical pre-recovery B gate raw2."""
import argparse
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

from bindings import boundary_event, recipe, s, validate_raw2_meta


class PhaseComplete(Exception):
    """Raised only after the requested genuine native save has returned."""


def execute(request_path, phase):
    request = s.read(request_path)
    output = Path(request['output_dir'])
    short = request['validation_only']
    s.require(Path(sys.executable).resolve() == s.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(request['gpu'])
        and os.environ.get('PYTHONHASHSEED') == '31', 'Use the bound bitvae/single-GPU/seed environment.')
    s.check_pins(request['source_sha256'])
    s.require(s.hf_identity(request['base_model_path']) == request['hf_identity'], 'Original HF input changed.')
    after_o, parent_event = (None, None) if phase == 'prefix' else s.own_after_o(request)
    values = recipe.argv('B', request['base_model_path'], output / phase / 'native', after_o=after_o)
    cat, hf, training, vae, cfgs, identity, normalized = recipe.parse('B', values)
    from train_utils import cat_train_pipeline as native
    from train_utils import cat_after_category_common as recovery
    import torch
    s.require(Path(native.__file__).resolve().is_relative_to(s.SNAPSHOT), 'Import the unchanged N runtime only.')
    path = output / (phase + '.json')
    s.require(not path.exists(), 'Do not overwrite phase evidence.')
    expected = s.CATEGORIES[:4] if phase == 'prefix' else ('gate_proj',)
    record = dict(status='running', phase=phase, arm='B', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        started_unix=time.time(), command=[sys.executable, *sys.argv],
        request_sha256=s.sha(request_path), native_argv=values, runtime_identity=identity,
        normalized=normalized, validation_only=short, formal_parent_eligible=not short,
        parent_event=parent_event, seed_calls=[], actual_groups=[], saved_boundaries=[], calibration=None, recoveries=[])
    s.write(path, record)
    originals = {name: getattr(native, name) for name in ('set_seed', 'build_cat_cross_category_runtime_identity',
        'train_group_vae_payload', 'collect_activation_stats_for_linears', 'save_cat_v6_full_checkpoint', 'run_after_category_distill')}
    original_trainer = recovery.CanonicalCatSFTTrainer
    current_recovery = None

    def seeded(seed, *args, **kwargs):
        s.require(seed == 31 and not record['seed_calls'], 'Each process starts exactly once at seed31.')
        result = originals['set_seed'](seed, *args, **kwargs)
        record['seed_calls'].append(seed)
        return result

    def runtime_identity(**kwargs):
        actual = s.with_policy(originals['build_cat_cross_category_runtime_identity'](**kwargs), 'B')
        s.require(actual == identity, 'The live seven-category identity changed.')
        return actual

    def fit(**kwargs):
        cfg = kwargs['runtime_cfg']
        index = len(record['actual_groups'])
        s.require(index < len(expected) and cfg.category == expected[index] and cfg == cfgs[cfg.category]
            and len(kwargs['group_refs']) == 36
            and {ref.name for ref in kwargs['group_refs']} == s.projection_names((cfg.category,))
            and cfg.steps == 10000 and cfg.residual_stages == 2
            and kwargs['shuffle_seed'] == s.policy('B')['category_shuffle_seeds'][cfg.category],
            'Fit exactly the next full live M36 category using the original recipe.')
        batch = s.policy('B')['category_vector_batches'][cfg.category]
        kwargs['batch_size'] = batch
        if short:
            kwargs['runtime_cfg'] = replace(cfg, steps=2)
        event = dict(category=cfg.category, dim=cfg.codebook_dim, bits=cfg.codebook_bits, vector_batch=batch,
            scalars_per_linear_update=cfg.codebook_dim * batch, shuffle_seed=kwargs['shuffle_seed'],
            planned_steps_per_stage=10000, actual_steps_per_stage=2 if short else 10000,
            stages=2, status='fitting', started_unix=time.time())
        record['actual_groups'].append(event)
        s.write(path, record)
        result = originals['train_group_vae_payload'](**kwargs)
        event.update(status='fitted_and_encoded', elapsed_seconds=time.time() - event['started_unix'])
        s.write(path, record)
        return result

    def calibration(**kwargs):
        s.require(phase == 'prefix' and record['calibration'] is None and kwargs['cache'] is None
            and kwargs['dataset'] == 'alpaca=1' and kwargs['nsamples'] == 512 and kwargs['seqlen'] == 256
            and kwargs['seed'] == 31 and {name for name, _ in kwargs['linear_items']} == s.projection_names(s.CATEGORIES),
            'Generate fresh full512x256 train calibration for all 252 projections.')
        stats, cache = originals['collect_activation_stats_for_linears'](**kwargs)
        digest = hashlib.sha256()
        for tokens in cache.input_ids:
            digest.update(tokens.contiguous().numpy().tobytes())
        record['calibration'] = dict(dataset=cache.dataset, samples=len(cache.input_ids), seqlen=cache.seqlen,
            seed=cache.seed, stats_linears=len(stats), input_ids_sha256=digest.hexdigest())
        s.write(path, record)
        return stats, cache

    def distill(**kwargs):
        nonlocal current_recovery
        s.require(phase == 'prefix' and kwargs['category'] == expected[len(record['recoveries'])],
            'Only the four attention recoveries are enabled; stop before gate recovery.')
        current_recovery = dict(category=kwargs['category'], started_unix=time.time(), status='running')
        record['recoveries'].append(current_recovery)
        result = originals['run_after_category_distill'](**kwargs)
        current_recovery.update(status='completed', elapsed_seconds=time.time() - current_recovery['started_unix'])
        s.write(path, record)
        return result

    def save(*args, **kwargs):
        if short:
            annotation = dict(validation_only=True, actual_updates=2, schedule_max_steps=500, formal_parent_eligible=False)
            if kwargs.get('distill_stage_meta') is not None:
                kwargs['distill_stage_meta'] = {**kwargs['distill_stage_meta'], **annotation}
            kwargs['distill_stage_history'] = [{**item, **annotation} for item in kwargs['distill_stage_history']]
        result = originals['save_cat_v6_full_checkpoint'](*args, **kwargs)
        kind, category = kwargs['checkpoint_kind'], kwargs.get('category')
        if kind not in ('category_boundary', 'round_base'):
            return result
        event = boundary_event(result['output_dir'])
        s.require(Path(event['path']).is_relative_to(output / phase / 'native'), 'Save escaped the new phase.')
        record['saved_boundaries'].append(event)
        s.write(path, record)
        if (phase == 'prefix' and kind == 'category_boundary' and category == 'o_proj') or (
                phase == 'gate_raw2' and kind == 'round_base' and category == 'gate_proj'):
            meta = s.read(Path(event['path']) / 'checkpoint_meta.json')
            (s.validate_after_o_meta if phase == 'prefix' else validate_raw2_meta)(meta, request['base_model_path'])
            s.require([group['category'] for group in record['actual_groups']] == list(expected)
                and all(group['status'] == 'fitted_and_encoded' for group in record['actual_groups'])
                and record['seed_calls'] == [31], 'Incomplete true phase computation.')
            raise PhaseComplete()
        return result

    if short:
        from validation import RecoveryCheck

        class ShortTrainer(original_trainer):
            def __init__(self, *args, **kwargs):
                kwargs['callbacks'] = [*kwargs.get('callbacks', []), RecoveryCheck(current_recovery, current_recovery['category'])]
                super().__init__(*args, **kwargs)

    try:
        native.set_seed = seeded
        native.build_cat_cross_category_runtime_identity = runtime_identity
        native.train_group_vae_payload = fit
        native.collect_activation_stats_for_linears = calibration
        native.run_after_category_distill = distill
        native.save_cat_v6_full_checkpoint = save
        if short:
            recovery.CanonicalCatSFTTrainer = ShortTrainer
        native.run_cat_train(cat_args=cat, hf_args=hf, training_args=training, vae_args=vae)
        raise ValueError('Native execution returned without the requested real boundary.')
    except PhaseComplete:
        s.require(len(record['recoveries']) == (4 if phase == 'prefix' else 0), 'Unexpected recovery count.')
        record.update(status='prefix_completed' if phase == 'prefix' else 'raw2_completed', exit_code=0)
    except BaseException as error:
        record.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for name, original in originals.items():
            setattr(native, name, original)
        recovery.CanonicalCatSFTTrainer = original_trainer
        record.update(elapsed_seconds=time.time() - record['started_unix'],
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        s.write(path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--phase', choices=('prefix', 'gate_raw2'), required=True)
    args = parser.parse_args()
    execute(args.request.resolve(strict=True), args.phase)
