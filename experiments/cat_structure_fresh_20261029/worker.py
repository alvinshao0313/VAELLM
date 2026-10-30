"""One native process: fresh attention prefix or the same run's MLP suffix."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

import fresh_support as s
import recipe


class PrefixComplete(Exception):
    """The true native after_o save has returned; terminate this process."""


def execute(request_path, phase):
    request = s.read(request_path)
    output = Path(request['output_dir'])
    arm = request['arm']
    s.require(Path(sys.executable).resolve() == s.PYTHON.resolve()
              and os.environ.get('CUDA_VISIBLE_DEVICES') == str(request['gpu'])
              and os.environ.get('PYTHONHASHSEED') == '31', 'Use the recorded bitvae/single-GPU/seed environment.')
    s.check_pins(request['source_sha256'])
    s.require(s.hf_identity(request['base_model_path']) == request['hf_identity'], 'Original HF input identity changed.')
    after_o, parent_event = (None, None) if phase == 'prefix' else s.own_after_o(request)
    values = recipe.argv(arm, request['base_model_path'], output / phase / 'native', after_o=after_o)
    cat, hf, training, vae, cfgs, identity, normalized = recipe.parse(arm, values)
    s.require(cat.resume_from_checkpoint == (None if after_o is None else str(after_o)), 'Unexpected initial checkpoint.')
    from train_utils import cat_train_pipeline as native
    s.enable_snapshot()
    path = output / (phase + '.json')
    s.require(not path.exists(), 'Never overwrite a phase execution record.')
    record = dict(status='running', phase=phase, arm=arm, pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        command=[sys.executable, *sys.argv], started_unix=time.time(),
        request_sha256=s.sha(request_path), native_argv=values, runtime_identity=identity,
        normalized=normalized, parent_event=parent_event, seed_calls=[], actual_groups=[],
        saved_boundaries=[], calibration=None)
    s.write(path, record)
    original_identity = native.build_cat_cross_category_runtime_identity
    original_fit = native.train_group_vae_payload
    original_save = native.save_cat_v6_full_checkpoint
    original_seed = native.set_seed
    original_calibration = native.collect_activation_stats_for_linears
    expected = s.CATEGORIES[:4] if phase == 'prefix' else s.CATEGORIES[4:]

    def seeded(seed, *args, **kwargs):
        s.require(seed == 31 and not record['seed_calls'], 'Each native process starts once with seed31.')
        result = original_seed(seed, *args, **kwargs)
        record['seed_calls'].append(seed)
        s.write(path, record)
        return result

    def identity_with_policy(**kwargs):
        actual = s.with_policy(original_identity(**kwargs), arm)
        s.require(actual == identity, 'The live native seven-category runtime differs from the parsed recipe.')
        return actual

    def fit(**kwargs):
        cfg = kwargs['runtime_cfg']
        index = len(record['actual_groups'])
        s.require(index < len(expected) and cfg.category == expected[index]
                  and len(kwargs['group_refs']) == 36
                  and {ref.name for ref in kwargs['group_refs']} == s.projection_names((cfg.category,)),
                  'Fit the complete next M36 live category in this process, without donor payloads.')
        batch = s.policy(arm)['category_vector_batches'][cfg.category]
        s.require(cfg == cfgs[cfg.category] and cfg.steps == 10000 and cfg.residual_stages == 2
                  and kwargs['shuffle_seed'] == s.policy(arm)['category_shuffle_seeds'][cfg.category]
                  and cfg.codebook_dim * batch == s.SCALARS, 'Changed stage/shuffle/scalar-exposure contract.')
        kwargs['batch_size'] = batch
        event = dict(category=cfg.category, dim=cfg.codebook_dim, bits=cfg.codebook_bits,
                     vector_batch=batch, scalars_per_linear_update=cfg.codebook_dim * batch,
                     shuffle_seed=kwargs['shuffle_seed'], steps_per_stage=cfg.steps,
                     stages=cfg.residual_stages, status='fitting', started_unix=time.time())
        record['actual_groups'].append(event)
        s.write(path, record)
        result = original_fit(**kwargs)
        event.update(status='fitted_and_encoded', elapsed_seconds=time.time() - event['started_unix'])
        s.write(path, record)
        return result

    def calibration(**kwargs):
        s.require(phase == 'prefix' and record['calibration'] is None and kwargs['cache'] is None
                  and kwargs['dataset'] == 'alpaca=1' and kwargs['nsamples'] == 512
                  and kwargs['seqlen'] == 256 and kwargs['seed'] == 31
                  and {name for name, _ in kwargs['linear_items']} == s.projection_names(s.CATEGORIES),
                  'The fresh prefix must generate the original full calibration from train data.')
        stats, cache = original_calibration(**kwargs)
        digest = hashlib.sha256()
        for tokens in cache.input_ids:
            digest.update(tokens.contiguous().numpy().tobytes())
        record['calibration'] = dict(source='New native calibration, no checkpoint cache input.',
            dataset=cache.dataset, samples=len(cache.input_ids), seqlen=cache.seqlen,
            seed=cache.seed, input_ids_sha256=digest.hexdigest(), stats_linears=len(stats))
        s.write(path, record)
        return stats, cache

    def save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        if kwargs['checkpoint_kind'] not in ('category_boundary', 'final_model'):
            return result
        checkpoint = Path(result['output_dir']).resolve(strict=True)
        s.require(checkpoint.is_relative_to(output / phase / 'native'), 'Native save escaped this new phase.')
        meta = s.read(checkpoint / 'checkpoint_meta.json')
        event = dict(path=str(checkpoint), checkpoint_id=meta['checkpoint_id'], kind=meta['checkpoint_kind'],
            completed_categories=meta['completed_categories'], metadata_sha256=s.sha(checkpoint / 'checkpoint_meta.json'),
            runtime_sha256=s.sha(checkpoint / 'cat_runtime_state.pt'), weight_stat=s.stat(checkpoint / meta['state_dict_file']))
        record['saved_boundaries'].append(event)
        s.write(path, record)
        if phase == 'prefix' and kwargs.get('category') == 'o_proj':
            s.validate_after_o_meta(meta, request['base_model_path'])
            s.require([group['category'] for group in record['actual_groups']] == list(expected)
                      and all(group['status'] == 'fitted_and_encoded' for group in record['actual_groups'])
                      and record['calibration'] is not None and record['seed_calls'] == [31],
                      'Prefix did not complete its genuine fitting/calibration path.')
            raise PrefixComplete()
        return result

    try:
        native.build_cat_cross_category_runtime_identity = identity_with_policy
        native.train_group_vae_payload = fit
        native.save_cat_v6_full_checkpoint = save
        native.set_seed = seeded
        native.collect_activation_stats_for_linears = calibration
        native.run_cat_train(cat_args=cat, hf_args=hf, training_args=training, vae_args=vae)
        s.require(phase == 'suffix' and record['seed_calls'] == [31]
                  and [group['category'] for group in record['actual_groups']] == list(expected)
                  and all(group['status'] == 'fitted_and_encoded' for group in record['actual_groups'])
                  and record['saved_boundaries'][-1]['kind'] == 'final_model'
                  and record['saved_boundaries'][-1]['completed_categories'] == list(s.CATEGORIES),
                  'Missing the complete live-MLP native final endpoint.')
        record.update(status='completed', exit_code=0)
    except PrefixComplete:
        record.update(status='prefix_completed', exit_code=0)
    except BaseException as error:
        record.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        native.build_cat_cross_category_runtime_identity = original_identity
        native.train_group_vae_payload = original_fit
        native.save_cat_v6_full_checkpoint = original_save
        native.set_seed = original_seed
        native.collect_activation_stats_for_linears = original_calibration
        record['elapsed_seconds'] = time.time() - record['started_unix']
        s.write(path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--phase', choices=('prefix', 'suffix'), required=True)
    arguments = parser.parse_args()
    execute(arguments.request.resolve(strict=True), arguments.phase)
