"""Run the unchanged native B resume path, stopping at its genuine raw2 save."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b


class RawComplete(Exception):
    """The genuine complete gate round_base was saved."""


def execute(arm):
    settings = b.ARMS[arm]
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(settings['gpu'])
        and os.environ.get('PYTHONHASHSEED') == '31', 'Use caller bitvae and this arm assigned physical GPU.')
    prepared = b.read(b.RESULT / 'prepared.json')
    b.require(prepared['status'] == 'cpu_verified' and b.sources() == prepared['source_binding'], 'CPU fork or source changed.')
    b.check_pins(prepared['source_sha256'])
    output = b.OUTPUT / arm / 'formal'
    b.require(not output.exists(), 'Never overwrite a formal arm.')
    output.mkdir()
    argv, parsed, identity, normalized = b.recipe(arm, output / 'native')
    b.require(identity == prepared['arms'][arm]['runtime_identity'], 'Actual recipe differs from saved fork.')
    from train_utils import cat_train_pipeline as native
    from experiments.adaptive_class_20261030.initialization import observe
    b.require(Path(native.__file__).resolve().is_relative_to(b.SNAPSHOT), 'Native execution escaped N snapshot.')
    record = dict(schema='cat_gate_hyperparam_fit_v1', status='running', arm=arm, settings=settings,
        pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        started_unix=time.time(), command=[sys.executable, *sys.argv], native_argv=argv,
        source_binding=prepared['source_binding'], prepared_sha256=b.sha(b.RESULT / 'prepared.json'),
        source_sha256=prepared['source_sha256'], runtime_identity=identity, normalized=normalized,
        seed_calls=[], actual_groups=[], gate_recovery_updates=0, stage_update_budget=2*settings['steps'],
        scheduler_interpretation='Fresh per-stage linear schedule over the full configured horizon, no warmup; not continuation.',
        control_initialization_boundary='Same original native run_cat_train startup path. No historical control init digest exists; no bitwise initialization claim.')
    path = output / 'report.json'
    b.write(path, record)
    originals = {name: getattr(native, name) for name in ('set_seed', 'build_cat_cross_category_runtime_identity',
        'train_group_vae_payload', 'save_cat_v6_full_checkpoint', 'run_after_category_distill')}

    def seeded(seed, *args, **kwargs):
        b.require(seed == 31 and not record['seed_calls'], 'Native startup must seed exactly once at31.')
        result = originals['set_seed'](seed, *args, **kwargs)
        record['seed_calls'].append(seed)
        return result

    def live_identity(**kwargs):
        result = originals['build_cat_cross_category_runtime_identity'](**kwargs)
        result['experiment_runtime'] = deepcopy(identity['experiment_runtime'])
        b.require(result == identity, 'Live seven-category identity changed.')
        return result

    def fit(**kwargs):
        cfg = kwargs['runtime_cfg']
        b.require(not record['actual_groups'] and cfg == parsed[-1]['gate_proj']
            and len(kwargs['group_refs']) == 36 and all(ref.name.endswith('.gate_proj') for ref in kwargs['group_refs'])
            and kwargs['shuffle_seed'] == 400031, 'Fit only the original M36 gate group and shuffle400031.')
        kwargs['batch_size'] = 4096
        row = dict(category='gate_proj', linears=36, dim=64, bits=64, stages=2,
            vector_batch_per_linear=4096, scalar_exposure_per_linear_update=262144,
            actual_updates_per_stage=settings['steps'], shuffle_seed=400031, started_unix=time.time(), status='fitting')
        record['actual_groups'].append(row)
        b.write(path, record)
        with observe(native) as initializations:
            result = originals['train_group_vae_payload'](**kwargs)
        b.require(len(initializations) == 2, 'Both genuine residual stages must initialize their native optimizers.')
        row.update(status='fitted_and_encoded', elapsed_seconds=time.time()-row['started_unix'],
            actual_optimizer_initializations=initializations)
        b.write(path, record)
        return result

    def distill(**kwargs):
        raise ValueError('Gate recovery and future suffix are excluded from this fixed cohort.')

    def save(*args, **kwargs):
        result = originals['save_cat_v6_full_checkpoint'](*args, **kwargs)
        if kwargs['checkpoint_kind'] == 'round_base' and kwargs.get('category') == 'gate_proj':
            directory = Path(result['output_dir'])
            b.validate_raw2(b.read(directory / 'checkpoint_meta.json'))
            b.require(record['seed_calls'] == [31] and len(record['actual_groups']) == 1
                and record['actual_groups'][0]['status'] == 'fitted_and_encoded', 'Incomplete native gate fit.')
            meta = b.read(directory / 'checkpoint_meta.json')
            state = b.torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
            # Native saving is complete here; this read has no influence on fitting RNG.
            record.update(raw2=b.event(directory), state=b.fork.state_digest(state))
            raise RawComplete()
        return result

    try:
        native.set_seed, native.build_cat_cross_category_runtime_identity = seeded, live_identity
        native.train_group_vae_payload, native.run_after_category_distill = fit, distill
        native.save_cat_v6_full_checkpoint = save
        cat, hf, training, vae, _ = parsed
        native.run_cat_train(cat_args=cat, hf_args=hf, training_args=training, vae_args=vae)
        raise ValueError('Native execution returned without a genuine raw2 save.')
    except RawComplete:
        record.update(status='raw2_completed', exit_code=0)
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        for name, value in originals.items():
            setattr(native, name, value)
        record.update(elapsed_seconds=time.time()-record['started_unix'],
            peak_allocated=b.torch.cuda.max_memory_allocated(), peak_reserved=b.torch.cuda.max_memory_reserved())
        b.write(path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(b.ARMS), required=True)
    execute(parser.parse_args().arm)
