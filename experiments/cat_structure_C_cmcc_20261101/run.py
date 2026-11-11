"""Original native C: M36 encoder fitting, then its own three original500 recoveries."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

from . import bindings as b


class ShortComplete(Exception):
    """The real new-shape encoding and original native round_base save completed."""


def execute(phase):
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'C training uses caller bitvae and physical GPU0 only.')
    prepared = b.read(b.RESULT / 'prepared.json')
    b.require(prepared['status'] == 'cpu_verified' and b.source_binding() == prepared['source_binding'],
        'Require the actual CPU-verified original after_o fork.')
    b.check_pins(prepared['source_sha256'])
    if phase == 'formal':
        proof = b.read(b.RESULT / 'short_validation.json')
        b.require(proof['status'] == 'verified_new_E256_native_shape'
            and proof['prepared_sha256'] == b.sha(b.RESULT / 'prepared.json')
            and proof['steps_per_stage'] == 2 and proof['stages'] == 2
            and proof['fresh_full_state_exact'], 'The real new-shape encoding/fresh evidence is missing.')
    output = b.OUTPUT / phase
    b.require(not output.exists(), 'Never overwrite an existing short or formal C run.')
    output.mkdir()
    argv, parsed, identity, normalized = b.recipe(output / 'native')
    b.require(identity == prepared['runtime_identity'], 'Live recipe differs from the strict saved fork.')
    from train_utils import cat_train_pipeline as native
    from train_utils import cat_checkpoint_v6 as saver
    from train_utils import cat_after_category_common as recovery
    from transformers import TrainerCallback
    from experiments.adaptive_class_20261030.initialization import observe
    b.require(Path(native.__file__).resolve().is_relative_to(b.SNAPSHOT), 'Native execution escaped the original N snapshot.')
    started = time.monotonic()
    report_path = output / 'report.json'
    provenance = dict(schema='original_C_native_provenance_v1', arm='C', validation_only=phase == 'short',
        source_binding=prepared['source_binding'], originalA_control=prepared['originalA_control'],
        training_contract=b.CONTRACT, actual_fit_updates_per_stage=2 if phase == 'short' else 10000,
        actual_recovery_steps_per_category=0 if phase == 'short' else 500)
    record = dict(schema='cat_structure_C_native_pipeline_v1', status='running', arm='C', phase=phase,
        validation_only=phase == 'short', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        actual_argv=[sys.executable, *sys.argv], native_argv=argv, normalized=normalized,
        source_binding=prepared['source_binding'], originalA_control=prepared['originalA_control'],
        training_contract=b.CONTRACT, pipeline_source_sha256=prepared['source_sha256'],
        prepared_sha256=b.sha(b.RESULT / 'prepared.json'), runtime_identity=identity,
        fork=prepared['fork'], actual_groups=[], recoveries=[], saved_boundaries=[],
        final_strict_reload_report=str(output / 'strict_reload.json'), started_unix=time.time())
    b.write(report_path, record)
    originals = {name: getattr(native, name) for name in ('build_cat_cross_category_runtime_identity',
        'train_group_vae_payload', 'save_cat_v6_full_checkpoint', 'run_after_category_distill')}
    low_save = saver.save_v6_full_checkpoint
    original_trainer = recovery.CanonicalCatSFTTrainer

    class ObserveSteps(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            row = record['recoveries'][-1]
            b.require(state.global_step == state.max_steps == 500, 'Actual native Trainer did not finish the original500 updates.')
            row.update(actual_steps=int(state.global_step), actual_schedule_max_steps=int(state.max_steps))

    class ObservedTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            kwargs['callbacks'] = [*kwargs.get('callbacks', []), ObserveSteps()]
            super().__init__(*args, **kwargs)

    def live_identity(**kwargs):
        result = b.with_policy(originals['build_cat_cross_category_runtime_identity'](**kwargs))
        b.require(result == identity, 'Actual native seven-category identity changed.')
        return result

    def fit(**kwargs):
        cfg = kwargs['runtime_cfg']
        index = len(record['actual_groups'])
        b.require(index < 3 and cfg.category == b.CATEGORIES[index+4]
            and len(kwargs['group_refs']) == 36 and cfg.codebook_dim == 32
            and cfg.residual_stages == 2 and cfg.steps == 10000
            and kwargs['shuffle_seed'] == b.CONTRACT['shuffle_seeds'][cfg.category],
            'Fit the actual next complete M36 live category with the original shuffle and stages.')
        b.require(cfg.base_ch == (256 if cfg.category == 'gate_proj' else 128)
            and cfg.decoder_type == ('asymmetric' if cfg.category == 'gate_proj' else 'symmetric')
            and (cfg.category != 'gate_proj' or cfg.decoder_base_ch == 128), 'Incorrect actual C encoder/decoder geometry.')
        kwargs['batch_size'] = 8192
        if phase == 'short':
            b.require(cfg.category == 'gate_proj', 'Short C stops before recovery or own up/down fitting.')
            kwargs['runtime_cfg'] = replace(cfg, steps=2)
        row = dict(category=cfg.category, linears=36, dim=32, bits=cfg.codebook_bits,
            encoder=cfg.base_ch, decoder=128, vector_batch_per_linear=8192,
            scalar_exposure_per_linear_update=262144, shuffle_seed=kwargs['shuffle_seed'],
            planned_updates_per_stage=10000, actual_updates_per_stage=2 if phase == 'short' else 10000,
            stages=2, started_unix=time.time(), status='fitting')
        record['actual_groups'].append(row)
        b.write(report_path, record)
        updates = []
        with observe(native) as initializations:
            capture = native.create_optimizer

            def checked_optimizer(*args, **options):
                optimizer = capture(*args, **options)
                stage_updates = []
                updates.append(stage_updates)
                original_step = optimizer.step

                def step(this, *positional, **keywords):
                    parameters = [p for group in this.param_groups for p in group['params'] if p.grad is not None]
                    b.require(parameters and all(b.torch.isfinite(p.grad).all().item() for p in parameters)
                        and any(b.torch.count_nonzero(p.grad).item() for p in parameters),
                        'New E256 actual backward produced missing, nonfinite or entirely zero gradients.')
                    before = [p.detach().clone() for p in parameters]
                    result = original_step(*positional, **keywords)
                    changed = sum(not b.torch.equal(old,p) for old,p in zip(before,parameters,strict=True))
                    states = [state for state in this.state.values() if 'step' in state]
                    actual_steps = sorted({int(state['step']) for state in states})
                    b.require(changed and all(b.torch.isfinite(p).all().item() for p in parameters)
                        and actual_steps == [len(stage_updates)+1], 'New E256 actual parameter/Adam update failed.')
                    stage_updates.append(dict(actual_optimizer_step=actual_steps[0], gradient_tensors=len(parameters),
                        all_gradients_finite=True, changed_parameter_tensors=changed,
                        parameter_dtypes=sorted({str(p.dtype) for p in parameters}),
                        Adam_moment_dtypes=sorted({str(value.dtype) for state in states
                            for name,value in state.items() if name != 'step' and isinstance(value,b.torch.Tensor)})))
                    return result

                optimizer.step = MethodType(step,optimizer)
                return optimizer

            if phase == 'short':
                native.create_optimizer = checked_optimizer
            try:
                result = originals['train_group_vae_payload'](**kwargs)
            finally:
                native.create_optimizer = capture
        b.require(len(initializations) == 2, 'Both real residual stages must create their original optimizers.')
        if phase == 'short':
            b.require(len(updates) == 2 and all(len(value) == 2 for value in updates),
                'New E256 short must complete two genuine optimizer updates in both residual stages.')
        row.update(status='fitted_and_encoded', elapsed_seconds=time.time()-row['started_unix'],
            actual_optimizer_initializations=initializations,
            new_shape_actual_updates=updates,
            optimizer_rng='Observed without reseeding the original continuous native RNG.')
        b.write(report_path, record)
        return result

    def distill(**kwargs):
        b.require(phase == 'formal' and kwargs['category'] == b.CATEGORIES[4+len(record['recoveries'])],
            'Recover only this branch gate/up/down in original order.')
        row = dict(category=kwargs['category'], planned_steps=500, started_unix=time.time(), status='running')
        record['recoveries'].append(row)
        b.write(report_path, record)
        result = originals['run_after_category_distill'](**kwargs)
        b.require(result.did_train and row.get('actual_steps') == 500, 'The original recovery did not complete its real500 updates.')
        row.update(status='completed', original_distill_meta=result.distill_meta,
            elapsed_seconds=time.time()-row['started_unix'])
        b.write(report_path, record)
        return result

    def save_provenance(model, directory, **kwargs):
        kwargs['extra_meta'] = {**kwargs.get('extra_meta', {}), 'cat_structure_C': provenance}
        return low_save(model, directory, **kwargs)

    def save(*args, **kwargs):
        result = originals['save_cat_v6_full_checkpoint'](*args, **kwargs)
        event = b.event(result['output_dir'])
        b.require(Path(event['path']).is_relative_to(output / 'native'), 'Actual native save escaped this new run.')
        if event['kind'] in ('category_boundary', 'final_model') or phase == 'short':
            record['saved_boundaries'].append(event)
        if phase == 'short' or event['kind'] == 'final_model':
            event['state'] = b.fork.state_digest({name:value.detach().cpu()
                for name,value in args[0].state_dict().items()})
            b.write(output / 'expected_state.json', dict(state=event['state'], checkpoint_id=event['checkpoint_id']))
        b.write(report_path, record)
        if phase == 'short':
            b.require(event['kind'] == 'round_base' and kwargs['category'] == 'gate_proj',
                'The new-shape short endpoint must be the genuine native gate round_base.')
            raise ShortComplete()
        return result

    try:
        native.build_cat_cross_category_runtime_identity = live_identity
        native.train_group_vae_payload = fit
        native.run_after_category_distill = distill
        native.save_cat_v6_full_checkpoint = save
        saver.save_v6_full_checkpoint = save_provenance
        recovery.CanonicalCatSFTTrainer = ObservedTrainer
        native.run_cat_train(cat_args=parsed[0], hf_args=parsed[1], training_args=parsed[2], vae_args=parsed[3])
        final = record['saved_boundaries'][-1]
        b.require(phase == 'formal' and final['kind'] == 'final_model'
            and len(record['actual_groups']) == len(record['recoveries']) == 3,
            'The complete original C native suffix is missing.')
        record.update(status='completed_C_native_pipeline', exit_code=0,
            final_model=final['path'], final_checkpoint_id=final['checkpoint_id'], final_native_source=final)
    except ShortComplete:
        record.update(status='completed_C_native_shape_short', exit_code=0,
            final_model=record['saved_boundaries'][-1]['path'],
            final_checkpoint_id=record['saved_boundaries'][-1]['checkpoint_id'])
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        for name, value in originals.items():
            setattr(native, name, value)
        saver.save_v6_full_checkpoint = low_save
        recovery.CanonicalCatSFTTrainer = original_trainer
        record.update(elapsed_seconds=time.monotonic()-started, peak_allocated=b.torch.cuda.max_memory_allocated(),
            peak_reserved=b.torch.cuda.max_memory_reserved())
        b.write(report_path, record)
    print(dict(status=record['status'], elapsed_seconds=record['elapsed_seconds']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('short', 'formal'), required=True)
    args = parser.parse_args()
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    execute(args.phase)


if __name__ == '__main__':
    main()
