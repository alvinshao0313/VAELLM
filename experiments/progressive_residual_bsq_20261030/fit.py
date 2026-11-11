"""Run the unchanged N fitter, recording genuine cumulative S4 boundaries."""
import argparse
from dataclasses import replace
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from bindings import CATEGORIES, HF, boundary, check_environment, process_event, s
import recipe


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def execute(request_path, event_fd):
    request = s.read(request_path)
    check_environment(request['fit_gpu'])
    s.check_pins(request['source_sha256'])
    s.require(s.hf_identity(HF) == request['hf_identity'], 'Original BF16 identity changed.')
    output = Path(request['output_dir'])
    short = request['validation_only']
    cat, hf, training, vae, cfgs, identity, normalized, values = recipe.parse(output / 'native')
    from train_utils import cat_train_pipeline as native
    import torch
    import transformers
    s.require(Path(native.__file__).resolve().is_relative_to(s.SNAPSHOT), 'Only frozen N native fitting is allowed.')
    path = output / 'fit.json'
    record = dict(schema='progressive_bf16_s4_fit_v1', status='running', **process_event(os.getpid()),
        started_unix=time.time(), request_sha256=s.sha(request_path), validation_only=short,
        formal_parent_eligible=not short, native_argv=values, runtime_identity=identity,
        normalized=normalized, groups=[], boundaries=[], calibration=None, recoveries=0,
        actual_updates_per_stage=2 if short else 10000, schedule_steps_per_stage=10000)
    s.write(path, record)
    names = ('train_group_vae_payload', 'apply_group_vae_payload', 'collect_activation_stats_for_linears',
        'save_cat_v6_full_checkpoint', 'run_after_category_distill',
        'materialize_prepared_group_data', '_convert_stage_stacked_to_common_stacked',
        '_group_data_to_compressed_weights', '_fuse_norm_into_decoder', 'create_optimizer',
        '_build_vae_linear_from_stage_payload')
    originals = {name: getattr(native, name) for name in names}
    original_scheduler = transformers.get_scheduler
    current = None
    residual = None
    contributions = []
    extracted = []
    materialize_calls = 0
    complement_before = None
    calibration_ids = None
    stream = os.fdopen(event_fd, 'w', encoding='utf-8', buffering=1)

    def distill(**kwargs):
        raise ValueError('No category recovery or future-weight update is permitted in initialization.')

    def scheduler(name, optimizer, **kwargs):
        if current is not None:
            s.require(str(name) == 'linear' and kwargs['num_warmup_steps'] == 0
                and kwargs['num_training_steps'] == (2 if short else 10000), 'Unexpected stage LR schedule.')
            kwargs['num_training_steps'] = 10000
        return original_scheduler(name, optimizer, **kwargs)

    def optimizer(parameters, args, lr):
        result = originals['create_optimizer'](parameters, args, lr)
        s.require(isinstance(result, torch.optim.AdamW) and result.defaults['betas'] == (0.9, 0.95)
            and result.defaults['weight_decay'] == 0 and lr == .003, 'Unexpected optimizer recipe.')
        current['optimizer_initializations'] += 1
        return result

    def materialize(**kwargs):
        nonlocal materialize_calls, residual
        result = originals['materialize_prepared_group_data'](**kwargs)
        materialize_calls += 1
        if materialize_calls > 1:
            stage = materialize_calls - 1
            actual = result.stacked_data
            if residual is not None:
                s.require(torch.equal(actual, residual), 'Next stage is not the actual preceding cumulative residual.')
            residual = actual.detach().clone()
            current['stages'].append(dict(stage=stage, residual_shape=list(actual.shape),
                residual_dtype=str(actual.dtype), residual_rms_before=float(actual.float().square().mean().sqrt()),
                input_sha256=tensor_digest({'residual': actual}) if short else None))
        return result

    def convert(**kwargs):
        result = originals['_convert_stage_stacked_to_common_stacked'](**kwargs)
        contributions[:] = [result.detach().cpu()]
        return result

    def outgoing(**kwargs):
        nonlocal residual
        actual = kwargs['stacked_data']
        contribution = contributions[-1].to(dtype=residual.dtype)
        expected = (residual - contribution).contiguous()
        s.require(torch.equal(actual, expected), 'Actual residual subtraction changed its cumulative semantics.')
        current['stages'][-1].update(residual_rms_after=float(actual.float().square().mean().sqrt()),
            contribution_rms=float(contribution.float().square().mean().sqrt()),
            cumulative_residual_exact=True)
        residual = actual.detach().clone()
        return originals['_group_data_to_compressed_weights'](**kwargs)

    def fuse(decoder, *args, **kwargs):
        result = originals['_fuse_norm_into_decoder'](decoder, *args, **kwargs)
        if short:
            extracted.append((decoder, tensor_digest(decoder.state_dict())))
        return result

    def fit(**kwargs):
        nonlocal current, residual, contributions, extracted, materialize_calls, complement_before
        cfg = kwargs['runtime_cfg']
        index = len(record['groups'])
        s.require(cfg.category == CATEGORIES[index] and cfg == cfgs[cfg.category]
            and cfg.steps == 10000 and cfg.residual_stages == 4
            and len(kwargs['group_refs']) == 36
            and {ref.name for ref in kwargs['group_refs']} == s.projection_names((cfg.category,))
            and kwargs['shuffle_seed'] == 31 + index * 100000,
            'Fit only the next complete M36 category from the original no-recovery model.')
        if short:
            current_prefix = tuple(name + '.' for name in s.projection_names((cfg.category,)))
            complement_before = tensor_digest({name: value for name, value in kwargs['model'].state_dict().items()
                if not name.startswith(current_prefix)})
        current = dict(category=cfg.category, dim=cfg.codebook_dim, bits=cfg.codebook_bits,
            vector_batch=4096 if cfg.category == 'gate_proj' else 8192,
            shuffle_seed=kwargs['shuffle_seed'], stages=[], optimizer_initializations=0,
            actual_updates_per_stage=2 if short else 10000, schedule_steps_per_stage=10000,
            started_unix=time.time(), status='fitting')
        record['groups'].append(current)
        residual, contributions, extracted, materialize_calls = None, [], [], 0
        kwargs['batch_size'] = current['vector_batch']
        if short:
            kwargs['runtime_cfg'] = replace(cfg, steps=2)
        s.write(path, record)
        result = originals['train_group_vae_payload'](**kwargs)
        s.require(result['residual_stages'] == 4 and len(result['all_stage_bits']) == 4
            and materialize_calls == 5 and current['optimizer_initializations'] == 4,
            'The complete four-stage fit and encoding did not execute.')
        current['hard_code_shapes'] = [list(value.shape) for value in result['all_stage_bits']]
        s.require(all(value.shape[1] == 36 and value.shape[2] == cfg.codebook_bits
            and torch.all((value == 0) | (value == 1)) for value in result['all_stage_bits']),
            'All four stages must contain complete hard binary codes for every vector and layer.')
        if short:
            s.require(len(extracted) == 144 and all(tensor_digest(decoder.state_dict()) == before
                for decoder, before in extracted), 'An earlier extracted stage decoder changed during later-stage fitting.')
            current['extracted_prior_stages_frozen_exact'] = True
        current.update(status='fitted_and_full_encoded', elapsed_seconds=time.time() - current['started_unix'])
        s.write(path, record)
        residual, contributions, extracted = None, [], []
        return result

    def apply(**kwargs):
        result = originals['apply_group_vae_payload'](**kwargs)
        if short:
            prefix = tuple(name + '.' for name in s.projection_names((current['category'],)))
            after = tensor_digest({name: value for name, value in kwargs['model'].state_dict().items()
                if not name.startswith(prefix)})
            s.require(after == complement_before, 'Prior categories, future dense weights or outside weights changed.')
            current['complete_noncurrent_complement_frozen_exact'] = True
        return result

    def build(**kwargs):
        result = originals['_build_vae_linear_from_stage_payload'](**kwargs)
        target = kwargs['old_module'].weight.detach()
        rows = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            target_cuda = target.cuda().float()
            energy = float(target_cuda.double().square().sum())
            for depth in range(1, 5):
                if depth == 4:
                    prefix = result.to('cuda')
                else:
                    arguments = dict(kwargs)
                    for field in ('stage_split_metas', 'stage_part_bits_payload', 'stage_codebook_dims'):
                        arguments[field] = kwargs[field][:depth]
                    arguments['stage_part_decoders_payload'] = deepcopy(kwargs['stage_part_decoders_payload'][:depth])
                    prefix = originals['_build_vae_linear_from_stage_payload'](**arguments).to('cuda')
                decoded = prefix._decode_weight(dtype=torch.bfloat16)
                s.require(decoded.dtype == torch.bfloat16 and decoded.shape == target.shape and torch.isfinite(decoded).all(),
                    'Actual native hard-prefix BF16 deployment decoding is invalid.')
                error = float((decoded.float() - target_cuda).double().square().sum())
                rows.append(dict(depth=depth, SSE=error, target_energy=energy, NMSE=error / energy,
                    weights=target.numel(), decoded_dtype=str(decoded.dtype)))
                prefix.clear_decoded_weight_cache()
                prefix.to('cpu')
                del decoded, prefix
            del target_cuda
        current.setdefault('BF16_deployment_prefixes', []).append(dict(name=kwargs['split_meta'].linear_name,
            prefixes=rows, consumer='Original N packed hard codes and fused decoder _decode_weight BF16; inverse rotation/protection included.'))
        if len(current['BF16_deployment_prefixes']) == 36:
            current['BF16_deployment_category_prefixes'] = [dict(depth=depth,
                SSE=sum(item['prefixes'][depth - 1]['SSE'] for item in current['BF16_deployment_prefixes']),
                target_energy=sum(item['prefixes'][depth - 1]['target_energy'] for item in current['BF16_deployment_prefixes']))
                for depth in range(1, 5)]
            for row in current['BF16_deployment_category_prefixes']:
                row['NMSE'] = row['SSE'] / row['target_energy']
            s.write(path, record)
        return result

    def calibration(**kwargs):
        nonlocal calibration_ids
        s.require(record['calibration'] is None and kwargs['dataset'] == 'alpaca=1'
            and kwargs['nsamples'] == 512 and kwargs['seqlen'] == 256 and kwargs['seed'] == 31
            and {name for name, _ in kwargs['linear_items']} == s.projection_names(CATEGORIES),
            'Keep original BF16 train-only 512x256 calibration for all252 projections.')
        stats, cache = originals['collect_activation_stats_for_linears'](**kwargs)
        calibration_ids = cache.input_ids[0]
        record['calibration'] = dict(dataset=cache.dataset, samples=len(cache.input_ids), seed=cache.seed,
            seqlen=cache.seqlen, linears=len(stats), input_ids_sha256=tensor_digest(
                {str(index): value for index, value in enumerate(cache.input_ids)}))
        s.write(path, record)
        return stats, cache

    def save(*args, **kwargs):
        result = originals['save_cat_v6_full_checkpoint'](*args, **kwargs)
        if kwargs['checkpoint_kind'] == 'category_boundary':
            event = boundary(result['output_dir'])
            event.update(category=kwargs['category'], validation_only=short)
            record['boundaries'].append(event)
            if short and kwargs['category'] == 'down_proj':
                model = args[0]
                original_cache = model.config.use_cache
                model.config.use_cache = False
                model.eval()
                from prewarm import prepare
                prepare(model, native.log)
                with torch.inference_mode():
                    ids = calibration_ids.to('cuda')
                    logits = model(input_ids=ids).logits.detach().cpu()
                torch.save(dict(input_ids=calibration_ids.cpu(), logits=logits), output / 'short_live_logits.pt')
                model.config.use_cache = original_cache
                event['short_live_logits_sha256'] = s.sha(output / 'short_live_logits.pt')
            s.write(path, record)
            stream.write(json.dumps(event) + '\n'); stream.flush()
        return result

    try:
        for name, replacement in dict(train_group_vae_payload=fit, apply_group_vae_payload=apply,
                collect_activation_stats_for_linears=calibration, save_cat_v6_full_checkpoint=save,
                run_after_category_distill=distill, materialize_prepared_group_data=materialize,
                _convert_stage_stacked_to_common_stacked=convert, _group_data_to_compressed_weights=outgoing,
                _fuse_norm_into_decoder=fuse, create_optimizer=optimizer,
                _build_vae_linear_from_stage_payload=build).items():
            setattr(native, name, replacement)
        transformers.get_scheduler = scheduler
        native.run_cat_train(cat_args=cat, hf_args=hf, training_args=training, vae_args=vae)
        s.require(len(record['boundaries']) == 7 and len(record['groups']) == 7, 'Missing category boundaries.')
        record.update(status='complete_bf16_s4_no_recovery_fit', exit_code=0)
    except BaseException as error:
        record.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        stream.close()
        for name, original in originals.items():
            setattr(native, name, original)
        transformers.get_scheduler = original_scheduler
        record.update(elapsed_seconds=time.time() - record['started_unix'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        s.write(path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--event-fd', type=int, required=True)
    args = parser.parse_args()
    execute(args.request.resolve(strict=True), args.event_fd)
