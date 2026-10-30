"""Matched real L18/L35 third-stage fitting: sequential versus grouped GEMMs."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import get_scheduler
from safetensors import safe_open

from experiments.adaptive_grouped_fit_20261029.grouped import IndependentGroupedVAE, fresh_models
from experiments.adaptive_layer_transfer_20261028 import helpers as h
from experiments.adaptive_vector_rvq import codec, common as c
from experiments.adaptive_vector_rvq.training import fit_stage, encode, finalize, mse
from experiments.adaptive_vector_rvq.third_stage import vector_errors
from experiments.adaptive_vector_rvq.observe import original_weight
from experiments.adaptive_vector_rvq.third_stage_export import save, load, inventory
from experiments.adaptive_sparse_pair_20261028.allocate import iterate
from train_utils.cat_train_data import compute_stage_norm_stats
from train_utils.train_args import create_optimizer

HERE = Path(__file__).resolve().parent
PREPARED = Path('/data/star/VAELLM/adaptive_layer_transfer_20261028/prepared')
DEVICE = 'cuda:0'


def prepare_inputs(*, normalize):
    prepared = h.check_prepared(PREPARED)
    values = []
    for layer in (18, 35):
        row = prepared['layers'][str(layer)]
        source = codec.load(Path(row['fixed2_path']))
        data = torch.load(row['path'], map_location='cpu', weights_only=True, mmap=True)
        target = data['vectors']
        c.require(c.digest({'vectors': target}) == row['target_digest']
                  and source['geometry'] == data['geometry'] == row['geometry'], 'Real frozen source changed.')
        full = torch.arange(len(target))
        prefix = torch.zeros_like(target, dtype=torch.bfloat16)
        errors = []
        for stage in source['stages']:
            codec.add_stage(prefix, stage, full, DEVICE)
            errors.append(vector_errors(target, prefix))
        value = dict(layer=layer, row=row, source=source, target=target, prefix=prefix, errors=errors)
        if normalize:
            residual = (target.float() - prefix.float()).contiguous()
            mean, scale = compute_stage_norm_stats(residual.unsqueeze(1))
            mean, scale = mean.reshape(()), scale.reshape(())
            value.update(mean=mean, scale=scale, normalized=((residual-mean)/scale).bfloat16().unsqueeze(1))
        values.append(value)
    return prepared, values


def compare_vectors(reference, candidate):
    first = torch.cat([item.detach().float().reshape(-1).cpu() for item in reference])
    second = torch.cat([item.detach().float().reshape(-1).cpu() for item in candidate])
    difference = (second-first).double()
    norm = float(first.double().norm())
    return dict(reference_norm=norm, difference_norm=float(difference.norm()),
        max_abs=float(difference.abs().max()), relative_l2=float(difference.norm())/norm if norm else None)


def check_math(values):
    originals = fresh_models(2, 'fp32')
    grouped = IndependentGroupedVAE(originals)
    inputs = torch.cat([value['normalized'][:128].float() for value in values], dim=1)
    reconstruction, losses = grouped(inputs)
    losses.sum().backward()
    records = []
    for index, original in enumerate(originals):
        expected, parts = original(inputs[:, index:index+1])
        parts['loss'].backward()
        for name, parameter in original.named_parameters():
            c.require(torch.equal(parameter, grouped.single_parameters(index)[name]), 'Grouped initialization differs.')
        torch.testing.assert_close(reconstruction[:, index:index+1], expected, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(losses[index], parts['loss'], rtol=2e-5, atol=1e-6)
        ref = dict(original.named_parameters())
        actual = grouped.single_parameters(index, gradients=True)
        metrics = compare_vectors([ref[name].grad for name in ref], [actual[name] for name in ref])
        c.require(metrics['relative_l2'] <= 1e-4, 'FP32 grouped gradients differ beyond affine/LN/reduction rounding.')
        records.append(dict(layer=values[index]['layer'], loss_reference=float(parts['loss'].detach()),
            loss_grouped=float(losses[index].detach()), gradient=metrics))
    return dict(status='verified_FP32_math', real_residual_rows_per_layer=128, layers=records,
        tolerance_basis='FP32 batched affine, LayerNorm affine, and reduction rounding; per-layer losses and all12 leaf gradients are checked.')


def check_updates(values):
    originals = fresh_models(2, device=DEVICE)
    grouped = IndependentGroupedVAE(originals).to(DEVICE)
    optimizers = [create_optimizer(model.parameters(), c.arguments(), .003) for model in originals]
    optimizer = create_optimizer(grouped.parameters(), c.arguments(), .003)
    schedulers = [get_scheduler('linear', opt, num_warmup_steps=0, num_training_steps=6000) for opt in optimizers]
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=6000)
    generator = torch.Generator().manual_seed(2031)
    order = torch.randperm(len(values[0]['target']), generator=generator)
    rows = []
    initial = {name: parameter.detach().clone() for name, parameter in grouped.named_parameters()}
    for step in range(2):
        ids = order[step*c.BATCH:(step+1)*c.BATCH]
        batches = [value['normalized'].index_select(0, ids).to(DEVICE) for value in values]
        optimizer.zero_grad(set_to_none=True)
        _, losses = grouped(torch.cat(batches, dim=1))
        losses.sum().backward()
        records = []
        for index, (model, opt, batch) in enumerate(zip(originals, optimizers, batches, strict=True)):
            opt.zero_grad(set_to_none=True)
            _, parts = model(batch)
            parts['loss'].backward()
            params = dict(model.named_parameters())
            gradients = grouped.single_parameters(index, gradients=True)
            c.require(all(torch.isfinite(parameter.grad).all().item() for parameter in params.values())
                      and all(torch.isfinite(value).all().item() for value in gradients.values()), 'Nonfinite real BF16 gradient.')
            records.append(dict(layer=values[index]['layer'], serial_loss=float(parts['loss'].detach()),
                grouped_loss=float(losses[index].detach()),
                gradient=compare_vectors([params[name].grad for name in params], [gradients[name] for name in params])))
            opt.step(); schedulers[index].step()
            del parts
        optimizer.step(); scheduler.step()
        rows.append(dict(step=step+1, layers=records, learning_rate=scheduler.get_last_lr()[0]))
        c.require(all(int(optimizer.state[p]['step']) == step+1 for p in grouped.parameters()), 'Grouped Adam update counter changed.')
        del losses
    c.require(any(not torch.equal(parameter, initial[name]) for name, parameter in grouped.named_parameters()),
              'Grouped actual BF16 parameters did not update.')
    return dict(status='verified_real_BF16_two_updates', batch_per_layer=c.BATCH, rows=rows,
        boundary='Grouped/fused normalization rounding is recorded; no bitwise equality of BF16 trajectories or quality claim.')


def fit_group(values, output):
    started = time.monotonic()
    source_models = fresh_models(2, device=DEVICE)
    model = IndependentGroupedVAE(source_models).to(DEVICE)
    del source_models
    data = [value['normalized'].to(DEVICE) for value in values]
    optimizer = create_optimizer(model.parameters(), c.arguments(), .003)
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=6000)
    generators = [torch.Generator().manual_seed(2031) for _ in values]
    orders, positions = [None, None], [0, 0]
    observations, stages, recent = [[], []], {}, []
    torch.cuda.reset_peak_memory_stats()
    compute_start = time.monotonic()
    for step in range(1, 6001):
        batches = []
        for index, value in enumerate(data):
            if orders[index] is None or positions[index] >= len(value):
                orders[index] = torch.randperm(len(value), generator=generators[index])
                positions[index] = 0
            ids = orders[index][positions[index]:positions[index]+c.BATCH]
            positions[index] += len(ids)
            batches.append(value.index_select(0, ids.to(DEVICE)))
        c.require(batches[0].shape == batches[1].shape, 'Keep original per-layer tail batches; do not pad or resample.')
        model.train(); optimizer.zero_grad(set_to_none=True)
        _, losses = model(torch.cat(batches, dim=1))
        c.require(torch.isfinite(losses).all().item(), 'Nonfinite independent grouped losses.')
        losses.sum().backward()
        if step == 1 or step % 100 == 0:
            c.require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                      'Nonfinite or disconnected actual encoder/decoder gradient.')
        optimizer.step(); scheduler.step()
        recent.append(losses.detach().float().tolist())
        if step in (3000, 6000):
            for index, value in enumerate(values):
                single = model.single_model(index)
                full = torch.arange(len(value['target']))
                residual = value['target'].float()-value['prefix'].float()
                codes = encode(single, residual, full, value['mean'], value['scale'], DEVICE)
                spec, state = finalize(single, value['mean'], value['scale'])
                stage = dict(decoder_spec=spec, decoder_state=state, codes=codes,
                    normalization=dict(mean=float(value['mean']), scale=float(value['scale']), active_count=len(full)))
                following = value['prefix'].clone()
                codec.add_stage(following, stage, full, DEVICE)
                observations[index].append(dict(step=step, normalized_whole_MSE=mse(value['target'], following)/value['row']['source_variance']))
                stages[value['layer']] = stage
                print(dict(event='group_stage_observation', layer=value['layer'], **observations[index][-1]), flush=True)
                del single, following, residual
        del losses
    torch.cuda.synchronize()
    report = dict(elapsed_seconds=time.monotonic()-started, updates_per_wall_second=6000/(time.monotonic()-compute_start),
        actual_sampled_vectors_per_layer=6000*c.BATCH, observations=observations,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        timing_scope='Both layers together; includes original3000/6000 full encoding/decoder export and residual readouts.')
    return stages, report


@torch.no_grad()
def readout(value, stage3, output, prepared):
    output.mkdir()
    source, row, target = value['source'], value['row'], value['target']
    banks = [*source['stages'], stage3]
    following = value['prefix'].clone()
    codec.add_stage(following, stage3, torch.arange(len(target)), DEVICE)
    errors = torch.stack([*value['errors'], vector_errors(target, following)], dim=1)
    energy = torch.load(row['energy']['path'], map_location='cpu', weights_only=True)
    vector_weight = energy['normalized_h'].repeat_interleave(128)
    cap = Path(f'/data/star/VAELLM/adaptive_sparse_layers_20261028/layer_{value["layer"]}/sparse_abs005.avrqsp').stat().st_size
    lengths, allocation = iterate(errors/row['source_variance']*vector_weight[:, None], row['budget'], cap, output/'allocation')
    selected = [{**stage, 'codes': stage['codes'].index_select(0, torch.nonzero(lengths >= index+1).flatten()).contiguous()}
                for index, stage in enumerate(banks)]
    frame = Path(allocation['chosen']['length_frame']['artifact']['path']).read_bytes()
    bundle = dict(n_vectors=len(target), geometry=source['geometry'], fixed=source['fixed'], stages=selected,
                  length_frame=torch.frombuffer(bytearray(frame), dtype=torch.uint8).clone())
    path = output/'weighted_dynamic3.avrvqz'
    size = save(bundle, path)
    fresh, fresh_lengths = load(path)
    c.require(size == allocation['chosen']['actual_complete_bytes'] <= cap
              and torch.equal(lengths, fresh_lengths) and c.digest(inventory(fresh)) == c.digest(inventory(bundle)),
              'Complete packed budget or fresh persistent fields changed.')
    vectors = h.decode_compact(fresh, fresh_lengths, DEVICE)
    actual_weight = original_weight(fresh, vectors, DEVICE)
    key = source['geometry']['source_key']
    shard = c.read(c.HF/'model.safetensors.index.json')['weight_map'][key]
    with safe_open(str(c.HF/shard), framework='pt', device='cpu') as handle:
        reference = handle.get_tensor(key).to(DEVICE)
    archive = torch.load(prepared['activations']['path'], map_location='cpu', weights_only=True, mmap=True)
    x, split = h.activation_inputs(archive, c.read(prepared['activations']['manifest']), source['geometry']['name'])
    metrics = h.output_metrics(actual_weight, reference, x, split, DEVICE)
    return dict(layer=value['layer'], metrics=metrics, complete_file=dict(path=str(path), bytes=size, sha256=c.sha(path)),
        common_cap_bytes=cap, allocation=allocation, dense_stage3_NMSE=float(errors[:, 2].mean()/row['source_variance']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'serial', 'grouped'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
              and Path(sys.executable).resolve() == Path('/home/star/miniconda3/envs/bitvae/bin/python').resolve(),
              'Use caller-activated bitvae and the assigned single GPU.')
    c.require(not args.output.exists(), 'Never replace an earlier comparison.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    record = dict(status='running', pid=os.getpid(), started_unix=time.time(), mode=args.mode,
        command=[sys.executable, *sys.argv], source_sha256={str(path): c.sha(path) for path in HERE.glob('*.py')})
    c.write(args.output/'report.json', record)
    try:
        prepared, values = prepare_inputs(normalize=args.mode != 'serial')
        record['input_prepared'] = dict(path=str(PREPARED/'prepared.json'), sha256=c.sha(PREPARED/'prepared.json'))
        record['original_input_and_implementation_sha256'] = prepared['source_sha256']
        record['input_stats'] = prepared['input_stats']
        if args.mode == 'check':
            record.update(math=check_math(values), real_updates=check_updates(values))
        else:
            if args.mode == 'serial':
                stages, train = {}, {}
                for value in values:
                    full = torch.arange(len(value['target']))
                    stage, following, info = fit_stage(value['target'], value['prefix'], full, steps=6000,
                        seed=2031, device=DEVICE, output=args.output/f'layer_{value["layer"]}'/'stage3',
                        source_variance=value['row']['source_variance'])
                    stages[value['layer']], train[value['layer']] = stage, info
                    del following
            else:
                stages, train = fit_group(values, args.output)
            record['training'] = train
            record['endpoints'] = [readout(value, stages[value['layer']], args.output/f'endpoint_{value["layer"]}', prepared)
                                   for value in values]
        c.require(all(c.sha(path) == expected for path, expected in record['source_sha256'].items()), 'Source changed during the run.')
        record.update(status='completed', exit_code=0)
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        c.write(args.output/'report.json', record)


if __name__ == '__main__':
    main()
