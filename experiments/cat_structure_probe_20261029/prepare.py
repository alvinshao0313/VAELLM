"""Bind live after_o weights and the original CAT protection ranking, once."""
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from experiments.cat_structure_probe_20261029 import common as c
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.activation_utils import activation_stats_to_views, subset_activation_stats
from train_utils.cat_data_prep import LinearPrepRef, build_outlier_channel_index_plan, prepare_group_linear_entries
from rotation.weight_preprocess import rotation_spec


def main():
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only input preparation.')
    output = c.OUTPUT/'prepared'
    c.require(not output.exists(), 'Do not overwrite prepared inputs.')
    torch.set_num_threads(4)
    started = time.monotonic()
    meta = c.source()
    state = torch.load(c.SOURCE/meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    runtime = torch.load(c.SOURCE/'cat_runtime_state.pt', map_location='cpu', weights_only=False)
    activation = runtime['activation_runtime']
    names = [f'model.layers.{layer}.mlp.gate_proj' for layer in c.LAYERS]
    targets, refs = {}, []
    for name in names:
        weight = state[name+'.weight'].clone()
        c.require(name in meta['pending_dense_targets'] and tuple(weight.shape) == (12288, 4096)
                  and weight.dtype == torch.bfloat16, 'Require true live dense gate weights.')
        targets[name] = dict(reference_weight=weight)
        refs.append(LinearPrepRef(name, weight, 4096, 12288, False))
    actmax, actmean, _ = activation_stats_to_views(subset_activation_stats(activation['stats_by_linear'], names))
    plan = build_outlier_channel_index_plan(group_refs=refs, activation_weight_by_linear=actmax,
        activation_abs_mean_by_linear=actmean, channel_protect_count=64, channel_axis='input',
        channel_scope='layer', channel_rank_metric='channel_weight_actmean_abs')
    entries = prepare_group_linear_entries(group_refs=refs, activation_weight_by_linear=None,
        channel_protect_count=64, channel_axis='input', channel_quant='int8', recon_loss_type='mse', channel_plan=plan)
    for name, entry in zip(names, entries, strict=True):
        p = entry.prepared_weight
        targets[name].update(protected_input_indices=plan[name].clone(),
            protected_input_qvalues=p.protected_input_qvalues.clone(), protected_input_scales=p.protected_input_scales.clone(),
            rotation=rotation_spec('two_sided', 32, 31, name))
    # MLP readouts use this same parent's still-dense up/down, not an HF substitute.
    other = {f'model.layers.{layer}.mlp.{category}.weight': state[f'model.layers.{layer}.mlp.{category}.weight'].clone()
             for layer in c.LAYERS for category in ('up_proj', 'down_proj')}
    manifest = c.read(c.TOKEN_MANIFEST)
    c.require(c.sha(c.TOKENS) == manifest['archives']['tokens.pt']['sha256']
              and manifest['data_source']['split'] == 'train' and manifest['valid_tokens'] == 30675,
              'Use the unchanged already-audited FineWeb train tokens.')
    output.mkdir(parents=True)
    archive = output/'weights.pt'
    torch.save(dict(targets=targets, other_MLP=other), archive)
    record = dict(status='prepared_weights_waiting_live_capture', source_checkpoint_id=c.SOURCE_ID,
        source_checkpoint=str(c.SOURCE), source_weights=c.stat(c.SOURCE/meta['state_dict_file']),
        source_meta_sha256=c.sha(c.SOURCE/'checkpoint_meta.json'), runtime_sha256=c.sha(c.SOURCE/'cat_runtime_state.pt'),
        weights=dict(**c.stat(archive), sha256=c.sha(archive)),
        targets={name: {key: tensor_record(value) if isinstance(value, torch.Tensor) else value for key, value in item.items()}
                 for name, item in targets.items()},
        original_channel_statistics=dict(dataset=activation['dataset'], nsamples=activation['nsamples'],
            seqlen=activation['seqlen'], seed=activation['seed'], metric='channel_weight_actmean_abs',
            use='Same original CAT statistics with actual live weights; fixed plan for every arm.'),
        token_archive=dict(**c.stat(c.TOKENS), sha256=c.sha(c.TOKENS)), token_manifest_sha256=c.sha(c.TOKEN_MANIFEST),
        data_source=manifest['data_source'], documents=manifest['documents'], valid_tokens=manifest['valid_tokens'],
        arm_specs=c.ARMS, group_semantics='Original CAT M3 pooled BSQ entropy in every arm; not independent group2.',
        source_sha256={str(path): c.sha(path) for path in (Path(__file__), c.HERE/'common.py')},
        elapsed_seconds=time.monotonic()-started, CUDA_initialized=False)
    c.write(output/'prepared.json', record)
    print(dict(status=record['status'], elapsed_seconds=record['elapsed_seconds']), flush=True)


if __name__ == '__main__':
    main()
