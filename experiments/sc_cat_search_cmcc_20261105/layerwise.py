"""Full512 same-BF16-input transpose readout; no fitted-bank composition."""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_offline_execution_cmcc_20261104.protocol import binding, read, verify_files, write
from . import evidence

FP32_ARITHMETIC = 'BF16 deployment forward; FP32 difference/square, batch sum and across-batch accumulation; binary nonpadding mask; no KL weighting'


def choose_direction(A_SSE, B_SSE, entering):
    require(type(entering) is bool and all(math.isfinite(value) and value >= 0 for value in (A_SSE, B_SSE)),
        'Finite nonnegative paired SSE and an actual boolean entering direction are required.')
    return entering if A_SSE == B_SSE else B_SSE < A_SSE


def _canonical(value):
    if 'canonical_reference' in value:
        return read(evidence._path(value['canonical_reference']))
    if 'path' in value:
        return read(evidence._path(value))
    return value


def paired_sources(first, second, inventory):
    first, second = _canonical(first), _canonical(second)
    a, b = deepcopy(first['recipe']), deepcopy(second['recipe'])
    names = a['group_signature']
    require(evidence.normalized_transpose(a.pop('transpose'), names) is False
        and evidence.normalized_transpose(b.pop('transpose'), names) is True and a == b,
        'A/B must differ only by uniform False/True transpose in the full resolved recipe.')
    category = first['category_id']
    actual = [row for row in inventory['projections'] if row['category_id'] == category]
    require(category == second['category_id'] and set(names) == {row['actual_name'] for row in actual}
        and first['run_identity'] == second['run_identity']
        and first['selection_data_identity'] == second['selection_data_identity'],
        'A/B original BF16/data/source9/teacher or complete projection inventory differs.')
    for name in names:
        count = a['protection']['counts'][name]
        require(type(count) is int and count >= 0 and count % a['D'] == 0, 'Protection count must be an integer multiple of D.')
    return first, second, actual


def merge_output_statistics(local, candidate, projections, *, require_fp32=True):
    identity = candidate['run_identity']; expected_names = {row['actual_name'] for row in projections}
    require(local['candidate_id'] == candidate['candidate_id'] and local['selection_halves'] == [256, 256]
        and local['data_identity'] == identity['data_identity']
        and local['parent_identity'] == dict(uuid=identity['parent_uuid'], weights_digest=identity['parent_weights_digest'])
        and local['parent_input_identity'] and set(local['sufficient_statistics']) == expected_names,
        'Local statistics omit actual projections or mismatch BF16 parent/data/candidate/512 inputs.')
    fp32 = local.get('arithmetic') == FP32_ARITHMETIC and local.get('reduction_dtype') == 'torch.float32'
    require(not require_fp32 or fp32, 'Legacy FP32-square/FP64-sum statistics require the bounded FP32 readout; fit/full8 remain reusable.')
    result = {}
    for projection in projections:
        name, shape = projection['actual_name'], projection['shape']
        require(local['parameter_counts'][name] == math.prod(shape), 'Local parameter count differs from actual BF16 matrix shape.')
        halves = local['sufficient_statistics'][name]
        require(len(halves) == 2, 'Both original256 halves are required.')
        for half in halves:
            require(half['records'] == 256 and type(half['valid_tokens']) is int and half['valid_tokens'] > 0
                and math.isfinite(half['error_squared']) and half['error_squared'] >= 0
                and math.isfinite(half['target_squared']) and half['target_squared'] > 0,
                'Missing/nonfinite SSE, valid token count, energy or full512 coverage is incomplete readout.')
        tokens = sum(half['valid_tokens'] for half in halves); elements = tokens * shape[0]
        if fp32:
            full = local['full_statistics'][name]
            require(full['records'] == 512 and full['valid_tokens'] == tokens and full['elements'] == elements
                and math.isfinite(full['error_squared']) and full['error_squared'] >= 0, 'Formal FP32 full512 reduction/count differs.')
            sse = full['error_squared']
        else:
            sse = math.fsum(half['error_squared'] for half in halves)
        result[name] = dict(SSE=sse, valid_tokens=tokens, elements=elements, MSE=sse / elements,
            half_valid_tokens=[half['valid_tokens'] for half in halves], records=512, weight_shape=list(shape),
            reduction='FP32' if fp32 else 'legacy_FP64_diagnostic_only')
    return result


def build_mixed(a_receipt, b_receipt, inventory, current_transpose, *, weight_readouts=None, readouts=None):
    first, second, projections = paired_sources(a_receipt, b_receipt, inventory)
    names = first['recipe']['group_signature']
    current = evidence.normalized_transpose(current_transpose, names)
    current = {name: current for name in names} if type(current) is bool else current
    local_a, local_b = (readouts[identifier] for identifier in (first['candidate_id'], second['candidate_id'])) if readouts else (
        first['layer_functional_readout'], second['layer_functional_readout'])
    require(local_a['parent_input_identity'] == local_b['parent_input_identity'], 'A/B did not consume the identical BF16 parent records/tokens/mask/order.')
    merged = [merge_output_statistics(local, source, projections) for local, source in ((local_a, first), (local_b, second))]
    require(weight_readouts is not None and set(weight_readouts) == {first['candidate_id'], second['candidate_id']},
        'Both original-coordinate weight readouts are required for the reported transpose evidence.')
    rows, directions, bridge_differences = [], {}, []
    for projection in projections:
        name = projection['actual_name']; a, b = (value[name] for value in merged)
        require(a['half_valid_tokens'] == b['half_valid_tokens'] and a['elements'] == b['elements'], 'A/B counted different valid tokens/output elements.')
        tie = a['SSE'] == b['SSE']
        chosen = choose_direction(a['SSE'], b['SSE'], current[name])
        directions[name] = bool(chosen)
        weights = []
        for source in (first, second):
            weight = weight_readouts[source['candidate_id']][name]
            require(weight['elements'] == projection['numel'] and math.isfinite(weight['SSE']) and weight['SSE'] >= 0
                and weight['MSE'] == weight['SSE'] / weight['elements'] and weight['original_coordinates'] is True,
                'Weight error must come from the actual decoded inverse-transformed/protected original matrix.')
            weights.append(weight)
        bridge = None
        if 'bridge_FP64_statistics' in local_a and 'bridge_FP64_statistics' in local_b:
            legacy_a = local_a['bridge_FP64_statistics'][name]['error_squared']
            legacy_b = local_b['bridge_FP64_statistics'][name]['error_squared']
            bridge_direction = current[name] if legacy_a == legacy_b else legacy_b < legacy_a
            bridge = dict(A_SSE=legacy_a, B_SSE=legacy_b, transpose=bool(bridge_direction), differs_from_FP32=bridge_direction != chosen)
            if bridge_direction != chosen:
                bridge_differences.append(name)
        rows.append(dict(layer_id=projection['layer_id'], projection=name, A_candidate_id=first['candidate_id'], B_candidate_id=second['candidate_id'],
            A=a, B=b, A_weight=weights[0], B_weight=weights[1], transpose=bool(chosen),
            exact_tie=tie, entering_transpose=current[name], bridge_FP64=bridge))
    transpose = evidence.normalized_transpose(directions, names)
    reused = first['candidate_id'] if transpose is False else second['candidate_id'] if transpose is True else None
    return dict(schema='sc_cat_full512_layerwise_transpose_v1', category_id=first['category_id'],
        A_candidate_id=first['candidate_id'], B_candidate_id=second['candidate_id'],
        parent_identity=local_a['parent_identity'], data_identity=local_a['data_identity'],
        parent_input_identity=local_a['parent_input_identity'], rows=rows, transpose=transpose, reuse_candidate_id=reused,
        reduction_dtype='torch.float32', FP64_direction_differences=bridge_differences,
        selection='Per-layer full512 output MSE only; exact tie retains entering direction; W-MSE is diagnostic only.',
        fitting='Any new legal mixed recipe requires fresh full-category pooled S2x10000; no A/B bank composition.')


def load_bank_modules(bank, device):
    """Rebuild the real PUBLIC modules and strictly load only the original bank."""
    import torch
    from train_utils import checkpoint_v6 as v6
    from experiments.cat_auto_protocol_cmcc_20261104.vae_muon import tensor_digest
    require(evidence.file_sha(bank['state_file']) == bank['state_sha256'], 'Actual bank bytes changed before deployment readout.')
    state = torch.load(bank['state_file'], map_location='cpu', weights_only=True, mmap=True)
    require(tensor_digest(state) == bank['tensor_digest'], 'Actual bank tensors changed before deployment readout.')
    holder = torch.nn.Module()
    for spec in bank['converted_modules']:
        node = holder; parts = spec['name'].split('.')
        for part in parts[:-1]:
            if part not in node._modules:
                node.add_module(part, torch.nn.Module())
            node = node._modules[part]
        node.add_module(parts[-1], torch.nn.Linear(0, 0, bias=False, dtype=torch.bfloat16))
    v6._rebuild_converted_modules(holder, bank['converted_modules'])
    incompatible = holder.load_state_dict(state, strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys, 'Actual PUBLIC bank strict state/topology load failed.')
    v6.refresh_vae_linear_runtime_after_state_load(holder)
    holder.to(device).eval()
    return {spec['name']: holder.get_submodule(spec['name']) for spec in bank['converted_modules']}


def score_FP32(parent, modules_by_candidate, batches, projections, parent_identity, data_identity):
    import torch
    from experiments.cat_auto_protocol_cmcc_20261104.data import batch_digest
    from experiments.cat_auto_protocol_cmcc_20261104.proposal_readout import preserved_eval, _model_logits
    names = [row['actual_name'] for row in projections]
    require(len(batches) == 128 and all(set(modules) == set(names) for modules in modules_by_candidate.values()), 'Full fixed512/B4 and complete target category required.')
    stats = {identifier: {name: [dict(error_squared=torch.zeros((), device='cuda', dtype=torch.float32),
        target_squared=0., valid_tokens=0, records=0) for _ in range(2)] for name in names} for identifier in modules_by_candidate}
    full = {identifier: {name: torch.zeros((), device='cuda', dtype=torch.float32) for name in names} for identifier in modules_by_candidate}
    bridge = {identifier: {name: 0. for name in names} for identifier in modules_by_candidate}
    handles, current, records, inputs = [], {}, [], []
    def hook(name):
        def observe(module, args, target):
            require(name not in current['seen'] and isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.bfloat16, 'Dense BF16 parent projection must run exactly once per batch.')
            current['seen'].add(name); x = args[0]; mask = current['mask'].to(target.device).bool().unsqueeze(-1)
            require(x.ndim == target.ndim == 3 and x.shape[:2] == target.shape[:2] == mask.shape[:2]
                and x.dtype == target.dtype == torch.bfloat16 and torch.isfinite(target).all().item(), 'Actual BF16 parent input/output shape or precision differs.')
            tokens = int(mask.sum()); energy = float(target.float().square().masked_fill(~mask, 0).sum(dtype=torch.float64))
            require(tokens > 0 and math.isfinite(energy) and energy > 0, 'Nonfinite/empty parent output.')
            for identifier, modules in modules_by_candidate.items():
                output = modules[name](x)
                require(output.shape == target.shape and output.dtype == target.dtype and torch.isfinite(output).all().item(), 'Actual deployed candidate output is invalid.')
                squared = (output.float() - target.float()).square().masked_fill(~mask, 0)
                batch_sse = squared.sum(dtype=torch.float32)
                require(torch.isfinite(batch_sse).item(), 'Nonfinite FP32 output error.')
                full[identifier][name].add_(batch_sse)
                row = stats[identifier][name][current['half']]; row['error_squared'].add_(batch_sse)
                row['target_squared'] += energy; row['valid_tokens'] += tokens; row['records'] += x.shape[0]
                bridge[identifier][name] += float(squared.sum(dtype=torch.float64))
        return observe
    try:
        for name in names:
            handles.append(parent.get_submodule(name).register_forward_hook(hook(name)))
        with preserved_eval(parent, *(module for modules in modules_by_candidate.values() for module in modules.values())):
            for index, batch in enumerate(batches):
                require(batch_digest(batch['inputs']) == batch['input_sha256'] and len(batch['records']) == 4, 'Frozen record/token/mask/order changed.')
                mask = batch['inputs']['attention_mask']; require(torch.all((mask == 0) | (mask == 1)).item(), 'Nonbinary input token mask.')
                current.update(mask=mask, half=index // 64, seen=set()); _model_logits(parent, batch['inputs'])
                require(current['seen'] == set(names), 'BF16 parent failed to reach every target projection.')
                records.extend(value['record_id'] for value in batch['records']); inputs.append(batch['input_sha256'])
        require(len(records) == len(set(records)) == 512, 'All512 distinct frozen train records required.')
    finally:
        for handle in handles:
            handle.remove()
    input_identity = digest(dict(parent_identity=parent_identity, data_identity=data_identity, record_ids=records, input_sha256=inputs))
    result = {}
    for identifier in stats:
        serial = {name: [dict(row, error_squared=float(row['error_squared'])) for row in halves] for name, halves in stats[identifier].items()}
        result[identifier] = dict(candidate_id=identifier, selection_halves=[256, 256], parent_identity=parent_identity,
            data_identity=data_identity, parent_input_identity=input_identity, arithmetic=FP32_ARITHMETIC, reduction_dtype='torch.float32',
            sufficient_statistics=serial, parameter_counts={row['actual_name']: row['numel'] for row in projections},
            full_statistics={row['actual_name']: dict(error_squared=float(full[identifier][row['actual_name']]), records=512,
                valid_tokens=sum(half['valid_tokens'] for half in serial[row['actual_name']]),
                elements=sum(half['valid_tokens'] for half in serial[row['actual_name']]) * row['shape'][0]) for row in projections},
            bridge_FP64_statistics={name: dict(error_squared=value) for name, value in bridge[identifier].items()})
    return result


def worker(request_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.cat_auto_protocol_cmcc_20261104.proposal_readout import proposal_batches, preserved_eval
    request = read(request_path); manifest = read(evidence._path(request['manifest']))
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == str(request['gpu'])
        and request['gpu'] in range(6) and torch.cuda.device_count() == 1 and time.time() < request['deadline'], 'Use exactly the assigned authorized GPU0..5 in caller bitvae.')
    verify_files(manifest['source_sha256'])
    a, b = (evidence.checked_result(reference, manifest) for reference in (request['A'], request['B']))
    inventory = read(evidence._path(manifest['inventory']))['inventory']; a, b, projections = paired_sources(a, b, inventory)
    evidence._path(manifest['data_plan'])
    prepared = torch.load(manifest['data_plan']['path'], map_location='cpu', weights_only=False)
    require(digest(prepared['manifest']) == prepared['data_identity'] == manifest['run_identity']['data_identity']
        and prepared['manifest']['model_identity'] == inventory['model_identity'] and prepared['manifest']['search_seed'] == manifest['run_identity']['search_seed'],
        'Frozen actual DataPlan/model/seed changed.')
    source = manifest['parent']['path']
    parent = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    batches = proposal_batches(prepared['proposal'], tokenizer)
    modules = {row['candidate_id']: load_bank_modules(row['bank'], 'cuda:0') for row in (a, b)}
    for projection in projections:
        name = projection['actual_name']; left, right = (modules[row['candidate_id']][name] for row in (a, b))
        for field in ('protected_input_indices', 'protected_output_indices'):
            x, y = getattr(left, field), getattr(right, field)
            require((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y)), 'Paired transpose candidates protect different actual channels.')
    parent_identity = dict(uuid=manifest['parent']['uuid'], weights_digest=manifest['parent']['weights_digest'])
    local = score_FP32(parent, modules, batches, projections, parent_identity, prepared['data_identity'])
    require(local[a['candidate_id']]['parent_input_identity'] == a['layer_functional_readout']['parent_input_identity']
        == b['layer_functional_readout']['parent_input_identity'], 'Supplemental output readout input identity differs from the original fitted A/B full512.')
    weights = {row['candidate_id']: {} for row in (a, b)}
    with preserved_eval(parent, *(module for values in modules.values() for module in values.values())):
        for projection in projections:
            require(time.time() < request['deadline'], 'Readout budget exhausted; do not publish a partial mapping.')
            name = projection['actual_name']; target = parent.get_submodule(name).weight
            for row in (a, b):
                decoded = modules[row['candidate_id']][name]._decode_weight(torch.bfloat16)
                require(list(decoded.shape) == projection['shape'] and decoded.dtype == target.dtype and torch.isfinite(decoded).all().item(), 'Actual original-coordinate deployment decode failed.')
                sse = float((decoded.float() - target.float()).square().sum(dtype=torch.float32))
                weights[row['candidate_id']][name] = dict(SSE=sse, elements=projection['numel'], MSE=sse / projection['numel'], original_coordinates=True,
                    definition='PUBLIC _decode_weight with inverse rotation and protected scatter; BF16 decoded weight vs original BF16; FP32 error square/sum')
    result = build_mixed(a, b, inventory, request['entering_transpose'], weight_readouts=weights, readouts=local)
    result.update(A_source=request['A'], B_source=request['B'], local_readouts=local, weight_readouts=weights,
        status='completed', exit_code=0, request_reference=binding(request_path), no_training=True, no_teacher_cache_recalculation=True)
    write(Path(request['output']) / 'layerwise.json', result)


def readout_run(manifest, Aref, Bref, entering_transpose, gpu, output, deadline):
    from experiments.cat_night_profiles_cmcc_20261104.run import wait_child
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    destination = output / 'layerwise.json'
    manifest_path = Path(manifest['output_root']) / 'manifest.json'
    require(read(manifest_path) == manifest, 'Readout requires the actual frozen new-run manifest.')
    if destination.is_file():
        value = read(destination)
        require(value['A_source'] == Aref and value['B_source'] == Bref and value['status'] == 'completed', 'Existing readout belongs to different actual A/B sources.')
        request = read(evidence._path(value['request_reference']))
        require(request['entering_transpose'] == entering_transpose and request['manifest'] == binding(manifest_path),
            'Cannot reuse another manifest or entering direction for exact ties.')
        exit_row = read(output / 'exit.json')
        require(exit_row.get('exit_code') == 0 and value.get('exit_code') == 0, 'A result written before a failed/timed-out child exited is not a completed readout.')
        return dict(transpose=value['transpose'], evidence=binding(destination), reuse_candidate_id=value['reuse_candidate_id'],
            owned_GPU_hours=(exit_row['finished_epoch'] - exit_row['started_epoch']) / 3600, status='completed', exit_code=0)
    require(type(gpu) is int and gpu in range(6), 'Readout only uses its assigned original authorized GPU0..5 slot.')
    started = time.time(); end = min(deadline, started + 1800)
    require(end > started, 'No remaining bounded readout time.')
    request = dict(schema='sc_cat_bounded_AB_readout_v1', manifest=binding(manifest_path), A=Aref, B=Bref,
        entering_transpose=entering_transpose, gpu=gpu, deadline=end, authorized_GPU_hours=.5, output=str(output.resolve()))
    registry = output / 'request.json'; write(registry, request)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), ACCELERATE_USE_CPU='false',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    failure = None
    try:
        code = wait_child([sys.executable, '-B', '-u', '-m', __name__, '--request', str(registry)], environment, output, end)
    except Exception as error:
        code = 1; failure = type(error).__name__ + ': ' + str(error)
    finally:
        finished = time.time(); cost = (finished - started) / 3600
        write(output / 'readout_execution.json', dict(status='completed' if not failure and code == 0 else 'failed',
            exit_code=code, error=failure, started_epoch=started, finished_epoch=finished, owned_GPU_hours=cost,
            request_reference=binding(registry), no_automatic_retry=True))
    if code:
        return dict(status='failed', exit_code=code, owned_GPU_hours=cost, evidence=None, error=failure)
    value = read(destination)
    return dict(transpose=value['transpose'], evidence=binding(destination), reuse_candidate_id=value['reuse_candidate_id'],
        owned_GPU_hours=cost, status='completed', exit_code=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--request', required=True)
    args = parser.parse_args(); worker(args.request)


if __name__ == '__main__':
    main()
