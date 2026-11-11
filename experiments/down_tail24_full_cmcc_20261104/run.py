"""Fixed once-only capture, twelve groups, full models/fresh and complete full8."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from . import bridge, capture, evaluate, fit
from experiments.down_tail23_short_cmcc_20261104 import allocation, artifact, readout
from experiments.down_tail23_short_cmcc_20261104 import fit as single
from experiments.adaptive_native_model_20261029 import objective
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from tools.cat_eval import _prepare_model_for_eval


def metadata():
    return dict(pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        command=[sys.executable, *sys.argv], runtime_environment={name: os.environ.get(name)
            for name in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF')})


def remaining(deadline):
    value = deadline - time.time()
    b.c.require(value > 0, 'The one common4h deadline expired; stop without retry or extension.')
    return value


def check_prepared():
    record = b.c.read(b.PREPARED)
    b.c.require(record['status'] == 'CPU_full_bridge_verified' and record['exit_code'] == 0, 'Require completed CPU full-bridge evidence.')
    b.check_pins(record['source_sha256'])
    b.c.require(record['native_full8_reference'] == b.native_reference(), 'The prepared native raw full8 identity changed.')
    return record


def common_report(source):
    return dict(schema=b.SCHEMA, phase='formal', source_binding=source,
        source_sha256=b.c.read(b.PREPARED)['source_sha256'], **metadata())


def wait_for_fresh(process, timeout):
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise


def capture_stage(root):
    output = root / 'capture'
    output.mkdir()
    prepared = check_prepared()
    b.check_pins(prepared['short_identity_files_sha256'])
    raw, _live, specs, source = b.source()
    bundles = {spec['name']: b.short.prefix(raw, spec, source)[0] for spec in specs}
    record = common_report(source)
    record.update(capture.execute(bundles, output))
    record.update(peak_allocated_bytes=b.torch.cuda.max_memory_allocated(), peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
    b.c.write(output / 'report.json', record)


def group_stage(root, index):
    output = root / 'groups' / f'g{index:02d}'
    output.mkdir(parents=True)
    started = time.time()
    raw, live, specs, source = b.source()
    by_name = {spec['name']: spec for spec in specs}
    names = b.GROUPS[index]
    captured = b.torch.load(root / 'capture/inputs.pt', map_location='cpu', weights_only=True, mmap=True)
    energies = b.torch.load(root / 'capture/h.pt', map_location='cpu', weights_only=True, mmap=True)
    values = []
    for name in names:
        bundle, _lengths, _resolved, _fields = b.short.prefix(raw, by_name[name], source)
        value = single.common_residual(bundle, live[name + '.weight'], device='cuda:0')
        value['name'] = name
        values.append(value)
    stages, training = fit.execute(values)
    record = dict(**common_report(source), status='running', group_index=index, names=names, training=training, layers=[])
    path = output / 'report.json'
    b.c.write(path, record)
    for value, stage in zip(values, stages, strict=True):
        name = value['name']
        scores, full = single.errors(value, stage, energies[name], device='cuda:0')
        lengths, selected, routing = allocation.allocate(scores, full, b.CAP)
        model_path = output / (name + '.pt')
        measured = artifact.save(selected, model_path)
        b.c.require(measured['payload_tensor_bytes'] <= b.CAP, 'A down layer exceeded its independent true-payload cap.')
        module = CompactRVQLinear(selected, lengths, 'cuda:0').eval()
        base_lengths = b.torch.full((b.N,), 2, dtype=b.torch.uint8)
        base = CompactRVQLinear(value['bundle'], base_lengths, 'cuda:0').eval()
        b.c.require(b.c.digest(bridge.frozen_down(module)) == b.c.digest(bridge.frozen_down(base))
            and bridge.frozen_metadata(module) == bridge.frozen_metadata(base), 'Actual routing changed a fixed stage or its normalization/layout.')
        reference = readout.outputs(live[name + '.weight'], captured[name], device='cuda:0')
        controls = []
        for current_bundle, current_lengths, current in ((value['bundle'], base_lengths, base), (selected, lengths, module)):
            weight, actual, fingerprint = readout.deployment(current_bundle, current_lengths, current, captured[name], device='cuda:0')
            controls.append(dict(metrics=readout.metrics(weight, live[name + '.weight'], actual, reference), fingerprint=fingerprint))
            del weight, actual
        chosen = scores.gather(1, (lengths.long() - 2).unsqueeze(1))
        record['layers'].append(dict(name=name, model_path=str(model_path), model_sha256=b.c.sha(model_path), artifact=measured,
            routing=routing, common_residual=value['info'], coverage=float((lengths == 3).double().mean()),
            control=dict(**controls[0], weighted_residual_NMSE=float(scores[:, 0].mean())),
            candidate=dict(**controls[1], weighted_residual_NMSE=float(chosen.mean())), all_frozen_fields_exact=True))
        del module, base, full, selected, scores, reference
        gc.collect()
        b.torch.cuda.empty_cache()
    record.update(status='completed_independent_M3_10000', exit_code=0, elapsed_seconds=time.time() - started)
    b.c.write(path, record)


def warm(model):
    result = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    b.c.require(result['total'] == result['warmed'] == 180 and not result['failed'], 'Require complete original group8 prewarm of180 native modules.')
    return result


@b.torch.no_grad()
def logits(model, tokens):
    value = model(input_ids=tokens.to('cuda:0'), use_cache=False, logits_to_keep=1).logits
    b.c.require(value.dtype == b.torch.bfloat16 and bool(b.torch.isfinite(value).all()), 'Actual complete-model logits must be finite BF16.')
    return value.detach().cpu().contiguous()


def fresh_stage(root, arm):
    directory = root / arm
    expected = b.torch.load(directory / 'deployment_evidence.pt', map_location='cpu', weights_only=True)
    model, meta, loaded, budget = b.checkpoint.load(directory / 'model')
    evaluate.inspect_topology(meta, budget, arm)
    b.c.require(meta['checkpoint_id'] == expected['checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys
        and b.c.digest(model.state_dict()) == expected['state_digest'], 'Independent full-model strict deployment state changed.')
    bridge.topology(model, mode=arm if arm == 'control' else 'candidate')
    bridge.verify_frozen(model, b.c.read(b.PREPARED)['frozen_identity'])
    warmed = warm(model)
    actual = logits(model, expected['tokens'])
    b.c.require(b.torch.equal(actual, expected['logits']), 'Independent full-model same-input BF16 logits changed.')
    ready = [name for name in b.NAMES if model.get_submodule(name)._decoded_cache is not None]
    b.c.require(len(ready) == 36, 'All36 actual compact down modules must execute the real full-model forward.')
    b.c.write(directory / 'fresh_report.json', dict(status='strict_full_state_and_BF16_logits_verified', exit_code=0,
        **metadata(), checkpoint_id=meta['checkpoint_id'], persistent_fields=len(model.state_dict()),
        full_state_digest=expected['state_digest'], same_input_logits_exact=True, actual_budget=budget,
        effective_weight_files=[str(directory / 'model' / b.checkpoint.STATE_FILE)], native_warmup=warmed,
        actual_compact_down_forward_count=len(ready), all_frozen_fields_exact=True,
        peak_allocated_bytes=b.torch.cuda.max_memory_allocated(), peak_reserved_bytes=b.torch.cuda.max_memory_reserved()))


def model_stage(root, arm, deadline):
    directory = root / arm
    directory.mkdir()
    raw, live, specs, source = b.source()
    model, _bundles, identity = bridge.build(raw, specs, source)
    b.c.require(identity == b.c.read(b.PREPARED)['frozen_identity'], 'The actual formal model has a different fixed predecessor.')
    training = []
    if arm == 'k24':
        for index, names in enumerate(b.GROUPS):
            group = b.c.read(root / 'groups' / f'g{index:02d}' / 'report.json')
            b.c.require(group['status'] == 'completed_independent_M3_10000' and group['exit_code'] == 0
                and group['names'] == names and group['training']['actual_updates'] == 10000
                and group['source_binding'] == source and group['source_sha256'] == b.c.read(b.PREPARED)['source_sha256'],
                'Do not publish an incomplete or differently sourced group as a36x10000 candidate.')
            training.append(group['training'])
            for row in group['layers']:
                path = Path(row['model_path'])
                b.c.require(b.c.sha(path) == row['model_sha256'], 'Actual trained layer artifact changed.')
                bundle, _lengths, module = artifact.load(path)
                b.c.require(artifact.measure(bundle)['payload_tensor_bytes'] <= b.CAP, 'Actual trained payload cap changed.')
                b.checkpoint.native.set_module_by_name(model, row['name'], module)
    bridge.topology(model, mode='control' if arm == 'control' else 'candidate')
    bridge.verify_frozen(model, identity)
    warmed = warm(model)
    tokens = b.torch.load(root / 'capture/tokens.pt', map_location='cpu', weights_only=True)[:1].clone()
    expected_logits = logits(model, tokens)
    bridge.verify_frozen(model, identity)
    provenance = bridge.provenance(arm, phase='formal', source=source)
    meta, budget, state = b.checkpoint.save(model, directory / 'model', provenance=provenance)
    evaluate.inspect_topology(meta, budget, arm)
    b.torch.save(dict(checkpoint_id=meta['checkpoint_id'], tokens=tokens, logits=expected_logits,
        state_digest=b.c.digest(state)), directory / 'deployment_evidence.pt')
    record = dict(**common_report(source), arm=arm, checkpoint_id=meta['checkpoint_id'], provenance=provenance,
        actual_budget=budget, group_training=training, all_frozen_fields_exact=True,
        native_warmup=warmed, status='saved_waiting_own_independent_fresh')
    b.c.write(directory / 'building.json', record)
    del model, state, raw, live, _bundles
    gc.collect()
    b.torch.cuda.empty_cache()
    command = [sys.executable, '-B', '-u', '-m', 'experiments.down_tail24_full_cmcc_20261104.run',
        '--component', 'fresh', '--root', str(root), '--arm', arm, '--deadline-unix', str(deadline)]
    with (directory / 'fresh.log').open('x') as log:
        process = subprocess.Popen(command, cwd=b.REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        record['fresh_process'] = dict(pid=process.pid, command=command,
            start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        b.c.write(directory / 'building.json', record)
        code = wait_for_fresh(process, remaining(deadline))
    record['fresh_process']['exit_code'] = code
    b.c.require(code == 0, 'Independent full-model fresh failed; stop without evaluation or retry.')
    record.update(status='completed_full_compact2_control' if arm == 'control' else 'completed_full_K24_raw', exit_code=0)
    # This immutable report is a source pin for all subsequent full8 children.
    b.c.write(directory / 'report.json', record)


def summary(root):
    groups = [b.c.read(root / 'groups' / f'g{index:02d}' / 'report.json') for index in range(12)]
    rows = [row for group in groups for row in group['layers']]
    result = {}
    for arm in ('control', 'candidate'):
        current = [dict(name=row['name'], **row[arm]) for row in rows]
        result[arm] = dict(output=readout.aggregate(current),
            weight_NMSE=sum(row['metrics']['weight']['SSE'] for row in current) / sum(row['metrics']['weight']['reference_energy'] for row in current),
            weighted_residual_NMSE=sum(row['weighted_residual_NMSE'] for row in current) / 36)
    c8 = b.c.read(root / 'control/full8/full8_summary.json')
    k8 = b.c.read(root / 'k24/full8/full8_summary.json')
    b.c.require(c8['leaf_count'] == k8['leaf_count'] == 64 and c8['sample_count'] == k8['sample_count'] == 24742,
        'Both endpoints need their own complete64-leaf/24742-example full8.')
    native_reference = b.c.read(b.PREPARED)['native_full8_reference']
    b.c.require(native_reference == b.native_reference(), 'The authoritative native reference changed.')
    native_path = Path(native_reference['path'])
    native = b.c.read(native_path)
    b.c.require(native['mean_percent'] == 67.01259272688102, 'The original native D16 raw reference changed.')
    b.c.write(root / 'result.json', dict(status='completed_full36_raw_phase', exit_code=0, schema=b.SCHEMA,
        source_raw_id=b.RAW_ID, live_target_id=b.LIVE_ID, native_reference=native_reference,
        compact2_full8=c8, K24_full8=k8, local_readout=result,
        candidate_beats_both_raw_controls=k8['mean_percent'] > c8['mean_percent'] and k8['mean_percent'] > native['mean_percent'],
        both_actual_budgets={arm: b.c.read(root / arm / 'report.json')['actual_budget'] for arm in ('control', 'k24')},
        all_actual_fit_updates_per_layer=10000, actual_layers=36, group_count=12,
        no_automatic_recovery_or_nine=True, boundary='Stage full8 and actual complete payload only. Development feedback; no500/nine or final whole-chain acceptance.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', required=True, choices=('capture', 'control', 'group', 'candidate', 'fresh', 'full8_setup', 'summary'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--group-index', type=int)
    parser.add_argument('--arm', choices=('control', 'k24'))
    args = parser.parse_args()
    b.gpu_environment()
    remaining(args.deadline_unix)
    check_prepared()
    root = args.root.resolve()
    if args.component == 'capture':
        capture_stage(root)
    elif args.component == 'group':
        b.c.require(args.group_index is not None and 0 <= args.group_index < 12, 'Use one of the fixed12 group indices.')
        group_stage(root, args.group_index)
    elif args.component in ('control', 'candidate'):
        model_stage(root, 'control' if args.component == 'control' else 'k24', args.deadline_unix)
    elif args.component == 'fresh':
        fresh_stage(root, args.arm)
    elif args.component == 'full8_setup':
        evaluate.full8(root, args.arm)
    else:
        summary(root)
    remaining(args.deadline_unix)
    print(dict(status='component_completed', component=args.component, arm=args.arm, group=args.group_index, **metadata()), flush=True)


if __name__ == '__main__':
    main()
