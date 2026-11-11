"""Two updates on each of adjacent real blocks 0 and 1; no formal BR option."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from . import data, export, fit, rolling, runtime

torch = b.torch
LAYERS = (0, 1)


def code_bindings():
    paths = set(b.HERE.glob('*.py'))
    paths.update(Path(module.__file__).resolve() for module in (data.legacy, data.original,
        data.legacy.core, data.legacy.mmlu, data.legacy.base))
    return {**b.runtime_bindings(), **{str(path.resolve()): b.sha(path) for path in sorted(paths)}}


def check_resources():
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '6'
        and Path(sys.executable).resolve() == Path('/home/cmcc/miniconda3/envs/bitvae/bin/python').resolve(),
        'Use caller-activated bitvae and the assigned physical GPU6 only.')
    result = subprocess.run(['nvidia-smi', '-i', '6', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    occupants = [line.strip() for line in result.stdout.splitlines()
                 if line.strip() and line.strip() != str(os.getpid())]
    b.require(not occupants, 'Assigned GPU6 has existing compute processes; do not interfere or retry: ' + str(occupants))
    b.require(torch.cuda.device_count() == 1, 'The BR adjacent-block short requires exactly one visible card.')


def input_fixture(pool):
    entry = next(entry for entry in pool.entries if entry.domain == 'mmlu')
    valid = entry.valid_mask[0]
    inputs = {name: value[0, valid].unsqueeze(0).clone() for name, value in entry.inputs.items()}
    return dict(schema='BR_BF16_same_input_v1', **inputs, source_case_id=entry.case_ids[0],
                input_signature=b.digest(inputs), source_entry_id=entry.entry_id)


@torch.no_grad()
def probe(model, fixture):
    export.deployment_contract(model)
    device = model.get_input_embeddings().weight.device
    inputs = {name: fixture[name].to(device) for name in ('input_ids', 'attention_mask', 'position_ids') if name in fixture}
    output = model(**inputs, use_cache=False, logits_to_keep=1)
    logits = output.logits.detach().cpu()
    b.require(output.past_key_values is None and logits.dtype == torch.bfloat16
        and logits.shape == (1, 1, model.config.vocab_size) and bool(torch.isfinite(logits).all()),
        'The same-input fixture requires the complete finite BF16 last-position logits.')
    return logits


def stream_inputs(streams, pool):
    rows = {}
    for index in fit.GROUPS:
        for entry_index, batch in zip(pool.group(index).entry_indices, streams.group(index).batches, strict=True):
            if entry_index in rows:
                continue
            rows[entry_index] = dict(entry_index=entry_index, entry_id=pool.entries[entry_index].entry_id,
                student=fit.tensor_record(batch.hidden_states),
                teacher_target=fit.tensor_record(batch.target_hidden_states),
                valid_mask=fit.tensor_record(batch.valid_mask))
    b.require(len(rows) == 18, 'The short must preserve exactly18 unique complete entries per stream.')
    return [rows[index] for index in sorted(rows)]


def require_transition(previous, current, previous_key, current_key, message):
    left = {row['entry_index']: (row['entry_id'], row[previous_key]) for row in previous}
    right = {row['entry_index']: (row['entry_id'], row[current_key]) for row in current}
    b.require(len(previous) == len(current) == len(left) == len(right) == 18 and left == right, message)


def run(args, identity, observe, started):
    student, source_meta, source_audit = b.load_student(identity['source_binding'], args.output/'source_baseline')
    export.deployment_contract(student)
    source_records = fit.state_records(student)
    allowed_by_layer = {layer: sorted(runtime.decoder_parameters(student, layer)) for layer in LAYERS}
    allowed = {name for names in allowed_by_layer.values() for name in names}
    b.require(len(allowed) == sum(map(len, allowed_by_layer.values())), 'The two local decoder ownership sets must be disjoint.')
    source_frozen = fit.proof_from_records(source_records, allowed)
    pool, evidence = data.load(student.config.pad_token_id)
    pool = pool.select_groups(fit.GROUPS)
    short_data = data.summary(pool)
    fixture = input_fixture(pool)
    teacher = b.load_teacher()
    teacher_frozen = fit.frozen_proof(teacher)
    streams = rolling.RollingStreams.capture(teacher, student, pool, device='cuda:0')
    observe('BR_independent_streams_captured', dict(initialization_passes=streams.initialization_passes,
        source_checkpoint_id=source_meta['checkpoint_id'], entries=len(pool.entries), memory=fit.memory()))
    layer_reports, completed_names = [], set()
    for layer in LAYERS:
        b.require(time.monotonic() < started+args.timeout_seconds,
                  'The bounded BR mechanism short exceeded its wall-clock budget.')
        layer_started = time.monotonic()
        teacher_block = teacher.model.layers[layer]
        teacher_block.to(device='cuda:0')
        teacher_started = time.monotonic()
        teacher_advance = streams.prepare_layer(layer, teacher_block)
        torch.cuda.synchronize()
        teacher_seconds = time.monotonic()-teacher_started
        teacher_block.to(device='cpu')
        inputs = stream_inputs(streams, pool)
        if layer == 0:
            transition = dict(source='Separate actual teacher/student embedding captures', entries=18)
        else:
            previous = layer_reports[-1]
            require_transition(previous['student_advance']['entries'], inputs, 'output', 'student',
                'Block1 must consume exactly the18 BF16 outputs committed by updated student block0.')
            require_transition(previous['teacher_advance']['entries'], teacher_advance['entries'], 'output', 'input',
                'Teacher block1 must consume exactly its independent original-BF16 teacher block0 outputs.')
            transition = dict(previous_layer=layer-1, entries=18,
                previous_student_BF16_output_equals_current_student_input=True,
                previous_teacher_output_equals_current_teacher_input=True)
        warm_before = runtime.prepare_deployment_block(student, layer, 'cuda:0')
        before_deployed = fit.evaluate_layer(student, layer, streams)
        active = runtime.enable_block_training(student, layer)
        b.require(set(active) == set(allowed_by_layer[layer]), 'Training initialization changed the actual local D topology.')
        before_live = fit.evaluate_layer(student, layer, streams)
        observe('BR_block_ready', dict(layer=layer, warmup=warm_before, before_deployed=before_deployed,
            before_live=before_live, active_leaves=len(active), memory=fit.memory()))
        training = fit.fit_layer(student, layer, streams, active, observe, started+args.timeout_seconds)
        expected_commit = {name: fit.tensor_record(value.detach().to(device='cpu', dtype=torch.bfloat16))
                           for name, value in active.items()}
        committed = runtime.commit_block(student, layer)
        b.require({name: fit.tensor_record(value) for name, value in runtime.decoder_parameters(student, layer).items()}
            == expected_commit, 'The local commit did not preserve every actual FP32 master\'s exact BF16 cast.')
        committed = dict(**committed, all_D_BF16_cast_exact=True, committed_D=expected_commit)
        del active
        warm_after = runtime.prepare_deployment_block(student, layer, 'cuda:0')
        student_started = time.monotonic()
        student_advance = streams.commit_and_advance(layer, student.model.layers[layer])
        after_deployed = student_advance['after_deployed']
        require_transition(inputs, student_advance['entries'], 'student', 'input',
            'Training must preserve detached current student inputs until the committed BF16 advance.')
        require_transition(teacher_advance['entries'], student_advance['entries'], 'output', 'teacher_target',
            'The committed student output must be compared with the actual independent teacher target of this layer.')
        torch.cuda.synchronize()
        student_seconds = time.monotonic()-student_started
        b.require(streams.layer_index == layer+1, 'Advance the submitted student prefix exactly once before the next real block.')
        runtime.offload_block(student, layer)
        completed_names.update(allowed_by_layer[layer])
        _actual, frozen = export.frozen_contract(student, source_records, completed_names)
        del _actual
        row = dict(layer=layer, input_stream=inputs, stream_transition=transition, teacher_advance=teacher_advance,
            teacher_seconds=teacher_seconds, before_deployed=before_deployed, before_live=before_live,
            training=training, commit=committed, deployment_warmup_before=warm_before,
            deployment_warmup_after=warm_after, after_deployed=after_deployed,
            student_advance=student_advance, student_advance_seconds=student_seconds,
            completed_prefix=list(range(streams.layer_index)), frozen_complement=frozen,
            seconds=time.monotonic()-layer_started, memory=fit.memory())
        layer_reports.append(row)
        observe('BR_block_committed_and_advanced', row)
    teacher_after = fit.frozen_proof(teacher)
    b.require(streams.layer_index == 2 and not streams.complete
        and teacher_after == teacher_frozen,
        'The independent BF16 teacher must remain unchanged; short execution stops after the real two-block prefix.')
    # No teacher or hidden pool remains on GPU while all216 deployment caches are prepared.
    del teacher_block, teacher, streams, pool
    gc.collect()
    torch.cuda.empty_cache()
    deployment_started = time.monotonic()
    warm = runtime.prepare_full_deployment(student, 'cuda:0')
    fixture['logits'] = probe(student, fixture)
    runtime.offload_full_deployment(student)
    b.require(all(value.device.type == 'cpu' for value in student.state_dict().values()),
              'Serialize the complete committed endpoint from CPU after releasing all decoded caches.')
    fixture_path = args.output/'same_input_output.pt'
    torch.save(fixture, fixture_path)
    final_directory = args.output/'final_model'
    meta, budget, expected, export_audit = export.save(student, final_directory, identity,
        source_audit, source_records, allowed)
    b.write(args.output/'expected_state.json', expected)
    b.require(export_audit['frozen_complement'] == source_frozen,
              'The final frozen complement differs from the exact converted initial source.')
    report = dict(**identity, status='training_and_BF16_export_complete_fresh_pending', exit_code=0,
        data=evidence, short_data=short_data,
        source_audit=source_audit, source_frozen_complement=source_frozen,
        trained_decoder_names=sorted(allowed), trained_decoder_names_by_layer=allowed_by_layer,
        teacher_frozen_before=teacher_frozen, teacher_frozen_after=teacher_after,
        teacher_frozen_exact=True, source_inventory=export_audit['groups'],
        layer_reports=layer_reports, completed_layers=list(LAYERS), updates_per_layer=2,
        deployment_prewarm=warm, deployment_and_export_seconds=time.monotonic()-deployment_started,
        export_audit=export_audit, final_memory=fit.memory(),
        export=dict(directory=str(final_directory), checkpoint_id=meta['checkpoint_id'], bit_budget=budget,
            expected_state=b.file_binding(args.output/'expected_state.json'), fixture=b.file_binding(fixture_path)),
        elapsed_seconds=time.monotonic()-started,
        scope='Full36-layer mixed topology; only blocks0/1 trained twice. No quality or formal32-update/36-block claim.')
    b.write(args.output/'report.json', report)
    observe('BR_adjacent_short_training_export_complete', dict(report=str(args.output/'report.json'), fresh_pending=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=b.SOURCE_B)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=14400)
    args = parser.parse_args()
    check_resources()
    args.output = args.output.resolve()
    b.require(not args.output.exists() and args.timeout_seconds > 0, 'Use a new isolated output and a positive short runtime bound.')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(137)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    binding = b.source_binding(args.checkpoint, prototype_only=args.checkpoint.resolve() == b.SOURCE_B.resolve())
    identity = dict(schema='BR_adjacent_blocks_short2_v1', prototype_only=binding['prototype_only'], validation_only=True,
        source_binding=binding, physical_GPUs=[6], logical_device='cuda:0', layers=list(LAYERS),
        group_indices=list(fit.GROUPS), seed=137, pid=os.getpid(), argv=sys.argv,
        python=sys.executable, cwd=os.getcwd(), source_sha256=code_bindings())
    b.write(args.output/'execution.json', identity)
    observe = lambda event, values: print(json.dumps(dict(event=event,
        elapsed_seconds=time.monotonic()-started, **values)), flush=True)
    try:
        run(args, identity, observe, started)
    except BaseException as error:
        b.write(args.output/'failure.json', dict(**identity, error=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started, memory=fit.memory()))
        (args.output/'run.exit').write_text('1\n', encoding='utf-8')
        raise
    (args.output/'run.exit').write_text('0\n', encoding='utf-8')


if __name__ == '__main__':
    main()
