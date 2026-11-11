"""The single fixed full-H9636-block/32-update BR candidate; no resume/grid."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b
from . import export, source, training
from experiments.block_reconstruction_cmcc_20261030.run_short import input_fixture, probe


def execute(args, parent, parent_meta, reused):
    started = time.monotonic()
    model, source_audit = b.load_student(parent, parent_meta)
    source_records = b.fit.state_records(model)
    allowed_by_layer = {index: sorted(b.runtime.decoder_parameters(model, index)) for index in b.LAYERS}
    allowed = {name for names in allowed_by_layer.values() for name in names}
    b.require(allowed == export.decoder_names(parent_meta)
        and len(allowed) == sum(map(len, allowed_by_layer.values())), 'Derive the exact disjoint actual D union from source modules.')
    source_frozen = b.fit.proof_from_records(source_records, allowed)
    pool, data_evidence = b.data.load(model.config.pad_token_id)
    full_data = b.data.summary(pool)
    b.require(len(pool.entries) == 160 and [group.group_index for group in pool.groups] == list(b.GROUPS),
              'Formal BR consumes all original H96 entries and32 groups without truncation or selection.')
    fixture = input_fixture(pool)
    identity = dict(schema='BR36_H96_32_execution_v1', phase='BR36', validation_only=False, prototype_only=False,
        source_C0_binding=parent, contract=b.CONTRACT, layer_ids=list(b.LAYERS), group_indices=list(b.GROUPS),
        physical_GPUs=[args.gpu], logical_device='cuda:0', seed=137, pid=os.getpid(),
        start_ticks=b.sequence.proc_start(os.getpid()), argv=sys.argv, python=sys.executable,
        cwd=os.getcwd(), source_sha256=b.code_bindings(), reused_short=reused)
    b.write(args.output/'source_state.json', source_records)
    source_state = b.file_binding(args.output/'source_state.json')
    b.write(args.output/'execution.json', identity)
    teacher = b.core.load_teacher()
    teacher_before = b.fit.frozen_proof(teacher)
    streams = b.rolling.RollingStreams.capture(teacher, model, pool, device='cuda:0')
    block_reports, completed, previous = [], set(), None
    blocks = args.output/'blocks'
    blocks.mkdir()

    def observe(event, values):
        print(json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started, **values)), flush=True)

    observe('BR36_independent_streams_captured', dict(initialization_passes=streams.initialization_passes,
        entries=160, memory=b.fit.memory()))
    for index in b.LAYERS:
        began = time.monotonic()
        teacher_block = teacher.model.layers[index]
        teacher_block.to(device='cuda:0')
        teacher_advance = streams.prepare_layer(index, teacher_block)
        teacher_block.cpu()
        if previous is not None:
            source.transition(previous['teacher_advance']['entries'], teacher_advance['entries'])
            current = {entry_index: batch.hidden_states for group in pool.groups
                for entry_index, batch in zip(group.entry_indices, streams.group(group.group_index).batches, strict=True)}
            inputs = [dict(entry_index=i, entry_id=pool.entries[i].entry_id,
                           input=b.fit.tensor_record(current[i])) for i in sorted(current)]
            source.transition(previous['student_advance']['entries'], inputs)
            del current, inputs
        warm_before = b.runtime.prepare_deployment_block(model, index, 'cuda:0')
        before_deployed = training.evaluate_block(model, index, streams)
        active = b.runtime.enable_block_training(model, index)
        b.require(set(active) == set(allowed_by_layer[index]), 'Only the actual current seven projection D leaves may train.')
        before_live = training.evaluate_block(model, index, streams)
        training_record = training.train_block(model, index, streams, active, observe)
        expected_commit = {name: b.fit.tensor_record(value.detach().to(device='cpu', dtype=b.torch.bfloat16))
                           for name, value in active.items()}
        committed = b.runtime.commit_block(model, index)
        b.require({name: b.fit.tensor_record(value) for name, value in b.runtime.decoder_parameters(model, index).items()}
            == expected_commit, 'Every committed D must equal its final FP32 master exact BF16 cast.')
        committed = dict(**committed, all_D_BF16_cast_exact=True, committed_D=expected_commit)
        del active
        warm_after = b.runtime.prepare_deployment_block(model, index, 'cuda:0')
        student_advance = streams.commit_and_advance(index, model.model.layers[index])
        source.transition(teacher_advance['entries'], student_advance['entries'], 'output', 'teacher_target')
        if previous is not None:
            source.transition(previous['student_advance']['entries'], student_advance['entries'])
        else:
            source.transition(teacher_advance['entries'], student_advance['entries'], 'input', 'input')
        b.require(streams.layer_index == index+1, 'Only a fully committed BF16 block advances the student prefix.')
        b.runtime.offload_block(model, index)
        completed.update(allowed_by_layer[index])
        _state, frozen = b.deployment.frozen_contract(model, source_records, completed)
        del _state
        row = dict(schema='BR36_H96_32_block_v1', validation_only=False, layer=index,
            completed_prefix=list(range(index+1)), teacher_advance=teacher_advance,
            before_deployed=before_deployed, before_live=before_live, training=training_record,
            commit=committed, deployment_warmup_before=warm_before, deployment_warmup_after=warm_after,
            student_advance=student_advance, after_deployed=student_advance['after_deployed'],
            frozen_complement=frozen, seconds=time.monotonic()-began, memory=b.fit.memory())
        path = blocks/f'layer_{index:02d}.json'
        b.write(path, row)
        block_reports.append(b.file_binding(path))
        previous = row
        observe('BR36_block_complete', dict(layer=index, completed_blocks=len(block_reports),
            block_report=block_reports[-1], seconds=row['seconds'], memory=row['memory']))
    teacher_after = b.fit.frozen_proof(teacher)
    b.require(streams.complete and teacher_before == teacher_after, 'All36 independent prefixes must finish with the original teacher frozen.')
    del teacher_block, teacher, streams, previous, row, pool
    gc.collect()
    b.torch.cuda.empty_cache()
    deployment_started = time.monotonic()
    warm = b.runtime.prepare_full_deployment(model, 'cuda:0')
    fixture['logits'] = probe(model, fixture)
    b.runtime.offload_full_deployment(model)
    fixture_path = args.output/'same_input_output.pt'
    b.torch.save(fixture, fixture_path)
    final = args.output/'final_model'
    meta, budget, expected, audit = export.save(model, final, identity, parent_meta,
        source_records, allowed, block_reports, source_state)
    b.require(audit['frozen_complement'] == source_frozen, 'The full D complement must remain exact against the original C0.')
    b.write(args.output/'expected_state.json', expected)
    report = dict(**identity, status='BR36_training_BF16_export_complete_fresh_pending', exit_code=0,
        source_state=source_state, source_audit=source_audit, source_geometry_sha256=export.geometry(parent_meta),
        source_frozen_complement=source_frozen, trained_decoder_names=sorted(allowed),
        trained_decoder_names_by_layer=allowed_by_layer, completed_layers=list(b.LAYERS),
        updates_per_layer=32, total_updates=1152, block_reports=block_reports,
        teacher_frozen_before=teacher_before, teacher_frozen_after=teacher_after, teacher_frozen_exact=True,
        data=data_evidence, full_data=full_data, deployment_prewarm=warm, export_audit=audit,
        deployment_and_export_seconds=time.monotonic()-deployment_started, elapsed_seconds=time.monotonic()-started,
        final_memory=b.fit.memory(), export=dict(directory=str(final), checkpoint_id=meta['checkpoint_id'], bit_budget=budget,
            expected_state=b.file_binding(args.output/'expected_state.json'), fixture=b.file_binding(fixture_path)),
        scope='One fixed full36x32 BR endpoint on actual C0; downstream quality and BR→E remain separate actual results.')
    b.write(args.output/'report.json', report)
    observe('BR36_export_complete', dict(checkpoint_id=meta['checkpoint_id'], report=str(args.output/'report.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pipeline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    args = parser.parse_args()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu),
        'Use caller-activated bitvae with exactly the assigned single physical GPU visible.')
    args.output = args.output.resolve()
    b.require(not os.path.lexists(args.output), 'Use a new BR output; no automatic resume, overwrite or retry.')
    reused = b.verified_short()
    parent, parent_meta = b.bind_C0(args.checkpoint, args.pipeline_report)
    b.sequence.check_resources([str(args.gpu)])
    b.require(b.torch.cuda.device_count() == 1, 'BR uses one actual visible GPU.')
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True)
    code = 1
    try:
        execute(args, parent, parent_meta, reused)
        code = 0
    except BaseException:
        b.write(args.output/'failure.json', dict(error=traceback.format_exc(), source_C0_binding=parent, memory=b.fit.memory()))
        raise
    finally:
        (args.output/'run.exit').write_text(str(code)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
