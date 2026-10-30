"""One frozen G02 load; genuine boolq train first-four cache and batch paths only."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import weakref

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTECTION = ROOT / 'experiments/cat_aligned_protection_20261028'
sys.path.insert(0, str(PROTECTION))
import bindings as b
import data as aligned_data
import native

sys.path.insert(0, str(HERE))
from audit import audit_capture


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


@b.torch.no_grad()
def capture(output, gpu):
    if Path(sys.executable).resolve() != b.PYTHON.resolve() or Path.cwd() != b.SNAPSHOT:
        raise ValueError('Use caller-activated bitvae and the established snapshot cwd.')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu) or b.torch.cuda.device_count() != 1 or b.PROFILE != 'G':
        raise ValueError('Use exactly the root-assigned single GPU and G profile.')
    if output.exists():
        raise ValueError('Do not overwrite an existing diagnostic capture directory.')
    directory, meta, source_audit = b.source()
    cache_path = b.OUTPUT / 'R896/cache.json'
    record = b.read(cache_path)
    b.require(record['status'] == 'verified' and record['profile'] == 'G'
        and record['source_checkpoint_id'] == meta['checkpoint_id'] and record['source_path'] == str(directory)
        and record['source_audit'] == b.binding(b.SOURCE_AUDIT) and record['restored_projection_count'] == 0,
        'Require the verified G02 initial native R cache.')
    aligned_data.check_binding(record['dataset_binding'])
    aligned_data.check_binding(record['input_prepared'])
    payload_path = Path(record['payload']['path'])
    b.require(payload_path.stat().st_size == record['payload']['bytes'], 'Changed R payload size.')
    archive = b.torch.load(record['dataset_binding']['path'], map_location='cpu', weights_only=True)
    targets = b.torch.load(payload_path, map_location='cpu', weights_only=True, mmap=True)
    cases = archive['tasks']['boolq'][:4]
    b.require(len(cases) == 4 and all(case['case_id'].startswith('boolq/train/') for case in cases),
        'Use original first4 boolq train documents without quality-based selection.')
    output.mkdir(parents=True)
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    started = time.monotonic()
    event('loading_frozen_native_G02', source=str(directory), assigned_gpu=gpu,
        case_ids=[case['case_id'] for case in cases])
    model, load_audit = native.load_student(b.model_config(directory, meta))
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    initial = b.core.state_proof(model)
    rows = []
    for case in cases:
        cached = targets[case['case_id']]
        b.require(cached['input_signature'] == case['input_signature'], 'Changed R input signature.')
        b.require(len(cached['branches']) == len(case['branches']), 'Changed all-candidate branch support.')
        for index, branch in enumerate(case['branches']):
            target = cached['branches'][index]
            b.require(b.core.tensor_record(target) == record['target_records'][case['case_id']][index],
                'Changed selected saved R target tensor.')
            logits = b.base.answer_logits(model, branch)
            b.require(logits.dtype == b.torch.bfloat16 and logits.shape == target.shape,
                'Cache replay must keep complete native BF16 continuation support.')
            rows.append(dict(case_id=case['case_id'], input_signature=case['input_signature'], branch_index=index,
                input_ids=branch['input_ids'].clone(), answer_query_positions=branch['answer_query_positions'].clone(),
                serial_logits=logits.cpu().contiguous(),
                replayed_logp=b.torch.nn.functional.log_softmax(logits.float(), -1).cpu().contiguous(),
                cached_logp=target.clone(), document_branch_weight=.5/(7*4*len(case['branches']))))
            event('serial_branch_captured', case_id=case['case_id'], branch_index=index, shape=list(logits.shape))
            del logits
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    inputs, count, task_rows = aligned_data.loss.task_batch(cases, pad)
    device = model.get_input_embeddings().weight.device
    result = model(**{key: value.to(device) for key, value in inputs.items()}, use_cache=False, logits_to_keep=count)
    b.require(result.past_key_values is None and result.logits.dtype == b.torch.bfloat16
        and result.logits.shape == (len(task_rows), count, 151936), 'Changed original padded native task forward.')
    for index, ((case, branch_index, branch), row) in enumerate(zip(task_rows, rows, strict=True)):
        b.require(row['case_id'] == case['case_id'] and row['branch_index'] == branch_index, 'Changed original batch order.')
        row['batch_logits'] = result.logits[index, -len(branch['continuation_ids']):].cpu().contiguous()
    event('native_task_batch_captured', branches=len(rows), input_shape=list(inputs['input_ids'].shape))
    del result
    terminal = b.core.state_proof(model)
    b.require(initial == terminal, 'Frozen G02 persistent state changed during read-only capture.')
    captured = dict(schema='softmax_R_boolq_first4_capture_v1', source=dict(checkpoint_dir=str(directory),
        checkpoint_id=meta['checkpoint_id'], source_audit=b.binding(b.SOURCE_AUDIT), R_cache=b.binding(cache_path)),
        protocol=dict(task='boolq', group_index=0, documents=4, case_ids=[case['case_id'] for case in cases],
            branches=len(rows), continuation_tokens=sum(len(row['serial_logits']) for row in rows),
            vocab_size=151936, logits_dtype='torch.bfloat16', target_dtype='torch.float32',
            reduction='sum vocabulary / true branch token count; .5/(7*4*document branch count); sum branches',
            batch_input_shape=list(inputs['input_ids'].shape), batch_logits_to_keep=count,
            no_training=True, no_QA_generation=True, no_FP32_decoder_live_path=True),
        initial_state=initial, terminal_state=terminal, load_audit=load_audit, branches=rows,
        execution=dict(command_argv=sys.argv, assigned_physical_gpu=gpu, pid=os.getpid(),
            cwd=str(Path.cwd()), python=sys.executable, device_name=b.torch.cuda.get_device_name(0)),
        implementation={str(path): b.binding(path) for path in (
            HERE/'capture_r.py', HERE/'audit.py', HERE/'numerics.py', PROTECTION/'bindings.py',
            PROTECTION/'native.py', PROTECTION/'data.py',
            Path(aligned_data.loss.__file__), Path(b.base.__file__))})
    reference = weakref.ref(model)
    del model
    gc.collect()
    b.torch.cuda.empty_cache()
    b.torch._C._cuda_clearCublasWorkspaces()
    b.torch.cuda.empty_cache()
    b.require(reference() is None and b.torch.cuda.memory_allocated() == 0,
        'Release the complete frozen native model before auditing detached logits.')
    for item in captured['implementation'].values():
        b.require(b.binding(item['path']) == item, 'Bound implementation changed during capture.')
    captured['capture_seconds'] = time.monotonic()-started
    b.torch.save(captured, output / 'capture.pt')
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    captured = capture(args.output.resolve(), args.gpu)
    # Only detached logits participate in this autograd audit, after the model was released.
    report = audit_capture(captured, b.torch.device('cuda:0'))
    b.core.write(args.output / 'audit_cuda.json', report)
    print(json.dumps(dict(status=report['status'], output=str(args.output), aggregate=report['aggregate'])), flush=True)


if __name__ == '__main__':
    main()
