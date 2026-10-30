"""Full-vocabulary R896 from this profile's complete frozen actual aligned02 initial state."""
import gc
from pathlib import Path
import time
import weakref
import bindings as b
import data

CACHE = b.OUTPUT / 'R896'
SCHEMA = 'CG_aligned02_native_R896_v1'


def validate_targets(archive, record, targets):
    wanted = [case['case_id'] for task in b.TASKS for case in archive['tasks'][task]]
    b.require(list(targets) == wanted and len(targets) == record['documents'] == 896, 'Changed complete R896 target identity/order.')
    maximum, rows, tokens = 0., {}, 0
    for task in b.TASKS:
        for case in archive['tasks'][task]:
            item = targets[case['case_id']]
            b.require(item['input_signature'] == case['input_signature'] and len(item['branches']) == len(case['branches']),
                'R target native input/all-branch identity changed.')
            rows[case['case_id']] = []
            for branch, target in zip(case['branches'], item['branches'], strict=True):
                k = len(branch['continuation_ids'])
                b.require(target.shape == (k, 151936) and target.dtype == b.torch.float32 and target.device.type == 'cpu'
                    and not target.requires_grad and b.torch.isfinite(target).all().item(), 'R must cover every continuation token/full vocabulary.')
                error = float((target.exp().sum(-1)-1).abs().max())
                b.require(error < 2e-6, 'R full-vocabulary probability is not normalized.')
                maximum = max(maximum, error)
                tokens += k
                rows[case['case_id']].append(b.core.tensor_record(target))
    b.require(tokens == 15462 and tokens*151936*4 == 9396937728 and rows == record['target_records'],
        'R continuation budget or actual saved tensor SHA changed.')
    return dict(documents=896, tokens=tokens, raw_FP32_bytes=tokens*151936*4,
        maximum_probability_normalization_error=maximum, target_records_sha256=b.core.mapping_digest(rows))


def load(archive, materials):
    directory, meta, _ = b.source()
    return load_native(archive, materials, directory=directory, meta=meta,
        source_audit=b.binding(b.SOURCE_AUDIT), cache_dir=CACHE, profile=b.PROFILE, schema=SCHEMA)


def load_native(archive, materials, *, directory, meta, source_audit, cache_dir, profile, schema):
    """Caller validates the real parent; this reader binds its complete native R cache."""
    data.check_binding(source_audit)
    record = b.read(Path(cache_dir) / 'cache.json')
    b.require(record['schema'] == schema and record['status'] == 'verified'
        and record['profile'] == profile and record['source_checkpoint_id'] == meta['checkpoint_id']
        and record['source_path'] == str(directory) and record['source_audit'] == source_audit
        and record['dataset_binding'] == materials['retention_dataset']
        and record['input_prepared'] == materials['retention_prepared']
        and record['restored_projection_count'] == 0 and record['teacher_fully_released']
        and record['source_initial_state'] == record['source_terminal_state'], 'Use only the same-profile actual parent initial native R; never another student/F.')
    data.check_binding(record['payload'])
    targets = b.torch.load(record['payload']['path'], map_location='cpu', weights_only=True, mmap=True)
    checked = validate_targets(archive, record, targets)
    b.require(checked == record['readback_validation'], 'Actual R payload changed after verified readback.')
    return targets, record


@b.torch.no_grad()
def build(archive, materials, observe):
    directory, meta, _ = b.source()
    return build_native(archive, materials, observe, directory=directory, meta=meta,
        source_audit=b.binding(b.SOURCE_AUDIT), cache_dir=CACHE, profile=b.PROFILE, schema=SCHEMA)


@b.torch.no_grad()
def build_native(archive, materials, observe, *, directory, meta, source_audit, cache_dir, profile, schema):
    """Generate once from a caller-validated real native parent, with zero HF restoration."""
    from native import load_student
    data.check_binding(source_audit)
    cache_dir = Path(cache_dir)
    b.require(profile in ('C', 'G') and not cache_dir.exists(), 'Never overwrite or automatically retry a profile R cache.')
    cache_dir.mkdir(parents=True)
    config = b.model_config(directory, meta)
    started = time.monotonic()
    model, audit = load_student(config)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    source, policy = b.core.state_proof(model), b.core.native_policy(model)
    b.require(audit['checkpoint_id'] == meta['checkpoint_id'] and audit['persistent_tensors'] == 3423,
        'R teacher must be the complete actual native initial model with zero HF restoration.')
    cached, records, maximum = {}, {}, 0.
    for task in b.TASKS:
        for case in archive['tasks'][task]:
            values = []
            for branch in case['branches']:
                logits = b.base.answer_logits(model, branch)
                b.require(logits.dtype == b.torch.bfloat16, 'R must start from original native BF16 logits.')
                probs = b.torch.nn.functional.log_softmax(logits.float(), dim=-1).cpu().contiguous()
                b.require(probs.shape == (len(branch['continuation_ids']), 151936)
                    and probs.dtype == b.torch.float32 and b.torch.isfinite(probs).all().item(), 'Changed actual R full-vocabulary support.')
                error = float((probs.exp().sum(-1)-1).abs().max())
                b.require(error < 2e-6, 'Actual R target is not normalized.')
                maximum = max(maximum, error)
                values.append(probs); del logits
            cached[case['case_id']] = dict(input_signature=case['input_signature'], branches=values)
            records[case['case_id']] = [b.core.tensor_record(value) for value in values]
            observe('native_initial_R_document_cached', dict(task=task, case_id=case['case_id'], documents_complete=len(cached)))
    terminal = b.core.state_proof(model)
    b.require(terminal == source and policy == b.core.native_policy(model), 'Frozen native parent state/numerical policy changed.')
    peak = dict(allocated=b.torch.cuda.max_memory_allocated(), reserved=b.torch.cuda.max_memory_reserved())
    reference, parameters = weakref.ref(model), [weakref.ref(p) for p in model.parameters()]
    del model
    gc.collect(); b.torch.cuda.empty_cache(); b.torch._C._cuda_clearCublasWorkspaces(); b.torch.cuda.empty_cache()
    b.require(reference() is None and all(ref() is None for ref in parameters) and b.torch.cuda.memory_allocated() == 0,
        'Release complete native parent R teacher after caching.')
    path = cache_dir / 'targets.pt'
    b.torch.save(cached, path)
    record = dict(schema=schema, status='verified', profile=profile,
        source_checkpoint_id=meta['checkpoint_id'], source_path=str(directory), source_audit=source_audit,
        source_initial_state=source, source_terminal_state=terminal, native_policy=policy, load_audit=audit,
        restored_projection_count=0, teacher_fully_released=True, dataset_binding=materials['retention_dataset'],
        input_prepared=materials['retention_prepared'], payload=b.binding(path), documents=len(cached),
        target_records=records, maximum_probability_normalization_error=maximum, peak_bytes=peak,
        elapsed_seconds=time.monotonic()-started, raw_FP32_probability_bytes=9396937728,
        semantics='BF16native logits -> FP32 full151936 log_softmax; original train896/allcandidatebranches/15462continuationtokens; noHFrestore')
    del cached
    reread = b.torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    record['readback_validation'] = validate_targets(archive, record, reread)
    b.core.write(cache_dir / 'cache.json', record)
    del reread
    return record
