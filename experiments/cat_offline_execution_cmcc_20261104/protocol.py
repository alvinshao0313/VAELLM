"""Separate common evaluation conditions, quality references and artifact facts."""
import importlib.metadata
import importlib.util
import hashlib
import json
import math
from pathlib import Path

from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha
from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require

REPO = Path(__file__).resolve().parents[2]
SAMPLE_HELPER = REPO / 'result/qwen3_8b_mmlu65_20261004/lm_eval_sample_artifacts_20261010/eval_utils_with_samples.py'
SAMPLE_WRITER = SAMPLE_HELPER.with_name('sample_artifacts.py')
DEFAULT_BASELINE = REPO / 'artifacts/cmcc_baselines_20261030/qwen3_8b/migration_binding.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'Never overwrite a frozen receipt/result: ' + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def verify_files(values):
    for path, expected in values.items():
        require(file_sha(path) == expected, 'Bound source/artifact changed: ' + path)


def code_identity():
    from experiments.cat_auto_protocol_cmcc_20261104.adapter import public_consumer_identity
    paths = set(Path(__file__).parent.glob('*.py')) | {SAMPLE_HELPER, SAMPLE_WRITER}
    paths.update(REPO / name for name in ('tools/cat_eval.py', 'train_utils/v6_model_loader.py',
        'train_utils/checkpoint_v6.py', 'rotation/model_utils.py', 'experiments/w25_eval/acceptance.py',
        'experiments/w25_eval/model_inventory.py', 'experiments/w25_eval/transfer_nine_task_acceptance.py',
        'experiments/qwen3_w25/summarize_results.py','experiments/cmcc_runtime_20261030/sequence.py'))
    paths.update((REPO/'experiments/cat_auto_protocol_cmcc_20261104').glob('*.py'))
    paths.update(Path(importlib.util.find_spec(name).origin) for name in ('lm_eval.evaluator','lm_eval.models.huggingface'))
    return {**public_consumer_identity()['source_sha256'], **{str(path): file_sha(path) for path in sorted(paths)}}


def baseline_receipt(inventory_receipt, *, migration_binding=DEFAULT_BASELINE):
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.model_inventory import model_inventory
    frozen = read(inventory_receipt)
    inventory = frozen['inventory']
    source = Path(inventory['source_evidence']['path']).resolve()
    migration = read(migration_binding)
    require(migration['schema'] == 'cmcc_qwen3_bf16_baseline_migration_v1' and migration['status'] == 'verified',
        'Use the actual verified cmcc BF16 reference migration.')
    original=migration['original_HF_identity'];model=inventory['model_identity']
    require(original['repository']==model['repository'] and original['revision']==model['revision']
        and {name:row['sha256'] for name,row in original['weights'].items()}==model['weight_shards_sha256'],
        'Same config with different actual repository/revision/weight shards cannot inherit the BF16 reference score.')
    verified=original['completed_byte_verification']
    require(file_sha(verified['path'])==verified['sha256'],'Original completed BF16 byte verification receipt changed.')
    official={row['name']:row for row in read(verified['path'])['files']}
    for name,expected in {**model['tokenizer_files_sha256'],'config.json':model['config_sha256']}.items():
        data=(source/name).read_bytes();row=official[name]
        require(hashlib.sha256(data).hexdigest()==expected and len(data)==row['bytes']
            and ('sha256' not in row or row['sha256']==expected)
            and ('git_blob_sha1' not in row or hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==row['git_blob_sha1']),
            'Actual model/tokenizer metadata differs from the original verified BF16 bytes: '+name)
    paths = [Path(migration['bindings'][name]['path']) for name in ('mmlu', 'seven')]
    require(all(file_sha(path) == migration['bindings'][name]['sha256'] for name, path in zip(('mmlu', 'seven'), paths, strict=True)),
        'Original migrated BF16 result bytes changed.')
    shape_inventory = model_inventory(source)
    require(shape_inventory['config_sha256'] == inventory['model_identity']['config_sha256'], 'Reference model config differs from the actual frozen model content.')
    scores = collect_results(paths, shape_inventory)
    raw = [read(path) for path in paths]
    require(raw[0]['max_length'] == raw[1]['max_length'], 'Reference shards used different likelihood windows.')
    seeds={key:raw[0]['config'][key] for key in ('random_seed','numpy_seed','torch_seed','fewshot_seed')}
    require(seeds==dict(random_seed=0,numpy_seed=1234,torch_seed=1234,fewshot_seed=1234)
        and all({key:payload['config'][key] for key in seeds}==seeds for payload in raw),'Recorded BF16 shards do not share the original evaluation seed conditions.')
    config = read(source / 'config.json')
    versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'lm_eval')}
    require(all(payload['transformers_version']==versions['transformers'] for payload in raw), 'Actual Transformers runtime differs from the recorded BF16 reference.')
    evaluator=dict(installed_versions=versions,sample_preserving_helper_sha256=file_sha(SAMPLE_HELPER),sample_writer_sha256=file_sha(SAMPLE_WRITER),
        HFLM_sha256=file_sha(importlib.util.find_spec('lm_eval.models.huggingface').origin),evaluator_sha256=file_sha(importlib.util.find_spec('lm_eval.evaluator').origin))
    common = dict(schema='cat_common_full8_protocol_v0', model_identity=inventory['model_identity'],evaluator=evaluator,
        tokenizer_metadata_sha256=inventory['model_identity']['tokenizer_files_sha256'],
        tasks={task: dict(config=value, version=payload['versions'][task], samples=payload['n-samples'][task], shots=payload['n-shot'][task])
            for payload in raw for task, value in payload['configs'].items()},
        task_metric_keys=scores['task_metric_keys'], leaf_count=scores['leaf_count'], sample_count=scores['sample_count'],
        execution=dict(batch_by_shard=dict(mmlu=64, seven=32), num_fewshot=0, limit=None, dtype='bfloat16',
            attn_implementation='flash_attention_2' if config['model_type'] == 'qwen3' else 'sdpa',
            rope_buffers_dtype='float32', rope_scaling=config.get('rope_scaling'), rope_theta=config['rope_theta'],
            model_type=config['model_type'], max_length=raw[0]['max_length'], add_bos_token=False, use_fast_tokenizer=True,
            use_cache=False, chat_template=None, system_instruction=None, fewshot_as_multiturn=False,
            **seeds),
        comparison_evidence_level='Actual full task/config/version and recorded execution/source conditions; historical BF16 ordered pairing unavailable')
    return dict(schema='cat_bf16_full8_quality_receipt_v0', kind='BF16', reference_id='BF16:' + digest(inventory['model_identity']),
        quality_reference_verified=True, ordered_pairing_verified=False,
        ordered_sample_identity_evidence=dict(status='unavailable', reason='The original evaluator did not serialize BF16 full8 samples; no retrospective hashes or new inference.'),
        model_identity=inventory['model_identity'], protocol=common, protocol_identity=digest(common),
        **scores, source_evidence=dict(migration=binding(migration_binding), inventory_receipt=binding(inventory_receipt),
            model_path=str(source), result_files=[binding(path) for path in paths]),
        tokenizer_inventory=shape_inventory,
        evaluator_identity=dict(current_candidate_sample_preserving_helper=binding(SAMPLE_HELPER),current_candidate_sample_writer=binding(SAMPLE_WRITER),
            current_installed_versions=versions,historical_recorded=[dict(transformers_version=payload['transformers_version'],git_hash=payload.get('git_hash'),execution_config=payload['config']) for payload in raw]),
        boundary='Verified historical BF16 aggregate quality reference; no v6 UUID, ordered-pairing or retrospective prediction claim.')


def load_baseline(path):
    receipt=read(path)
    source=receipt['source_evidence'];inventory=source['inventory_receipt'];migration=source['migration']
    require(file_sha(inventory['path'])==inventory['sha256'] and file_sha(migration['path'])==migration['sha256'], 'Bound BF16 inventory/migration receipt changed.')
    require(receipt==baseline_receipt(inventory['path'],migration_binding=migration['path']), 'BF16 quality/protocol receipt differs from its actual source artifacts.')
    return receipt


def sample_evidence(paths):
    """Validate real serialized JSONL bytes and retain input identities separately."""
    files, leaves = {}, {}
    for path in paths:
        result = read(path)['evals']['lm_eval']
        samples, counts = result['sample_artifacts'], result['n_samples']
        require(samples['total_count'] == sum(row['effective'] for row in counts.values()) and set(samples['tasks']) == set(counts), 'Missing actual sample artifacts.')
        for task, item in samples['tasks'].items():
            source = Path(item['path']).resolve()
            require(source.is_relative_to(Path(path).resolve().parent) and source.is_file() and file_sha(source) == item['sha256'], 'Raw sample path or SHA changed.')
            with source.open(encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream]
            require(len(rows) == item['count'] == counts[task]['effective'] and task not in leaves, 'Raw sample count/leaf duplicated or incomplete.')
            identity = [[row[key] for key in ('doc_id', 'doc_hash', 'prompt_hash', 'target_hash')] for row in rows]
            leaves[task] = dict(count=len(rows), ordered_input_digest=digest(identity))
            files[str(source)] = item['sha256']
    require(len(leaves) == 64 and sum(row['count'] for row in leaves.values()) == 24742, 'All64 actual leaves/24742 raw predictions required.')
    return dict(status='verified_actual_serialized_samples', leaf_count=64, sample_count=24742, files_sha256=files,
        per_leaf=leaves, ordered_input_identity=digest(leaves))


def collect_native(paths, *, endpoint, baseline):
    from experiments.w25_eval.acceptance import collect_results
    require(baseline['quality_reference_verified'] and baseline['protocol_identity'] == digest(baseline['protocol']), 'Invalid BF16 common protocol receipt.')
    meta = read(Path(endpoint['path']) / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == endpoint['checkpoint_id'] and file_sha(Path(endpoint['path']) / 'checkpoint_meta.json') == endpoint['metadata_sha256'], 'Native endpoint UUID/metadata changed.')
    metrics = collect_results(paths, baseline['tokenizer_inventory'], checkpoint=(Path(endpoint['path']).resolve(), meta))
    for path in paths:
        payload = read(path)
        require(payload['endpoint'] == endpoint and payload['protocol_identity'] == baseline['protocol_identity'], 'Different native artifact or protocol.')
        audit = payload['load_audit']
        require(audit['checkpoint_id'] == endpoint['checkpoint_id'] and audit['strict_missing'] == audit['strict_unexpected'] == []
            and audit['PUBLIC_consumer_verified'] and audit['dtype'] == 'torch.bfloat16' and audit['rope_FP32_verified']
            and audit['attn_implementation'] == baseline['protocol']['execution']['attn_implementation'], 'Unverified real PUBLIC/precision/backend consumer.')
        conditions=baseline['protocol']['execution'];lm=audit['HFLM']
        require(lm['max_length']==conditions['max_length'] and lm['add_bos_token'] is False and lm['log_samples'] is True
            and all(lm[key]==conditions[key] for key in ('random_seed','numpy_seed','torch_seed','fewshot_seed')),
            'Actual HFLM window/BOS/seed/sample execution differs.')
        require(payload['execution_conditions'] == baseline['protocol']['execution'], 'Actual native execution conditions differ from reference.')
        result = payload['evals']['lm_eval']
        for task, cfg in result['configs'].items():
            expected = baseline['protocol']['tasks'][task]
            require(cfg == expected['config'] and result['versions'][task] == expected['version'], 'Task/template/version differs from original BF16.')
    samples = sample_evidence(paths)
    require(math.isfinite(metrics['mean_percent']), 'Nonfinite full8 score.')
    return dict(schema='cat_native_full8_receipt_v0', checkpoint_id=endpoint['checkpoint_id'], endpoint=endpoint,
        **metrics, complete=True, protocol_identity=baseline['protocol_identity'], protocol=baseline['protocol'],
        ordered_sample_identity_evidence=samples, BF16_ordered_pairing_verified=False,
        quality_reference_verified=True, baseline_reference_id=baseline['reference_id'],
        evaluation_digest=digest(dict(endpoint=endpoint, shards=[binding(path) for path in paths], samples=samples)),
        formal_eligible=endpoint['formal_eligible'], phase=endpoint['phase'])


def read_native_eight(*, summary_path, evaluation_path):
    """Re-read actual artifacts and return the existing recovery score type."""
    from experiments.cat_auto_protocol_cmcc_20261104.nodes import load_complete_eight_score
    from .evaluate import verify_plan
    evaluation=read(evaluation_path)
    plan_path=Path(evaluation['source_plan']['path'])
    require(file_sha(plan_path)==evaluation['source_plan']['sha256'], 'Actual full8 execution plan changed.')
    plan=read(plan_path);baseline,_=verify_plan(plan)
    require(plan['endpoint']['kind']=='native_v6' and plan['formal_eligible'] and plan['phase']=='formal', 'Only a formal real native endpoint becomes a recovery full-eight score.')
    summary=read(summary_path)
    paths=[Path(row['path']) for row in summary['input_results']]
    actual=collect_native(paths,endpoint=plan['endpoint'],baseline=baseline)
    require(summary==actual, 'Written native full8 summary differs from its actual complete artifacts.')
    manifest=dict(plan_path=str(plan_path),plan_sha256=file_sha(plan_path),checkpoint_meta_sha256=plan['endpoint']['metadata_sha256'],
        protocol_id=baseline['protocol_identity'],leaf_count=64,sample_count=24742,
        task_metric_keys=baseline['task_metric_keys'],expected_stages=plan['expected_stages'])
    score=load_complete_eight_score(summary_path=summary_path,evaluation_path=evaluation_path,
        checkpoint_dir=plan['endpoint']['path'],source_checkpoint_id=plan['endpoint']['checkpoint_id'],protocol_manifest=manifest)
    return score,actual


def collect_dense(paths, *, endpoint, baseline):
    """Collect a newly budgeted actual BF16 run without inventing a v6 UUID."""
    from experiments.w25_eval.acceptance import leaf_metrics,_group_agrees
    from experiments.qwen3_w25.summarize_results import TASK_METRICS
    scores,subjects,versions,sources={},{},{},[]
    require(endpoint['kind']=='BF16' and endpoint['model_identity']==baseline['model_identity'], 'Original BF16 content identity required.')
    for path in paths:
        payload=read(path);result=payload['evals']['lm_eval'];audit=payload['load_audit']
        require(payload['model_kind']=='BF16' and payload['reference_id']==endpoint['reference_id'] and payload['endpoint']==endpoint
            and payload['protocol_identity']==baseline['protocol_identity'] and payload['execution_conditions']==baseline['protocol']['execution'], 'Wrong actual BF16 endpoint/protocol.')
        require(audit['PUBLIC_consumer_verified'] and audit['rope_FP32_verified'] and audit['dtype']=='torch.bfloat16'
            and audit['attn_implementation']==baseline['protocol']['execution']['attn_implementation'], 'Actual original BF16 precision/backend not verified.')
        current,sub,batch=leaf_metrics(result['raw_results'],result['n_samples'],result['n_shot'],result['configs'])
        _group_agrees(current,result['raw_results'],result['group_results'])
        require(result['num_fewshot']==0 and result['limit'] is None and str(result['batch_size'])==str(batch)
            and not set(current)&set(scores), 'Incomplete, truncated or duplicate BF16 shard.')
        for task,cfg in result['configs'].items():
            expected=baseline['protocol']['tasks'][task]
            require(cfg==expected['config'] and result['versions'][task]==expected['version'], 'BF16 task/template/version changed.')
        for task,value in current.items():
            require(result['task_metric_keys'][task]==TASK_METRICS[task] and math.isclose(result['task_metrics'][task],value,rel_tol=0,abs_tol=1e-12), 'BF16 serializer/raw metric mismatch.')
        lm=audit['HFLM'];policy=baseline['protocol']['execution']
        require(lm['max_length']==policy['max_length'] and lm['add_bos_token'] is False and lm['log_samples'] is True
            and all(lm[key]==policy[key] for key in ('random_seed','numpy_seed','torch_seed','fewshot_seed')), 'Real BF16 HFLM conditions differ.')
        scores.update(current);subjects.update(sub);versions.update({task:result['versions'][task] for task in result['n_samples']})
        sources.append(dict(**binding(path),tasks=list(current),leaf_count=len(result['n_samples']),sample_count=sum(row['effective'] for row in result['n_samples'].values()),batch_size=batch,format='actual_PUBLIC_BF16_full8'))
    require(set(scores)==set(TASK_METRICS) and len(subjects)==57 and sum(row['leaf_count'] for row in sources)==64
        and sum(row['sample_count'] for row in sources)==24742,'Both complete BF16 shards required.')
    samples=sample_evidence(paths)
    return dict(schema='cat_actual_BF16_full8_receipt_v0',kind='BF16',reference_id=endpoint['reference_id'],endpoint=endpoint,
        model_identity=baseline['model_identity'],complete=True,quality_reference_verified=True,protocol_identity=baseline['protocol_identity'],
        protocol=baseline['protocol'],task_metrics=scores,task_metric_keys=TASK_METRICS,mmlu_subject_metrics=subjects,versions=versions,
        mean_percent=math.fsum(scores.values())*100/8,input_results=sources,leaf_count=64,sample_count=24742,
        ordered_sample_identity_evidence=samples,evaluation_digest=digest(dict(endpoint=endpoint,shards=[binding(path) for path in paths],samples=samples)),
        boundary='New actual budgeted BF16 execution; the historical BF16 reference remains separate and has no retrospective samples.')
