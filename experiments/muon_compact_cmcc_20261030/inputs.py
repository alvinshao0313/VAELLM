"""Read migrated immutable HOT artifacts without changing their historical identity."""
from copy import deepcopy

from safetensors import safe_open

from . import bindings as b

HOT = b.R / 'qwen_hotpot_recovery_20261012'


def checked(binding):
    path = b.resolve_legacy_path(binding['path'])
    b.require(path.stat().st_size == binding['bytes'] and b.sha256(path) == binding['sha256'],
              'Changed migrated artifact: ' + str(path))
    return path


def load_inputs():
    # Same validated train cases/groups and once-only F-M mixture as aligned.load_inputs.
    # The immutable files retain old paths; only physical file resolution is rebound.
    recipe = b.aligned.reference.original_record()
    manifest = b.read(HOT / 'dataset_prepared.json')
    dataset = b.torch.load(checked(manifest['dataset_binding']), map_location='cpu', weights_only=True)
    control_manifest = b.read(b.R / 'qwen_teacher_scope_qa128_20261011/dataset_prepared.json')
    control = b.torch.load(checked(control_manifest['dataset_binding']), map_location='cpu', weights_only=True)
    hot_manifest = b.read(b.R / 'qwen_hotpot_train_cpu_20261011/prepared.json')
    hot = b.torch.load(checked(hot_manifest['dataset_binding']), map_location='cpu', weights_only=True)
    properties = b.aligned.reference.data.validate(dataset, (control, hot, control_manifest, hot_manifest))
    b.require(dataset['groups'] == recipe['groups'] and dataset['train_ids'] == recipe['train_ids']
              and dataset['evidence'] == manifest['data_contract']
              and properties == manifest['CPU_validation']['data_properties'], 'Changed actual HOT train cases/groups.')
    composed = b.read(HOT / 'teacher_targets.json')
    b.require(composed['status'] == 'complete' and composed['dataset_binding'] == manifest['dataset_binding'],
              'Changed original composed full-F target manifest.')
    validators = b.module(HOT / 'teacher_targets.py', '_mixed_muon_target_validator')
    old = {}
    records = []
    for index, component in enumerate(composed['components']):
        record = b.read(checked(component['cache_binding']))
        payload = b.torch.load(checked(component['payload_binding']), map_location='cpu', weights_only=True, mmap=True)
        cases = dataset['training_cases'][:1056] if index == 0 else dataset['training_cases'][1056:]
        b.require(record['teacher_kind'] == payload['teacher_kind'] == 'full7'
                  and record['restored_projection_count'] == 252 and record['remaining_compressed_projection_count'] == 0
                  and record['payload'] == component['payload_binding'] and record['teacher_fully_released']
                  and record['source_original_state'] == payload['source_original_state'] == recipe['validated_source_state']
                  and record['teacher_projection_names'] == payload['teacher_projection_names'] == recipe['teacher_projection_names']
                  and component['case_ids'] == [case['case_id'] for case in cases]
                  and validators.verify_targets(record, payload, cases) == component['target_properties'],
                  'Changed original full-F cached support/identity.')
        for case in cases:
            old[case['case_id']] = dict(case=case, **payload['targets'][case['case_id']])
        records.append(record)
    mixed, evidence = b.aligned.reference.mixed.actual_targets(old)
    b.require(evidence == recipe['actual_target_evidence'], 'M mixture was applied differently or twice.')
    recovery = {key: dict(case=value['case'], **({'target_log_probs': value['target_log_probs']}
                if value['case']['domain'] == 'mmlu' else {})) for key, value in mixed.items()}
    b.require(len(recovery) == 1152 and sum('target_log_probs' in value for value in recovery.values()) == 1024,
              'Only F-M1024 targets may survive; recompute every student rollout and QA full-F target.')
    return dataset, recovery, dict(manifest=b.binding(HOT / 'teacher_targets.json'),
        source='Original full-F cache, QA discarded before training', native_policy=records[0]['native_policy']), evidence


def teacher_identity(source_state, names):
    recipe = b.aligned.reference.original_record()
    old_config = recipe['model_config']
    metadata = {}
    for old_path, digest in old_config['model']['hf_metadata_sha256'].items():
        path = b.resolve_legacy_path(old_path)
        b.require(b.sha256(path) == digest, 'Current HF/tokenizer differs from cached teacher: ' + str(path))
        metadata[str(path)] = digest
    verified_path = b.ROOT / 'result/prepare_eval_20261030/qwen_verified.json'
    verified = b.read(verified_path)
    b.require(verified['status'] == 'passed' and verified['repository'] == 'Qwen/Qwen3-8B', 'Missing current HF identity.')
    shards = []
    for row in verified['files']:
        if row['name'].endswith('.safetensors'):
            path = b.HF / row['name']
            b.require(path.stat().st_size == row['bytes'] and b.sha256(path) == row['sha256'], 'HF shard identity changed.')
            shards.append(b.binding(path))
    outside = {key: value for key, value in source_state.items() if not key.startswith(tuple(name + '.' for name in names))}
    b.require(len(outside) == 147, 'Actual nonprojection backbone must contain147 fields.')
    index = b.read(b.HF / 'model.safetensors.index.json')['weight_map']
    for key, value in outside.items():
        with safe_open(str(b.HF / index[key]), framework='pt', device='cpu') as stream:
            expected = stream.get_tensor(key)
        b.require(expected.dtype == value.dtype and expected.shape == value.shape and b.torch.equal(expected, value),
                  'B source does not match full-HF teacher backbone: ' + key)
    config = deepcopy(old_config)
    config['model']['hf_path'] = str(b.HF)
    return dict(hf_metadata_sha256=metadata, hf_shards=shards, migration_verified=b.binding(verified_path),
        nonprojection_tensors_compared=len(outside), nonprojection_SHA=b.core.mapping_digest({key:b.core.tensor_record(value)
            for key,value in outside.items()}), source_backbone_matches_current_HF_exact=True,
        model_config=config, protocol=config['protocol'], teacher='Original unmodified full-HF BF16 FA2; FP32 RoPE')
