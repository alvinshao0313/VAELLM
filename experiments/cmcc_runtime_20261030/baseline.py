"""Relocate the fixed Qwen BF16 evidence without rerunning any evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .paths import ARTIFACTS, EVALUATION, MODELS, REPO, TOPIC, resolve_legacy_path


QWEN_BASELINE = ARTIFACTS / 'cmcc_baselines_20261030/qwen3_8b'
HF = MODELS / 'Qwen/Qwen3-8B'
MANIFEST = EVALUATION / 'longbench/manifest.json'
OLD_MANIFEST = REPO / 'result/qwen3_8b_longbench_20261004/data/manifest.json'
OLD_DENSE = OLD_MANIFEST.parent.parent / 'dense'
TEMPLATE = TOPIC / 'qwen_a0e5_baseline_s400_longbench21/merge_plan.json'
OFFICIAL_SOURCE = EVALUATION / 'qwen_source.json'
HF_VERIFICATION = EVALUATION / 'qwen_verified.json'
INPUT_VERIFICATION = REPO / 'result/migration_audit_20261030/baseline_input_validation.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def binding(path):
    from experiments.longbench_v1.common import sha256
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def _named_hashes(pins):
    names = {Path(path).name: digest for path, digest in pins.items()}
    require(len(names) == len(pins), 'Repeated basename in the fixed input identity.')
    return names


def _lm_sources():
    argv = read(TEMPLATE)['nine_task_argv']
    index = argv.index('--baseline_results')
    return [resolve_legacy_path(value) for value in argv[index + 1:index + 3]]


def _relocate_lm(payload):
    value = deepcopy(payload)
    args = value['config']['model_args']
    old = '/home/star/models/Qwen/Qwen3-8B'
    new = str(HF.resolve())
    if isinstance(args, str):
        fields = args.split(',')
        found = False
        for index, field in enumerate(fields):
            name, separator, data = field.partition('=')
            if separator and name in ('pretrained', 'tokenizer'):
                require(data == old, 'Unexpected old HF identity in the baseline arguments.')
                fields[index] = name + '=' + new
                found = True
        require(found, 'Missing old HF identity in the baseline arguments.')
        value['config']['model_args'] = ','.join(fields)
    else:
        require(isinstance(args, dict) and args.get('pretrained') == old, 'Missing fixed HF arguments.')
        args['pretrained'] = new
        if 'tokenizer' in args:
            require(args['tokenizer'] == old, 'Unexpected old tokenizer identity.')
            args['tokenizer'] = new
    unchanged = deepcopy(value)
    unchanged['config']['model_args'] = payload['config']['model_args']
    require(unchanged == payload, 'Relocation changed downstream results or task configuration.')
    return value


def validate_inputs():
    """Prove the new token bytes/order and historical predictions share one input."""
    from experiments.longbench_v1.common import sha256
    from experiments.longbench_v1.data import load_archive
    from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS
    from experiments.longbench_v1.scoring import check_prediction

    old, current = read(OLD_MANIFEST), read(MANIFEST)
    old_scores, old_config = read(OLD_DENSE / 'scores.json'), read(OLD_DENSE / 'config.json')
    require(sha256(OLD_MANIFEST) == old_scores['archive_sha256'] == old_config['archive']['sha256'],
            'Historical baseline no longer binds the original input manifest.')
    require(sha256(OLD_DENSE / 'config.json') == old_scores['config_sha256'],
            'Historical baseline config differs from its score record.')
    require(old_scores['model'] == old_config['model'] and old_scores['model']['kind'] == 'dense',
            'Historical BF16 baseline identity differs between config and scores.')
    for key in ('schema', 'status', 'protocol', 'upstream', 'sample_count', 'statistics'):
        require(old[key] == current[key], 'Changed LongBench input semantics: ' + key)
    require(old['protocol'] == old_scores['protocol'] == old_config['protocol'], 'Historical LongBench protocol differs.')
    require({key: value for key, value in old['inputs'].items() if key != 'path'} ==
            {key: value for key, value in current['inputs'].items() if key != 'path'}
            and old['index']['sha256'] == current['index']['sha256'],
            'New LongBench token bytes or index/order differ from the historical identity.')
    for key in ('repository', 'revision', 'zip_sha256', 'counts'):
        require(old['data'][key] == current['data'][key], 'Changed official LongBench source: ' + key)
    require(_named_hashes(old['data']['raw_sha256']) == _named_hashes(current['data']['raw_sha256'])
            and _named_hashes(old['tokenizer']['sha256']) == _named_hashes(current['tokenizer']['sha256'])
            and old['tokenizer']['use_fast'] == current['tokenizer']['use_fast'],
            'Changed raw tasks or tokenizer identity.')
    actual, rows = load_archive(MANIFEST)
    require(actual == current and len(rows) == 4750, 'Changed complete LongBench archive.')
    by_task = {name: [row for row in rows if row['dataset'] == name] for name in DATASETS}
    predictions = {}
    require(len(old_scores['prediction_sha256']) == len(DATASETS), 'Missing historical prediction inventory.')
    for source, expected in old_scores['prediction_sha256'].items():
        path = resolve_legacy_path(source)
        name = path.stem
        require(name in by_task and name not in predictions and sha256(path) == expected,
                'Missing, repeated or changed historical prediction: ' + str(path))
        values = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        require(len(values) == EXPECTED_COUNTS[name], 'Incomplete historical prediction task: ' + name)
        for predicted, row in zip(values, by_task[name], strict=True):
            check_prediction(predicted, row, old_scores['config_sha256'])
        predictions[name] = dict(path=path, values=values, original_path=source, sha256=expected)
    require(set(predictions) == set(DATASETS), 'Historical predictions omit an official task.')
    proof = dict(status='verified', old_manifest=binding(OLD_MANIFEST), new_manifest=binding(MANIFEST),
        old_inputs_identity=old['inputs'], old_index_identity=old['index'],
        new_input_file=binding(current['inputs']['path']), new_index_file=binding(current['index']['path']),
        sample_count=4750, task_count=21, protocol_upstream_and_statistics_equal=True,
        token_bytes_and_order_match_historical_SHA256=True,
        prediction_input_fields_and_order_verified=4750,
        boundary='Old token/index files are not duplicated; their recorded SHA256 is anchored by the original baseline config/scores. New actual bytes match both recorded hashes.')
    return old_config, old_scores, predictions, by_task, proof


def _current_model(old_model):
    from experiments.longbench_v1.common import file_stat, sha256
    source = read(OFFICIAL_SOURCE)
    require(source['id'] == 'Qwen/Qwen3-8B' and source['sha'] == 'b968826d9c46dd6066d109eabc6255188de91218',
            'Use the original fixed official Qwen revision.')
    official = {item['rfilename']: item for item in source['siblings']}
    verified = read(HF_VERIFICATION)
    require(verified['status'] == 'passed' and verified['repository'] == source['id']
            and verified['revision'] == source['sha'] and Path(verified['path']).resolve() == HF.resolve(),
            'Require the completed official-byte verification for this original HF model.')
    verified_files = {item['name']: item for item in verified['files']}
    metadata = {}
    for original, expected in old_model['hf_metadata_sha256'].items():
        path = HF / Path(original).name
        require(sha256(path) == expected, 'Current HF metadata differs from the BF16 baseline: ' + path.name)
        metadata[str(path.resolve())] = expected
    stats, weights = [], {}
    for old in old_model['hf_weight_stats']:
        path = HF / Path(old['path']).name
        require(path.is_file() and path.stat().st_size == old['size'], 'Finalized original BF16 shard is missing: ' + str(path))
        expected = official[path.name]['lfs']['sha256']
        actual = verified_files[path.name]
        require(actual['sha256'] == expected and actual['bytes'] == path.stat().st_size,
                'Original BF16 shard differs from its completed official verification: ' + path.name)
        stats.append(file_stat(path))
        weights[path.name] = dict(bytes=path.stat().st_size, sha256=actual['sha256'], official_lfs_sha256=expected)
    require({Path(item['path']).name for item in stats} == set(read(HF / 'model.safetensors.index.json')['weight_map'].values()),
            'Current HF shard inventory is incomplete.')
    model = deepcopy(old_model)
    model.update(hf_path=str(HF.resolve()), hf_metadata_sha256=metadata, hf_weight_stats=stats)
    require(set(model) == set(old_model) and model['kind'] == old_model['kind'] == 'dense', 'Changed baseline representation.')
    return model, dict(repository=source['id'], revision=source['sha'], official_source=binding(OFFICIAL_SOURCE),
        completed_byte_verification=binding(HF_VERIFICATION), weights=weights,
        verification_reused=True, current_stat_newly_measured=True)


def _reused_inputs():
    """Reuse the completed 4750-row/input-byte check while relocating its files."""
    from experiments.longbench_v1.common import sha256
    from experiments.longbench_v1.protocol import DATASETS
    require(INPUT_VERIFICATION.is_file(), 'Run --check-inputs-only before writing the terminal baseline binding.')
    proof = read(INPUT_VERIFICATION)
    require(proof['status'] == 'verified' and proof['sample_count'] == 4750 and proof['task_count'] == 21
            and proof['prediction_input_fields_and_order_verified'] == 4750
            and proof['token_bytes_and_order_match_historical_SHA256']
            and binding(OLD_MANIFEST) == proof['old_manifest'] and binding(MANIFEST) == proof['new_manifest'],
            'Completed shared-input verification does not cover these manifests.')
    old_scores, old_config = read(OLD_DENSE / 'scores.json'), read(OLD_DENSE / 'config.json')
    require(old_scores['archive_sha256'] == old_config['archive']['sha256'] == proof['old_manifest']['sha256']
            and sha256(OLD_DENSE / 'config.json') == old_scores['config_sha256'],
            'Historical baseline inputs/config changed after the completed input check.')
    predictions = {}
    for source, expected in old_scores['prediction_sha256'].items():
        path = resolve_legacy_path(source)
        require(path.stem not in predictions and sha256(path) == expected,
                'Historical prediction changed after the completed input check.')
        predictions[path.stem] = dict(path=path, sha256=expected,
            values=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()])
    require(set(predictions) == set(DATASETS), 'Historical prediction inventory changed.')
    return old_config, old_scores, predictions, proof


def prepare(output=QWEN_BASELINE):
    from experiments.compare_longbench_v1 import _load_scores
    from experiments.longbench_v1.common import sha256
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.model_inventory import model_inventory

    output = Path(output).resolve()
    require(not output.exists(), 'Refusing to replace an existing baseline migration: ' + str(output))
    old_config, old_scores, predictions, input_proof = _reused_inputs()
    model, hf_proof = _current_model(old_scores['model'])
    inventory = model_inventory(HF)
    old_lm_paths = _lm_sources()
    lm = [_relocate_lm(read(path)) for path in old_lm_paths]
    config = deepcopy(old_config)
    config['model'] = model
    config['archive'] = dict(path=str(MANIFEST.resolve()), sha256=sha256(MANIFEST))
    # The original inference source hashes remain historical provenance. Only
    # the new comparison identity is rebound; no new inference is claimed.
    stripped = deepcopy(config)
    stripped['model'], stripped['archive'] = old_config['model'], old_config['archive']
    require(stripped == old_config, 'Changed historical generation configuration.')
    output.mkdir(parents=True)
    longbench = output / 'longbench'
    (longbench / 'predictions').mkdir(parents=True)
    write(output / 'mmlu.json', lm[0])
    write(output / 'seven.json', lm[1])
    write(longbench / 'config.json', config)
    config_sha = sha256(longbench / 'config.json')
    prediction_hashes, original_prediction_hashes = {}, {}
    for name, source in predictions.items():
        path = longbench / 'predictions' / (name + '.jsonl')
        with path.open('x', encoding='utf-8') as stream:
            for old in source['values']:
                current = {**old, 'config_sha256': config_sha}
                require({**current, 'config_sha256': old['config_sha256']} == old,
                        'Changed generated text/tokens or nonidentity prediction fields.')
                stream.write(json.dumps(current, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        prediction_hashes[str(path)] = sha256(path)
        original_prediction_hashes[str(source['path'])] = source['sha256']
    scores = deepcopy(old_scores)
    scores.update(model=model, archive_sha256=sha256(MANIFEST), config_sha256=config_sha,
                  prediction_sha256=prediction_hashes)
    unchanged = deepcopy(scores)
    for key in ('model', 'archive_sha256', 'config_sha256', 'prediction_sha256'):
        unchanged[key] = old_scores[key]
    require(unchanged == old_scores, 'Changed BF16 metrics, task configuration or aggregation.')
    write(longbench / 'scores.json', scores)
    collected = collect_results([output / 'mmlu.json', output / 'seven.json'], inventory)
    verified_scores, _ = _load_scores(longbench / 'scores.json', MANIFEST, read(MANIFEST))
    require(verified_scores == scores, 'Migrated baseline failed the unchanged acceptance loader.')
    files = dict(mmlu=output / 'mmlu.json', seven=output / 'seven.json',
                 longbench_config=longbench / 'config.json', longbench_scores=longbench / 'scores.json',
                 longbench_manifest=MANIFEST)
    proof = dict(schema='cmcc_qwen3_bf16_baseline_migration_v1', status='verified',
        bindings={key: binding(path) for key, path in files.items()}, prediction_sha256=prediction_hashes,
        old_sources=dict(mmlu=binding(old_lm_paths[0]), seven=binding(old_lm_paths[1]),
            longbench_config=binding(OLD_DENSE / 'config.json'), longbench_scores=binding(OLD_DENSE / 'scores.json'),
            predictions=original_prediction_hashes, template=binding(TEMPLATE)),
        input_identity=input_proof, original_HF_identity=hf_proof,
        completed_input_verification=binding(INPUT_VERIFICATION), completed_input_verification_reused=True,
        exact_invariants=dict(downstream_results_and_task_configs=True, longbench_metrics_and_protocol=True,
            prediction_order_text_tokens_and_all_nonidentity_fields=True, historical_source_hashes_preserved=True),
        rebound_fields=['LM config.model_args pretrained/tokenizer paths', 'LongBench archive path/SHA256',
            'HF paths and actual current stat after official byte verification', 'config/prediction file identity hashes'],
        baseline_task_metrics=collected['task_metrics'], longbench_task_macro_mean=scores['task_macro_mean'],
        inference_or_scoring_reexecuted=False, implementation=binding(Path(__file__)))
    write(output / 'migration_binding.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs-only', action='store_true')
    args = parser.parse_args()
    if args.check_inputs_only:
        *_, value = validate_inputs()
        destination = REPO / 'result/migration_audit_20261030/baseline_input_validation.json'
        if destination.exists():
            require(read(destination) == value, 'Changed baseline input verification; preserve the existing evidence.')
        else:
            write(destination, value)
    else:
        value = prepare()
        destination = QWEN_BASELINE / 'migration_binding.json'
    print(json.dumps(dict(status=value['status'], output=str(destination)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
