"""Frozen H50 row selection and partial scoring on the original LongBench archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from core import load_selection
from experiments.longbench_v1 import common, data, protocol, scoring

SELECTION_SHA = 'be201d967e9aae091fd668c50bb3db54896bfbf60aacbb508ea5db9845d10837'
INDICES_SHA = 'e3dec00d4f095ebd856554eb675e0d12ba53eb30277b6c78ab82c9f15231d9a4'
RULE = "Within task sorted sha256(UTF8(f'137:{dataset}:{row_index}')).digest(), first50; no answers or predictions used."


def indices_for(selection):
    return {task: sorted(range(protocol.EXPECTED_COUNTS[task]),
        key=lambda index: hashlib.sha256(f'137:{task}:{index}'.encode('utf-8')).digest())[:50]
        for task in selection['high_tasks']}


def indices_sha(indices):
    return hashlib.sha256(json.dumps(indices, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def prepare_manifest(selection_path, archive_path, calibration_log, output):
    selection_path, archive_path, calibration_log, output = map(lambda p: Path(p).resolve(),
        (selection_path, archive_path, calibration_log, output))
    if output.exists():
        raise FileExistsError('Keep the frozen sample manifest: ' + str(output))
    if common.sha256(selection_path) != SELECTION_SHA:
        raise ValueError('Use the existing frozen12 sensitive selection.')
    selection = load_selection(selection_path)
    archive, _rows = data.load_archive(archive_path)
    if common.sha256(archive_path) != selection['archive_sha256']:
        raise ValueError('Sample selection/archive mismatch.')
    reports = [json.loads(line.split('FINAL_REPORT', 1)[1].lstrip(' :\t'))
        for line in calibration_log.read_text(encoding='utf-8').splitlines() if line.startswith('FINAL_REPORT')]
    if len(reports) != 1:
        raise ValueError('Require the single actual historical calibration FINAL_REPORT.')
    analysis = reports[0]
    indices = indices_for(selection)
    if (indices_sha(indices) != INDICES_SHA or analysis['plans']['H50']['indices_sha256'] != INDICES_SHA
            or analysis['archive_sha256'] != selection['archive_sha256']
            or analysis['plans']['H50']['samples'] != 600
            or analysis['plans']['H50']['estimates']['a0'] != 53.692011958798304):
        raise ValueError('Historical H50 rule/archive/baseline differs.')
    for record in analysis['paths'].values():
        directory = Path(record['path'])
        if (common.sha256(directory / 'config.json') != record['config_sha256']
                or common.sha256(directory / 'scores.json') != record['scores_sha256']):
            raise ValueError('Historical calibration input changed: ' + str(directory))
    evidence = output.with_suffix('')
    evidence.mkdir(parents=True, exist_ok=True)
    analysis_path, log_path = evidence / 'analysis.json', evidence / 'calibration.log'
    if analysis_path.exists() or log_path.exists():
        raise FileExistsError('Do not overwrite historical calibration evidence.')
    common.write_json(analysis_path, analysis)
    shutil.copyfile(calibration_log, log_path)
    baseline = analysis['paths']['a0']
    record = {'schema': 'longbench_v1_h50_selection_v1', 'status': 'frozen', 'mode': 'sample_partial',
        'kind': 'sample_partial', 'is_full_benchmark': False, 'purpose': 'checkpoint_budget_screen_only',
        'selection': {'path': str(selection_path), 'sha256': SELECTION_SHA},
        'archive': {'path': str(archive_path), 'sha256': selection['archive_sha256']},
        'protocol': archive['protocol'], 'sampling_seed': 137, 'samples_per_task': 50,
        'selection_rule': RULE, 'indices_serialization': "json.dumps(indices,sort_keys=True,separators=(',',':')).encode('utf-8')",
        'high_tasks': selection['high_tasks'], 'task_count': 12, 'sample_count': 600,
        'indices': indices, 'indices_sha256': INDICES_SHA,
        'paired_source': {'checkpoint_id': selection['source_checkpoint_id'], 'h50_macro_mean': 53.692011958798304,
            'prediction_dir': baseline['path'], 'config_sha256': baseline['config_sha256'],
            'full_scores_sha256': baseline['scores_sha256']},
        'historical_calibration': {'analysis': {'path': str(analysis_path), 'sha256': common.sha256(analysis_path)},
            'core_log': {'path': str(log_path), 'sha256': common.sha256(log_path)},
            'original_log': str(calibration_log), 'original_log_sha256': common.sha256(calibration_log),
            'historical_endpoints': analysis['paths'], 'elapsed_cpu_seconds': analysis['elapsed_cpu_seconds']},
        'acceptance_note': 'H50 is only a cheap independent screen. Final acceptance requires full21/4750 and all nine metrics; do not compare H50 with fullH.',
        'source_sha256': {str(Path(__file__).resolve()): common.sha256(__file__),
            str(Path(__file__).with_name('core.py').resolve()): common.sha256(Path(__file__).with_name('core.py'))}}
    common.write_json(output, record)
    return record


def load_manifest(path):
    path = Path(path).resolve()
    sample = json.loads(path.read_text(encoding='utf-8'))
    if (sample['schema'] != 'longbench_v1_h50_selection_v1' or sample['status'] != 'frozen'
            or sample['mode'] != 'sample_partial' or sample['kind'] != 'sample_partial'
            or sample['is_full_benchmark'] is not False or sample['sampling_seed'] != 137
            or sample['samples_per_task'] != 50 or sample['selection_rule'] != RULE
            or sample['task_count'] != 12 or sample['sample_count'] != 600
            or sample['selection']['sha256'] != SELECTION_SHA
            or common.sha256(sample['selection']['path']) != SELECTION_SHA):
        raise ValueError('Require the fixed sample_partial H50 manifest.')
    selection = load_selection(sample['selection']['path'])
    if (sample['high_tasks'] != selection['high_tasks'] or sample['indices'] != indices_for(selection)
            or sample['indices_sha256'] != INDICES_SHA or indices_sha(sample['indices']) != INDICES_SHA
            or sample['archive']['sha256'] != selection['archive_sha256']
            or common.sha256(sample['archive']['path']) != sample['archive']['sha256']
            or sample['paired_source']['checkpoint_id'] != selection['source_checkpoint_id']
            or sample['paired_source']['h50_macro_mean'] != 53.692011958798304):
        raise ValueError('Frozen task/index/archive/paired source identity changed.')
    for name in ('analysis', 'core_log'):
        record = sample['historical_calibration'][name]
        if common.sha256(record['path']) != record['sha256']:
            raise ValueError('Historical H50 evidence changed: ' + name)
    archive, rows = data.load_archive(sample['archive']['path'])
    if sample['protocol'] != archive['protocol']:
        raise ValueError('H50 protocol changed.')
    return sample, archive, rows


def selected_rows(sample, rows):
    # Execute in original row order; manifest and score aggregation retain hash rank.
    selected = {(task, index) for task, values in sample['indices'].items() for index in values}
    result = [row for row in rows if (row['dataset'], row['row_index']) in selected]
    if len(selected) != len(result) or len(result) != 600:
        raise ValueError('Missing or duplicate H50 archive rows.')
    return result


def check_score_config(config, sample, archive):
    if (config['schema'] != 'longbench_v1_run_v1' or config['mode'] != 'full'
            or config['archive']['sha256'] != sample['archive']['sha256']
            or Path(config['archive']['path']).resolve() != Path(sample['archive']['path']).resolve()
            or config['protocol'] != archive['protocol'] or config['model']['kind'] not in ('dense', 'compressed')
            or Path(config['model']['hf_path']).resolve() != Path(archive['tokenizer']['path']).resolve()
            or config['prewarm_group_size'] != 8 or config['device'] != 'cuda:0'
            or (config['model']['kind'] == 'compressed' and not config['model'].get('checkpoint_id'))):
        raise ValueError('Only original full-protocol same-family configs may supply H50 predictions.')


def check_rows(predictions, references, config_sha256, *, sample_sha256=None, complete=True):
    if len(predictions) > len(references) or (complete and len(predictions) != len(references)):
        raise ValueError('Missing or extra H50/official prediction rows.')
    for predicted, source in zip(predictions, references):
        scoring.check_prediction(predicted, source, config_sha256)
        if sample_sha256 is not None and (predicted.get('evaluation_kind') != 'sample_partial'
                or predicted.get('sample_manifest_sha256') != sample_sha256):
            raise ValueError('Sample prediction mode/manifest differs.')


def score_directory(manifest_path, prediction_dir, output):
    started = time.monotonic()
    manifest_path, directory, output = map(lambda p: Path(p).resolve(), (manifest_path, prediction_dir, output))
    if output.exists():
        raise FileExistsError('Use a fresh H50 report: ' + str(output))
    sample, archive, rows = load_manifest(manifest_path)
    config_path = directory / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    check_score_config(config, sample, archive)
    config_sha, sample_sha = common.sha256(config_path), common.sha256(manifest_path)
    marker = directory / 'sample_manifest.json'
    is_sample = marker.exists()
    if is_sample and marker.read_bytes() != manifest_path.read_bytes():
        raise ValueError('Prediction directory sample manifest differs.')
    tasks, pins = {}, {}
    for task in sample['high_tasks']:
        references = [row for row in rows if row['dataset'] == task]
        chosen = set(sample['indices'][task])
        if is_sample:
            references = [row for row in references if row['row_index'] in chosen]
        path = directory / 'predictions' / (task + '.jsonl')
        with path.open(encoding='utf-8') as stream:
            predictions = [json.loads(line) for line in stream]
        check_rows(predictions, references, config_sha, sample_sha256=sample_sha if is_sample else None)
        by_index = {item['row_index']: item for item in predictions}
        refs = {item['row_index']: item for item in references}
        values = [scoring.score_example(task, by_index[index]['pred'], refs[index]['answers'], refs[index]['all_classes'])
            for index in sample['indices'][task]]
        score = 100 * sum(values) / len(values)
        tasks[task] = {'samples': 50, 'score': score, 'official_rounded_score': round(score, 2),
            'metric': scoring.METRICS[task].__name__}
        pins[str(path)] = common.sha256(path)
    mean = sum(item['score'] for item in tasks.values()) / len(tasks)
    source_files = [Path(__file__), Path(__file__).with_name('sample_score.py'), Path(scoring.__file__),
        Path(scoring.metrics.__file__), Path(data.__file__), Path(protocol.__file__), Path(common.__file__)]
    report = {'schema': 'longbench_v1_h50_scores_v1', 'status': 'sample_partial_complete', 'mode': 'sample_partial',
        'kind': 'sample_partial', 'is_full_benchmark': False, 'purpose': 'checkpoint_budget_screen_only',
        'sample_manifest': {'path': str(manifest_path), 'sha256': sample_sha}, 'indices_sha256': INDICES_SHA,
        'selected_tasks': sample['high_tasks'], 'task_count': 12, 'sample_count': 600, 'tasks': tasks,
        'h50_macro_mean': mean, 'paired_source_h50_macro_mean': sample['paired_source']['h50_macro_mean'],
        'paired_source_h50_delta_pp': mean - sample['paired_source']['h50_macro_mean'],
        'paired_source_checkpoint_id': sample['paired_source']['checkpoint_id'],
        'input_mode': 'h50_predictions' if is_sample else 'selected_from_complete_high_task_predictions',
        'archive': sample['archive'], 'protocol': archive['protocol'], 'model': config['model'],
        'config_sha256': config_sha, 'prediction_sha256': pins,
        'sourcefiles_sha256': {str(p.resolve()): common.sha256(p) for p in source_files},
        'aggregation_note': 'Equal12-task mean; sum each official metric in fixed hash-rank order. Compare only same H50 source, not fullH.',
        'acceptance_note': sample['acceptance_note'], 'duration_seconds': time.monotonic() - started}
    output.parent.mkdir(parents=True, exist_ok=True); common.write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare the permanent H50 selection from completed historical calibration.')
    parser.add_argument('--selection', required=True); parser.add_argument('--archive', required=True)
    parser.add_argument('--calibration-log', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    record = prepare_manifest(args.selection, args.archive, args.calibration_log, args.output)
    print(json.dumps({key: record[key] for key in ('schema', 'mode', 'task_count', 'sample_count', 'indices_sha256')}))
