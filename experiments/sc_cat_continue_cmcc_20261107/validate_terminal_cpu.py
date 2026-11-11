"""Actual after_o config, frozen recipes and archived full21 partition checks; no GPU."""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time

import torch

from experiments.cat_auto_protocol_cmcc_20261104.replay import StopAtSelectedNode
from experiments.sc_cat_live_cmcc_20261106.recovery_io import binding, read_json, write_json
from . import final_score, longbench, replay


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31'
FIRST = RUN / 'live_k_o_20261106'
INVENTORY = ROOT / 'result/qwen3_8b_mmlu65_20261004/cat_auto_protocol_cmcc_20261104/model_inventory_cpu.json'
DATA = INVENTORY.parent / 'full_data_v0/data_plan.pt'
ARCHIVE = ROOT / 'result/prepare_eval_20261030/longbench/manifest.json'


def rejects(call):
    try:
        call()
    except (ValueError, KeyError, OSError, RuntimeError):
        return True
    raise AssertionError('An invalid/partial input escaped its declared boundary.')


def run(output):
    report_path = FIRST / 'formal/artifacts/o_proj/different/baseline/node_125/export/deployment_report.json'
    config = longbench.make_config(binding(report_path), binding(INVENTORY), binding(ARCHIVE))
    actual = read_json(FIRST / 'formal/after_o.json')
    assert config['model']['checkpoint_id'] == actual['uuid']
    assert config['model']['checkpoint'] == actual['path']
    assert not config['model']['sc_cat']['complete']
    partial_rejected = rejects(lambda: longbench.load_model(config, None))
    resolved = replay.resolve_categories([binding(FIRST / 'formal/frozen_recipe.json')],
        binding(RUN / 'selected_settings.json'), binding(INVENTORY), binding(DATA))
    k, o = [replay.stage_plan(resolved, category) for category in ('k_proj', 'o_proj')]
    assert k['selected_step'] == 250 and o['selected_step'] == 125
    assert k['schedule_max'] == o['schedule_max'] == 500
    partial_manifest_rejected = rejects(lambda: replay.freeze_manifest(
        [binding(FIRST / 'formal/frozen_recipe.json')], binding(RUN / 'selected_settings.json'),
        binding(INVENTORY), binding(DATA), binding(FIRST / 'formal/after_o.json')))
    callback_proofs = []
    for maximum, selected in ((500, 250), (1000, 125)):
        callback = StopAtSelectedNode(schedule_max=maximum, selected_step=selected)
        args, state = SimpleNamespace(max_steps=maximum), SimpleNamespace(global_step=0)
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        callback.on_train_begin(args, state, control)
        state.global_step = selected
        callback.on_step_end(args, state, control)
        assert control.should_save and control.should_training_stop and args.max_steps == maximum
        callback_proofs.append(dict(original_schedule_max=maximum, selected_stop=selected))
    zero = read_json(FIRST / 'technical/export_k_raw0/deployment_report.json')
    original = zero['selected_recipe']
    zero_row = dict(category_id='k_proj', resolved_recipe=resolved['categories'][0]['resolved_recipe'], selected=dict(
        settings=original['recovery'], selected_step=zero['export_evidence']['actual_updates'],
        seed_identity=original['seed_identity'], source_candidate_id=zero['export_evidence']['source_candidate_id'],
        recipe_id=zero['export_evidence']['recipe_id']))
    assert replay.stage_plan(dict(categories=[zero_row]), 'k_proj')['skip_KD']
    assert final_score.quality_flags(Fraction(9, 10)) == dict(quality_passed=True, research_goal_quality_passed=False)
    assert final_score.quality_flags(Fraction(1))['quality_passed']
    assert not final_score.quality_flags(Fraction(1) + Fraction(1, 10**12))['quality_passed']
    baseline = read_json(ROOT / 'result/qwen3_8b_mmlu65_20261004/cat_offline_execution_cmcc_20261104/current/bf16_quality_receipt.json')
    dense_lb = read_json('/data1/cmcc/VAELLM/artifacts/cmcc_baselines_20261030/qwen3_8b/longbench/scores.json')
    dense_nine = final_score.exact_nine(baseline['task_metrics'], {task: row['score'] for task, row in dense_lb['tasks'].items()})
    assert abs(float(dense_nine) - 68.04714491790848) < 1e-10
    historical = ROOT / 'result/qwen3_8b_mmlu65_20261004/block_reconstruction_sequence_cmcc_20261031/single_C0_v3/evaluation_BR'
    plan_path = historical / 'longbench/merge_plan.json'
    plan = read_json(plan_path)
    config_ref = binding(plan['config_path'])
    with tempfile.TemporaryDirectory(prefix='terminal_partition_CPU_', dir=output.parent) as directory:
        directory = Path(directory)
        refs = []
        for index, unit in enumerate(plan['execution_units']):
            reports = list((Path(unit['output_dir']) / 'runs').glob('*.json'))
            assert len(reports) == 1
            part = dict(status='completed_same_artifact_LongBench_partition', exit_code=0,
                config=config_ref, datasets=unit['datasets'], output=unit['output_dir'],
                native_report=binding(reports[0]), checkpoint_id=plan['checkpoint_id'],
                prediction_sha256={str(Path(unit['output_dir']) / 'predictions' / (task + '.jsonl')):
                    longbench.sha256(Path(unit['output_dir']) / 'predictions' / (task + '.jsonl')) for task in unit['datasets']})
            path = directory / f'actual_archived_partition_{index}.json'
            write_json(path, part)
            refs.append(binding(path))
        _old_config, sources, _partitions = longbench.partition_sources(config_ref, refs)
        assert len(sources) == 21
        duplicate_rejected = rejects(lambda: longbench.partition_sources(config_ref, [refs[0], refs[0]]))
        missing_rejected = rejects(lambda: longbench.partition_sources(config_ref, refs[:1]))
        changed = deepcopy(read_json(refs[1]['path']))
        changed['output'] = read_json(refs[0]['path'])['output']
        shared = directory / 'invalid_shared_output.json'
        write_json(shared, changed)
        shared_output_rejected = rejects(lambda: longbench.partition_sources(config_ref, [refs[0], binding(shared)]))
    assert not torch.cuda.is_initialized()
    return dict(status='SC_CAT_terminal_and_replay_interfaces_CPU_verified', exit_code=0,
        actual_after_o_uuid=actual['uuid'], actual_after_o_config_preserves_UUID_and_kind=True,
        partial_endpoint_blocked_from_full21=partial_rejected, actual_k_o_fixed_stops=[k, o],
        partial_recipe_not_mislabeled_complete=partial_manifest_rejected,
        unchanged_schedule_callback_checks=callback_proofs, actual_zero_plan_skips_KD=True,
        exact_standard_and_strict_research_goal_boundaries=True, original_BF16_nine_score=float(dense_nine),
        actual_archived_partitions=len(plan['execution_units']), actual_archived_full21_sources=len(sources),
        actual_archived_full21_samples=4750, duplicate_owner_rejected=duplicate_rejected,
        missing_partition_rejected=missing_rejected, shared_output_rejected=shared_output_rejected,
        archived_partition_plan=binding(plan_path), no_prediction_or_official_score_recomputed=True,
        source_sha256={str(path): binding(path)['sha256'] for path in
            (Path(longbench.__file__), Path(final_score.__file__), Path(replay.__file__))},
        GPU_executed=False, CUDA_initialized=False,
        limitation='No complete SC-CAT product exists yet. Genuine replay stop/fresh is validated on the first authorized k250 stage; future final artifacts/LB are not claimed tested.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    began = time.monotonic()
    report = run(args.output.resolve())
    report['elapsed_seconds'] = time.monotonic() - began
    write_json(args.output, report)
    print(json.dumps(dict(status=report['status'], elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
