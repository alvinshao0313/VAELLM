"""Close the reviewed metadata flag and small runtime readback, without model rebuild."""
import gc
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cat_category_selection_20261029 import sources as s


def equal_runtime(actual, expected, counts, path='runtime'):
    if isinstance(expected, s.torch.Tensor):
        s.require(isinstance(actual, s.torch.Tensor) and actual.dtype == expected.dtype
                  and actual.shape == expected.shape and s.torch.equal(actual, expected), 'Runtime tensor changed: ' + path)
        counts['tensors'] += 1
        counts['tensor_bytes'] += expected.numel() * expected.element_size()
    elif isinstance(expected, dict):
        s.require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'Runtime mapping changed: ' + path)
        for key in expected:
            equal_runtime(actual[key], expected[key], counts, path + '.' + str(key))
    elif isinstance(expected, (tuple, list)):
        s.require(type(actual) is type(expected) and len(actual) == len(expected), 'Runtime sequence changed: ' + path)
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            equal_runtime(left, right, counts, path + '.' + str(index))
    else:
        s.require(type(actual) is type(expected) and actual == expected, 'Runtime scalar changed: ' + path)
        counts['scalars'] += 1


def finish():
    s.cpu_only()
    report_path = s.DATA / 'report.json'
    report = s.read(report_path)
    s.require(report['status'] == 'CPU_exports_fresh_budget_and_full8_inputs_prepared'
              and report['exit_code'] == 0 and 'CPU_review_closure' not in report, 'Complete the reviewed CPU output exactly once.')
    directory = Path(report['endpoints']['250']['checkpoint_dir'])
    meta_path = directory / 'checkpoint_meta.json'
    meta = s.native.load_v6_meta(str(directory))
    s.require(meta['checkpoint_id'] == report['endpoints']['250']['checkpoint_id']
              and meta['extra_meta']['stage'] == 'round_base', 'Only correct the specifically reviewed inherited metadata flag.')
    weight_stat = s.stat(directory / meta['state_dict_file'])
    before = s.binding(meta_path)
    original = s.torch.load(s.RAW / 'cat_runtime_state.pt', weights_only=True, map_location='cpu')
    serialized = s.native.load_v6_cat_runtime_state(str(directory), map_location='cpu', required=True)
    counts = dict(tensors=0, tensor_bytes=0, scalars=0)
    equal_runtime(serialized, original, counts)
    from train_utils.cat_train_runtime import load_cat_resume_distill_progress
    from train_utils.cat_step_resume_v6 import validate_cat_step_immutable_resume_contract
    source_contract = s.native.load_v6_training_step_meta(str(s.NODES / 'gate250'))['immutable_resume_contract']
    validate_cat_step_immutable_resume_contract(meta['extra_meta']['distill_stage']['source_immutable_contract'], source_contract)
    for step in (250, 500):
        child_contract = s.native.load_v6_training_step_meta(str(s.NODES / f'gate{step}'))['immutable_resume_contract']
        validate_cat_step_immutable_resume_contract(source_contract, child_contract)
    revision = dict(reason='CPU review: category_boundary inherited the raw-only extra.stage flag; corrected to after_category. No tensor or training history change.',
                    prior_metadata=before, prior_independent_model_fresh_evidence_preserved=True,
                    revision_source=s.binding(Path(__file__)), no_model_rebuild_or_resave=True, revised_unix=time.time())
    meta['extra_meta']['stage'] = 'after_category'
    meta['extra_meta']['selected_step_export']['metadata_review_revision'] = revision
    s.native.validate_v6_meta(meta, expected_kind='category_boundary')
    s.write(meta_path, meta)
    actual = s.native.load_v6_meta(str(directory))
    s.require(actual == meta and actual['extra_meta']['stage'] == 'after_category'
              and actual['immutable_resume_contract'] is None, 'Revised metadata native readback failed.')
    progress = load_cat_resume_distill_progress(str(directory))
    s.require(progress.lora_round_idx == 5 and len(progress.distill_stage_history) == 5
              and progress.distill_stage_history[-1]['actual_updates'] == 250
              and progress.distill_stage_history[-1]['planned_steps'] == 500, 'Revised metadata lost truthful progress.')
    s.require(s.stat(directory / meta['state_dict_file']) == weight_stat, 'Metadata correction changed the deployment weight payload.')
    report['endpoints']['250']['checkpoint_meta_sha256'] = s.sha(meta_path)
    old_plan = s.read(s.DATA / 'evaluation_plan.json')
    from experiments.cat_category_selection_20261029.evaluate import prepare
    prepare(report)
    closure = dict(status='reviewed_metadata_and_new_runtime_serialization_exact', pid=os.getpid(),
        metadata_revision=revision, revised_metadata=s.binding(meta_path), runtime_fields_exact=counts,
        raw_runtime=s.binding(s.RAW / 'cat_runtime_state.pt'), serialized_runtime=s.binding(directory / 'cat_runtime_state.pt'),
        weight_payload_unchanged=True, revised_progress_lora_round_idx=progress.lora_round_idx,
        prior_evaluation_plan_metadata_sha256=old_plan['endpoints']['250']['checkpoint_meta_sha256'],
        evaluation_plan_rebound_after_metadata_revision=True, source_sha256=s.own_pins(), CUDA_initialized=False)
    for step in (250, 500):
        row = report['endpoints'][str(step)]
        endpoint = Path(row['checkpoint_dir'])
        files = {path.name: dict(bytes=path.stat().st_size) for path in sorted(endpoint.iterdir()) if path.is_file()}
        row['actual_serialized_directory'] = dict(files=files, total_bytes=sum(item['bytes'] for item in files.values()),
            weight_payload_sha256=s.sha(endpoint / 'pytorch_model.bin'),
            boundary='All actual files listed, including tokenizer and CAT runtime provenance; distinct from tensor/logical/packed bit ledger.')
    failed = Path('/data/star/VAELLM/cat_category_selection_20261029_failed_initial_contract')
    if failed.is_dir():
        failure_path = failed / 'report.json'
        history = s.read(failure_path)
        s.require(history['status'] == 'failed' and history['exit_code'] == 1 and not history['endpoints']
                  and set(failed.iterdir()) == {failure_path}, 'Do not remove an unexpected prior attempt payload.')
        closure['initial_failed_attempt'] = history
        closure['initial_failed_attempt_cleanup'] = dict(path=str(failure_path), bytes=failure_path.stat().st_size,
            sha256=s.sha(failure_path), reason='Complete original failure evidence is nested here; no generated model existed.')
    report['CPU_review_closure'] = closure
    s.write(report_path, report)
    if failed.is_dir():
        (failed / 'report.json').unlink()
        failed.rmdir()
    s.require(not s.torch.cuda.is_initialized(), 'Review closure initialized CUDA.')
    print(dict(status=closure['status'], runtime_fields=counts, metadata_SHA=s.sha(meta_path)), flush=True)


if __name__ == '__main__':
    s.torch.set_num_threads(4)
    finish()
