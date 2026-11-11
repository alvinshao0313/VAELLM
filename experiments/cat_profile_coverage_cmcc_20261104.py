"""One-shot actual 21-profile coverage index, outside every active source pin set.

Reuses producer-certified tensor/sample digests without rereading those payloads.
This is an evidence index, never a joint proposer, cap allocator or GPU runner.

Caller activates bitvae, hides CUDA, and runs in tmux/nohup, for example:
  python -B -m experiments.cat_profile_coverage_cmcc_20261104 --validate --output NEW.json
Each invocation reads the declared up-fit/up-full8/night manifests exactly once;
only terminal completed night candidates can publish complete coverage. Failed,
running, unstarted, invalid and missing-attachment entries remain in the index.
No retries, waiting, GPU calls, sample JSONL parsing or state/bank tensor loads.

The full night queue started-to-finished ownership window is the primary cost;
each queue is counted once. Candidate windows and evaluation.elapsed_seconds
are diagnostic subwindows, never added to the queue window. For the separate up
batch, fit chains and auth-start-to-controller-exit full8 windows are distinct.
Running windows are elapsed_to_observation, never complete costs or forecasts.
"""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, inventory_id, require, validate_ledger, validate_recipe, validate_run_identity
from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id, semantic_seed
from experiments.cat_auto_protocol_cmcc_20261104.nodes import load_complete_eight_score
from experiments.w25_eval.acceptance import collect_results

DEFAULT_UP = ROOT/'artifacts/cat_batch_profiles_cmcc_20261104/proposal_20261104_1759/proposal.json'
DEFAULT_EIGHT = ROOT/'artifacts/cat_profile_full8_cmcc_20261104/proposal_20261104_1837/proposal.json'
DEFAULT_NIGHT = ROOT/'artifacts/cat_night_profiles_cmcc_20261104/night_20261104_1918/manifest.json'


class IncompleteRead(ValueError):
    pass


class Snapshot:
    """Read each small source once, including mutable atomic queue snapshots."""
    def __init__(self):
        self.values, self.bindings = {}, {}
        self.observations = {}
        self.observed_epoch = time.time()

    def path(self, path):
        return str(Path(path).resolve())

    def bytes(self, path):
        path = self.path(path)
        require(Path(path).stat().st_size <= 32*1024*1024, 'Index accepts small JSON/code sources, never a tensor/sample payload: '+path)
        data = Path(path).read_bytes()
        self.bindings[path] = dict(path=path, sha256=hashlib.sha256(data).hexdigest())
        self.observations[path] = time.time()
        return data

    def read(self, path):
        path = self.path(path)
        if path not in self.values:
            try:
                self.values[path] = json.loads(self.bytes(path))
            except json.JSONDecodeError as error:
                raise IncompleteRead('Unreadable one-shot JSON snapshot: '+path) from error
        return self.values[path]

    def ref(self, path):
        path = self.path(path)
        if path not in self.bindings:
            self.bytes(path)
        return self.bindings[path]

    def bound(self, reference):
        value = self.read(reference['path'])
        require(self.ref(reference['path'])['sha256'] == reference['sha256'], 'Bound small source SHA changed: '+reference['path'])
        return value

    def optional(self, path):
        return self.read(path) if Path(path).is_file() else None

    def code(self, sources):
        for path, expected in sources.items():
            require(Path(path).suffix == '.py', 'Producer code pins must not redirect index hashing to payloads.')
            require(self.ref(path)['sha256'] == expected, 'Producer/reader source changed: '+path)


def unique_expected(rows, inventory):
    keys = [(row['category_id'], Fraction(row['rate'])) for row in rows]
    expected = {(category, rate) for category in inventory['category_ids'] for rate in (Fraction(2), Fraction(5, 2), Fraction(3))}
    require(len(rows) == len(set(keys)) == len({row['candidate_id'] for row in rows}) == 21 and set(keys) == expected,
        'The explicit manifests must cover all seven actual categories and three rates once, not the files that finished first.')


def expected_sources(snapshot, up_path, eight_path, night_path):
    up, eight, night = (snapshot.read(path) for path in (up_path, eight_path, night_path))
    for value, key in ((up, 'proposal_identity'), (eight, 'proposal_identity'), (night, 'manifest_identity')):
        require(value[key] == digest({name: item for name, item in value.items() if name != key}), 'Explicit candidate manifest identity changed.')
    identity = validate_run_identity(up['run_identity'])
    require(identity == night['run_identity'] and len(identity['code_identity']['source_sha256']) == 9,
        'All profiles require the same actual source9/BF16/data/consumer/search identity.')
    inventory = snapshot.bound(night['inventory'])['inventory']; baseline = snapshot.bound(night['baseline'])
    require(identity['model'] == inventory['model_identity'] == baseline['model_identity']
        and identity['parent_weights_digest'] == digest(identity['model']['weight_shards_sha256'])
        and up['parent'] == night['parent'] and up['parent']['kind'] == 'BF16', 'Cross-model or foreign BF16 profile source.')
    require(eight['baseline_receipt'] == night['baseline'] and eight['common_protocol_identity'] == night['common_protocol_identity']
        == baseline['protocol_identity'] == digest(baseline['protocol']), 'All21 profiles require the same actual complete8 protocol.')
    require(snapshot.ref(up_path) == eight['actual_fit_batch_proposal'], 'Up full8 card differs from the actual up fit manifest.')
    snapshot.code(up['source_sha256']); snapshot.code(night['source_sha256'])
    rows = []
    for arm in eight['arms']:
        planned = up['recipes'][arm['label']]; directory = Path(up['batch_root'])/arm['label']
        require(planned['candidate_id'] == arm['candidate_id'], 'Up fit and full8 declared candidate sets differ.')
        rows.append(dict(**planned, label=arm['label'], category_id=planned['recipe']['category_id'],
            rate=planned['recipe']['nominal_main_rate'], gpu=arm['physical_GPU'], origin='up', candidate_dir=str(directory),
            formal_dir=str(directory/'formal'), request=str(directory/'formal_request.json'), full8_dir=str(Path(eight['output_root'])/arm['label']),
            control=str(directory/'chain.json'), source_declared_refs={key:arm[key] for key in ('actual_profile_result','actual_completed','actual_request')}))
    for gpu, planned_rows in night['queues'].items():
        for planned in planned_rows:
            directory = Path(night['output_root'])/('gpu'+gpu)/planned['label']
            rows.append(dict(**planned, category_id=planned['recipe']['category_id'], rate=planned['recipe']['nominal_main_rate'],
                gpu=int(gpu), origin='night', candidate_dir=str(directory), formal_dir=str(directory/'formal'),
                request=str(directory/'formal_request.json'), full8_dir=str(directory/'full8'),
                control=str(directory.parent/'queue.json'), source_declared_refs={}))
    for row in rows:
        recipe = validate_recipe(row['recipe'])
        require(recipe['S'] == 2 and row['candidate_id'] == candidate_id(identity, recipe), 'Declared profile recipe/ID/fixed S2 identity mismatch.')
    unique_expected(rows, inventory)
    return rows, dict(identity=identity, inventory=inventory, baseline=baseline, baseline_ref=night['baseline'],
        protocol_identity=baseline['protocol_identity'], selection_identity=None, ordered_full8_inputs=None)


def profile_contract(row, context, canonical, request, completed):
    recipe, identity = row['recipe'], context['identity']
    require(canonical['candidate_id'] == row['candidate_id'] and canonical['recipe'] == recipe
        and canonical['run_identity'] == request['run_identity'] == identity, 'Foreign candidate/recipe/parent/source9 identity.')
    require(completed['status'] == 'completed' and completed['exit_code'] == 0 and completed['phase'] == request['phase'] == 'formal'
        and completed['request_identity'] == digest(request) and len(completed['candidates']) == len(request['candidates']) == 1,
        'Require actual successful complete formal producer and its exact request.')
    require(completed['model_identity'] == identity['model'] and completed['data_identity'] == identity['data_identity']
        and completed['consumer_identity'] == identity['consumer_identity'] and completed['fit_code_identity'] == identity['code_identity'],
        'Actual completed model/data/consumer/fit-source headers differ from the common identity.')
    planned, actual = request['candidates'][0], completed['candidates'][0]
    fit, bank = actual['fit_report'], actual['bank']
    require(planned['candidate_id'] == actual['candidate_id'] == fit['candidate_id'] == bank['candidate_id'] == row['candidate_id']
        and planned['recipe'] == fit['resolved_recipe'] == bank['recipe'] == recipe and planned['purpose'] == actual['purpose'] == canonical['purpose'] == 'profile',
        'Actual planned/fit/bank/canonical profile endpoints differ.')
    require(request['physical_GPU'] == row['gpu'] and request['parent']['kind'] == 'BF16'
        and request['parent']['uuid'] == identity['parent_uuid'] and request['parent']['weights_digest'] == identity['parent_weights_digest'], 'Different actual resource or BF16 parent.')
    require(canonical['formal_eligible'] and canonical['phase'] == 'formal' and fit['formal_eligible'] and bank['formal_eligible']
        and fit['status'] == 'completed' and fit['exit_code'] == actual['exit_code'] == 0
        and fit['technical_steps'] is None and fit['actual_stage_steps'] == fit['original_schedule_steps'] == 10000
        and len(fit['stages']) == 2 and all(stage['actual_steps'] == 10000 for stage in fit['stages']), 'Technical/incomplete fits cannot count as a complete profile.')
    require(fit['parent_identity'] == bank['parent_identity'] == identity and fit['frozen_prefix_before_digest'] == fit['frozen_prefix_after_digest'], 'Actual fitted source or frozen complement differs.')
    for index, stage in enumerate(fit['stages'], 1):
        require(stage['init_seed'] == semantic_seed(identity['model'], row['category_id'], 'residual_stage_'+str(index), 'vae_initialization', identity['search_seed'])
            and stage['shuffle_seed'] == semantic_seed(identity['model'], row['category_id'], 'residual_stage_'+str(index), 'vae_shuffle', identity['search_seed']), 'Semantic fit seeds differ.')
    require(canonical['BSQ_diagnostics'] == fit['BSQ_diagnostics'] and [item['step'] for item in fit['BSQ_diagnostics']] == [5000, 10000], 'Actual formal diagnostics were lost or replaced.')
    fresh, endpoint = actual['fresh_evidence'], canonical['endpoint']
    require(all(fresh[key] for key in ('state_exact','forward_executed','gpu_validation','same_input_same_BF16_path_logits_exact'))
        and fresh['checkpoint_id'] == actual['raw_checkpoint']['checkpoint_id'] == endpoint['checkpoint_id'], 'Actual independent fresh or raw UUID differs.')
    ledger = validate_ledger(canonical['ledger']); inventory = context['inventory']
    require(ledger == actual['actual_ledger'] == endpoint['ledger'] and ledger['N'] == inventory['total_target_numel']
        and ledger['source']['kind'] == 'actual_native_v6_tensor_storage' and set(ledger['per_category']) == set(inventory['category_ids'])
        and sum(item['N'] for item in ledger['per_category'].values()) == ledger['N']
        and all(sum(item[key] for item in ledger['per_category'].values()) == ledger[key] for key in ('logical_bits','packed_bits')),
        'Actual full-inventory ledger no longer closes.')
    fee = ledger['per_category'][row['category_id']]
    require(fee['N'] == sum(item['numel'] for item in inventory['projections'] if item['category_id'] == row['category_id'])
        and all(type(fee[key]) is int and fee[key] >= 0 for key in ('logical_bits','packed_bits'))
        and fee['logical_bits'] <= fee['packed_bits'] and canonical['logical_bits'] == fee['logical_bits'] and canonical['packed_bits'] == fee['packed_bits'],
        'Use actual category fees, never whole partial-model bpw or a geometry forecast.')
    projections = canonical['per_projection_fees']
    actual_names = {item['actual_name'] for item in inventory['projections'] if item['category_id'] == row['category_id']}
    require(set(projections) == set(recipe['group_signature']) == actual_names and len(projections) == len(actual_names)
        and all(sum(item[key] for item in projections.values()) == fee[key] for key in ('logical_bits','packed_bits'))
        and all(sum(item[key] for item in fee['component_fees'].values()) == fee[key] for key in ('logical_bits','packed_bits')), 'Actual class/projection/component fees do not close.')
    components = {}
    for projection in projections.values():
        for name, values in projection['component_fees'].items():
            total = components.setdefault(name, dict(logical_bits=0,packed_bits=0))
            for key in total: total[key] += values[key]
    require(components == fee['component_fees'], 'Actual per-projection components differ from the measured class ledger.')
    require(bank['tensor_digest'] == canonical['bank_digest'] and bank == canonical['bank']
        and bank['inventory_identity'] == inventory_id(inventory) and bank['model_identity'] == identity['model']
        and fresh['weight_digest'] == endpoint['state_sha256'] == ledger['source']['state_file_sha256'], 'Producer-certified bank/payload digest bindings differ.')
    readout = actual['joint_readout']
    selection = dict(data_identity=readout['data_identity'],teacher_cache_identity=readout['teacher_cache_identity'],record_order_digest=readout['record_order_digest'],loss_contract=readout['loss_contract'])
    require(selection == canonical['selection_data_identity'] and readout['full']['records'] == 512
        and selection['data_identity'] == identity['data_identity'] and canonical['joint_KL'] == readout['full']['KL'], 'Actual common512 diagnostic source differs.')
    return actual, fee, selection


def check_scores(canonical, summary, derived):
    require(canonical['complete_full8_evaluated'] and canonical['full8_mean'] == summary['mean_percent'] == derived['mean_percent']
        and canonical['full8_evidence']['task_metrics'] == summary['task_metrics'] == derived['task_metrics'], 'Attached/summary/raw full8 scores differ; arbitrary means cannot enter the index.')


def verify_complete(snapshot, row, context):
    full8, formal = Path(row['full8_dir']), Path(row['formal_dir'])
    canonical_path = full8/'profile_full8_result.json' if row['origin'] == 'up' else formal/'profile_full8_result.json'
    canonical = snapshot.read(canonical_path); request = snapshot.read(row['request']); completed = snapshot.read(formal/'completed.json')
    for name, path in (('request',row['request']),('completed',formal/'completed.json')):
        reference = canonical['artifact_evidence'][name]
        require(snapshot.path(reference['path']) == snapshot.path(path) and reference['sha256'] == snapshot.ref(path)['sha256'], 'Canonical producer request/completed binding differs.')
    for reference in row['source_declared_refs'].values(): snapshot.bound(reference)
    snapshot.code(request['source_sha256'])
    actual, fee, selection = profile_contract(row,context,canonical,request,completed)
    endpoint = canonical['endpoint']; meta_path = Path(endpoint['path'])/'checkpoint_meta.json'; meta = snapshot.read(meta_path)
    require(snapshot.ref(meta_path)['sha256'] == endpoint['metadata_sha256'] == canonical['ledger']['source']['metadata_sha256']
        and meta['checkpoint_id'] == endpoint['checkpoint_id'] and endpoint['model_identity'] == context['identity']['model']
        and {item['name']:item for item in meta['converted_modules']} == {item['name']:item for item in actual['bank']['converted_modules']}, 'Actual metadata/UUID/native topology changed.')
    require(canonical['artifact_evidence']['bank_state']['sha256'] == actual['bank']['state_sha256']
        and snapshot.path(canonical['artifact_evidence']['bank_state']['path']) == snapshot.path(actual['bank']['state_file']), 'Certified bank file reference changed.')
    payloads = [Path(endpoint['path'])/meta['state_dict_file'],Path(actual['bank']['state_file'])]
    require(all(path.is_file() for path in payloads), 'A declared profile payload is unavailable for subsequent real loading; no payload is reconstructed by this index.')
    plan_path, summary_path, evaluation_path = full8/'plan.json', full8/'full8_summary.json', full8/'evaluation.json'
    plan, summary, evaluation = (snapshot.read(path) for path in (plan_path,summary_path,evaluation_path))
    require(plan['schema'] == 'cat_PUBLIC_native_full8_plan_v0' and plan['phase'] == 'formal' and plan['formal_eligible']
        and plan['endpoint'] == endpoint and plan['baseline_receipt'] == context['baseline_ref']
        and plan['protocol_identity'] == context['protocol_identity'] and plan['authorization']['physical_GPU'] == row['gpu']
        and plan['expected'] == dict(leaf_count=64,sample_count=24742), 'Actual evaluation plan used another artifact/common protocol/resource.')
    snapshot.code(plan['source_sha256'])
    paths = [full8/'mmlu.json',full8/'seven.json']
    require([snapshot.path(item['path']) for item in summary['input_results']] == [snapshot.path(path) for path in paths], 'Summary must bind its own two actual result shards.')
    for stage, name, count in zip(plan['expected_stages'], ('mmlu','seven'), (14042,10700), strict=True):
        require(stage['name'] == name and stage['sample_count'] == count and stage['argv'] == [str(Path(sys.executable)), '-B','-u','-m',
            'experiments.cat_offline_execution_cmcc_20261104.evaluate','--component','shard','--plan',str(plan_path),'--shard',name], 'Actual planned full8 CLI/coverage changed.')
    manifest = dict(plan_path=str(plan_path),plan_sha256=snapshot.ref(plan_path)['sha256'],checkpoint_meta_sha256=endpoint['metadata_sha256'],
        protocol_id=context['protocol_identity'],leaf_count=64,sample_count=24742,task_metric_keys=context['baseline']['task_metric_keys'],expected_stages=plan['expected_stages'])
    score = load_complete_eight_score(summary_path=summary_path,evaluation_path=evaluation_path,checkpoint_dir=endpoint['path'],
        source_checkpoint_id=endpoint['checkpoint_id'],protocol_manifest=manifest)
    derived = collect_results(paths,context['baseline']['tokenizer_inventory'],checkpoint=(Path(endpoint['path']).resolve(),meta))
    check_scores(canonical,summary,derived)
    samples = summary['ordered_sample_identity_evidence']; seen, leaves = {}, {}
    for path in paths:
        payload = snapshot.read(path); result = payload['evals']['lm_eval']; audit = payload['load_audit']
        require(payload['endpoint'] == endpoint and payload['protocol_identity'] == context['protocol_identity']
            and payload['execution_conditions'] == context['baseline']['protocol']['execution']
            and audit['PUBLIC_consumer_verified'] and audit['strict_missing'] == audit['strict_unexpected'] == []
            and audit['checkpoint_id'] == endpoint['checkpoint_id'] and audit['dtype'] == 'torch.bfloat16' and audit['rope_FP32_verified'], 'Actual shard execution/consumer/source differs.')
        expected_execution = context['baseline']['protocol']['execution']; hflm = audit['HFLM']
        require(audit['attn_implementation'] == expected_execution['attn_implementation'] and hflm['max_length'] == expected_execution['max_length']
            and hflm['add_bos_token'] is False and hflm['log_samples'] is True
            and all(hflm[key] == expected_execution[key] for key in ('random_seed','numpy_seed','torch_seed','fewshot_seed')), 'Actual HFLM execution changed.')
        for task, cfg in result['configs'].items():
            expected_task = context['baseline']['protocol']['tasks'][task]
            require(cfg == expected_task['config'] and result['versions'][task] == expected_task['version'], 'Actual task/prompt/version changed.')
        for task, sample in result['sample_artifacts']['tasks'].items():
            sample_path = snapshot.path(sample['path'])
            require(task not in leaves and sample_path not in seen and Path(sample_path).is_relative_to(full8.resolve())
                and Path(sample_path).is_file() and sample['count'] == result['n_samples'][task]['effective'], 'Producer sample manifest is incomplete, duplicated, unavailable or outside this endpoint.')
            seen[sample_path] = sample['sha256']; leaves[task] = sample['count']
    require(len(leaves) == 64 and sum(leaves.values()) == 24742 and seen == samples['files_sha256']
        and samples['leaf_count'] == 64 and samples['sample_count'] == 24742 and set(samples['per_leaf']) == set(leaves)
        and all(samples['per_leaf'][name]['count'] == count for name,count in leaves.items())
        and samples['ordered_input_identity'] == digest(samples['per_leaf']) and canonical['full8_evidence']['ordered_sample_identity_evidence'] == samples,
        'Reuse only mutually consistent original complete sample receipts; no retrospective sample proof invented.')
    evaluation_digest = digest(dict(endpoint=endpoint,shards=[snapshot.ref(path) for path in paths],samples=samples))
    require(canonical['full8_evidence']['evaluation_digest'] == summary['evaluation_digest'] == evaluation_digest
        and canonical['full8_evidence']['protocol_identity'] == context['protocol_identity']
        and canonical['full8_evidence']['leaf_count'] == 64 and canonical['full8_evidence']['sample_count'] == 24742
        and canonical['full8_evidence']['checkpoint_id'] == endpoint['checkpoint_id'] and math.isclose(float(score.mean)*100,canonical['full8_mean'],abs_tol=1e-12,rel_tol=0), 'Actual attached evaluation digest/protocol/arithmetic differs.')
    sources = dict(canonical=snapshot.ref(canonical_path),request=snapshot.ref(row['request']),completed=snapshot.ref(formal/'completed.json'),
        plan=snapshot.ref(plan_path),evaluation=snapshot.ref(evaluation_path),summary=snapshot.ref(summary_path),metadata=snapshot.ref(meta_path),shards=[snapshot.ref(path) for path in paths])
    return dict(status='complete',endpoint=endpoint,bank_digest=canonical['bank_digest'],bank_reference=canonical['artifact_evidence']['bank_state'],
        actual_class_fee=dict(**fee,logical_bpw=fee['logical_bits']/fee['N'],packed_bpw=fee['packed_bits']/fee['N']),
        actual_partial_model_fee=dict(N=canonical['ledger']['N'],logical_bits=canonical['ledger']['logical_bits'],packed_bits=canonical['ledger']['packed_bits'],final_budget_applicable=False),
        full8=dict(mean_percent=canonical['full8_mean'],task_metrics=derived['task_metrics'],protocol_identity=context['protocol_identity'],
            leaf_count=64,sample_count=24742,ordered_candidate_input_identity=samples['ordered_input_identity'],evaluation_digest=evaluation_digest),
        selection_data_identity=selection,diagnostic_KL=canonical['joint_KL'],sources=sources,
        available_payload_stats={str(path):dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,inode=path.stat().st_ino) for path in payloads},
        payload_bytes_rehashed=False,sample_bytes_reverified=False)


def observed_status(snapshot, row):
    full8, formal = Path(row['full8_dir']), Path(row['formal_dir'])
    control = snapshot.optional(row['control'])
    if control:
        item = control if row['origin'] == 'up' else next((item for item in control['candidates'] if item['candidate_id'] == row['candidate_id']),None)
        if item and item.get('status') == 'failed': return 'failed_candidate'
        if item and item.get('status') == 'completed' and (item.get('finished_epoch') or row['origin']=='up'): return 'terminal_completed'
        if item and item.get('status') == 'running': return 'pending_in_progress'
        if control.get('status') == 'stopped': return 'pending_queue_stopped'
    if row['origin'] == 'night': return 'pending_not_started'
    evaluation = snapshot.optional(full8/'evaluation.json')
    if evaluation and evaluation.get('status') == 'failed': return 'failed_full8'
    if evaluation and evaluation.get('status') == 'complete_original_full8': return 'awaiting_canonical_attachment'
    if evaluation: return 'pending_full8'
    if (formal/'completed.json').is_file(): return 'awaiting_formal_mapping_or_full8'
    return 'pending_not_started'


def window(snapshot, source, start_key, end_key, terminal):
    value = snapshot.optional(source)
    if not value or value.get(start_key) is None: return None
    start, end = value[start_key], value.get(end_key)
    complete = terminal(value) and end is not None
    observed = snapshot.observations[snapshot.path(source)]
    elapsed = (end if complete else observed)-start
    require(elapsed >= 0, 'Actual cost window has a future start or reversed endpoint.')
    return dict(seconds=elapsed,GPU_hours=elapsed/3600,status='complete_conservative_wall_window' if complete else 'elapsed_to_observation',source=snapshot.ref(source),
        definition='One assigned GPU wall ownership; includes CPU overhead, not utilization-integrated kernel time.')


def costs(snapshot, row):
    if row['origin'] == 'night':
        queue = snapshot.optional(row['control']); item = next((item for item in queue['candidates'] if item['candidate_id'] == row['candidate_id']),None) if queue else None
        if not item: return dict(chain=None)
        end = item.get('finished_epoch'); done = item['status'] in ('completed','failed') and end is not None
        observed = snapshot.observations[snapshot.path(row['control'])]
        elapsed = (end if done else observed)-item['started_epoch'];require(elapsed >= 0,'Actual queue cost window is reversed.')
        return dict(chain=dict(seconds=elapsed,GPU_hours=elapsed/3600,
            status='complete_conservative_wall_window' if done else 'elapsed_to_observation',source=snapshot.ref(row['control']),
            observed_epoch=observed,
            definition='Actual candidate started-to-finished or observed window; excludes other candidates and prequeue waits.'))
    result = dict(fit_chain=window(snapshot,row['control'],'started_epoch','ended_epoch',lambda value:value.get('status') in ('completed','failed')))
    plan_path, exit_path = Path(row['full8_dir'])/'plan.json', Path(row['full8_dir'])/'controller.exit'
    if plan_path.is_file():
        plan = snapshot.read(plan_path); start = plan['authorization']['authorized_start_epoch']; end = exit_path.stat().st_mtime if exit_path.is_file() else snapshot.observed_epoch
        result['full8_controller_upper_window'] = dict(seconds=end-start,GPU_hours=(end-start)/3600,
            status='complete_conservative_wall_window' if exit_path.is_file() else 'elapsed_to_observation',plan=snapshot.ref(plan_path),
            exit_file=snapshot.ref(exit_path) if exit_path.is_file() else None,definition='Actual plan authorization start to controller.exit mtime; conservative upper bound including launch/verification overhead.')
        evaluation = snapshot.optional(Path(row['full8_dir'])/'evaluation.json')
        result['evaluation_reported_subwindow_seconds'] = evaluation.get('elapsed_seconds') if evaluation else None
    return result


def collect(up_path, eight_path, night_path):
    snapshot = Snapshot(); expected, context = expected_sources(snapshot,up_path,eight_path,night_path); rows=[]
    for row in expected:
        base = dict(category_id=row['category_id'],rate=row['rate'],candidate_id=row['candidate_id'],recipe=row['recipe'],gpu=row['gpu'],origin=row['origin'])
        result = dict(base)
        try:
            canonical = Path(row['full8_dir'])/'profile_full8_result.json' if row['origin']=='up' else Path(row['formal_dir'])/'profile_full8_result.json'
            terminal = observed_status(snapshot,row)
            if row['origin']=='night' and terminal!='terminal_completed':
                result.update(status=terminal,sources=dict(queue=snapshot.ref(row['control']) if Path(row['control']).is_file() else None,
                    expected_canonical=str(canonical)))
            elif canonical.is_file():
                verified = verify_complete(snapshot,row,context)
                selection, inputs = digest(verified['selection_data_identity']), verified['full8']['ordered_candidate_input_identity']
                if context['selection_identity'] is None: context['selection_identity'], context['ordered_full8_inputs'] = selection, inputs
                require(selection==context['selection_identity'] and inputs==context['ordered_full8_inputs'], 'Complete profiles differ in actual selection512/teacher or ordered candidate evaluation inputs.')
                result.update(verified)
            else: result.update(status='awaiting_canonical_attachment' if terminal=='terminal_completed' else terminal,sources=dict(expected_canonical=str(canonical)))
            result['costs'] = costs(snapshot,row)
        except IncompleteRead as error: result=dict(base,status='incomplete_source_read',error=str(error))
        except (ValueError,KeyError,TypeError,FileNotFoundError) as error: result=dict(base,status='invalid',error=type(error).__name__+': '+str(error))
        rows.append(result)
    queue_paths = {snapshot.path(row['control']):row['gpu'] for row in expected if row['origin']=='night'}
    queue_costs = [dict(gpu=gpu,queue_path=path,window=window(snapshot,path,'started_epoch','finished_epoch',
        lambda value:value.get('status') in ('completed_finite_queue','completed_with_failures','stopped'))) for path,gpu in queue_paths.items()]
    counts=dict(Counter(row['status'] for row in rows)); complete=counts.get('complete',0)==len(expected)
    return dict(schema='cat_actual_profile_coverage_snapshot_v1',status='complete' if complete else 'incomplete',observed_epoch=snapshot.observed_epoch,
        expected_profile_count=len(expected),expected_categories=context['inventory']['category_ids'],rates=['2','5/2','3'],status_counts=counts,profiles=rows,
        inputs={name:snapshot.ref(path) for name,path in [('up_fit',up_path),('up_full8',eight_path),('night',night_path)]},
        common_run_identity=context['identity'],common_protocol_identity=context['protocol_identity'],baseline_reference=context['baseline_ref'],
        queue_costs=queue_costs,cost_aggregation_rule='Count each complete night queue window once; candidate windows are diagnostic subwindows and must not be added to queue costs. Up fit and up full8 windows are separate sources.',
        payload_bytes_rehashed=False,sample_bytes_reverified=False,original_producer_receipts_reused=True,
        limitation='Index validates small canonical/request/completed/meta/plan/eval/raw-shard contracts and recalculates eight scores. Tensor/sample payload digests are reused from original validated receipts, not re-audited. Future joint must actually load/install/measure its banks.',
        next_scope='Coverage only; no joint proposals, caps, winners, HPO generation, GPU work, monitoring, waits or retries.',
        implementation=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def validate_rules(up_path, eight_path, night_path, snapshot_result):
    snapshot=Snapshot();expected,context=expected_sources(snapshot,up_path,eight_path,night_path)
    def rejects(function,*args):
        try:function(*args)
        except (ValueError,KeyError,TypeError,FileNotFoundError):return
        raise AssertionError('Coverage accepted a deliberately invalid small-source counterexample.')
    rejects(unique_expected,expected+[expected[0]],context['inventory'])
    row=expected[0];formal=Path(row['formal_dir']);canonical=snapshot.read(Path(row['full8_dir'])/'profile_full8_result.json')
    request=snapshot.read(row['request']);completed=snapshot.read(formal/'completed.json')
    foreign=deepcopy(canonical);foreign['candidate_id']='CPU_foreign_candidate_counterexample'
    rejects(profile_contract,row,context,foreign,request,completed)
    foreign=deepcopy(canonical);foreign['endpoint']['checkpoint_id']='CPU_foreign_UUID_counterexample'
    rejects(profile_contract,row,context,foreign,request,completed)
    wrong_fee=deepcopy(canonical);wrong_fee['logical_bits']=wrong_fee['ledger']['logical_bits']
    rejects(profile_contract,row,context,wrong_fee,request,completed)
    summary=snapshot.read(Path(row['full8_dir'])/'full8_summary.json')
    arbitrary=deepcopy(canonical);arbitrary['full8_mean']+=.01
    rejects(check_scores,arbitrary,summary,summary)
    bad=dict(canonical['artifact_evidence']['request'],sha256='0'*64);rejects(snapshot.bound,bad)
    require(snapshot_result['expected_profile_count']==21 and len(snapshot_result['profiles'])==21,'Incomplete items disappeared from the explicit coverage set.')
    require(all('full8' not in item for item in snapshot_result['profiles'] if item['status']!='complete'),'Pending/failed/invalid profiles received invented scores.')
    require(len(snapshot_result['queue_costs']) == len({row['control'] for row in expected if row['origin']=='night'})
        and len({item['queue_path'] for item in snapshot_result['queue_costs']}) == len(snapshot_result['queue_costs']), 'Night queue ownership windows were duplicated across candidates.')
    closed=window(snapshot,row['control'],'started_epoch','ended_epoch',lambda value:value.get('status') in ('completed','failed'))
    require(closed['status']=='complete_conservative_wall_window' and closed['seconds']>=0, 'Actual terminal up-chain cost was not recognized as complete.')
    return dict(actual_up_complete_sources_tested=True,duplicate_expected_rejected=True,foreign_candidate_and_UUID_rejected=True,
        partial_model_bits_as_class_fee_rejected=True,arbitrary_full8_mean_rejected=True,wrong_request_SHA_rejected=True,
        missing_running_or_failed_expected_items_preserved_without_score=True,
        unique_per_queue_cost_windows_and_real_terminal_chain_verified=True,
        payload_and_sample_bytes_reverified=False,scope='Small actual-source and explicit CPU counterexamples only; no fabricated successful model result.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--up-fit',type=Path,default=DEFAULT_UP)
    parser.add_argument('--up-full8',type=Path,default=DEFAULT_EIGHT);parser.add_argument('--night',type=Path,default=DEFAULT_NIGHT)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Run this single CPU snapshot in caller bitvae with CUDA hidden.')
    require(not args.output.exists(),'Write one new immutable snapshot; never overwrite a result or run automatically.')
    started=time.time(); result=collect(args.up_fit,args.up_full8,args.night)
    if args.validate:result['CPU_validation']=validate_rules(args.up_fit,args.up_full8,args.night,result)
    import torch
    require(not torch.cuda.is_initialized(),'Coverage indexing must not initialize CUDA.')
    result['CUDA_initialized']=False
    result['CPU_seconds']=time.time()-started
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print({key:result[key] for key in ('status','expected_profile_count','status_counts','CPU_seconds')},flush=True)
    return 2 if result['status_counts'].get('invalid') else 0


if __name__=='__main__':raise SystemExit(main())
