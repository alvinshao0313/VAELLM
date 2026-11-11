"""Finite explicit candidate runner; never launches a full generated search grid.

Caller activates bitvae and invokes this in tmux/nohup. The controller enforces
one shared absolute deadline on its worker process group, without retries.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .adapter import file_sha
from .contracts import digest, inventory_id, require, validate_recipe, validate_run_identity
from .identity import candidate_id


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class _StageAccounting:
    def __init__(self, output, started):
        self.output, self.started, self.rows = output, started, []

    def write(self, active=None):
        report = dict(pid=os.getpid(), elapsed_seconds=time.monotonic() - self.started, active_stage=active,
            stages=self.rows, peak_allocated_bytes=max((row.get('peak_allocated_bytes') or 0 for row in self.rows), default=0),
            peak_reserved_bytes=max((row.get('peak_reserved_bytes') or 0 for row in self.rows), default=0))
        temporary = self.output / 'progress.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(self.output / 'progress.json')
        return report

    @contextmanager
    def stage(self, name):
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        self.write(dict(name=name, started_epoch=time.time()))
        print(dict(event='stage_start', name=name, pid=os.getpid(), started_epoch=time.time()), flush=True)
        status = 'completed'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            peak_allocated = peak_reserved = None
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            row = dict(name=name, status=status, seconds=time.monotonic() - began,
                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved)
            self.rows.append(row); self.write()
            print(dict(event='stage_end', **row), flush=True)


def resource_path_signature(recipe):
    """Match the actual execution path, while reusing routine coefficient work."""
    path = deepcopy(validate_recipe(recipe))
    # These registered scalar changes use the same graph, owners and buffers.
    # Kappa's fixed reference/owner/precision contract remains fully bound.
    for key in ('base_lr', 'multiplier'):
        path['optimizer'].pop(key, None)
    for key in ('commitment_loss_weight', 'entropy_loss_weight'):
        path['loss'].pop(key, None)
    return digest(dict(schema='cat_auto_same_resource_path_v0', recipe=path))


def _path_covered(candidate, observed):
    if candidate['candidate_id'] == observed['candidate_id']:
        return True
    fitted_recipe = observed.get('fit_report', {}).get('resolved_recipe')
    return fitted_recipe is not None and resource_path_signature(candidate['recipe']) == resource_path_signature(fitted_recipe)


def _two_step_fit_completed(fitted):
    stages = fitted.get('stages', [])
    return (fitted.get('status') == 'completed' and fitted.get('exit_code') == 0 and fitted.get('technical_steps') == 2
        and fitted.get('actual_stage_steps') == 2 and fitted.get('original_schedule_steps') == 10000
        and len(stages) == fitted.get('resolved_recipe', {}).get('S')
        and all(stage.get('actual_steps') == 2 for stage in stages))


def _completion_scope(phase, technical_steps):
    return ('actual_model_GPU_fixed_schedule_technical' + str(technical_steps)) if phase == 'technical_validation' else 'actual_model_GPU_complete_fit_proxy_not_full8'


def _validate_model_path_evidence(evidence, inventory, identity, candidates):
    require(evidence.get('exit_code') == 0 and evidence.get('status') == 'completed'
        and evidence.get('phase') == 'technical_validation' and evidence.get('verification_scope') == 'actual_model_GPU_fixed_schedule_technical2'
        and evidence.get('model_identity') == inventory['model_identity'] and evidence.get('data_identity') == identity['data_identity']
        and evidence.get('fit_code_identity') == identity['code_identity'] and evidence.get('consumer_identity') == identity['consumer_identity'],
        'A CPU fixture, old source or different model/data is not applicable model GPU evidence.')
    observed = evidence.get('candidates', [])
    require(observed, 'Model-path evidence contains no actual completed candidate.')
    for row in observed:
        fitted = row.get('fit_report', {})
        recipe = fitted.get('resolved_recipe', {})
        require(_two_step_fit_completed(fitted) and fitted.get('verification_scope') == 'GPU_model_fit'
            and fitted.get('parent_identity') == identity
            and row.get('candidate_id') == fitted.get('candidate_id') == candidate_id(identity, recipe),
            'Formal model-path reuse needs the actual two updates in every real stage; a scope label or one-step diagnostic is insufficient.')
    require(all(any(_path_covered(row, old) for old in observed) for row in candidates),
        'Required consumer/optimizer/geometry-resource paths lack applicable model GPU evidence; routine registered LR/BSQ coefficient changes may reuse the same path.')


def validate_request(request):
    require(request['schema'] == 'cat_auto_explicit_candidates_v0' and request['phase'] in ('technical_validation', 'formal'), 'Explicit versioned model-stage request required.')
    inventory = _read(request['inventory_receipt'])['inventory']
    require(file_sha(request['inventory_receipt']) == request['inventory_receipt_sha256'], 'Frozen actual model inventory receipt changed.')
    require(file_sha(request['data_plan']) == request['data_plan_sha256'], 'Frozen actual data plan changed.')
    identity = validate_run_identity(request['run_identity'])
    require(identity['model'] == inventory['model_identity'], 'Candidate identity model differs from inventory.')
    candidates = request['candidates']
    require(1 <= len(candidates) <= 4 and len({row['candidate_id'] for row in candidates}) == len(candidates),
        'This finite batch accepts1..4 explicitly frozen distinct candidates, never an automatic175-fit run.')
    require(request['parent']['kind'] in ('BF16', 'native_v6') and request['parent']['uuid'] == identity['parent_uuid']
        and request['parent']['weights_digest'] == identity['parent_weights_digest'], 'Own exact parent identity required.')
    require(request['parent']['kind'] == 'BF16' and all(row['purpose'] in ('profile', 'offline') for row in candidates),
        'This finite bank runner serves original-BF16 profile/offline candidates; live uses live.fit_next_live_category for its complete selected-prefix runtime contract.')
    require(request['absolute_deadline_epoch'] > time.time() and request['authorized_gpu_hours'] > 0
        and request['authorization_reference'] and request['physical_GPU'] in (1, 3, 4, 5), 'An explicit new finite batch GPU/deadline/budget authorization is required; protected cards are unavailable.')
    require(request['absolute_deadline_epoch'] - request['authorized_start_epoch'] <= request['authorized_gpu_hours'] * 3600,
        'Absolute deadline exceeds this single-card shared authorization budget.')
    needed = 0.
    for row in candidates:
        require(row['candidate_id'] == candidate_id(identity, row['recipe']) and row['purpose'] in ('profile', 'offline', 'live'), 'Full resolved planned identity/purpose mismatch.')
        require(row['purpose'] == 'live' or request['parent']['kind'] == 'BF16', 'Offline/profile banks must use original BF16, not a historical live point.')
        require(row.get('complete_dependency_gpu_hours', 0) > 0 and row.get('cost_basis'), 'Each candidate must reserve fit/capture/readout/save/fresh costs before launch.')
        needed += row['complete_dependency_gpu_hours']
        if row['purpose'] == 'offline':
            require(isinstance(row.get('class_cap_bits'), int) and row['class_cap_bits'] > 0, 'Offline candidates require their own fixed actual class cap.')
    require(needed + request['shared_overhead_gpu_hours'] <= request['authorized_gpu_hours'], 'Explicit candidates and all shared dependencies exceed the finite batch budget.')
    require(all(Path(path).is_file() and file_sha(path) == sha for path, sha in request['source_sha256'].items()), 'Actual caller/consumer source pins changed.')
    own_sources = {str(path.resolve()) for path in Path(__file__).parent.glob('*.py')}
    require(own_sources <= set(request['source_sha256']), 'Request must pin every actual isolated package Python source, including runtime hooks.')
    if request['phase'] == 'formal':
        require(request['technical_steps'] is None and request['required_model_path_evidence'], 'Formal batch requires applicable model-path evidence, not only CPU fixtures.')
        proof = request['required_model_path_evidence']
        require(file_sha(proof['path']) == proof['sha256'], 'Model-path technical evidence changed.')
        evidence = _read(proof['path'])
        _validate_model_path_evidence(evidence, inventory, identity, candidates)
        for row in candidates:
            if row['purpose'] == 'offline':
                proof = row['class_cap_evidence']
                require(file_sha(proof['path']) == proof['sha256'], 'Frozen actual joint-budget cap evidence changed.')
                caps = _read(proof['path'])
                require(caps.get('formal_eligible') and caps.get('source_proposal_id') and caps.get('joint_evidence_identity')
                    and caps['categories'][row['recipe']['category_id']]['cap_bits'] == row['class_cap_bits'],
                    'CPU/nominal/another-class budget cannot authorize a formal offline candidate.')
    else:
        require(request['technical_steps'] in (1, 2), 'Technical model-path validation uses explicit original-schedule1/2 updates.')
    return inventory, identity


def execute_candidates(request, *, registry_path=None):
    started = time.monotonic()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_utils import checkpoint_v6 as v6
    from train_utils.cat_runtime_state_v6 import build_cat_runtime_state
    from .fit_adapter import fit_category, install_category_bank
    from .ledger import measure_checkpoint
    from .live import architecture, fresh_native
    from .proposal_readout import proposal_batches, build_teacher_cache, score_proposal, score_layer_functions
    inventory, identity = validate_request(request)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(request['physical_GPU'])
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True', 'Caller must select the authorized physical GPU and original allocator exactly.')
    prepared = torch.load(request['data_plan'], map_location='cpu', weights_only=False)
    require(digest(prepared['manifest']) == prepared['data_identity'] == identity['data_identity']
        and prepared['manifest']['model_identity'] == inventory['model_identity']
        and prepared['manifest']['search_seed'] == identity['search_seed'], 'Use the root-frozen actual manifest/tokens and semantic seed, never resample.')
    source = Path(inventory['source_evidence']['path']).resolve()
    require(all(file_sha(source / name) == record['sha256'] for name, record in inventory['source_evidence']['files'].items()), 'Actual source bytes changed after inventory verification.')
    parent = request['parent']
    if parent['kind'] == 'BF16':
        require(Path(parent['path']).resolve() == source and parent['weights_digest'] == digest(inventory['model_identity']['weight_shards_sha256']), 'BF16 parent is the actual frozen original, not a converted donor.')
    else:
        native_meta = v6.load_v6_meta(parent['path'])
        require(native_meta['checkpoint_id'] == parent['uuid'] and file_sha(Path(parent['path']) / native_meta['state_dict_file']) == parent['weights_digest'], 'Actual native parent UUID/weights changed.')

    def load_parent():
        if parent['kind'] == 'BF16':
            return AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
        model = architecture(parent['path'], attn_implementation='sdpa')
        return v6.load_v6_full_checkpoint_into_model(model, parent['path'], strict=True)[0]

    output = Path(request['output_dir']).resolve()
    require(output.is_dir() and not (output / 'completed.json').exists(), 'Controller must create one independent incomplete output directory.')
    accounting = _StageAccounting(output, started)
    accounting.rows.append(dict(name='source_and_data_binding', status='completed', seconds=time.monotonic() - started,
        peak_allocated_bytes=None, peak_reserved_bytes=None))
    accounting.write()
    calibration = dict(input_ids=prepared['calibration'], manifest=prepared['manifest']['calibration'], data_identity=prepared['data_identity'],
        model_path=str(source), dataset_spec='alpaca=1')
    if request['phase'] == 'formal':
        require(registry_path, 'The frozen formal request file is the recipe registry evidence.')
        calibration['formal_recipe_certificate'] = dict(phase='formal', formal_eligible=True,
            registry_path=str(Path(registry_path).resolve()), registry_sha256=file_sha(registry_path))
    with accounting.stage('proposal_pack'):
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
        batches = proposal_batches(prepared['proposal'], tokenizer)
    with accounting.stage('teacher'):
        teacher = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda:0')
        cache = build_teacher_cache(teacher, batches, teacher_identity=inventory['model_identity'], data_identity=prepared['data_identity'])
        del teacher; torch.cuda.empty_cache()
    results = []
    for row in request['candidates']:
        require(time.time() < request['absolute_deadline_epoch'], 'Shared deadline exhausted before the next candidate; no retry.')
        destination = output / row['candidate_id']
        require(not destination.exists(), 'Never overwrite a candidate artifact.')
        with accounting.stage('fit_and_encode:' + row['candidate_id'][:12]):
            model = load_parent()
            result = fit_category(model, inventory=inventory, recipe=row['recipe'], run_identity=identity, calibration_bundle=calibration,
                device='cuda:0', output_dir=destination, technical_steps=request['technical_steps'])
        with accounting.stage('proxy:' + row['candidate_id'][:12]):
            model = result['model'].to('cuda:0').eval()
            score = score_proposal(model, batches, cache, data_identity=prepared['data_identity'])
            fresh_input = batches[0]['inputs']['input_ids'][:1, :8].contiguous()
            with torch.no_grad():
                expected_logits = model(input_ids=fresh_input.to('cuda:0'), use_cache=False).logits.detach().cpu()
        # Read local errors at the actual same parent inputs, not downstream
        # shifted candidate inputs. Keep candidate modules only during this pass.
        with accounting.stage('local:' + row['candidate_id'][:12]):
            candidates = {name: deepcopy(model.get_submodule(name)) for name in row['recipe']['group_signature']}
            model.to('cpu'); torch.cuda.empty_cache()
            actual_parent = load_parent().to('cuda:0').eval()
            local = score_layer_functions(actual_parent, {row['candidate_id']: candidates}, batches,
                projection_names=row['recipe']['group_signature'], parent_identity=dict(uuid=parent['uuid'], weights_digest=parent['weights_digest']),
                data_identity=prepared['data_identity'])[row['candidate_id']]
            del actual_parent, candidates; torch.cuda.empty_cache()
        raw = destination / 'raw_model'
        specs = v6._collect_vae_linear_specs(model)
        compressed = [spec['name'] for spec in specs]
        all_names = [item['actual_name'] for item in inventory['projections']]
        runtime = build_cat_runtime_state(activation_runtime=result['activation_bundle']['activation_runtime'], global_adaptive_plan=None,
            runtime_identity=dict(protocol='cat_auto_explicit_candidates_v0', candidate_id=row['candidate_id'], recipe=row['recipe'], parent_identity=identity))
        with accounting.stage('save:' + row['candidate_id'][:12]):
            saved = v6.save_v6_full_checkpoint(model, str(raw), checkpoint_kind='round_base', compressed_targets=compressed,
                pending_dense_targets=[name for name in all_names if name not in compressed],
                completed_categories=[] if parent['kind'] == 'BF16' else native_meta['completed_categories'],
                compression_categories=inventory['category_ids'] if parent['kind'] == 'BF16' else native_meta['compression_categories'], target_layers=sorted({item['layer_id'] for item in inventory['projections']}),
                target_modules=inventory['category_ids'], base_model_path=str(source), tokenizer=tokenizer, cat_runtime_state=runtime,
                extra_meta=dict(cat_auto_phase=request['phase'], candidate_id=row['candidate_id'], own_parent=identity, purpose=row['purpose'],
                    formal_search_boundary=False, single_category_raw_only=row['purpose'] != 'live'))
            ledger = measure_checkpoint(raw, inventory)
        if row['purpose'] == 'offline':
            require(ledger['per_category'][row['recipe']['category_id']]['logical_bits'] <= row['class_cap_bits'],
                'Actual complete class deployment fees exceed its fixed cap; no borrowing or quality comparison.')
        del model; torch.cuda.empty_cache()
        with accounting.stage('fresh:' + row['candidate_id'][:12]):
            fresh, evidence = fresh_native(raw, model_factory=lambda directory: architecture(directory, attn_implementation='sdpa'))
            require(evidence, 'Independent actual raw reload evidence missing.')
            fresh.to('cuda:0').eval()
            with torch.no_grad():
                actual_logits = fresh(input_ids=fresh_input.to('cuda:0'), use_cache=False).logits.detach().cpu()
            require(torch.equal(expected_logits, actual_logits), 'Independent fresh same-path BF16 logits changed.')
            evidence.update(forward_executed=True, gpu_validation=True, same_input_same_BF16_path_logits_exact=True,
                input_identity=digest(fresh_input.tolist()))
        record = dict(candidate_id=row['candidate_id'], purpose=row['purpose'], fit_report=result['fit_report'],
            bank=result['category_bank'], joint_readout=score, layer_functional_readout=local, actual_ledger=ledger,
            raw_checkpoint=saved, fresh_evidence=evidence, exit_code=0, status='completed',
            complete_full8_evaluated=False, formal_result_eligibility=request['phase'] == 'formal',
            quality_scope='Fixed train512 proxy/local errors only; full8 remains a separately budgeted dependency')
        (destination / 'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        results.append(record)
        del fresh, result; torch.cuda.empty_cache()
    completed = dict(status='completed', exit_code=0, candidates=results, absolute_deadline_epoch=request['absolute_deadline_epoch'],
        inventory_identity=inventory_id(inventory), data_identity=prepared['data_identity'], no_automatic_followups=True,
        phase=request['phase'], model_identity=inventory['model_identity'], fit_code_identity=identity['code_identity'], consumer_identity=identity['consumer_identity'],
        actual_technical_steps=request['technical_steps'],
        stage_accounting=accounting.write(),
        verification_scope=_completion_scope(request['phase'], request['technical_steps']),
        actual_end_epoch=time.time(), request_identity=digest(request))
    (output / 'completed.json').write_text(json.dumps(completed, ensure_ascii=False, indent=2) + '\n')
    return completed


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(); request = _read(args.request)
    validate_request(request)
    if args.worker:
        execute_candidates(request, registry_path=args.request)
        return
    output = Path(request['output_dir']).resolve()
    require(not output.exists(), 'Use an independent new finite batch output.')
    output.mkdir(parents=True)
    (output / 'request.json').write_text(json.dumps(request, indent=2, ensure_ascii=False) + '\n')
    log = (output / 'worker.log').open('w')
    worker = subprocess.Popen([sys.executable, '-m', __package__ + '.search', '--request', str(output / 'request.json'), '--worker'],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state = dict(pid=worker.pid, controller_pid=os.getpid(), started_epoch=time.time(), command=worker.args, deadline_epoch=request['absolute_deadline_epoch'])
    (output / 'process.json').write_text(json.dumps(state, indent=2) + '\n')
    try:
        code = worker.wait(timeout=max(0., request['absolute_deadline_epoch'] - time.time()))
    except BaseException as error:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        worker.wait()
        (output / 'exit.json').write_text(json.dumps(dict(exit_code=worker.returncode, end_epoch=time.time(),
            actual_completed=(output / 'completed.json').is_file(), controller_exception=type(error).__name__,
            reason=str(error), absolute_deadline_epoch=request['absolute_deadline_epoch'])) + '\n')
        raise
    finally:
        log.close()
    if code:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    (output / 'exit.json').write_text(json.dumps(dict(exit_code=code, end_epoch=time.time(), actual_completed=(output / 'completed.json').is_file())) + '\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
