"""First-class exact BF16 bank reuse; all later fitting uses the true boundary."""
from copy import deepcopy
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.fit_adapter import install_category_bank
from experiments.cat_auto_protocol_cmcc_20261104.ledger import measure_checkpoint
from experiments.sc_cat_live_cmcc_20261106.live_io import source9_live_fit, live_identity, verify_fit_sources
from experiments.sc_cat_live_cmcc_20261106.fit_worker import probe
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native
from train_utils import checkpoint_v6 as v6
from train_utils.cat_runtime_state_v6 import build_cat_runtime_state
from . import io


def first_raw(request):
    manifest, _ = io.validate_request(request)
    handoff = io.read(io.bound_path(manifest['selected_settings']))
    category = request['category_id']
    require(category==handoff['frozen_order'][0] and request['source_candidate_id'] in handoff['categories'][category]['top2'],
        'Only the first class of the BF16-root chain can reuse these offline banks.')
    source = handoff['candidates'][request['source_candidate_id']]
    original = io.read(io.bound_path(source['canonical']))
    old = io.read(io.bound_path(handoff['source_manifest']))
    verify_fit_sources(old)
    audit=io.read(io.bound_path(manifest['artifact_cpu']))
    for item in audit['source_files']:
        stat=Path(item['path']).stat()
        require((stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)==(item['bytes'],item['mtime_ns'],item['ctime_ns']),
            'The CPU-verified BF16 source changed; rehash this source before reuse.')
    inventory, prepared, _, tokenizer = io.load_inputs(request, with_sources=False)
    require(original['run_identity']==old['run_identity']==handoff['run_identity']
        and prepared['data_identity']==original['run_identity']['data_identity']
        and original['recipe']==source['recipe'], 'Full original fit/data/init/consumer contract required for reuse.')
    model_path = inventory['source_evidence']['path']
    # The bank installer hashes the actual target matrices and bank tensors and
    # verifies every untouched model field, rather than relabeling an old raw.
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True, trust_remote_code=False)
    model = install_category_bank(model, original['bank'], inventory=inventory, expected_parent_identity=original['run_identity'])
    compressed = list(original['recipe']['group_signature'])
    pending = [r['actual_name'] for r in inventory['projections'] if r['actual_name'] not in compressed]
    runtime_identity = dict(schema='sc_cat_BF16_bank_reuse_v1', phase=request['phase'],
        original_parent=handoff['parent'], source_candidate=source['canonical'],
        source_bank_state_sha256=original['bank']['state_sha256'], complete_fit_identity=original['run_identity'],
        frozen_order=handoff['frozen_order'], completed_prefix=[], new_run_id=manifest['run_id'])
    runtime = build_cat_runtime_state(activation_runtime=None, global_adaptive_plan=None, runtime_identity=runtime_identity)
    destination = Path(request['output'])/'round_base'
    require(not destination.exists(), 'Never replace a new-chain raw.')
    saved = v6.save_v6_full_checkpoint(model.eval().requires_grad_(False), str(destination), checkpoint_kind='round_base',
        compressed_targets=compressed, pending_dense_targets=pending, completed_categories=[],
        compression_categories=handoff['frozen_order'], target_layers=sorted({r['layer_id'] for r in inventory['projections']}),
        target_modules=inventory['category_ids'], cat_runtime_state=runtime, base_model_path=model_path, tokenizer=tokenizer,
        extra_meta=dict(cat_auto_phase=request['phase'], sc_cat_run_id=manifest['run_id'], distill_stage_history=[], round_idx=0,
            category=category, active_category=category, cat_auto_bank_reuse=runtime_identity))
    ids = prepared['calibration'][0]
    expected, _ = probe(model,ids)
    del model
    gc.collect(); torch.cuda.empty_cache()
    fresh, proof = fresh_native(destination,model_factory=lambda path:architecture(path,attn_implementation='sdpa'))
    actual, _ = probe(fresh,ids)
    require(torch.equal(expected,actual), 'Installed bank and independently loaded raw have different BF16 outputs.')
    proof.update(gpu_validation=True, forward_executed=True, same_input_logits_exact=True)
    result = dict(round_base_dir=str(destination), round_base_uuid=saved['checkpoint_id'],
        source_parent_uuid=handoff['parent']['uuid'], current_category=category, phase=request['phase'],
        source_candidate_id=request['source_candidate_id'], recipe=original['recipe'], run_identity=original['run_identity'],
        actual_ledger=measure_checkpoint(destination,inventory), bank_reuse=runtime_identity,
        formal_eligible=request['phase']=='formal', fresh_evidence=proof, raw_full8_required=True)
    require(result['actual_ledger']['per_category'][category]['logical_bits']<=handoff['categories'][category]['cap']['cap_bits'], 'Reused bank exceeds class cap.')
    io.write(Path(request['output'])/'live_raw.json',result)
    return result


def refit(request, parent):
    """No historical live parent or suffix is accepted by this fitting bridge."""
    manifest, _ = io.validate_request(request)
    handoff = io.read(io.bound_path(manifest['selected_settings']))
    order = handoff['frozen_order']
    replay=request.get('replay_manifest')
    require(parent['formal_eligible'] and parent['completed_prefix']==order[:order.index(request['category_id'])]
        and (parent['completed_prefix'] or replay), 'Later classes require the actual selected new-chain prefix.')
    meta = v6.load_v6_meta(parent['path'])
    if replay:
        frozen=io.read(io.bound_path(replay))
        selected=next(c for c in frozen['categories'] if c['category_id']==request['category_id'])
        require(frozen['run_id']==manifest['run_id'] and frozen['original_BF16_root']==handoff['parent']
            and selected['source_candidate_id']==request['source_candidate_id']
            and Path(parent['path']).resolve().is_relative_to(Path(manifest['output_root']).resolve()/'remaining'/'replay')
            and meta['extra_meta'].get('cat_auto_bf16_root',{}).get('original_BF16_root'),
            'Fixed replay requires its own original-BF16 prefix and frozen selected recipe, never a search donor.')
    else:
        require(meta['extra_meta'].get('sc_cat_run_id')==manifest['run_id'], 'A different run boundary cannot seed this chain.')
    source = handoff['candidates'][request['source_candidate_id']]
    if replay:
        require(source['recipe']==selected['resolved_recipe'],'Replay changed the frozen VAE recipe.')
    require(request['source_candidate_id'] in handoff['categories'][request['category_id']]['top2'], 'Not an offline Top2 recipe.')
    inventory, prepared, _, _ = io.load_inputs(request,with_sources=False)
    old = io.read(io.bound_path(handoff['source_manifest']))
    verify_fit_sources(old)
    identity = live_identity(old,parent)
    output = Path(request['output'])
    output.mkdir(parents=True,exist_ok=False)
    from experiments.cat_auto_protocol_cmcc_20261104.identity import candidate_id
    registry = output/'formal_registry.json'
    io.write(registry,dict(phase=request['phase'],run_identity=identity,technical_steps=None,
        data_plan=str(io.bound_path(manifest['data_plan'])),data_plan_sha256=manifest['data_plan']['sha256'],
        candidates=[dict(candidate_id=candidate_id(identity,source['recipe']),recipe=source['recipe'])]))
    calibration = dict(input_ids=prepared['calibration'],manifest=prepared['manifest']['calibration'],
        data_identity=prepared['data_identity'],model_path=inventory['source_evidence']['path'],dataset_spec='alpaca=1',
        formal_recipe_certificate=dict(phase='formal',formal_eligible=True,
            registry_path=str(registry),registry_sha256=io.binding(registry)['sha256']))
    result = source9_live_fit(boundary_reference=parent,run_identity=identity,resolved_recipe=source['recipe'],
        inventory=inventory,calibration_bundle=calibration,frozen_order=order,device='cuda:0',output_dir=output/'fit',
        attention_implementation='sdpa',phase='formal')
    model = result.pop('_fitted_model'); result.pop('_category_bank'); result.pop('activation_bundle')
    expected, _ = probe(model, prepared['calibration'][0])
    del model
    gc.collect(); torch.cuda.empty_cache()
    fresh, proof = fresh_native(result['round_base_dir'],model_factory=lambda path:architecture(path,attn_implementation='sdpa'))
    actual, _ = probe(fresh, prepared['calibration'][0])
    require(torch.equal(expected,actual), 'Own-live fitted raw differs after fresh load.')
    proof.update(gpu_validation=True, forward_executed=True, same_input_logits_exact=True)
    require(result['actual_ledger']['per_category'][request['category_id']]['logical_bits']<=handoff['categories'][request['category_id']]['cap']['cap_bits'], 'Own-live class exceeds cap.')
    result.update(phase='formal',run_identity=identity,recipe=source['recipe'],formal_eligible=True,
        fresh_evidence=proof,raw_full8_required=True)
    io.write(output/'live_raw.json',result)
    return result
