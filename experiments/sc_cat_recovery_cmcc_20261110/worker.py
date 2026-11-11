"""Actual new-batch KD, node export and independent fresh operations."""
import argparse
from dataclasses import asdict
import functools
import gc
import logging
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native
from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds
from experiments.sc_cat_live_cmcc_20261106.recovery_worker import technical_peak_proof
from train_utils import checkpoint_v6 as v6
from . import io
from .builder import make_native_recovery_trainer, RecoveryExecutionPolicy
from .boundary import export_selected_node, export_zero_kd_boundary
from .selection import SelectedCategory
from .settings import RecoverySettings


def train(request):
    inventory, prepared, sources, tokenizer = io.load_inputs(request)
    raw = io.read(io.bound_path(request['live_raw']))
    require(raw['phase']==request['phase'] and raw['current_category']==request['category_id'], 'Training source phase/category differs.')
    settings = RecoverySettings(**request['settings'])
    require(request['phase']!='formal' or not request.get('technical_stop_after'), 'No short state may seed a formal run.')
    require(v6.load_v6_meta(raw['round_base_dir'])['checkpoint_id']==raw['round_base_uuid'], 'Own raw UUID changed.')
    run = make_native_recovery_trainer(round_base_dir=raw['round_base_dir'], inventory=inventory,
        category_id=raw['current_category'], settings=settings, prepared_data=prepared, sources=sources,
        tokenizer=tokenizer, output_dir=request['output'], execution_policy=RecoveryExecutionPolicy(**request['execution_policy']),
        phase=request['phase'], technical_stop_after=request.get('technical_stop_after'),
        stop_at=request.get('stop_at'), resume_from=request.get('resume_from'),
        replay_selected_step=request.get('replay_selected_step'), fixed_calibrations=request.get('fixed_calibrations'))
    began = time.monotonic()
    try:
        result = run.train()
        peak = technical_peak_proof(run, prepared) if request.get('technical_peak_proof') else None
        nodes = []
        for audit in run.trainer.node_save_audits:
            path = Path(audit['directory'])
            meta = v6.load_v6_training_step_meta(str(path))
            nodes.append(dict(step=audit['step'], directory=str(path), node_id=meta['checkpoint_id'], metadata=io.binding(path/'checkpoint_meta.json')))
        report = dict(status='complete_segment', phase=request['phase'], settings=asdict(settings),
            live_raw=request['live_raw'], trajectory_id=request['trajectory_id'], source_candidate_id=request['source_candidate_id'],
            actual=result, nodes=nodes, optimizer=run.optimizer.effective_contract(),
            peak_proof=peak, elapsed_seconds=time.monotonic()-began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        io.write(Path(request['output'])/'training_result.json', report)
        return report
    finally:
        run.close()


def export(request):
    inventory, prepared, _, tokenizer = io.load_inputs(request, with_sources=False)
    raw = io.read(io.bound_path(request['live_raw']))
    settings = RecoverySettings(**request['settings'])
    require(raw['phase']==request['phase'] and raw['current_category']==request['category_id'], 'Export source phase/category differs.')
    seeds = recovery_seeds(inventory['model_identity'], request['category_id'], prepared['manifest']['search_seed'])
    zero = request['operation']=='export_zero'
    if zero:
        node_id, step = raw['round_base_uuid'], 0
    else:
        metadata = io.bound_path(request['node_directory'])
        require(metadata.name=='checkpoint_meta.json', 'Bind the exact native node metadata.')
        directory = metadata.parent
        node = v6.load_v6_training_step_meta(str(directory))
        require(node['immutable_resume_contract']['sc_cat_recovery_settings']==request['settings'], 'Export must reconstruct the saved complete settings.')
        state = io.read(directory/'trainer_state.json')
        node_id, step = node['checkpoint_id'], state['global_step']
        require(state['max_steps']==settings.schedule_max, 'Export changed the original schedule.')
    selected = SelectedCategory(request['category_id'], request['recipe_id'], request['source_candidate_id'],
        settings, step, request['trajectory_id'], node_id, raw['round_base_uuid'], seeds, request['phase'])
    current = [r['actual_name'] for r in inventory['projections'] if r['category_id']==request['category_id']]
    common = dict(destination=request['destination'], selected=selected, current_targets=current,
        phase=request['phase'], model_factory=functools.partial(architecture,attn_implementation='flash_attention_2'))
    began = time.monotonic()
    report = export_zero_kd_boundary(round_base_dir=raw['round_base_dir'], **common) if zero else export_selected_node(node_directory=directory, **common)
    tokenizer.save_pretrained(request['destination'])
    report.update(source_raw_reference=request['live_raw'], selected=asdict(selected))
    io.write(Path(request['output'])/'deployment_export.json', report)
    fresh_request = dict(request, operation='fresh', checkpoint_dir=report['path'],
        expected_uuid=report['uuid'], expected_digest=report['digest'], output=str(Path(request['output'])/'fresh'))
    path = Path(request['output'])/'fresh_request.json'
    io.write(path, fresh_request)
    gc.collect()
    subprocess.run([sys.executable,'-B','-u','-m',__package__+'.worker','--request',str(path)],
        check=True, timeout=max(.1,request['deadline_epoch']-time.time()), stdin=subprocess.DEVNULL)
    proof_path = Path(fresh_request['output'])/'fresh_result.json'
    proof = io.read(proof_path)
    report.update(fresh_forward_evidence=proof, fresh_proof_reference=io.binding(proof_path),
        elapsed_seconds=time.monotonic()-began, gpu_validation=True)
    io.write(Path(request['output'])/'deployment_report.json', report)
    return report


def fresh(request):
    _, prepared, _, _ = io.load_inputs(request, with_sources=False)
    model, proof = fresh_native(request['checkpoint_dir'], model_factory=functools.partial(architecture,attn_implementation='flash_attention_2'))
    require(proof['checkpoint_id']==request['expected_uuid'] and proof['weight_digest']==request['expected_digest'], 'Fresh source changed.')
    from tools.cat_eval import _prepare_model_for_eval
    model.to('cuda:0').eval().requires_grad_(False)
    warm = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    require(not warm['failed'] and warm['warmed']==warm['total'], 'Deployment warmup incomplete.')
    ids = prepared['calibration'][0].to('cuda:0')
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits
    require(bool(torch.isfinite(logits).all()), 'Fresh deployment has nonfinite logits.')
    proof.update(exit_code=0, forward_executed=True, finite_logits=True, gpu_validation=True,
        device='cuda:0', input_shape=list(ids.shape), warmup=warm, independent_process=True)
    io.write(Path(request['output'])/'fresh_result.json', proof)
    return proof


def main():
    from . import EXECUTION
    print({'reproducibility_execution': EXECUTION}, flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    args = parser.parse_args()
    request = io.read(args.request)
    io.validate_request(request)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Allocated worker window expired.')))
    signal.setitimer(signal.ITIMER_REAL, max(.1,request['deadline_epoch']-time.time()))
    torch.set_num_threads(4)
    try:
        from .raw import first_raw, refit
        result = {'train':train,'export':export,'export_zero':export,'fresh':fresh,'first_raw':first_raw,
            'refit':lambda r:refit(r,io.read(io.bound_path(r['parent'])))}[request['operation']](request)
        print(result.get('status',request['operation']+'_completed'),flush=True)
    except BaseException:
        path = Path(request['output'])/'failed.json'
        if not path.exists():
            io.write(path,dict(exit_code=1,request=io.binding(args.request),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
