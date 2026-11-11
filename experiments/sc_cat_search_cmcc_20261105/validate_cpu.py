"""Necessary CPU contracts using real sources and labelled comparator-only values."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import sys
import time

from . import bindings as b
from .run import score_key,select_round,advance,distinct_count,wait_for_prior


def rejects(function,*args,**kwargs):
    try:function(*args,**kwargs)
    except (ValueError,FileNotFoundError):return
    raise AssertionError('Invalid contract was accepted: '+function.__name__)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();started = time.time();torch.set_num_threads(2)
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU validation needs caller bitvae and hidden CUDA.')
    root = b.REPO/'artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31'
    authorization = root/'authorization.json';auth = b.validate_authorization(b.read(authorization))
    manifest = b.make_manifest(root,authorization,auth['prior_manifest']['path'],auth['result_document'])
    inventory = b.read(manifest['inventory']['path'])['inventory']
    b.require(b.worst_case_GPU_hours(len(inventory['category_ids'])) == 350,'New-order/width candidate budget is350, not old292.25.')
    cache = dict(results={},failures={});first = []
    for category in inventory['category_ids']:
        rows = b.proposals(manifest,category,'protection',b.DEFAULT_SETTINGS,cache)
        b.require(len(rows) == 1 and rows[0]['status'] == 'declared_pending','Exactly the declared real first protection candidate is required.')
        row = rows[0];recipe = row['recipe']
        b.require(recipe['S'] == 2 and recipe['linear_group_size'] == 36 and recipe['steps'] == 10000
            and all(count % recipe['D'] == 0 for count in recipe['protection']['counts'].values())
            and row['predicted_logical_bits'] <= manifest['caps'][category]['cap_bits'],'First-round real geometry/cap/fullM changed.')
        first.append({key:row[key] for key in ('candidate_id','recipe','predicted_logical_bits','predicted_packed_bits')})
    for dimension in (8,16,32,64,128):
        settings = dict(b.DEFAULT_SETTINGS,dimension=dimension,transpose={name:index%2 == 0 for index,name in enumerate(first[0]['recipe']['group_signature'])})
        recipe = b.recipe(inventory,'q_proj',manifest['rates']['q_proj'],settings)
        b.require(recipe['B']*dimension == 262144 and all(count % dimension == 0 for count in recipe['protection']['counts'].values()),'Mixed-map D remapping failed.')
        quote = b.component_plan(inventory,recipe);b.require(quote['capability']['constructible'],'Actual component geometry cannot retain the full mixed map.')
    uniform = b.recipe(inventory,'q_proj',manifest['rates']['q_proj'],dict(b.DEFAULT_SETTINGS,transpose={name:False for name in first[0]['recipe']['group_signature']}))
    base = b.recipe(inventory,'q_proj',manifest['rates']['q_proj'],b.DEFAULT_SETTINGS)
    b.require(uniform == base and b.recipe_key(uniform) == b.recipe_key(base),'Uniform per-layer map must alias the original bool recipe/ID.')
    # These numbers are pure ordering inputs, never fabricated model evaluation receipts.
    b.require(score_key(70.,100,104,'z',current_id='z') < score_key(70.,100,104,'a',current_id='z'),'Exact score/fee tie must keep current before stable ID.')
    b.require(score_key(70.,99,104,'a',current_id='z') < score_key(70.,100,104,'z',current_id='z'),'Actual lower logical fee precedes current.')
    b.require(score_key(70.000001,100,104,'a',current_id='z') < score_key(70.,100,104,'z',current_id='z'),'Any actual positive score improvement must count.')
    rejects(score_key,float('nan'),100,104,'a');rejects(score_key,True,100,104,'a')
    # Same real artifact with different requested provenance must retain the entering target fraction.
    probe = deepcopy(manifest);probe['output_root'] = str(args.output.parent/'executor_import_probe')
    anchor = deepcopy(manifest['anchors']['q_proj']);anchor['settings']['protection'] = '1/64'
    alias_settings = dict(anchor['settings'],protection='1/32')
    stage = dict(candidates=[dict(status='reused',candidate_id=anchor['candidate_id'],result=anchor['result'],settings=alias_settings)])
    winner,_ = select_round(probe,anchor,stage)
    b.require(winner['settings']['protection'] == '1/64','A same-resolved provenance alias overwrote round-entry target protection.')
    stage['candidates'][0]['status'] = 'declared_pending';rejects(select_round,probe,anchor,stage)
    state = dict(status='running',next_parameter=0,current=anchor,anchor=anchor,known={anchor['candidate_id']:anchor['result']},
        rounds=[dict(status='running',parameter='protection',input=anchor,candidates=[dict(status='failed',error='Labelled dependency-state rejection only')])])
    advance(probe,'q_proj',state,cache)
    b.require(state['status'] == 'failed' and state['next_parameter'] == 0 and len(state['rounds']) == 1 and distinct_count(state) == 1,'A failed dependency must block only this category without claiming a score or adding a configuration.')
    # A real still-running prior controller must be refused by the no-wait path.
    old_state = b.read(Path(manifest['old_output_root'])/'search.json')
    prior_refused = old_state['status'] not in ('completed','completed_with_failures') or Path('/proc',str(old_state['controller_pid'])).exists()
    if prior_refused:rejects(b.prior_complete,manifest)
    # Exercise an actually exited subprocess through the mature pidfd boundary.
    import subprocess
    from experiments.cmcc_runtime_20261030.sequence import proc_start,wait_identity
    child = subprocess.Popen([sys.executable,'-c','pass'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    ticks = proc_start(child.pid);child.wait()
    exited = wait_identity(child.pid,ticks)
    rejects(wait_identity,os.getpid(),-1)
    b.require(not torch.cuda.is_initialized(),'CPU verification initialized CUDA.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    b.write_new(args.output,dict(status='CPU_verified',exit_code=0,elapsed_seconds=time.time()-started,
        source_sha256=b.pins(),authorization=b.bound(authorization),first_seven_protection_candidates=first,
        worst_case_candidate_GPU_hours=350.,readout_GPU_hours=3.5,source9_unchanged=manifest['run_identity']['code_identity'],
        current_tie_and_requested_target_retention=True,mixed_D_geometry_and_uniform_alias_verified=True,
        failed_and_pending_dependencies_rejected=True,prior_running_no_wait_refused=prior_refused,
        actual_exited_dependency_boundary=exited,PID_generation_mismatch_rejected=True,CUDA_initialized=False,GPU_launched=False,
        boundary='CPU real-source/configuration/evidence/comparator contracts only; no new model GPU path or quality result.'))
    print(dict(status='CPU_verified',seconds=time.time()-started),flush=True)


if __name__ == '__main__':main()
