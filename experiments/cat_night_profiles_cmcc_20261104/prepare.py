"""Actual CPU source/geometry/tail/router/owned-process checks; never a GPU smoke."""
import argparse
import ast
import dis
from copy import deepcopy
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from . import bindings as b, search
from .run import wait_child
from experiments.cat_auto_protocol_cmcc_20261104.geometry import component_plan


def rejects(function,*args,**kwargs):
    try:function(*args,**kwargs)
    except (ValueError,KeyError,TypeError,FileNotFoundError):return
    raise AssertionError('Invalid explicit night input accepted.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue-root',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();started=time.time()
    b.require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Use caller bitvae with CUDA hidden.')
    import torch
    torch.set_num_threads(2)
    original,inventory,identity=b.prior.common();manifest=b.make_manifest(args.queue_root)
    source=Path(inventory['source_evidence']['path'])
    for name,row in inventory['source_evidence']['files'].items():b.require(b.sha(source/name)==row['sha256'],'Actual original BF16 source changed.')
    b.require(b.sha(original['data_plan'])==original['data_plan_sha256'],'Actual root DataPlan changed.')
    data=torch.load(original['data_plan'],map_location='cpu',weights_only=False)
    b.require(b.digest(data['manifest'])==data['data_identity']==identity['data_identity'] and len(data['calibration'])==len(data['proposal'])==512,'Actual common data state changed.')
    checks=[]
    for gpu,rows in manifest['queues'].items():
        for row in rows:
            recipe=row['recipe'];quote=component_plan(inventory,recipe)
            b.require(quote['capability']['constructible'] and recipe['linear_group_size']==36 and recipe['S']==2
                and recipe['K'] in (32,40,48) and recipe['B']==8192,'Actual full-category recipe/factories changed.')
            expected=192 if recipe['category_id']=='down_proj' else 64
            b.require(set(recipe['protection']['counts'].values())=={expected},'Protection must come from each actual model shape, never copied up counts.')
            checks.append(dict(gpu=int(gpu),label=row['label'],candidate_id=row['candidate_id'],input_protection=expected,
                actual_n_vectors=quote['rows'][0]['n_vectors'],tail_vectors=quote['rows'][0]['n_vectors']%8192,
                predicted_class_logical_bits=quote['logical_bits'],fee_scope='actual factory/geometry forecast; not measured checkpoint'))
    native=search.batched.native.validate_request;private=search._private_validate()
    original_tree=ast.parse(inspect.getsource(native));changed=deepcopy(original_tree);hook=search._Cards();changed=hook.visit(changed)
    b.require(hook.count==1,'Only one actual physical_GPU tuple can change.')
    matches=[node for node in ast.walk(changed) if isinstance(node,ast.Compare) and isinstance(node.left,ast.Subscript)
        and isinstance(node.left.slice,ast.Constant) and node.left.slice.value=='physical_GPU']
    b.require(len(matches)==1,'The actual GPU membership predicate is ambiguous.')
    domain=ast.literal_eval(matches[0].comparators[0]);b.require(domain==tuple(range(6)) and domain in private.__code__.co_consts
        and 0 in domain and 2 in domain and 6 not in domain and 7 not in domain,'Compiled GPU0/2 admission or6/7 refusal failed.')
    matches[0].comparators[0]=ast.Tuple(elts=[ast.Constant(value=value) for value in (1,3,4,5)],ctx=ast.Load())
    b.require(ast.dump(changed,include_attributes=False)==ast.dump(original_tree,include_attributes=False),
        'Normalized actual validator AST changed beyond the one host-domain tuple.')
    before=dict(search.batched.native.execute_candidates.__globals__);entry=search.batched._private_execute()
    imports=[item.argval for item in dis.get_instructions(entry) if item.opname=='IMPORT_NAME']
    b.require(imports.count('experiments.cat_batch_gather_cmcc_20261104.fit')==1 and imports.count('fit_adapter')==1,'Preserve actual batched fit and original bank installer.')
    b.require(all(search.batched.native.execute_candidates.__globals__[key] is value for key,value in before.items()),'Shared runner globals were mutated.')
    from experiments.cat_batch_gather_cmcc_20261104.data import native_loaders,build_block_data_loaders
    tensor=torch.zeros((129024,36,32),dtype=torch.bfloat16);tensor[:,0,0]=torch.arange(129024).remainder(251).to(torch.bfloat16)
    traces=[]
    for factory in (native_loaders,build_block_data_loaders):
        torch.manual_seed(31);loader,_=factory(tensor,8192,104729);trace=[]
        for x,index in loader:
            b.require(torch.equal(x,tensor.index_select(0,index)),'Actual tail gather values differ.')
            trace.append(index.clone())
        b.require(len(trace)==16 and trace[-1].numel()==6144 and torch.unique(torch.cat(trace)).numel()==129024,'Native drop_last=False/full tail coverage changed.')
        traces.append(torch.cat(trace))
    b.require(torch.equal(*traces),'Actual k/v129024-vector sampler/tail order changed.')
    del tensor,traces
    with tempfile.TemporaryDirectory(prefix='cat_night_CPU_process_') as temporary:
        output=Path(temporary)
        try:wait_child([sys.executable,'-B','-c','import subprocess,time; p=subprocess.Popen(["sleep","60"]); print(p.pid,flush=True); time.sleep(60)'],
            dict(os.environ,CUDA_VISIBLE_DEVICES=''),output,time.time()+.3)
        except subprocess.TimeoutExpired:pass
        else:raise AssertionError('The owned failing CPU process was not bounded.')
        result=b.read(output/'exit.json');b.require(result['exit_code']<0,'Timed-out owned worker was not killed/reaped.')
        processes=subprocess.check_output(['ps','-eo','pgid=,pid=,stat='],text=True)
        active=[line for line in processes.splitlines() if int(line.split()[0])==result['process_group_id'] and not line.split()[2].startswith('Z')]
        b.require(not active,'Owned CPU descendants remained active after group cleanup.')
    rejects(b.phase_window,0,0,16201,'formal')
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    for module in ('run','search'):
        subprocess.run([sys.executable,'-B','-m',__package__+'.'+module,'--help'],check=True,stdout=subprocess.DEVNULL)
    b.require(not torch.cuda.is_initialized(),'CPU validation initialized CUDA.')
    args.queue_root.mkdir(parents=True,exist_ok=False);b.write_new(args.queue_root/'manifest.json',manifest)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='CPU_night18_sources_geometry_tail_private_router_and_PG_verified',exit_code=0,CUDA_initialized=False,
        elapsed_seconds=time.time()-started,manifest=b.bound(args.queue_root/'manifest.json'),source_sha256=b.pins(),checks=checks,
        actual_fit_identity_source_files=len(identity['code_identity']['source_sha256']),GPU0_2_domain_admitted_GPU6_7_rejected=True,
        remaining_native_validator_normalized_AST_unchanged=True,shared_source_namespace_unchanged=True,
        native_and_private_bytecode_identical=native.__code__.co_code==private.__code__.co_code,
        bytecode_boundary='Compilation/constant offsets may differ; reverting the one tuple yields exact normalized original AST.',
        actual_KV_129024_vectors_6144_tail_native_and_batched_exact=True,actual_owned_timeout_descendants_killed_and_worker_reaped=True,
        total_GPU_hours=27,per_candidate_seconds=5400,per_GPU_seconds=16200,actual_executable_requests_created=False,GPU_launched=False,
        boundary='Real CPU sources/config/data/factory/tail/process paths only; each actual candidate still needs its own real GPU technical2 before formal/full8.')
    b.write_new(args.report,report);print({key:report[key] for key in ('status','elapsed_seconds','CUDA_initialized')},flush=True)


if __name__=='__main__':main()
