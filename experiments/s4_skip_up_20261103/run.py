"""One incremental skip-path validation, then a fresh-HF complete candidate."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from . import bindings as b,fixture

_core=b.bound_implementation('run')
_original_cpu_stage=_core.cpu_stage


def cpu_stage(root,name,argv):
    if name=='merge_LB_nine':
        path=Path(argv[argv.index('--plan')+1]);plan=b.read(path)
        command=plan['nine_task_argv']
        b.require(command[3]=='experiments.s4_from_bf16_20261102.run','Unexpected mature summary caller.')
        command[3]='experiments.s4_skip_up_20261103.run'
        b.write(path,plan)
    return _original_cpu_stage(root,name,argv)


_core.cpu_stage=cpu_stage


def loop_equivalence():
    def function(path):
        return next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='run')
    original=function(b.base.HERE/'recovery.py');new=function(b.HERE/'recovery.py')
    b.require(isinstance(new.body[0],ast.If) and ast.unparse(new.body[0].test)=='stage == 6',
        'The new zero-update branch must execute before original training-data access.')
    new.body.pop(0)
    before=next(node for node in original.body if isinstance(node,ast.Try))
    after=next(node for node in new.body if isinstance(node,ast.Try))
    old_assignment=next(node for node in before.body if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='short_indices')
    index=next(i for i,node in enumerate(after.body) if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='short_indices')
    b.require(ast.unparse(after.body[index].value)=='[0]','Incremental validation has exactly one real down update.')
    after.body[index]=deepcopy(old_assignment)
    b.require(ast.dump(new)==ast.dump(original),'Original recovery optimization/schedule/freezing/state loop changed.')
    return dict(formal_non_up_loop_AST_exact=True,changes=['early up zero-update return','validation only single group0'],
        old_source=b.binding(b.base.HERE/'recovery.py'),new_source=b.binding(b.HERE/'recovery.py'))


def prepare(root):
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not root.exists()
        and root.resolve().is_relative_to(b.OUTPUT),'Prepare a new isolated CPU-only candidate root.')
    proof=b.read(b.CONTROL/'short/validation.json');prior=b.read(b.CONTROL/'plan.json')
    b.require(proof['status']=='verified_fresh_BF16_S4_prefix_GPU_short'
        and proof['source_sha256']==prior['source_sha256'],'Require the completed unchanged original short proof.')
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text())
    for key,value in b.base.CONTRACT.items():
        b.require(all(b.CONTRACT[key][field]==item for field,item in value.items()) if isinstance(value,dict)
            else b.CONTRACT[key]==value,'Skip-up must not change any inherited fit/recovery/budget/data setting: '+key)
    equivalent=loop_equivalence();pins=b.pins()
    root.mkdir(parents=True)
    prepared=fixture.prepare(root/'fixture')
    shutil.copyfile(b.CONTROL/'fixture.pt',root/'probe_inputs.pt')
    source=b.read(b.CONTROL/'formal/request.json')
    for limited,name in [(True,'validation'),(False,'formal')]:
        output=root/name;output.mkdir();(output/'fixture.pt').symlink_to(root/'probe_inputs.pt')
        request=dict(source,output=str(output),source_sha256=pins,contract=b.CONTRACT,validation_only=limited,
            incremental_fixture=prepared if limited else None,old_compressed_inputs=[prepared['model']['path']] if limited else [])
        b.write(output/'request.json',request)
    b.require(not b.torch.cuda.is_initialized(),'CPU source/fixture preparation must not initialize CUDA.')
    cpu=dict(status='verified_skip_up_incremental_CPU',exit_code=0,source_sha256=pins,
        loop_equivalence=equivalent,inherited_contract_fields_exact=True,fixture=b.binding(root/'fixture/fixture.json'),
        reused_original_short=b.binding(b.CONTROL/'short/validation.json'),
        reused_original_CPU=b.binding(b.CONTROL/'CPU_validation.json'),CUDA_initialized=False,
        scope='Only new skip/metadata/bindings/fixture interfaces; original optimizer/data/peak/B64/resume proofs reused.')
    b.write(root/'CPU_validation.json',cpu)
    plan=dict(root=str(root),source_sha256=pins,contract=b.CONTRACT,CPU_validation=b.binding(root/'CPU_validation.json'),
        validation_request=b.binding(root/'validation/request.json'),formal_request=b.binding(root/'formal/request.json'),
        automatic_retry=False,automatic_research=False,physical_GPUs=[6,7])
    b.write(root/'plan.json',plan)
    print(dict(status=cpu['status'],plan_sha256=b.sha(root/'plan.json'),source_count=len(pins)),flush=True)


def execute(path):
    plan=b.read(path);root=Path(plan['root'])
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (root/'run.json').exists(),'New CPU controller only.')
    b.parent.s.check_pins(plan['source_sha256'])
    report=dict(status='validating_incremental_up_skip_down',**b.parent.process_event(os.getpid()),
        started_unix=time.time(),plan_sha256=b.sha(path),source_sha256=plan['source_sha256'],processes={})
    b.write(root/'run.json',report);code=1
    try:
        for name in ['validation','formal']:
            if name=='formal':
                proof=b.read(root/'validation/validation.json')
                b.require(proof['status']=='verified_incremental_up_skip_down' and proof['source_sha256']==plan['source_sha256'],
                    'Formal requires the new actual skip/up/down path proof.')
                report['status']='formal_complete_HF_skip_up_started';b.write(root/'run.json',report)
            _core.stage(root,name+'_fit',[str(b.PYTHON),'-B','-u','-m','experiments.s4_skip_up_20261103.fit',
                '--request',str(root/name/'request.json')],'6,7',report)
            fitted=b.read(root/name/'fit.json')
            b.require(fitted['status']=='complete_fresh_BF16_S4_prefix_recovery','Actual scoped fit/hook chain failed.')
            if name=='validation':
                up,down=fitted['recoveries'];groups=fitted['groups']
                fresh=root/name/'nodes/down_proj/post/evaluation/fresh.json'
                history=b.read(root/name/'nodes/up_proj/post/model/checkpoint_meta.json')['extra_meta']['distill_stage_history']
                b.require([row['category'] for row in groups]==['up_proj','down_proj']
                    and all(row['complete_noncurrent_fields_exact_after_fit'] for row in groups)
                    and up['actual_updates']==0 and up['optimizer_created'] is False and up['complete_state_exact']
                    and up['RNG_exact'] and up['training_groups_consumed']==0
                    and history[-1]['did_train'] is False and history[-1]['actual_updates']==0
                    and down['actual_updates']==1 and down['full_frozen_complement_exact'] and b.read(fresh)['full_state_exact'],
                    'Actual up-fit/skip/down-fit/update/export/fresh contracts must pass.')
                proof=dict(status='verified_incremental_up_skip_down',source_sha256=plan['source_sha256'],
                    actual_new_fits=['up_proj','down_proj'],zero_update_up=True,original_model_state_and_RNG_exact=True,
                    actual_down_updates=1,final_fresh=b.binding(fresh),
                    reused_other_paths=b.binding(b.CONTROL/'short/validation.json'),elapsed_seconds=fitted['elapsed_seconds'])
                b.write(root/name/'validation.json',proof)
                b.cleanup_model(fitted['nodes'][-1]['post']['checkpoint']['path'],'Incremental fixture validation is complete; formal starts again from original HF.')
                b.cleanup_model(root/'fixture/model','The scoped up/down validation has retired; fixture is never a formal parent.')
            else:
                b.require(not fitted['old_compressed_inputs'] and len(fitted['groups'])==7
                    and fitted['recoveries'][5]['actual_updates']==0,'Formal must be the full new HF algorithm with only up skipped.')
        formal=root/'formal';checkpoint=b.read(formal/'fit.json')['nodes'][-1]['post']['checkpoint']['path']
        _core.longbench(formal/'request.json',formal,checkpoint,report)
        report.update(status='complete_HF_skip_up_candidate',exit_code=0);code=0
    except BaseException:
        report.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.time()-report['started_unix'];b.write(root/'run.json',report)
        (root/'run.exit').write_text(str(code)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--request',type=Path);parser.add_argument('--summarize',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare(args.prepare.resolve())
    elif args.summarize:_core.summarize(args.request,args.summarize)
    else:execute(args.plan.resolve(strict=True))


if __name__=='__main__':main()
