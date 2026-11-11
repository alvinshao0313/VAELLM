"""Fresh actual-C0 two-card arm: short2 bridge or fixed continuous128 learning."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from . import bindings as b
from . import fit, readout


def execute(args, identity, observe, started):
    dataset,recovery,evidence = b.data.load()
    model,meta,budget = b.core.load_source(identity['source_binding'])
    inventory = b.core.mlp_inventory(model)
    names = [name for group in b.checkpoint.inventories(model).values() for name in group]
    teacher = b.data.original.teacher_identity(model.state_dict(),names)
    non_mlp = b.life.non_mlp_proof(model)
    case = recovery[dataset['groups'][2]['case_ids'][0]]['case']
    fixture = dict(schema='all_MLP_learning_same_input_v1',input_ids=case['input_ids'].unsqueeze(0),
        attention_mask=b.torch.ones_like(case['input_ids']).unsqueeze(0),source_case_id=case['case_id'],
        input_signature=case['input_signature'])
    handles,device_map = b.life.apply_map(model)
    b.core.install_precision_runtime(model,b.torch.bfloat16)
    prewarm = b.core.prewarm_deployment(model)
    references,protection = b.life.capture_source(model)
    source_logits = b.run_short.probe(model,fixture)
    indices = [2,10] if args.mode == 'short' else list(range(32))
    source_readout = readout.evaluate(model,dataset,recovery,indices)
    if args.arm == 'dense':
        b.free_dense.replace_all_mlp(model,identity['source_binding'])
        b.require(b.life.non_mlp_proof(model) == non_mlp, 'Dense initialization changed the frozen C0 backbone.')
    active = b.life.training_mode(model,args.arm)
    live_logits = b.run_short.probe(model,fixture)
    live_readout = readout.evaluate(model,dataset,recovery,indices,source_readout)
    initial = b.run_short.comparison(source_logits,live_logits)
    if args.arm == 'dense':
        b.require(initial['exact'] and live_readout['M_cases'] == source_readout['M_cases']
            and live_readout['domain_KL'] == source_readout['domain_KL'],
            'Dense must initialize from the exact common C0 deployment function.')
    b.write(args.output/'initialization.json',dict(source_deployed=source_readout,zero_live=live_readout,
        source_to_zero_live_probe=initial,non_MLP_source=non_mlp,protection=protection,
        source_inventory=inventory,source_budget=budget,teacher_identity=teacher,prewarm=prewarm,
        layer_device_map=device_map,data=evidence,active_leaves=len(active),
        active_elements=sum(p.numel() for p in active.values())))
    observe('actual_C0_all108_initialized',dict(arm=args.arm,live_to_deployed=initial,
        active_leaves=len(active),memory=b.life.memory(),initial_objective=live_readout['weighted_objective']))
    training,optimizer = fit.fit(model,args.arm,active,dataset,recovery,references,identity,args.output,
        non_mlp,protection,fixture,source_readout,observe,started+args.timeout_seconds)
    b.require(training['steps'] == identity['planned_updates'] and training['frozen_before'] == training['frozen_after'],
        'Finish this actual fixed budget with its complete frozen proof.')
    report = dict(**identity,status='all_MLP_learning_training_complete_fresh_pending',exit_code=0,
        training=training,elapsed_seconds=time.monotonic()-started,initialization=b.binding(args.output/'initialization.json'),
        scope='Fixed same-C0 static-TF parameterization comparison; source/quality consumers remain independent.')
    b.write(args.output/'report.json',report)
    del optimizer,active,model,handles,recovery,references
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('decoder','dense'),required=True)
    parser.add_argument('--mode',choices=('short','formal'),required=True)
    parser.add_argument('--checkpoint',type=Path,default=b.C0)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',type=b.gpus,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=43200)
    args = parser.parse_args()
    args.output = args.output.resolve()
    b.require(not os.path.lexists(args.output) and args.timeout_seconds>0, 'Use a fresh output and a positive runtime budget.')
    source = b.source_C0(args.checkpoint)
    resources = b.resources(args.gpus)
    b.torch.set_num_threads(4)
    random.seed(137)
    np.random.seed(137)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True)
    (args.output/'updates').mkdir()
    started = time.monotonic()
    identity = dict(schema='all_MLP_learning_execution_v1',arm=args.arm,validation_only=args.mode=='short',
        diagnostic_overbudget=args.arm=='dense',planned_updates=2 if args.mode=='short' else 128,
        source_binding=source,source_C0_binding=source['fresh_C0'],training_contract=b.CONTRACT,
        physical_GPUs=list(map(int,args.gpus)),layer_device_map=b.life.LAYER_MAP,seed=137,
        source_sha256=b.code_bindings(),pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),
        argv=[sys.executable,*sys.argv],python=sys.executable,cwd=os.getcwd(),resource_check=resources)
    b.write(args.output/'execution.json',identity)
    observe = lambda event,value:print(json.dumps(dict(event=event,elapsed_seconds=time.monotonic()-started,**value)),flush=True)
    code = 1
    try:
        execute(args,identity,observe,started)
        code = 0
    except BaseException:
        b.write(args.output/'failure.json',dict(**identity,error=traceback.format_exc(),memory=b.life.memory()))
        raise
    finally:
        (args.output/'run.exit').write_text(str(code)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
