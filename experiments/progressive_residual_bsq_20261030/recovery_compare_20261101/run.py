"""One genuine fixed-bank placement arm; no refitting or automatic recovery."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from . import bindings as b, runtime, training, export


def execute(args,identity,observe,started):
    dataset,recovery,data_evidence=b.data.load()
    model,bank,evidence=runtime.build(args.arm)
    fixture_data={key:evidence[key] for key in ('nonprojection_exact','nonprojection_fields','factory','teacher_identity')}
    cpu=b.read(args.cpu_proof)
    b.require(data_evidence['actual_input_target_records_sha256']==cpu['data_evidence']['actual_input_target_records_sha256'],
        'Use exactly the verified original static inputs/targets; never mix M again.')
    b.write(args.output/'factory.json',fixture_data)
    observe('original_bank_and_BF16_factory_verified',fixture_data | dict(memory=b.old_life.memory()))
    handles,mapping=b.old_life.apply_map(model)
    b.install_precision_runtime(model,b.torch.bfloat16)
    active,optimizer,trained,fixture=training.fit(model,bank,evidence,args.arm,args.mode=='short',dataset,recovery,
        args.output,observe,started+args.timeout_seconds)
    b.write(args.output/'training.json',trained)
    if args.mode=='short':
        # Keep autograd enabled: native packed-u8 training and no_grad inference
        # can select different consumers even though the fixture is read-only.
        with b.torch.enable_grad():
            b.require(model.training and all(p.requires_grad for p in active.values()), 'Observe the real live training mode.')
            output=model(input_ids=fixture['input_ids'].to(model.get_input_embeddings().weight.device),
                attention_mask=fixture['attention_mask'].to(model.get_input_embeddings().weight.device),
                use_cache=False,logits_to_keep=1)
            b.require(output.logits.requires_grad and b.torch.isfinite(output.logits).all().item(),
                'Live fixture must traverse the actual finite autograd training consumer.')
            fixture['live_logits']=output.logits.detach().cpu()
            fixture['live_path']=dict(grad_enabled=True,model_training=True,logits_to_keep=1,
                active_projections=252,backward=False,consumer='Native packed-vq trainable_decoder execution plan')
            del output
        b.require(all(p.grad is None for p in active.values()), 'Read-only live fixture must not accumulate gradients.')
    exported=export.save(model,active,optimizer,identity,trained,fixture,args.output)
    report=dict(**identity,status='S4_recovery_training_complete_fresh_pending',exit_code=0,
        training=trained,export=exported,layer_device_map=mapping,
        factory=b.binding(args.output/'factory.json'),data_evidence=b.binding(args.cpu_proof),
        elapsed_seconds=time.monotonic()-started)
    b.write(args.output/'report.json',report)
    observe('BF16_recovery_export_complete',dict(arm=args.arm,checkpoint_id=exported['checkpoint_id'],
        actual_updates=trained['actual_updates'],seconds=report['elapsed_seconds']))
    for handle in handles:handle.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('terminal','sequential'),required=True)
    parser.add_argument('--mode',choices=('short','formal'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-proof',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=43200)
    args=parser.parse_args();args.output=args.output.resolve()
    b.require(not args.output.exists() and args.output.is_relative_to(b.OUTPUT), 'Use a new isolated arm directory.')
    source=b.source();resources=b.resources()
    cpu=b.read(args.cpu_proof)
    b.require(cpu['status']=='verified_S4_recovery_compare_CPU' and cpu['source_sha256']==b.pins(),
        'Require current scope-specific CPU source/data/control evidence.')
    b.torch.set_num_threads(4);random.seed(137);np.random.seed(137);b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32=False;b.torch.backends.cudnn.allow_tf32=False
    b.torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True);(args.output/'updates').mkdir();(args.output/'segments').mkdir()
    started=time.monotonic()
    identity=dict(schema='S4_fixed_bank_placement_execution_v1',arm=args.arm,validation_only=args.mode=='short',
        source_binding=source,source_sha256=b.pins(),CPU_validation=b.binding(args.cpu_proof),
        contract=b.CONTRACT,segments=b.segments(args.mode=='short'),
        planned_updates=4 if args.mode=='short' else 56,physical_GPUs=[6,7],resources=resources,
        **b.parent.process_event(__import__('os').getpid()),started_unix=time.time(),argv=[sys.executable,*sys.argv])
    b.write(args.output/'execution.json',identity)
    observe=lambda event,value:print(json.dumps(dict(event=event,elapsed_seconds=time.monotonic()-started,**value)),flush=True)
    code=1
    try:
        execute(args,identity,observe,started);code=0
    except BaseException:
        b.write(args.output/'failure.json',dict(error=traceback.format_exc(),memory=b.old_life.memory(),
            elapsed_seconds=time.monotonic()-started))
        raise
    finally:(args.output/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':main()
