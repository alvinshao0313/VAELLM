"""One fresh all-MLP arm, exactly two full static-TF updates on physical GPU6/7."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from . import data,export,fit,lifecycle as life
from .free_dense import replace_all_mlp,verify_trainable_contract

torch = b.torch


def code_bindings():
    paths = set(b.HERE.glob('*.py'))
    paths.update(Path(module.__file__).resolve() for module in (data.legacy,data.original,
        data.legacy.core,data.legacy.mmlu,data.legacy.base))
    paths.add(b.ROOT/'compressed_e2e_fintuning/device_map.py')
    return {**b.runtime_bindings(),**{str(path.resolve()):b.sha(path) for path in sorted(paths)}}


def check_resources():
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '6,7'
        and Path(sys.executable).resolve() == Path('/home/cmcc/miniconda3/envs/bitvae/bin/python').resolve(),
        'Use caller-activated bitvae and exactly assigned physical GPU6/7.')
    result = subprocess.run(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                            capture_output=True,text=True,check=True)
    occupants = [line.strip() for line in result.stdout.splitlines() if line.strip() and line.strip()!=str(os.getpid())]
    b.require(not occupants, 'Assigned GPU6/7 have existing compute processes; do not interfere or retry: '+str(occupants))
    b.require(torch.cuda.device_count() == 2, 'The real layer-MP short requires exactly two visible cards.')


@torch.no_grad()
def probe(model,fixture):
    previous = model.training
    model.eval()
    device = model.get_input_embeddings().weight.device
    output = model(input_ids=fixture['input_ids'].to(device),attention_mask=fixture['attention_mask'].to(device),
                   use_cache=False,logits_to_keep=1)
    logits = output.logits.detach().cpu()
    model.train(previous)
    b.require(output.past_key_values is None and logits.dtype == torch.bfloat16
              and logits.shape == (1,1,model.config.vocab_size) and torch.isfinite(logits).all().item(), 'Invalid actual BF16 full-model probe.')
    return logits


def comparison(reference,candidate):
    difference = candidate.float()-reference.float()
    norm = float(reference.double().norm())
    return dict(exact=torch.equal(reference,candidate),relative_L2=float(difference.double().norm())/norm if norm else None,
        max_abs=float(difference.abs().max()),changed_top1_positions=int((reference.argmax(-1)!=candidate.argmax(-1)).sum()),
        finite=bool(torch.isfinite(candidate).all()))


def run(args,identity,observe,started):
    dataset,recovery,evidence = data.load()
    model,source_meta,source_budget = b.load_source(identity['source_binding'])
    inventory = b.mlp_inventory(model)
    projection_names = [name for group in b.checkpoint.inventories(model).values() for name in group]
    teacher = data.original.teacher_identity(model.state_dict(),projection_names)
    source_non_mlp = life.non_mlp_proof(model)
    first_case = recovery[dataset['groups'][2]['case_ids'][0]]['case']
    fixture = dict(schema='all_MLP_BF16_same_input_v1',input_ids=first_case['input_ids'].unsqueeze(0),
        attention_mask=torch.ones_like(first_case['input_ids']).unsqueeze(0),
        source_case_id=first_case['case_id'],input_signature=first_case['input_signature'])
    handles,device_map = life.apply_map(model)
    b.install_precision_runtime(model,torch.bfloat16)
    prewarm = b.prewarm_deployment(model)
    references,protection_records = life.capture_source(model)
    source_logits = probe(model,fixture)
    first_m_cases = [recovery[key]['case'] for key in dataset['groups'][2]['case_ids'][:32]]
    with torch.no_grad(): source_m32 = data.mmlu_log_distribution(model,first_m_cases).cpu()
    observe('complete_compressed_source_deployed',dict(source_checkpoint_id=source_meta['checkpoint_id'],
        prewarm=prewarm,inventory_count=len(inventory),device_map=device_map))
    if args.arm == 'dense':
        _active,_inventory = replace_all_mlp(model,identity['source_binding'])
        del _active,_inventory
        b.require(life.non_mlp_proof(model) == source_non_mlp, 'Replacing all108 MLP changed frozen attention or the non-MLP backbone.')
    active = life.training_mode(model,args.arm)
    zero_logits = probe(model,fixture)
    model.eval()
    with torch.no_grad(): zero_m32 = data.mmlu_log_distribution(model,first_m_cases).cpu()
    model.train()
    initial_comparison = dict(full_vocabulary_probe=comparison(source_logits,zero_logits),
        M32_four_way_log_distribution=comparison(source_m32,zero_m32),
        interpretation='D live FP32-master decode may differ from deployed BF16/group8; dense must preserve the source function exactly.')
    if args.arm == 'dense':
        b.require(initial_comparison['full_vocabulary_probe']['exact']
            and initial_comparison['M32_four_way_log_distribution']['exact'], 'Free-dense conversion changed the actual full-model deployed initial function.')
        verify_trainable_contract(model)
    init_memory = life.memory()
    observe('all108_training_initialization_verified',dict(arm=args.arm,initial_comparison=initial_comparison,
        active_leaves=len(active),active_elements=sum(p.numel() for p in active.values()),initialization_memory=init_memory))
    training,optimizer = fit.fit(model,args.arm,active,dataset,recovery,references,observe,started+args.timeout_seconds)
    fixture['live_logits'] = probe(model,fixture)
    frozen_end = life.frozen_proof(model,set(active))
    b.require(frozen_end == training['frozen_before'] and life.non_mlp_proof(model) == source_non_mlp,
              'Final live frozen state differs from the actual compressed source.')
    deployment,export_audit = export.build(model,args.arm,identity['source_binding'],active,source_non_mlp,protection_records)
    for p in active.values(): p.grad = None
    del optimizer,active,model,handles,p
    gc.collect()
    for device in range(2):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    deployment_handles,deployment_map = life.apply_map(deployment)
    deployment_prewarm = b.prewarm_deployment(deployment)
    b.require(life.non_mlp_proof(deployment) == source_non_mlp, 'Independent deployment changed source attention/backbone fields.')
    motion = life.weight_motion(deployment,references)
    fixture['logits'] = probe(deployment,fixture)
    fixture['source_logits'] = source_logits
    fixture_path = args.output/'same_input_output.pt'
    torch.save(fixture,fixture_path)
    final_directory = args.output/'final_model'
    meta,budget,expected = export.save(deployment,final_directory,identity,source_non_mlp,protection_records)
    b.write(args.output/'expected_state.json',expected)
    report = dict(**identity,status='training_and_BF16_export_complete_fresh_pending',exit_code=0,
        data=evidence,teacher_identity=teacher,source_inventory=inventory,source_bit_budget=source_budget,
        source_non_mlp=source_non_mlp,protection_records=protection_records,initial_prewarm=prewarm,
        initial_comparison=initial_comparison,initialization_memory_by_logical_device=init_memory,
        training=training,export_audit=export_audit,deployment_device_map=deployment_map,
        deployment_prewarm=deployment_prewarm,actual_BF16_W_motion=motion,
        live_to_deployed_probe=comparison(fixture['live_logits'],fixture['logits']),
        export=dict(directory=str(final_directory),checkpoint_id=meta['checkpoint_id'],bit_budget=budget,
            expected_state=b.file_binding(args.output/'expected_state.json'),fixture=b.file_binding(fixture_path)),
        elapsed_seconds=time.monotonic()-started,
        scope='Complete36-layer/108-MLP mechanism short only; no quality evaluation, formal budget decision or exact-resume claim.')
    b.write(args.output/'report.json',report)
    del references,deployment,deployment_handles,recovery
    gc.collect()
    observe('all_MLP_short2_training_export_complete',dict(report=str(args.output/'report.json'),fresh_pending=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('decoder','dense'),required=True)
    parser.add_argument('--checkpoint',type=Path,default=b.SOURCE_B)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=14400)
    args = parser.parse_args()
    check_resources()
    args.output = args.output.resolve()
    b.require(not args.output.exists() and args.timeout_seconds>0, 'Choose a new isolated output and a positive bounded runtime.')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(137)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    binding = b.source(args.checkpoint,prototype_only=args.checkpoint.resolve()==b.SOURCE_B.resolve())
    identity = dict(schema='all_MLP_short2_v1',arm=args.arm,prototype_only=binding['prototype_only'],validation_only=True,
        diagnostic_overbudget=args.arm=='dense',source_binding=binding,physical_GPUs=[6,7],layer_device_map=life.LAYER_MAP,
        group_indices=[2,10],seed=137,pid=os.getpid(),argv=sys.argv,python=sys.executable,cwd=os.getcwd(),
        source_sha256=code_bindings())
    b.write(args.output/'execution.json',identity)
    observe = lambda event,values: print(json.dumps(dict(event=event,elapsed_seconds=time.monotonic()-started,**values)),flush=True)
    try:
        run(args,identity,observe,started)
    except BaseException as error:
        b.write(args.output/'failure.json',dict(**identity,error=str(error),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started,memory_by_logical_device=life.memory() if torch.cuda.is_initialized() else {}))
        (args.output/'run.exit').write_text('1\n',encoding='utf-8')
        raise
    (args.output/'run.exit').write_text('0\n',encoding='utf-8')


if __name__ == '__main__':
    main()
