"""Minimal CPU evidence for the new parent and unchanged mature Adam500/readers."""
import ast
from copy import deepcopy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import sys
import time

from . import bindings as b
from . import evaluate,sequence
from experiments.adaptive_native_model_20261029.training_checkpoint import MixedCatTrainer
from experiments.adaptive_native_model_20261029.handoff_validation import frozen_digest
from transformers.optimization import get_scheduler


def deadline_proof():
    from experiments.adaptive_class_evaluation_20261030 import evaluate as original_eval
    evaluate.setup(10000.)
    control_config=b.CONTROL_ACCEPTANCE.parents[2]/'full8/config.json'
    config=b.read(control_config)
    binding=dict(checkpoint_dir=str(b.OUTPUT/'formal/final_model'),pipeline_report=str(b.OUTPUT/'formal/report.json'),checkpoint_id='CPU_plan_only')
    path=b.RESULT/'deadline_cpu_config.json'
    b.write(path,config)
    try:
        original=original_eval.make_plan(config,path,binding,b.RESULT/'evaluation',['2'],{})
        bounded=evaluate.bound_longbench_plan(deepcopy(original),10000.,now_unix=1000.)
    finally:
        path.unlink()
    changes=[]
    def differences(left,right,where=''):
        if type(left) is not type(right):
            changes.append(where)
        elif isinstance(left,dict):
            b.require(left.keys()==right.keys(),'Deadline wrapper altered plan keys.')
            for key in left:
                differences(left[key],right[key],where+'/'+str(key))
        elif isinstance(left,list):
            b.require(len(left)==len(right),'Deadline wrapper altered plan list lengths.')
            for index,(a,c) in enumerate(zip(left,right,strict=True)):
                differences(a,c,where+'/'+str(index))
        elif left!=right:
            changes.append(where)
    differences(original,bounded)
    unit=original['execution_units'][0]
    time_index=unit['argv'].index('--max-runtime-seconds')+1
    expected=['/execution_units/0/argv/'+str(time_index),'/execution_units/0/native_runtime_seconds','/execution_units/0/outer_timeout_seconds']
    b.require(set(changes)==set(expected) and bounded['execution_units'][0]['native_runtime_seconds']==8760
        and bounded['execution_units'][0]['outer_timeout_seconds']==8940
        and bounded['execution_units'][0]['termination_grace_seconds']==20,
        'Only the three native/outer deadline values may change; preserve original grace/protocol/data/batches.')
    request=sequence.requests()
    b.require(request[0]['gpus']==request[1]['gpus']==['2']
        and request[0]['environment']['CUDA_VISIBLE_DEVICES']=='2' and request[1]['environment']['CUDA_VISIBLE_DEVICES']==''
        and request[1]['argv'][-1]=='2' and request[0]['argv'][-1].endswith('.run')
        and original['expected_samples']==4750 and len(original['expected_tasks'])==21,
        'Only GPU2 and the independent actual final reader/output may be scheduled.')
    return dict(changed_fields=changes,only_three_time_fields_changed=True,termination_grace_seconds=20,
        absolute_deadline_propagated_to_all_consumer_commands=True,actual_original_plan_protocol=original['protocol'],
        actual_original_config=str(control_config),actual_original_config_sha256=b.sha(control_config),
        expected_tasks=original['expected_tasks'],expected_samples=4750,
        all_reader_modules='experiments.cat_down_vae_muon_closure_cmcc_20261103.evaluate',requests=request)


def main():
    b.cpu_environment()
    b.torch.set_num_threads(4)
    b.require(not b.PREPARED.exists() and not b.OUTPUT.exists(),'Do not overwrite this one-shot preparation/output.')
    started=time.monotonic()
    source=b.source()
    cat,hf,training,vae,runtime,stage=b.recipe(b.OUTPUT/'formal')
    reference=b.torch.load(b.REFERENCE_ARGS,map_location='cpu',weights_only=False)
    saved_optimizer=b.torch.load(b.REFERENCE_ARGS.with_name('optimizer.pt'),map_location='cpu',weights_only=True)
    actual_args={name:getattr(reference,name) for name in ('adam_beta1','adam_beta2','adam_epsilon','learning_rate',
        'weight_decay','max_steps','warmup_ratio','max_grad_norm','seed','data_seed','per_device_train_batch_size')}
    b.require(actual_args==dict(adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,learning_rate=1e-4,
        weight_decay=0.,max_steps=500,warmup_ratio=.01,max_grad_norm=1.5,seed=37,data_seed=31,per_device_train_batch_size=4),
        'The actual original saved Trainer arguments differ from the fixed Adam500 reference.')
    model,meta,strict,budget=b.checkpoint.load(b.RAW)
    b.require(meta['checkpoint_id']==b.RAW_ID and not strict.missing_keys and not strict.unexpected_keys,
        'CPU strict load must resolve the complete new V raw UUID.')
    selected,native,compact,lr,inventory=b.recovery.selection(model,stage)
    active={name:value for name,value in selected.peft_model.named_parameters() if value.requires_grad}
    b.require(len(active)==1728 and sum(value.numel() for value in active.values())==5336064
        and len(native)==216 and len(compact)==36 and not selected.lora_parameters and not selected.norm_parameters
        and not selected.lm_head_parameters and not selected.residual_lora_parameters
        and {id(value) for value in active.values()}=={id(value) for value in selected.decoder_parameters.values()}
        and all(value.dtype==b.torch.float32 for value in active.values()),'Require the actual original all252 FP32 D and LoRA0 topology.')
    groups=b.recovery.build_model_level_param_groups(selected,lr_config=lr,model=selected.peft_model)
    cls,kwargs=MixedCatTrainer.get_optimizer_cls_and_kwargs(reference,model)
    b.require(cls is b.torch.optim.AdamW,'The mature actual Trainer must construct original torch AdamW.')
    kwargs={key:value for key,value in kwargs.items() if key not in ('params','model','optimizer_dict')}
    optimizer=cls(groups,**kwargs)
    b.require(not optimizer.state and all(group['lr']==1e-5 and group['weight_decay']==0. for group in optimizer.param_groups),
        'Create a fresh original Adam optimizer; never load VAE or category-Muon state.')
    scheduler=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=5,num_training_steps=500)
    actual_group=b.normalized(optimizer.state_dict()['param_groups'])
    original_group=b.normalized(saved_optimizer['param_groups'])
    b.require(actual_group==original_group and not optimizer.state,
        'Every actual Adam group field/order/default must match original checkpoint500; both scheduled LR0, fresh states empty.')
    b.require(b.normalized(kwargs)==dict(lr=1e-4,betas=[.9,.999],eps=1e-8)
        and all(group['initial_lr']==1e-5 and group['lr']==0 for group in optimizer.param_groups),
        'Original actual constructor kwargs/base-group LR/warmup0 contract changed.')
    frozen=frozen_digest(model,{id(value) for value in active.values()})
    modules={name:str(module.__file__) for name,module in sys.modules.items()
        if name.split('.')[0] in ('train_utils','litebsq','rotation','e2e_common') and getattr(module,'__file__',None)}
    b.require(all(Path(path).resolve().is_relative_to(b.base.SNAPSHOT) for path in modules.values()),
        'The CPU contract escaped the original bound N runtime.')
    forbidden=('experiments.cat_down_recovery_muon_cmcc_20261103.optimizer','experiments.cat_down_vae_muon_cmcc_20261103.optimizer')
    b.require(all(name not in sys.modules for name in forbidden),'The closure must not import either Muon optimizer adapter.')
    deadline=deadline_proof()
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    effective=dict(cls=cls.__module__+'.'+cls.__name__,constructor_kwargs=b.normalized(kwargs),
        defaults=b.normalized(optimizer.defaults),param_groups=actual_group,
        reference_group_fields_JSON_normalized_exact=True,reference_optimizer_sha256=b.sha(b.REFERENCE_ARGS.with_name('optimizer.pt')),
        fresh_state_entries=0,scheduled_initial_lr=0.,decoder_base_lr=1e-5,weight_decay=0.,
        full_group_comparison_ignored_fields=[],standard_scheduler_initial_state=scheduler.state_dict())
    record=dict(schema='VAE_Muon_original_Adam500_prepared_v1',status='verified_CPU_V_parent_and_original_Adam500_contract',
        command=[sys.executable,*sys.argv],pid=__import__('os').getpid(),created_unix=time.time(),contract=b.CONTRACT,
        source_binding=source,source_budget=budget,original_stage=asdict(stage),
        full_stage_comparison=dict(all_fields_JSON_normalized_exact=True,only_ignored_field='output_dir'),
        original_saved_training_args=b.normalized(reference.to_dict()),original_saved_argument_checks=actual_args,
        original_data_config=asdict(stage.config.data),original_loss_config=asdict(stage.config.loss),
        effective_optimizer=effective,actual_inventory=inventory,trainable_leaves=len(active),
        trainable_elements=sum(value.numel() for value in active.values()),frozen_state=frozen,
        original_runtime_identity=runtime['runtime_identity'],deadline_and_readers=deadline,
        implementations=dict(recovery=str(Path(b.recovery.__file__).resolve()),mature_trainer=inspect.getsourcefile(MixedCatTrainer)),
        source_sha256=b.pins(),CUDA_initialized=b.torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-started,
        boundary='Only new parent/layout/full original stage and actual Adam fields/readers verified on CPU. Mature recovery/fresh reused; no GPU short or repeated exact-resume audit.')
    b.require(not record['CUDA_initialized'],'CPU proof initialized CUDA.')
    b.OUTPUT.mkdir(parents=True)
    b.write(b.PREPARED,record)
    print(json.dumps(dict(status=record['status'],trainable_leaves=len(active),trainable_elements=record['trainable_elements'],
        actual_Adam_group_fields_exact=True,only_three_deadline_fields_changed=True,seconds=record['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
