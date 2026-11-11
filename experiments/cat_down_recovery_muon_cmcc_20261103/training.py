"""Narrow process-only optimizer override; original canonical class/data/loss/save."""
from copy import deepcopy
import json

from . import bindings as b
from .optimizer import CategoryMuon
from experiments.adaptive_native_model_20261029 import training_checkpoint as original
from experiments.adaptive_class_20261030 import continuation as flow
from train_utils.model_level_optimizer import build_model_level_param_groups


def unwrap(optimizer):
    while not isinstance(optimizer,CategoryMuon) and hasattr(optimizer,'optimizer'):
        optimizer=optimizer.optimizer
    b.require(isinstance(optimizer,CategoryMuon),'Require the actual category Muon adapter, including under Accelerate.')
    return optimizer


class MuonMixedCatTrainer(original.MixedCatTrainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        selection=self.model_level_trainable_selection
        model=getattr(self,'model_wrapped',None) or self.model
        groups=build_model_level_param_groups(selection,lr_config=self.model_level_optimizer_lr_config,model=model)
        b.require(len(groups)==1 and groups[0]['group_name']=='decoder' and groups[0]['lr']==1e-5
            and groups[0]['weight_decay']==0.,'Only original all252 decoder inventory may be optimized.')
        active={name:p for name,p in model.named_parameters() if p.requires_grad}
        b.require(len(active)==1728 and {id(p) for p in active.values()}=={id(p) for p in selection.decoder_parameters.values()},
            'Derive canonical actual decoder ownership, not name/dimension heuristics.')
        layout=b.mature.derive_layout(model,self.mixed_round_meta)
        cls,kwargs=self.get_optimizer_cls_and_kwargs(self.args,model)
        b.require(cls is b.torch.optim.AdamW and self.args.adam_beta1==.9 and self.args.adam_beta2==.999
            and self.args.adam_epsilon==1e-8,'Original Trainer reference defaults differ.')
        kwargs=dict(kwargs)
        for key in ('params','model','optimizer_dict'):
            kwargs.pop(key,None)
        self.optimizer=CategoryMuon(active,layout,groups[0]['lr'],kwargs)
        return self.optimizer

    def save_model(self,output_dir=None,_internal_call=False):
        result=super().save_model(output_dir,_internal_call)
        from pathlib import Path
        path=Path(output_dir or self.args.output_dir)/'mixed_training_meta.json'
        manifest=b.read(path)
        optimizer=unwrap(self.optimizer)
        manifest.update(effective_optimizer=optimizer.effective_contract(),
            trainer_config_optim_reference=str(self.args.optim),
            effective_optimizer_override=f'{type(self).__module__}.{type(self).__qualname__}.create_optimizer',
            original_stage_optimizer_is_reference_only=True)
        b.write(path,manifest)
        return result


class ObserveMuon(flow.ObserveRecovery):
    def on_train_begin(self,args,state,control,**kwargs):
        result=super().on_train_begin(args,state,control,**kwargs)
        optimizer=unwrap(kwargs['optimizer'])
        b.require(optimizer.param_groups[0]['lr']==optimizer.param_groups[1]['lr']==0.,'Original cosine warmup must start at LR0.')
        scheduler=self.trainer.lr_scheduler
        if hasattr(scheduler,'scheduler'):
            scheduler=scheduler.scheduler
        b.require(all(value==1e-5 for value in scheduler.base_lrs),'HF scheduler base_lrs must remain original1e-5.')
        self.evidence['effective_optimizer_override']=f'{MuonMixedCatTrainer.__module__}.{MuonMixedCatTrainer.__qualname__}.create_optimizer'
        self.evidence['original_stage_optimizer_is_reference_only']=True
        self.evidence['optimizer_updates']=[]
        return result

    def on_step_end(self,args,state,control,optimizer=None,**kwargs):
        result=super().on_step_end(args,state,control,optimizer=optimizer,**kwargs)
        actual=unwrap(optimizer)
        if state.global_step<=2 or state.global_step in (250,500):
            row=deepcopy(actual.last_step)
            row['scheduler_next_lrs']=[group['lr'] for group in actual.param_groups]
            row['changed_master_elements']=self.evidence['updates'][-1]['changed_master_elements']
            row['BF16_decoder_changed_elements']=sum(int(b.torch.count_nonzero(
                p.detach().cpu().bfloat16()!=self.initial[name].bfloat16())) for name,p in self.parameters.items())
            if state.global_step==1:
                b.require(row['external_scheduled_lr']==row['matrix_effective_lr']==row['auxiliary_actual_lr']==0.
                    and row['changed_master_elements']==0,'Warmup step1 must advance states without moving real masters.')
            elif state.global_step==2:
                b.require(row['external_scheduled_lr']>0 and row['matrix_effective_lr']>0
                    and row['changed_master_elements']>0 and row['BF16_decoder_changed_elements']>0,
                    'Step2 must perform a genuine scheduled update surviving BF16 deployment.')
            self.evidence['optimizer_updates'].append(row)
            print(json.dumps(dict(event='canonical_category_Muon_update',**row)),flush=True)
        return result

    def on_train_end(self,args,state,control,optimizer=None,**kwargs):
        self.evidence['effective_optimizer']=unwrap(optimizer).effective_contract()
        return super().on_train_end(args,state,control,optimizer=optimizer,**kwargs)
