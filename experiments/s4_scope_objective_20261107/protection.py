"""External FP32 masters feeding original-coordinate native INT8 protection rows."""
from types import MethodType
from . import bindings as b


def effective_scale(source,delta):
    value=source.float()*delta.exp();hard=value.bfloat16()
    b.require(b.torch.isfinite(value).all() and (value>0).all()
        and b.torch.isfinite(hard).all() and (hard>0).all(),'Learned scale must remain finite and positive.')
    return hard


def integer_coordinate(master):
    clipped=master.clamp(-128.,127.)
    return clipped+(clipped.round()-clipped).detach()


class Protection:
    def __init__(self,model,modules,mode):
        self.mode=mode;self.entries={};self.parameters={};self.first_hard_step=None
        original_keys=set(model.state_dict());registered={n:id(p) for n,p in model.named_parameters()}
        for name,module in modules.items():
            q,s,index=module.protected_input_qvalues,module.protected_input_scales,module.protected_input_indices
            b.require(module.protected_channel_quant_format=='int8' and q.dtype==b.torch.uint8
                and q.shape[1]==module.out_features and s.dtype==b.torch.bfloat16 and s.shape==index.shape==(q.shape[0],)
                and module.protected_output_indices is None and module.trainable_decode and not module.cache_decoded_weight
                and module._cached_weight is None and '_decode_protected_input_weight_rows' not in module.__dict__,
                'Wrong S4 all252 protection layout/path: '+name)
            p=b.torch.nn.Parameter(b.torch.zeros_like(s,dtype=b.torch.float32)
                if mode=='scale' else q.view(b.torch.int8).float())
            suffix='protected_log_scale_delta' if mode=='scale' else 'protected_code_coordinate'
            self.parameters[name+'.'+suffix]=p
            self.entries[name]=dict(module=module,parameter=p,calls=0,
                previous=(s if mode=='scale' else q).detach().clone())
            with b.torch.no_grad():
                native=module._decode_protected_input_weight_rows(dtype=b.torch.bfloat16,device=q.device)
                b.require(b.torch.equal(native,self.rows(name,dtype=b.torch.bfloat16,device=q.device)),
                    'Initial live protection differs from native: '+name)
            def rows(instance,*,dtype,device,key=name):
                e=self.entries[key];b.require(instance is e['module'],'Protection bound to wrong module.')
                e['calls']+=1
                return self.rows(key,dtype=dtype,device=device)
            module._decode_protected_input_weight_rows=MethodType(rows,module)
        b.require(len(self.entries)==252 and sum(e['module'].protected_input_qvalues.numel() for e in self.entries.values())==117964800
            and sum(e['module'].protected_input_scales.numel() for e in self.entries.values())==23040,
            'Actual protection inventory changed.')
        b.require(set(model.state_dict())==original_keys and {n:id(p) for n,p in model.named_parameters()}==registered,
            'External masters must preserve native registration.')

    def rows(self,name,*,dtype,device):
        e=self.entries[name];m,p=e['module'],e['parameter']
        b.require(p.device==b.torch.device(device),'Protection master device mismatch.')
        q=m.protected_input_qvalues.view(b.torch.int8).float()
        s=effective_scale(m.protected_input_scales,p) if self.mode=='scale' else m.protected_input_scales
        if self.mode=='integer':q=integer_coordinate(p)
        return (q*s.float().unsqueeze(1)).to(dtype).contiguous()

    def payload(self):
        values={}
        for name,e in self.entries.items():
            suffix='scales' if self.mode=='scale' else 'qvalues'
            value=effective_scale(e['module'].protected_input_scales,e['parameter']) if self.mode=='scale' else (
                integer_coordinate(e['parameter']).detach().to(b.torch.int8).view(b.torch.uint8))
            values[name+'.protected_input_'+suffix]=value
        return values

    @b.torch.no_grad()
    def reset_previous(self):
        for (name,e),value in zip(self.entries.items(),self.payload().values()):e['previous'].copy_(value)

    @b.torch.no_grad()
    def statistics(self,step):
        totals=dict(master_changed=0,hard_changed_from_source=0,adjacent_hard_changed=0,master_abs_delta_max=0.,live_calls=0)
        for (name,e),(key,hard) in zip(self.entries.items(),self.payload().items()):
            source=e['module'].protected_input_scales if self.mode=='scale' else e['module'].protected_input_qvalues
            delta=e['parameter'] if self.mode=='scale' else e['parameter']-source.view(b.torch.int8).float()
            totals['master_changed']+=int(b.torch.count_nonzero(delta))
            totals['hard_changed_from_source']+=int(b.torch.count_nonzero(hard!=source))
            totals['adjacent_hard_changed']+=int(b.torch.count_nonzero(hard!=e['previous']))
            totals['master_abs_delta_max']=max(totals['master_abs_delta_max'],float(delta.abs().max()))
            totals['live_calls']+=e['calls'];e['previous'].copy_(hard)
        if totals['hard_changed_from_source'] and self.first_hard_step is None:self.first_hard_step=step
        totals.update(first_hard_change_step=self.first_hard_step,mode=self.mode)
        b.require(all(e['calls']>0 for e in self.entries.values()),'Protection adapter bypassed in real forward.')
        return totals
