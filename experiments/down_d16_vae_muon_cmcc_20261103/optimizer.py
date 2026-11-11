"""Per-linear Muon over BF16 VAE parameters, with original BF16 auxiliary AdamW."""
from contextlib import contextmanager
from copy import deepcopy
import math
import time

from . import bindings as b
from litebsq.parallel_layers import Normalize,ParallelLinear

torch=b.torch
SCHEMA='down_D16_vae_muon_bf16_v1'


def derive_layout(vae):
    named=dict(vae.named_parameters())
    layout,affine={},[]
    for role in ('encoder','decoder'):
        module=getattr(vae.model,role)
        b.require(module.num_models==36 and module.num_res_blocks==0,'Use the real M36/R0 encoder and decoder.')
        for part in ('linear_in','linear_out'):
            linear=getattr(module,part)
            b.require(isinstance(linear,ParallelLinear) and linear.num_models==36
                and linear.conv.groups==36 and linear.conv.kernel_size==(1,), 'Derive matrices from their actual ParallelLinear owner.')
            name=f'model.{role}.{part}.conv.weight'
            parameter=named[name]
            shape=(36,linear.out_features,linear.in_features)
            b.require(tuple(parameter.shape)==(shape[0]*shape[1],shape[2],1), 'Packed owner/shape mismatch: '+name)
            layout[name]=dict(role=role,num_models=36,matrix_shape=list(shape),packed_shape=list(parameter.shape))
            bias=f'model.{role}.{part}.conv.bias'
            b.require(named[bias] is linear.conv.bias,'Route actual linear bias to auxiliary AdamW.')
            affine.append(bias)
        norm=module.norm_out
        b.require(isinstance(norm,Normalize) and norm.norm_type=='layer' and norm.num_models==36,
            'The original packed two-dimensional layer-norm is auxiliary, not a Muon matrix.')
        for leaf in ('weight','bias'):
            name=f'model.{role}.norm_out.{leaf}'
            b.require(named[name] is getattr(norm,leaf) and tuple(named[name].shape)==(36,128),'Unexpected actual norm leaf.')
            affine.append(name)
    b.require(len(layout)==4 and len(affine)==8 and set(layout)|set(affine)==set(named)
        and len({id(p) for p in named.values()})==12
        and sum(p.numel() for p in named.values())==397728
        and all(p.dtype==torch.bfloat16 for p in named.values()),'Every real BF16 VAE parameter must have one explicit owner.')
    return named,layout,affine


def matrices(value,spec):
    b.require(list(value.shape)==spec['packed_shape'],'The packed matrix storage changed.')
    return value[...,0].reshape(spec['matrix_shape'])


def zeropower(value):
    b.require(value.ndim==3,'Keep each actual linear matrix independent.')
    x=value.to(torch.bfloat16)
    transpose=x.shape[-2]>x.shape[-1]
    if transpose:x=x.mT
    x=x/(x.norm(dim=(-2,-1),keepdim=True)+1e-7)
    for _ in range(5):
        a=x@x.mT
        x=3.4445*x+(-4.7750*a+2.0315*(a@a))@x
    return x.mT if transpose else x


def direction(gradient,momentum,spec):
    momentum.lerp_(gradient.float(),.05)
    nesterov=gradient.float().lerp(momentum,.95)
    shape=spec['matrix_shape']
    return (zeropower(matrices(nesterov,spec)).float()*(.2*math.sqrt(max(shape[-2:])))).reshape_as(gradient)


def candidate(parameter,update,lr):
    return (parameter.float()-float(lr)*update).to(torch.bfloat16)


def displacement(parameters,updates,lr):
    squares=[]
    for p,u in zip(parameters,updates,strict=True):
        delta=candidate(p,u,lr).float()-p.float()
        squares.append(delta.double().square().sum())
    return math.sqrt(float(torch.stack(squares).sum()))


def solve_kappa(parameters,updates,reference_norm,base_lr):
    unorm=math.sqrt(float(torch.stack([u.double().square().sum() for u in updates]).sum()))
    b.require(reference_norm>0 and unorm>0 and math.isfinite(reference_norm) and math.isfinite(unorm),
        'The actual BF16 Adam reference and fresh Muon direction must be finite and nonzero.')
    initial=reference_norm/(base_lr*unorm)
    observed={0.:0.}

    def evaluate(kappa):
        if kappa not in observed:
            observed[kappa]=displacement(parameters,updates,base_lr*kappa)
        return observed[kappa]

    evaluate(initial)
    low,high=0.,2*initial
    doublings=0
    while evaluate(high)<reference_norm and doublings<2:
        high*=2
        doublings+=1
    upper_covers=evaluate(high)>=reference_norm
    for _ in range(12):
        middle=(low+high)/2
        if evaluate(middle)<reference_norm:low=middle
        else:high=middle
    best=min(observed,key=lambda k:abs(observed[k]/reference_norm-1))
    return dict(kappa=best,reference_BF16_displacement_norm=reference_norm,
        fresh_Muon_FP32_direction_norm=unorm,kappa0=initial,matched_BF16_displacement_norm=observed[best],
        relative_error=abs(observed[best]/reference_norm-1),maximum_relative_error=.01,
        upper_doublings=doublings,bisections=12,upper_covers_reference=upper_covers,
        observed_amplitudes=[dict(kappa=k,norm=v) for k,v in observed.items()],
        metric='Actual one-cast BF16 matrix parameter displacement; not an FP32 master candidate norm.')


class VAEMuon(torch.optim.Optimizer):
    def __init__(self,vae,args,reference_factory,evidence,*,diagnostic_steps):
        named,self.layout,affine=derive_layout(vae)
        self.named=named
        self.matrix_names=list(self.layout)
        self.reference_factory,self.reference_args=reference_factory,deepcopy(args)
        self.evidence=evidence
        self.diagnostic_steps=set(diagnostic_steps)
        self.actual_step=0
        super().__init__([dict(params=[named[n] for n in self.matrix_names],group_name='muon_matrix',kappa=None),
            dict(params=[named[n] for n in affine],group_name='aux_adam')],dict(lr=float(args.lr),weight_decay=0.))
        self._adam=torch.optim.AdamW([self.param_groups[1]],lr=float(args.lr),betas=(args.beta1,args.beta2),eps=1e-8,weight_decay=args.weight_decay)
        self._adam.state=self.state

    @torch.no_grad()
    def calibrate(self):
        group=self.param_groups[0]
        b.require(group['kappa'] is None and not self.state,'Calibrate only a genuinely fresh stage optimizer.')
        before=b.c.digest(self.named)
        gradients=b.c.digest({name:p.grad for name,p in self.named.items()})
        cpu_rng=torch.get_rng_state()
        cuda_rng=torch.cuda.get_rng_state() if self.param_groups[0]['params'][0].is_cuda else None
        parameters=group['params']
        clones=[p.detach().clone().requires_grad_() for p in parameters]
        for p,q in zip(parameters,clones,strict=True):q.grad=p.grad.detach().clone()
        reference=self.reference_factory(clones,self.reference_args,self.reference_args.lr)
        b.require(isinstance(reference,torch.optim.AdamW) and reference.param_groups[0]['betas']==(.9,.95)
            and reference.param_groups[0]['eps']==1e-8 and reference.param_groups[0]['weight_decay']==0,
            'Use the actual original native BF16 AdamW as the first-step amplitude reference.')
        reference.step()
        b.require(all(p.dtype==q.dtype==torch.bfloat16 and reference.state[q]['exp_avg'].dtype==torch.bfloat16
            and reference.state[q]['exp_avg_sq'].dtype==torch.bfloat16 for p,q in zip(parameters,clones,strict=True)),
            'Do not substitute FP32 Adam moments/parameters for the original BF16 reference.')
        adam_norm=math.sqrt(float(torch.stack([(q.float()-p.float()).double().square().sum()
            for p,q in zip(parameters,clones,strict=True)]).sum()))
        fresh=[direction(self.named[name].grad,torch.zeros_like(self.named[name],dtype=torch.float32),self.layout[name])
            for name in self.matrix_names]
        b.require(torch.stack([torch.isfinite(u).all() for u in fresh]).all().item(),'Fresh Muon direction is nonfinite.')
        result=solve_kappa(parameters,fresh,adam_norm,float(self.reference_args.lr))
        result.update(calibration_step=1,reference_optimizer='Original native BF16 torch.optim.AdamW',
            reference_moment_dtype='torch.bfloat16',actual_parameters_gradients_RNG_and_optimizer_state_unchanged=True)
        b.require(b.c.digest(self.named)==before and b.c.digest({name:p.grad for name,p in self.named.items()})==gradients
            and torch.equal(cpu_rng,torch.get_rng_state()) and (cuda_rng is None or torch.equal(cuda_rng,torch.cuda.get_rng_state()))
            and not self.state,'Amplitude calibration changed actual parameters, gradients, RNG or optimizer state.')
        self.evidence['calibration']=result
        print(dict(event='VAE_Muon_stage_calibration',stage=self.evidence.get('stage_index','CPU_fixture'),**result),flush=True)
        b.require(result['relative_error']<=.01,'Nearest bounded BF16 amplitude match exceeds1%; stop for Astra review, without changing precision or searching LR.')
        group['kappa']=result['kappa']

    @torch.no_grad()
    def step(self,closure=None):
        b.require(closure is None,'The original native VAE loop supplies one real backward, without optimizer closures.')
        b.require(all(p.grad is not None and p.dtype==p.grad.dtype==torch.bfloat16 for p in self.named.values()),
            'Keep all actual VAE parameters and gradients BF16, without persistent FP32 parameter masters.')
        b.require(torch.stack([torch.isfinite(p.grad).all() for p in self.named.values()]).all().item(),'Nonfinite actual VAE gradient.')
        if self.param_groups[0]['kappa'] is None:self.calibrate()
        self.actual_step+=1
        audit=self.actual_step in self.diagnostic_steps
        before={n:p.detach().clone() for n,p in self.named.items()} if audit else None
        group=self.param_groups[0]
        updates=[]
        for name,p in zip(self.matrix_names,group['params'],strict=True):
            state=self.state[p]
            if not state:state.update(momentum_buffer=torch.zeros_like(p,dtype=torch.float32),step=0)
            update=direction(p.grad,state['momentum_buffer'],self.layout[name])
            updates.append(update)
            state['step']+=1
        b.require(torch.stack([torch.isfinite(u).all() for u in updates]).all().item(),'Nonfinite actual Muon update.')
        for p,u in zip(group['params'],updates,strict=True):p.copy_(candidate(p,u,group['lr']*group['kappa']))
        self._adam.step()
        if audit:
            roles={}
            for role in ('encoder','decoder'):
                names=[n for n in self.matrix_names if self.layout[n]['role']==role]
                delta=[(self.named[n].float()-before[n].float()) for n in names]
                changed=sum(int(torch.count_nonzero(x)) for x in delta)
                total=sum(x.numel() for x in delta)
                roles[role]=dict(BF16_displacement_norm=math.sqrt(float(torch.stack([x.double().square().sum() for x in delta]).sum())),
                    changed_elements=changed,total_matrix_elements=total,nonzero_fraction=changed/total)
            if self.actual_step<=2:
                b.require(all(row['changed_elements']>0 for row in roles.values()),'Both first real encoder and decoder matrix updates must be nonzero.')
            b.require(all(self.state[p]['momentum_buffer'].dtype==torch.float32 for p in group['params'])
                and all(self.state[p]['exp_avg'].dtype==self.state[p]['exp_avg_sq'].dtype==torch.bfloat16
                    for p in self.param_groups[1]['params']),'Muon momentum must remain FP32 and original auxiliary Adam moments BF16.')
            b.require(torch.stack([torch.isfinite(p).all() for p in self.named.values()]).all().item(),'Nonfinite updated BF16 parameters.')
            row=dict(step=self.actual_step,roles=roles,
                scheduled_base_lr=group['lr'],effective_matrix_lr=group['lr']*group['kappa'],aux_lr=self.param_groups[1]['lr'],
                parameters_and_gradients_BF16=True,persistent_FP32_parameter_master=False,
                Muon_momentum_FP32=True,aux_Adam_moments_BF16=True)
            self.evidence.setdefault('updates',[]).append(row)
            print(dict(event='VAE_Muon_real_update',stage=self.evidence.get('stage_index','CPU_fixture'),**row),flush=True)

    def state_dict(self):
        result=super().state_dict()
        result.update(schema=SCHEMA,layout=deepcopy(self.layout),actual_step=self.actual_step)
        return result

    def load_state_dict(self,state_dict):
        b.require(state_dict.get('schema')==SCHEMA and state_dict.get('layout')==self.layout,
            'Only this adapter schema and exact current matrix layout may restore optimizer state.')
        saved=state_dict['param_groups'][0]['params']
        # The generic Optimizer loader casts ordinary state to p.dtype. Preserve
        # original FP32 values before that cast, rather than upcasting rounded values.
        momentum={p:state_dict['state'][key]['momentum_buffer'].to(device=p.device,dtype=torch.float32).clone()
            for p,key in zip(self.param_groups[0]['params'],saved,strict=True)
            if 'momentum_buffer' in state_dict['state'].get(key,{})}
        b.require(all(state_dict['state'][key]['momentum_buffer'].dtype==torch.float32
            for key in saved if 'momentum_buffer' in state_dict['state'].get(key,{})), 'Saved Muon momentum must already be FP32.')
        super().load_state_dict(state_dict)
        for p,value in momentum.items():self.state[p]['momentum_buffer']=value
        self.actual_step=int(state_dict['actual_step'])
        self._adam.param_groups=[self.param_groups[1]]
        self._adam.state=self.state


@contextmanager
def install(native,records,*,short):
    import litebsq.llm_vae as model_api
    constructor,factory=model_api.MultiLayerVAE,native.create_optimizer
    pending=[]

    def construct(args):
        b.require(not pending,'Each real stage VAE must be consumed by its optimizer factory exactly once.')
        model=constructor(args)
        pending.append(model)
        return model

    def create(params,args,lr):
        b.require(len(pending)==1 and len(records)<2 and lr==args.lr==.003,'Only two original stage factories are replaced.')
        model=pending.pop()
        parameters=list(params)
        b.require([id(p) for p in parameters]==[id(p) for p in model.parameters()],'Factory parameters differ from the captured actual VAE.')
        _named,layout,_affine=derive_layout(model)
        flat={f'group0.parameter{i}':p.detach() for i,p in enumerate(parameters)}
        row=dict(stage_index=len(records),canonical_original_flat_order_initial_digest=b.c.digest(flat),
            canonical_parameter_shapes={k:list(v.shape) for k,v in flat.items()},parameter_elements=sum(p.numel() for p in parameters),
            parameter_dtype='torch.bfloat16',matrix_layout=layout,matrix_count=144,auxiliary_parameter_leaves=8,
            stage_started_unix=time.time(),updates=[])
        records.append(row)
        return VAEMuon(model,args,factory,row,diagnostic_steps=(1,2) if short else (1,2,10000))

    model_api.MultiLayerVAE,native.create_optimizer=construct,create
    try:
        yield
    finally:
        model_api.MultiLayerVAE,native.create_optimizer=constructor,factory
