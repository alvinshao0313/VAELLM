"""Standard HF optimizer state; mature stage-Muon plus unchanged original AdamW."""
from copy import deepcopy
import math

from . import bindings as b

torch=b.torch


def gradient_identity(active):
    return {name:dict(parameter=id(p),storage=p.data_ptr(),version=p._version,value=b.c.digest({name:p}),
        gradient=id(p.grad),gradient_storage=p.grad.data_ptr(),gradient_version=p.grad._version,
        gradient_value=b.c.digest({name:p.grad})) for name,p in active.items()}


class CategoryMuon(torch.optim.Optimizer):
    def __init__(self,active,layout,base_lr,adam_kwargs):
        matrices,auxiliary=b.mature.split(active,layout)
        b.require(base_lr==1e-5 and adam_kwargs['betas']==(.9,.999) and adam_kwargs['eps']==1e-8,
            'Reference and auxiliary must use the actual original class AdamW defaults.')
        self.active,self.layout=active,layout
        self.matrix_names=set(matrices)
        self.names={id(p):name for name,p in active.items()}
        self.reference_kwargs={**deepcopy(adam_kwargs),'lr':base_lr,'weight_decay':0.}
        groups=[dict(params=list(matrices.values()),group_name='muon_matrix',lr=base_lr,weight_decay=0.,muon_kappa=None),
            dict(params=list(auxiliary.values()),group_name='auxiliary_original_AdamW',lr=base_lr,weight_decay=0.)]
        super().__init__(groups,dict(lr=base_lr,betas=adam_kwargs['betas'],eps=adam_kwargs['eps'],weight_decay=0.))
        # The child uses the same group object and shared standard state mapping.
        # HF/Accelerate save only this parent's standard state/param_groups format.
        self.adam=torch.optim.AdamW([self.param_groups[1]],**adam_kwargs)
        self.adam.state=self.state
        self.calibration=None
        self.last_step=None

    @property
    def kappa(self):
        return self.param_groups[0]['muon_kappa']

    def load_state_dict(self,state_dict):
        result=super().load_state_dict(state_dict)
        self.adam.param_groups=[self.param_groups[1]]
        self.adam.state=self.state
        return result

    @torch.no_grad()
    def calibrate(self):
        b.require(self.kappa is None and not self.state,'Calibrate only from genuinely fresh real state.')
        before=gradient_identity(self.active)
        rng_before=b.c.digest(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
        adam_squares,muon_squares=[],[]
        base_lr=self.defaults['lr']
        for p in self.param_groups[0]['params']:
            name=self.names[id(p)]
            clone=p.detach().clone().requires_grad_()
            clone.grad=p.grad.detach().clone()
            reference=torch.optim.AdamW([clone],**self.reference_kwargs)
            reference.step()
            state=reference.state[clone]
            beta1,beta2=self.reference_kwargs['betas']
            candidate=-base_lr*(state['exp_avg']/(1-beta1))/(state['exp_avg_sq'].sqrt()/math.sqrt(1-beta2)+self.reference_kwargs['eps'])
            muon,_seconds=b.mature.direction(p.grad,torch.zeros_like(p),self.layout[name]['stages'])
            adam_squares.append(float(candidate.double().square().sum()))
            muon_squares.append(float((-base_lr*muon).double().square().sum()))
        norm_a,norm_m=math.sqrt(math.fsum(adam_squares)),math.sqrt(math.fsum(muon_squares))
        b.require(norm_a>0 and norm_m>0,'Calibration requires positive reference and Muon candidate norms.')
        kappa=norm_a/norm_m
        b.require(math.isfinite(kappa) and gradient_identity(self.active)==before
            and not self.state and rng_before==b.c.digest(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())),
            'Clone-only calibration changed a real parameter/gradient/state/RNG or produced an invalid norm.')
        self.param_groups[0]['muon_kappa']=kappa
        self.calibration=dict(kappa=kappa,reference_base_lr=base_lr,reference_betas=[beta1,beta2],
            reference_eps=self.reference_kwargs['eps'],reference_weight_decay=0.,AdamW_FP32_candidate_norm=norm_a,
            base_Muon_FP32_candidate_norm=norm_m,matched_Muon_FP32_candidate_norm=norm_m*kappa,
            actual_parameters_gradients_state_and_RNG_unchanged=True,
            real_clipped_gradient_sha256=b.c.digest({name:p.grad for name,p in self.active.items()}),
            external_scheduler_base_lrs_unchanged=True,
            metric='Total unrounded FP32 matrix-candidate Frobenius norm at nonzero baseLR; actual step1 retains warmup LR0.')

    @torch.no_grad()
    def step(self,closure=None):
        b.require(closure is None,'This fixed canonical category path does not use an optimizer closure.')
        b.require(all(p.grad is not None and p.dtype==p.grad.dtype==torch.float32
            and torch.isfinite(p.grad).all().item() for p in self.active.values()),'Every real D gradient must be finite FP32.')
        if self.kappa is None:
            self.calibrate()
        group=self.param_groups[0]
        effective_lr=group['lr']*self.kappa
        for p in group['params']:
            name=self.names[id(p)]
            state=self.state[p]
            if not state:
                state.update(momentum_buffer=torch.zeros_like(p),step=0)
            update,_seconds=b.mature.direction(p.grad,state['momentum_buffer'],self.layout[name]['stages'])
            if effective_lr!=0:
                p.add_(update,alpha=-effective_lr)
            state['step']+=1
        self.adam.step()
        step=int(self.state[group['params'][0]]['step'])
        summary=self.state_summary(step)
        self.last_step=dict(step=step,external_scheduled_lr=group['lr'],matrix_effective_lr=effective_lr,
            auxiliary_actual_lr=self.param_groups[1]['lr'],kappa=self.kappa,state=summary,
            calibration=self.calibration if step==1 else None)

    def state_summary(self,step):
        for name,p in self.active.items():
            state=self.state[p]
            keys=('momentum_buffer',) if name in self.matrix_names else ('exp_avg','exp_avg_sq')
            b.require(int(state['step'])==step and all(state[key].dtype==torch.float32
                and torch.isfinite(state[key]).all().item() for key in keys),'Actual standard Muon/AdamW state invalid: '+name)
        return dict(Muon_leaves=len(self.matrix_names),AdamW_leaves=len(self.active)-len(self.matrix_names),
            independent_matrices=sum(row['stages'] for row in self.layout.values()),step=step,FP32_finite=True)

    def effective_contract(self):
        return dict(schema='canonical_category_Muon_standard_optimizer_v1',effective_class=f'{type(self).__module__}.{type(self).__qualname__}',
            Muon=dict(NS_steps=5,momentum=.95,Nesterov=True,shape_scale='.2sqrt(max(rows,cols))',kappa=self.kappa),
            reference_base_lr=self.defaults['lr'],auxiliary_class='torch.optim.AdamW',
            auxiliary_defaults={key:value for key,value in self.adam.defaults.items()},
            auxiliary_group_reference_lr=self.param_groups[1].get('initial_lr',self.defaults['lr']),
            auxiliary_group_weight_decay=self.param_groups[1]['weight_decay'],
            shared_standard_state=True,external_schedule_unchanged=True,
            matrix_effective_lr='scheduled_group_lr*fixed_muon_kappa',calibration=self.calibration)
