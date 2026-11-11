"""Native standard-state AdamW/Muon groups, with four independent fixed kappas."""
from copy import deepcopy
from dataclasses import asdict
import math

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.recovery import _tensor_hash
from .muon_math import direction


def calibration_context(raw_id, category, data_identity, seeds, settings):
    configuration = asdict(settings)
    for name in ('decoder_lr','protection_multiplier','lora_lr','schedule_max'):
        configuration.pop(name)
    return dict(raw_id=raw_id,category=category,data_identity=data_identity,seeds=seeds,settings=configuration)


def owned_groups(selection, settings):
    active = {id(p): name for name, p in selection.model.named_parameters() if p.requires_grad}
    matrices, layout, auxiliary = {}, {}, {}
    for owner, module in selection.decoder_targets:
        decoder = module._parallel_stage_decoder
        count = module.residual_stages
        require(module.parallel_parts == 1 and module._parallel_stage_layout == [(i, 0) for i in range(count)], 'Invalid actual stage ownership.')
        matrix_ids = set()
        for field in ('linear_in', 'linear_out'):
            leaf = getattr(decoder, field)
            p = leaf.weight if isinstance(leaf, torch.nn.Linear) else leaf.conv.weight
            require(id(p) in active, 'Selected decoder matrix is frozen.')
            name = active[id(p)]
            require((p.ndim == 3 and p.shape[-1] == 1 and p.shape[0] % count == 0)
                or (p.ndim == 2 and count == 1), 'Unexpected packed matrix geometry.')
            matrices[name], layout[name] = p, count
            matrix_ids.add(id(p))
        for name, p in decoder.named_parameters():
            if id(p) in active and id(p) not in matrix_ids:
                require('bias' in name or 'norm' in name, 'Unknown decoder auxiliary owner: ' + name)
                auxiliary[active[id(p)]] = p
    lora = {active[id(p)]: p for p in selection.selection.lora_parameters.values()}
    require(all(p.ndim == 2 for p in lora.values()), 'LoRA A/B must be actual independent matrices.')
    protection = selection.protection.parameter_groups()
    groups = []
    for group, params, kind, reference, multiplier in (
        ('decoder', matrices, settings.bound_optimizer, 1e-5, settings.decoder_lr/1e-5),
        ('decoder_aux', auxiliary, 'adamw', 1e-5, settings.decoder_lr/1e-5),
        ('lora', lora, settings.lora_optimizer, 1e-4, settings.lora_lr/1e-4),
        ('q', protection['q'], settings.bound_optimizer, .03, settings.protection_multiplier),
        ('scale', protection['scale'], settings.bound_optimizer, .001, settings.protection_multiplier)):
        if params:
            names = [active[id(p)] for p in params.values()]
            groups.append(dict(params=list(params.values()), names=names, group_name=group, kind=kind,
                stages=[layout.get(name, 1) for name in names], reference_lr=reference, multiplier=multiplier,
                lr=reference*multiplier, weight_decay=0., muon_kappa=None, calibration=None))
    ids = [id(p) for group in groups for p in group['params']]
    require(len(ids) == len(set(ids)) and set(ids) == set(active), 'Every actual trainable must be owned exactly once.')
    return groups


def muon_direction(gradient, momentum, count, group):
    if group == 'scale':
        require(gradient.ndim == 1 and count == 1, 'Scale uses one row per actual projection.')
        update, _ = direction(gradient.reshape(1, -1), momentum.reshape(1, -1), 1)
        return update.reshape_as(gradient)
    return direction(gradient, momentum, count)[0]


class RecoveryOptimizer(torch.optim.Optimizer):
    def __init__(self, groups, fixed_calibrations=None, context=None):
        super().__init__(groups, dict(lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=0.))
        self.fixed_calibrations = deepcopy(fixed_calibrations or {})
        self.calibration_context = deepcopy(context)
        require(set(self.fixed_calibrations) <= {g['group_name'] for g in groups if g['kind']=='muon'}, 'Unused fixed calibration group.')
        self._bind_adam()

    def _bind_adam(self):
        groups = [group for group in self.param_groups if group['kind'] == 'adamw']
        self.adam = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, weight_decay=0.) if groups else None
        if self.adam is not None:
            # Same group objects and standard state mapping for Trainer/Accelerate.
            self.adam.param_groups = groups
            self.adam.state = self.state

    def load_state_dict(self, state_dict):
        require(len(state_dict['param_groups']) == len(self.param_groups), 'Optimizer group count changed.')
        for actual, saved in zip(self.param_groups, state_dict['param_groups'], strict=True):
            for key in ('names', 'group_name', 'kind', 'stages', 'reference_lr', 'multiplier'):
                require(actual[key] == saved[key], 'Optimizer ownership/config changed: ' + key)
        result = super().load_state_dict(state_dict)
        self._bind_adam()
        return result

    @torch.no_grad()
    def calibrate(self, group):
        require(group['muon_kappa'] is None and all(not self.state.get(p) for p in group['params']), 'Only fresh groups may calibrate.')
        identity = dict(reference_lr=group['reference_lr'], names=list(group['names']), stages=list(group['stages']),
            parameter_sha256=_tensor_hash(dict(zip(group['names'], group['params']))),
            clipped_gradient_sha256=_tensor_hash(dict(zip(group['names'], [p.grad for p in group['params']]))))
        fixed = self.fixed_calibrations.get(group['group_name'])
        if fixed is not None:
            require(all(fixed[k]==v for k,v in identity.items() if k!='clipped_gradient_sha256')
                and self.calibration_context is not None and fixed.get('context')==self.calibration_context
                and fixed['kappa']>0 and math.isfinite(fixed['kappa']),
                'A fixed kappa belongs to different initial parameters, data, seeds, or optimizer groups.')
        a, m = [], []
        for p, count in zip(group['params'], group['stages'], strict=True):
            grad = p.grad.detach()
            # Fresh AdamW with WD0: compute its unrounded candidate using the
            # actual first-moment operations, independently of scheduler LR0.
            avg, sq = torch.zeros_like(grad), torch.zeros_like(grad)
            avg.lerp_(grad, .1)
            sq.mul_(.999).addcmul_(grad, grad, value=.001)
            adam = -group['reference_lr'] * (avg/(1-.9)) / (sq.sqrt()/math.sqrt(1-.999)+1e-8)
            muon = -group['reference_lr'] * muon_direction(grad, torch.zeros_like(grad), count, group['group_name'])
            a.append(float(adam.double().square().sum()))
            m.append(float(muon.double().square().sum()))
        norm_a, norm_m = math.sqrt(math.fsum(a)), math.sqrt(math.fsum(m))
        require(norm_a > 0 and norm_m > 0 and math.isfinite(norm_a/norm_m), 'No finite nonzero group calibration: '+group['group_name'])
        if fixed is not None:
            # Same-input real BF16/FA2 repetitions changed the inferred ratio by
            # up to0.060%. Allow0.2% norm drift, below BF16 unit roundoff; retain
            # the original kappa exactly. Parameters/context remain exact.
            errors={k:abs(v/fixed[k]-1) for k,v in dict(AdamW_candidate_norm=norm_a,
                Muon_candidate_norm=norm_m,kappa=norm_a/norm_m).items()}
            require(all(math.isfinite(v) and v<=.002 for v in errors.values()),
                'Fixed-kappa initial update norms changed beyond measured numerical allowance: '+str(errors))
            group['muon_kappa'], group['calibration'] = fixed['kappa'], deepcopy(fixed)
            group['fixed_calibration_check']=dict(actual_clipped_gradient_sha256=identity['clipped_gradient_sha256'],
                exact_gradient=identity['clipped_gradient_sha256']==fixed['clipped_gradient_sha256'],
                relative_norm_errors=errors,tolerance=.002,original_kappa_retained=True,
                actual_AdamW_candidate_norm=norm_a,actual_Muon_candidate_norm=norm_m)
            return
        group['muon_kappa'] = norm_a/norm_m
        group['calibration'] = dict(**identity, context=deepcopy(self.calibration_context), AdamW_candidate_norm=norm_a,
            Muon_candidate_norm=norm_m, kappa=norm_a/norm_m,
            metric='Unrounded FP32 fresh update norm, multiplier1/nonzero reference LR; real warmup retained.')

    @torch.no_grad()
    def step(self, closure=None):
        require(closure is None, 'Recovery does not use optimizer closures.')
        require(all(p.grad is not None and p.dtype == p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all())
            for g in self.param_groups for p in g['params']), 'All actual active gradients must be finite FP32.')
        for group in self.param_groups:
            if group['kind'] != 'muon':
                continue
            if group['muon_kappa'] is None:
                self.calibrate(group)
            for p, count in zip(group['params'], group['stages'], strict=True):
                state = self.state[p]
                if not state:
                    state.update(momentum_buffer=torch.zeros_like(p), step=0)
                update = muon_direction(p.grad, state['momentum_buffer'], count, group['group_name'])
                p.add_(update, alpha=-group['lr']*group['muon_kappa'])
                state['step'] += 1
        if self.adam is not None:
            self.adam.step()

    def effective_contract(self):
        return dict(schema='sc_cat_DPL_optimizer_v1', NS_steps=5, momentum=.95, Nesterov=True,
            shape_scale='.2sqrt(max(rows,cols))', groups=[{k: deepcopy(v) for k, v in g.items() if k != 'params'} for g in self.param_groups])
