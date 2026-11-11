"""Mature stage-local Muon math, with layout derived from actual mixed-model metadata."""
import math
import time

from . import bindings as b

torch = b.torch
MATRIX_SUFFIXES = ('linear_in.conv.weight', 'linear_out.conv.weight')


def derive_layout(model, meta):
    """Use explicit decoder ownership and metadata, including single-stage compact leaves."""
    layout = {}
    native_specs = {item['name']: item for item in meta['native_modules']}
    compact_specs = {item['name']: item['spec'] for item in meta['compact_modules']}
    for name, module in b.selected_modules(model).items():
        if isinstance(module, b.CompactRVQLinear):
            spec = compact_specs[name]
            b.require(len(module.decoders) == len(spec['stages']), 'Compact metadata stage count changed.')
            decoders = [(f'decoders.{i}.', decoder, [item['decoder_spec']])
                for i, (decoder, item) in enumerate(zip(module.decoders, spec['stages'], strict=True))]
        else:
            spec = native_specs[name]
            count = spec['residual_stages']
            b.require(spec['parallel_parts'] == 1 and len(spec['stage_decoders']) == count
                and module._parallel_stage_layout == [(i, 0) for i in range(count)], 'Native actual stage layout changed.')
            decoders = [('_parallel_stage_decoder.', module._parallel_stage_decoder, spec['stage_decoders'])]
        for prefix, decoder, specs in decoders:
            count = len(specs)
            fields = ('in_dim', 'out_dim', 'hidden_dim', 'num_res_blocks', 'decoder_type')
            b.require(all(all(item[key] == specs[0][key] for key in fields) for item in specs)
                and specs[0]['num_res_blocks'] == 0 and specs[0]['decoder_type'] in ('symmetric', 'asymmetric'),
                'Use actual homogeneous R0 nonlinear decoder leaves; do not infer from ndim.')
            d = specs[0]
            expected = ((count, d['hidden_dim'], d['in_dim']), (count, d['out_dim'], d['hidden_dim']))
            parameters = dict(decoder.named_parameters())
            suffixes = tuple(part + ('.weight' if isinstance(getattr(decoder,part), torch.nn.Linear) else '.conv.weight')
                for part in ('linear_in','linear_out'))
            for suffix, shape in zip(suffixes, expected, strict=True):
                parameter = parameters[suffix]
                key = name + '.' + prefix + suffix
                expected_shape = (shape[1],shape[2]) if suffix.endswith('.weight') and '.conv.' not in suffix else (shape[0]*shape[1],shape[2],1)
                b.require((parameter.ndim != 2 or count == 1) and tuple(parameter.shape) == expected_shape,
                    'Metadata/live matrix mismatch: '+key)
                layout[key] = dict(stages=count, matrix_shape=list(shape), packed_shape=list(parameter.shape),
                    elements=parameter.numel(), stage_layout=[[i, 0] for i in range(count)],
                    consumer='compact' if isinstance(module, b.CompactRVQLinear) else 'native', decoder_type=d['decoder_type'])
    return layout


def stages(value, count):
    if value.ndim == 2:
        b.require(count == 1, 'An actual independent nn.Linear leaf is one matrix.')
        return value.unsqueeze(0)
    b.require(value.ndim == 3 and value.shape[-1] == 1 and count > 0 and value.shape[0] % count == 0,
        'Muon matrices must have the explicit source stage layout.')
    return value[...,0].reshape(count, value.shape[0]//count, value.shape[1])


def zeropower(value):
    # Same coefficients, order, precision and five iterations as optimizer_math.py.
    b.require(value.ndim == 3, 'NS receives separate stage matrices.')
    x = value.to(torch.bfloat16)
    transpose = x.size(-2) > x.size(-1)
    if transpose: x = x.mT
    x = x/(x.norm(dim=(-2,-1),keepdim=True)+1e-7)
    for _ in range(5):
        a = x@x.mT
        polynomial = -4.7750*a+2.0315*(a@a)
        x = 3.4445*x+polynomial@x
    if transpose: x = x.mT
    b.require(torch.isfinite(x).all().item(), 'Nonfinite stage-local BF16 NS update.')
    return x


def direction(gradient, momentum, count):
    momentum.lerp_(gradient, .05)
    nesterov = gradient.lerp(momentum, .95)
    matrix = stages(nesterov, count)
    if gradient.is_cuda: torch.cuda.synchronize()
    started = time.perf_counter()
    orthogonalized = zeropower(matrix)
    elapsed = time.perf_counter()-started
    result = orthogonalized.float()*(.2*math.sqrt(max(matrix.shape[-2:])))
    return result.reshape_as(gradient), elapsed


def split(active, layout):
    matrices, affine = {}, {}
    for name, value in active.items():
        if name in layout:
            b.require(list(stages(value, layout[name]['stages']).shape) == layout[name]['matrix_shape'],
                'Live matrix no longer matches the source layout: '+name)
            matrices[name] = value
        else:
            b.require(name.endswith(('.linear_in.conv.bias', '.linear_out.conv.bias', '.norm_out.weight', '.norm_out.bias',
                '.linear_in.bias','.linear_out.bias','.norm_out.norm.weight','.norm_out.norm.bias')),
                'Only explicit bias/norm leaves belong to auxiliary Adam: '+name)
            affine[name] = value
    b.require(set(matrices) == set(layout) and len(matrices)+len(affine) == len(active)
        and len({id(p) for p in active.values()}) == len(active), 'Each actual leaf must be owned exactly once.')
    return matrices, affine


class StageMuon(torch.optim.Optimizer):
    def __init__(self, matrices, layout):
        self.counts = {p:layout[name]['stages'] for name,p in matrices.items()}
        self.last_NS_seconds = 0.
        super().__init__([dict(params=list(matrices.values()),group_name='muon_matrix',lr_multiplier=1.)],
            dict(lr=1e-6,momentum=.95,nesterov=True,weight_decay=0,ns_steps=5))

    @torch.no_grad()
    def step(self):
        self.last_NS_seconds = 0.
        for group in self.param_groups:
            for p in group['params']:
                b.require(p.grad is not None and p.dtype == p.grad.dtype == torch.float32, 'Muon needs real FP32 gradients.')
                state = self.state[p]
                if not state: state.update(momentum_buffer=torch.zeros_like(p),step=0)
                update, seconds = direction(p.grad,state['momentum_buffer'],self.counts[p])
                self.last_NS_seconds += seconds
                p.add_(update,alpha=-group['lr'])
                state['step'] += 1


class MuonAdam:
    def __init__(self, active, layout):
        matrices, affine = split(active,layout)
        self.matrix_names = set(matrices)
        self.muon = StageMuon(matrices,layout)
        self.adam = torch.optim.AdamW([dict(params=list(affine.values()),group_name='aux_adam',lr_multiplier=1.)],
            lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=0)
        self.kappa = None

    @property
    def param_groups(self): return self.muon.param_groups+self.adam.param_groups

    @property
    def state(self): return {**self.muon.state,**self.adam.state}

    def zero_grad(self,set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        b.require(self.kappa is not None, 'Calibrate on the first actual clipped joint gradient before step1.')
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return dict(muon=self.muon.state_dict(),adam=self.adam.state_dict(),kappa=self.kappa)

    def load_state_dict(self,state):
        self.muon.load_state_dict(state['muon'])
        self.adam.load_state_dict(state['adam'])
        self.kappa = state['kappa']

    @torch.no_grad()
    def calibrate(self):
        b.require(self.kappa is None and not self.muon.state and not self.adam.state, 'Only a truly fresh state may calibrate.')
        adam_squares, muon_squares, ns_seconds = [], [], 0.
        for p in self.muon.param_groups[0]['params']:
            g = p.grad
            b.require(g is not None and g.dtype == torch.float32, 'Calibration needs the actual clipped matrix gradient.')
            temporary = p.detach().clone().requires_grad_()
            temporary.grad = g.detach().clone()
            candidate_optimizer = torch.optim.AdamW([temporary],lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=0)
            candidate_optimizer.step()
            state = candidate_optimizer.state[temporary]
            candidate = -1e-6*(state['exp_avg']/(1-.9))/(state['exp_avg_sq'].sqrt()/math.sqrt(1-.95)+1e-8)
            muon, seconds = direction(g,torch.zeros_like(p),self.muon.counts[p])
            ns_seconds += seconds
            adam_squares.append(float(candidate.double().square().sum()))
            muon_squares.append(float((-1e-6*muon).double().square().sum()))
        adam_norm, muon_norm = math.sqrt(math.fsum(adam_squares)),math.sqrt(math.fsum(muon_squares))
        b.require(adam_norm > 0 and muon_norm > 0 and math.isfinite(adam_norm/muon_norm), 'Fresh candidate norms must be finite/nonzero.')
        self.kappa = adam_norm/muon_norm
        self.muon.param_groups[0]['lr_multiplier'] = self.kappa
        b.require(not self.muon.state and not self.adam.state, 'Candidate calibration mutated actual optimizer state.')
        return dict(kappa=self.kappa,fresh_Adam_FP32_candidate_norm=adam_norm,baseLR_Muon_FP32_candidate_norm=muon_norm,
            matched_Muon_FP32_candidate_norm=muon_norm*self.kappa,calibration_step=1,
            NS_wall_seconds=ns_seconds,real_optimizer_still_fresh=True,
            metric='Unrounded FP32 matrix-candidate norm; not BF16 decoder or decoded-W displacement.')


def schedule(optimizer,step):
    base = 1e-6*(1-(step-1)/500)
    for group in optimizer.param_groups: group['lr'] = base*group['lr_multiplier']
    return {group['group_name']:group['lr'] for group in optimizer.param_groups}


def state_steps(active,optimizer,step):
    for name,p in active.items():
        state = optimizer.state.get(p,{})
        keys = ('momentum_buffer',) if name in optimizer.matrix_names else ('exp_avg','exp_avg_sq')
        b.require(int(state.get('step',-1)) == step and all(state[key].dtype == torch.float32
            and torch.isfinite(state[key]).all().item() for key in keys), 'Continuous optimizer state failed: '+name)
    return dict(Muon_leaves=len(optimizer.matrix_names),Adam_leaves=len(active)-len(optimizer.matrix_names),
        steps=[step],state_FP32_finite=True,kappa=optimizer.kappa)


def metadata(optimizer):
    return dict(kappa=optimizer.kappa,groups=[{key:value for key,value in group.items() if key != 'params'}
        for group in optimizer.param_groups])
