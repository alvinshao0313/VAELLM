"""One full BF16 GEMM with FP32 masters only for unprotected MLP columns."""
from copy import deepcopy

from . import bindings as b

torch = b.torch
nn = torch.nn
F = torch.nn.functional
SCHEMA = 'protected_MLP_free_dense_v1'
PROTECTION_FIELDS = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')


def protection_binding(module):
    fields = {name: getattr(module, name) for name in PROTECTION_FIELDS}
    return dict(sha256=b.digest(fields), fields={name:dict(shape=list(value.shape), dtype=str(value.dtype))
                                               for name, value in fields.items()})


@torch.no_grad()
def deployed_weight(module, name):
    """Read the existing deployment algorithm, including inverse rotation/protection."""
    b.validate_supported(module,name)
    b.require(all(not p.requires_grad and p.dtype == torch.bfloat16 for p in module.parameters()),
              'Materialize W0 from frozen deployed BF16 parameters, never FP32 masters: '+name)
    if isinstance(module, b.CompactRVQLinear):
        module.clear_decoded_weight_cache()
        weight = module.decode_weight()
    else:
        b.require(isinstance(module, b.VAELinear) and not module.trainable_decode
                  and module.cache_decoded_weight and module.original_weight is None
                  and module.low_rank_a is module.low_rank_b is None
                  and not module.always_use_original and module.temporary,
                  'Require the real native compressed deployment path without original weights or adapters: '+name)
        weight = module._cached_weight
        b.require(weight is not None, 'Prewarm the complete source on its actual device map with group8 before reading native W0: '+name)
    b.require(weight.shape == (module.out_features, module.in_features) and weight.dtype == torch.bfloat16
              and not weight.requires_grad and not torch.is_inference(weight) and torch.isfinite(weight).all().item(),
              'The actual original-coordinate deployed W0 must be finite ordinary BF16: '+name)
    return weight.detach().contiguous()


class FreeDenseLinear(nn.Module):
    def __init__(self, weight, protected_indices, bias, source):
        super().__init__()
        b.require(weight.ndim == 2 and weight.dtype == torch.bfloat16 and not weight.requires_grad
                  and not torch.is_inference(weight), 'Initialize only from the actual deployed ordinary BF16 W0.')
        self.out_features, self.in_features = weight.shape
        b.require(protected_indices.dtype == torch.int64 and protected_indices.ndim == 1
                  and protected_indices.numel() > 0 and protected_indices.unique().numel() == protected_indices.numel()
                  and int(protected_indices.min()) >= 0 and int(protected_indices.max()) < self.in_features,
                  'Protected input columns must be unique in-range int64 indices.')
        ids = protected_indices.detach().to(weight.device).contiguous().clone()
        keep = torch.ones(self.in_features, dtype=torch.bool, device=weight.device)
        keep[ids] = False
        unprotected = torch.nonzero(keep).flatten()
        template = torch.zeros_like(weight)
        template.index_copy_(1, ids, weight.index_select(1, ids))
        self.register_buffer('protected_indices', ids)
        self.register_buffer('unprotected_indices', unprotected)
        self.register_buffer('protection_template', template)
        self.register_buffer('bias', None if bias is None else bias.detach().to(device=weight.device, dtype=torch.bfloat16).clone())
        self.active_weight = nn.Parameter(weight.index_select(1, unprotected).float())
        self.source = deepcopy(source)
        self.validate_structure()

    @classmethod
    @torch.no_grad()
    def from_compressed(cls, module, source_binding, name):
        b.require(b.MLP_PATTERN.fullmatch(name) is not None, 'Only the actual MLP projection inventory may be replaced.')
        weight = deployed_weight(module, name)
        source = dict(schema=SCHEMA, source_checkpoint_id=source_binding['checkpoint_id'],
            source_manifest_sha256=source_binding['manifest']['sha256'], source_weight_sha256=source_binding['weight']['sha256'],
            prototype_only=source_binding['prototype_only'], module_name=name,
            consumer='compact' if isinstance(module, b.CompactRVQLinear) else 'native',
            protection=protection_binding(module), W0_sha256=b.digest({'W0':weight}),
            protected_BF16_sha256=b.digest({'columns':weight.index_select(1,module.protected_input_indices)}),
            protected_indices_sha256=b.digest({'indices':module.protected_input_indices}),
            bias_sha256=b.digest({} if module.bias is None else {'bias':module.bias}))
        result = cls(weight, module.protected_input_indices, module.bias, source)
        b.require(torch.equal(result.decode_weight(), weight), 'Initialization changed actual deployed BF16 W0: '+name)
        module.clear_decoded_weight_cache()
        return result

    def validate_structure(self):
        b.require(self.active_weight.dtype == torch.float32 and self.active_weight.requires_grad
                  and tuple(self.active_weight.shape) == (self.out_features, self.unprotected_indices.numel())
                  and self.protection_template.dtype == torch.bfloat16
                  and tuple(self.protection_template.shape) == (self.out_features, self.in_features)
                  and torch.isfinite(self.active_weight).all().item()
                  and torch.isfinite(self.protection_template).all().item(),
                  'Only the actual unprotected columns may have FP32 trainable masters.')
        b.require(self.protected_indices.dtype == self.unprotected_indices.dtype == torch.int64
                  and self.protected_indices.numel() + self.unprotected_indices.numel() == self.in_features
                  and torch.equal(torch.cat((self.protected_indices, self.unprotected_indices)).sort().values,
                                  torch.arange(self.in_features, device=self.protected_indices.device))
                  and not self.protection_template.index_select(1, self.unprotected_indices).count_nonzero().item()
                  and all(not value.requires_grad for value in self.buffers()), 'Frozen protection template/index complement changed.')
        b.require(set(dict(self.named_parameters())) == {'active_weight'}
                  and (self.bias is None or (self.bias.dtype == torch.bfloat16 and self.bias.shape == (self.out_features,))),
                  'Dense projection must own exactly one trainable Parameter and an optional fixed BF16 bias.')
        b.require(self.source['schema'] == SCHEMA
                  and b.digest({'columns':self.protection_template.index_select(1,self.protected_indices)}) == self.source['protected_BF16_sha256']
                  and b.digest({'indices':self.protected_indices}) == self.source['protected_indices_sha256']
                  and b.digest({} if self.bias is None else {'bias':self.bias}) == self.source['bias_sha256'],
                  'Persistent protected BF16 columns, indices or bias differ from the original compressed source.')

    def decode_weight(self):
        return self.protection_template.index_copy(1, self.unprotected_indices, self.active_weight.to(torch.bfloat16))

    def forward(self, inputs):
        return F.linear(inputs, self.decode_weight(), self.bias)

    def get_extra_state(self):
        return deepcopy(self.source)

    def set_extra_state(self, state):
        b.require(state == self.source, 'Free-dense state belongs to another compressed source/module/protection binding.')

    @classmethod
    def from_state(cls, state, expected_source):
        b.require(state['_extra_state'] == expected_source and expected_source['schema'] == SCHEMA,
                  'Reload requires the independently bound original source, never a claimed replacement.')
        weight = state['protection_template'].index_copy(1, state['unprotected_indices'], state['active_weight'].bfloat16())
        result = cls(weight, state['protected_indices'], state['bias'] if 'bias' in state else None, expected_source)
        loaded = result.load_state_dict(state, strict=True)
        b.require(not loaded.missing_keys and not loaded.unexpected_keys, 'Free-dense state did not reload strictly.')
        result.validate_structure()
        return result


def active_parameters(model):
    return {name+'.active_weight':module.active_weight for name,module in model.named_modules() if isinstance(module,FreeDenseLinear)}


def verify_trainable_contract(model, expected_elements=5341446144):
    active = active_parameters(model)
    actual = {name:p for name,p in model.named_parameters() if p.requires_grad}
    expected_names = {f'model.layers.{layer}.mlp.{category}.active_weight' for layer in range(36)
                      for category in ('gate_proj', 'up_proj', 'down_proj')}
    b.require(set(active) == set(actual) == expected_names and all(actual[name] is p for name,p in active.items())
              and sum(p.numel() for p in active.values()) == expected_elements
              and all(p.dtype == torch.float32 for p in active.values()), 'Exactly all108 unprotected MLP FP32 masters must train; attention/non-MLP stay frozen.')
    for module in model.modules():
        if isinstance(module,FreeDenseLinear):
            module.validate_structure()
    return active


@torch.no_grad()
def replace_all_mlp(model, source_binding):
    inventory = b.mlp_inventory(model)
    model.eval().requires_grad_(False)
    for row in inventory:
        name = row['name']
        dense = FreeDenseLinear.from_compressed(model.get_submodule(name), source_binding, name)
        b.checkpoint.native.set_module_by_name(model,name,dense)
    active = verify_trainable_contract(model)
    return active, inventory
