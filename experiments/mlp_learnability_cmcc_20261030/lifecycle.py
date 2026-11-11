"""Model-parallel ownership, immutable-state checks and bounded motion readouts."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math

from . import bindings as b
from .free_dense import FreeDenseLinear, deployed_weight, protection_binding, verify_trainable_contract
from compressed_e2e_fintuning.device_map import apply_layer_device_map, resolve_layer_device_map
from train_utils.decoder_execution import enable_vae_linear_by_execution_plan

torch = b.torch
LAYER_MAP = 'cuda:0=0-17,cuda:1=18-35'


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def tensor_record(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=b.digest({'value':value}))


def state_records(model,exclude=()):
    omitted = set(exclude)
    state = model.state_dict()
    tensors = {name:tensor_record(value) for name,value in state.items() if name not in omitted and torch.is_tensor(value)}
    extra = {name:deepcopy(value) for name,value in state.items() if name not in omitted and not torch.is_tensor(value)}
    return dict(tensors=tensors,extra=extra)


def frozen_proof(model,exclude=()):
    values = state_records(model,exclude)
    return dict(tensors=len(values['tensors']),tensor_sha256=json_sha(values['tensors']),
                extra_entries=len(values['extra']),extra_sha256=json_sha(values['extra']))


def non_mlp_proof(model):
    names = tuple(f'model.layers.{layer}.mlp.{category}.' for layer in range(36)
                  for category in ('gate_proj','up_proj','down_proj'))
    return frozen_proof(model,[name for name in model.state_dict() if name.startswith(names)])


def apply_map(model):
    # Numeric device tokens use the resolver's existing range grammar. Its
    # cuda: prefix branch accepts a single device before looking for ranges.
    mapping = resolve_layer_device_map('0=0-17,1=18-35',36)
    handles,actual = apply_layer_device_map(model,layer_device_map=mapping)
    b.require(all(str(next(model.model.layers[layer].parameters()).device) == str(mapping[layer]) for layer in range(36)),
              'All36 real layers must reside on the approved two-card split.')
    rotary = model.model.rotary_emb
    b.require(rotary.inv_freq.dtype == rotary.original_inv_freq.dtype == torch.float32
              and rotary.inv_freq.data_ptr() == rotary.original_inv_freq.data_ptr(), 'Keep the original FP32 RoPE buffer/alias contract.')
    return handles,actual


def mlp_modules(model):
    modules = {name:module for name,module in model.named_modules() if b.MLP_PATTERN.fullmatch(name)}
    b.require(len(modules) == 108, 'The selected family must cover all36 layers and108 MLP projections.')
    return modules


@torch.no_grad()
def capture_source(model):
    references,records = {},{}
    for name,module in mlp_modules(model).items():
        weight = deployed_weight(module,name).cpu()
        ids = module.protected_input_indices.detach().cpu().clone()
        references[name] = dict(weight=weight,protected_indices=ids)
        records[name] = dict(protection=protection_binding(module),protected_indices=ids.tolist(),
            protected_BF16_sha256=b.digest({'columns':weight.index_select(1,ids)}),
            W0_sha256=b.digest({'W0':weight}),shape=list(weight.shape))
    return references,records


def decoder_parameters(modules):
    values = {}
    for name,module in modules.items():
        prefix,decoder = ('decoders.',module.decoders) if isinstance(module,b.CompactRVQLinear) else ('_parallel_stage_decoder.',module._parallel_stage_decoder)
        values.update({name+'.'+prefix+relative:p for relative,p in decoder.named_parameters()})
    b.require(len(values) == len({id(p) for p in values.values()}), 'Decoder optimizer ownership overlaps.')
    return values


def training_mode(model,arm):
    if arm == 'decoder':
        model.eval().requires_grad_(False)
        modules = mlp_modules(model)
        for name,module in modules.items():
            b.validate_supported(module,name)
            if isinstance(module,b.CompactRVQLinear):
                module.enable_decoder_training()
            else:
                module.packed_vq_decoder_linear = True
                plan = enable_vae_linear_by_execution_plan(module,mode='trainable_decoder')
                b.require(plan.use_packed, 'This actual native MLP must keep its packed training path: '+name)
                module._parallel_stage_decoder.use_checkpoint = True
                for p in module._parallel_stage_decoder.parameters():
                    p.data = p.data.float()
        b.install_precision_runtime(model,torch.bfloat16)
        prewarm_attention(model)
        active = decoder_parameters(modules)
        b.require({name:p for name,p in model.named_parameters() if p.requires_grad}.keys() == active.keys(),
                  'Only the actual108 MLP decoder leaves may train.')
    else:
        active = verify_trainable_contract(model)
    b.require(all(p.dtype == torch.float32 and p.requires_grad for p in active.values()), 'All actual optimizer masters must be FP32.')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    model.config.use_cache = False
    b.require(model.is_gradient_checkpointing and model.model.gradient_checkpointing,
              'Use real full-model layer checkpointing for both arms.')
    verify_attention(model)
    return active


@torch.no_grad()
def prewarm_attention(model):
    by_device = defaultdict(list)
    for name,module in model.named_modules():
        if '.self_attn.' not in name or not isinstance(module,b.VAELinear):
            continue
        b.require(not module.trainable_decode and module.cache_decoded_weight
                  and all(not p.requires_grad and p.dtype == torch.bfloat16 for p in module.parameters()),
                  'Warm only the frozen attention after setting the final precision policy: '+name)
        by_device[str(next(module.parameters()).device)].append((name,module))
    b.require(sum(map(len,by_device.values())) == 144, 'Frozen attention warmup must cover exactly144 native modules.')
    for device,targets in by_device.items():
        result = b.prime_named_vae_linear_cache(targets,dtype=torch.bfloat16,clear_existing=True,
                                               group_size=8,compute_device=device)
        b.require(result == dict(total=len(targets),warmed=len(targets),skipped=0,failed=0), 'Frozen attention group8 warmup failed: '+device)


def verify_attention(model):
    attention = [(name,module) for name,module in model.named_modules() if '.self_attn.' in name and isinstance(module,b.VAELinear)]
    b.require(len(attention) == 144, 'All144 attention projections must remain native and frozen.')
    for name,module in attention:
        cache = module._cached_weight
        b.require(not module.trainable_decode and module.cache_decoded_weight
            and all(not p.requires_grad and p.grad is None for p in module.parameters())
            and cache is not None and cache.dtype == torch.bfloat16 and not cache.requires_grad
            and not torch.is_inference(cache), 'Frozen native attention needs an ordinary BF16 cache for hidden backward: '+name)


def synchronize():
    for device in range(2):
        torch.cuda.synchronize(device)


def memory():
    return {str(device):dict(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),
        peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device)) for device in range(2)}


def gradient_check(model,active):
    norms = defaultdict(float)
    for name,p in active.items():
        b.require(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all().item(),
                  'Missing or nonfinite actual FP32 gradient: '+name)
        projection = name.split('.mlp.')[0]+'.mlp.'+name.split('.mlp.')[1].split('.')[0]
        norms[projection] += float(torch.linalg.vector_norm(p.grad).square())
    b.require(len(norms) == 108 and all(math.isfinite(value) and value>0 for value in norms.values()),
              'Every one of108 MLP projections must receive a finite nonzero gradient.')
    b.require(all(p.grad is None for name,p in model.named_parameters() if name not in active), 'A frozen non-MLP/nondecoder Parameter received a gradient.')
    verify_attention(model)
    return {name:math.sqrt(value) for name,value in norms.items()}


def optimizer_check(optimizer,active,step):
    total = 0
    for name,p in active.items():
        state = optimizer.state[p]
        b.require(int(state['step']) == step and state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32
            and state['exp_avg'].device == state['exp_avg_sq'].device == p.device
            and torch.isfinite(state['exp_avg']).all().item() and torch.isfinite(state['exp_avg_sq']).all().item()
            and torch.isfinite(p).all().item(), 'Adam state/master failed to advance continuously: '+name)
        total += state['exp_avg'].numel()+state['exp_avg_sq'].numel()
    return dict(step=step,initialized_leaves=len(active),moment_elements=total,moment_dtype='torch.float32',
                every_state_nonempty=True,foreach=False)


@torch.no_grad()
def parameter_motion(model,active,source_weights,decoder_initial):
    rows = defaultdict(lambda:dict(fp32_delta_squared=0.,initial_squared=0.,bf16_changed_elements=0,elements=0))
    for name,p in active.items():
        projection = name.split('.mlp.')[0]+'.mlp.'+name.split('.mlp.')[1].split('.')[0]
        if isinstance(model.get_submodule(projection),FreeDenseLinear):
            module = model.get_submodule(projection)
            initial = source_weights[projection]['weight'].index_select(1,module.unprotected_indices.cpu())
        else:
            initial = decoder_initial[name]
        flat,old = p.detach().reshape(-1),initial.reshape(-1)
        row = rows[projection]
        for start in range(0,p.numel(),1<<20):
            now = flat[start:start+(1<<20)].cpu()
            reference = old[start:start+(1<<20)].float()
            row['fp32_delta_squared'] += float((now-reference).double().square().sum())
            row['initial_squared'] += float(reference.double().square().sum())
            row['bf16_changed_elements'] += int((now.bfloat16()!=reference.bfloat16()).sum())
        row['elements'] += p.numel()
    result = {name:dict(fp32_delta_L2=math.sqrt(row['fp32_delta_squared']),
        fp32_relative_L2=math.sqrt(row['fp32_delta_squared']/row['initial_squared']) if row['initial_squared'] else None,
        bf16_changed_elements=row['bf16_changed_elements'],bf16_survival_fraction=row['bf16_changed_elements']/row['elements'],
        elements=row['elements']) for name,row in rows.items()}
    b.require(sum(row['bf16_changed_elements'] for row in result.values())>0, 'No real master update survives BF16 deployment.')
    return result


@torch.no_grad()
def weight_motion(model,references):
    rows = []
    for name,old in references.items():
        module = model.get_submodule(name)
        weight = module.weight if type(module) is torch.nn.Linear else deployed_weight(module,name)
        current = weight.detach().cpu()
        ids = old['protected_indices']
        b.require(torch.equal(current.index_select(1,ids),old['weight'].index_select(1,ids)), 'Export changed protected BF16 columns: '+name)
        keep = torch.ones(current.shape[1],dtype=torch.bool)
        keep[ids] = False
        selected = torch.nonzero(keep).flatten()
        squared,base,changed = 0.,0.,0
        for start in range(0,len(current),256):
            now = current[start:start+256].index_select(1,selected).float()
            previous = old['weight'][start:start+256].index_select(1,selected).float()
            squared += float((now-previous).double().square().sum())
            base += float(previous.double().square().sum())
            changed += int((now!=previous).sum())
        elements = current.shape[0]*len(selected)
        rows.append(dict(name=name,category=name.rsplit('.',1)[1],delta_L2=math.sqrt(squared),source_L2=math.sqrt(base),
            relative_L2=math.sqrt(squared/base),changed_elements=changed,elements=elements,survival_fraction=changed/elements,
            protected_BF16_exact=True))
    b.require(len(rows) == 108 and sum(row['changed_elements'] for row in rows)>0, 'No actual deployed all-MLP BF16 W update survived.')
    return dict(per_projection=rows,per_category={category:dict(projections=36,
        relative_L2=math.sqrt(sum(row['delta_L2']**2 for row in rows if row['category']==category)/
                              sum(row['source_L2']**2 for row in rows if row['category']==category)),
        changed_elements=sum(row['changed_elements'] for row in rows if row['category']==category),
        elements=sum(row['elements'] for row in rows if row['category']==category))
        for category in ('gate_proj','up_proj','down_proj')},scope='Actual BF16 deployed original-coordinate nonprotected W; not decoder/master displacement.')
