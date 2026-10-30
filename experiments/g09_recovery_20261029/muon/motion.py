"""Read existing native BF16 caches at the source and endpoint; no extra LM call."""
import math

import bindings as b


def native_weight(model,spec):
    module = model.get_submodule(spec['name'])
    weight = module._cached_weight
    b.require(module.cache_decoded_weight and not module.trainable_decode and module.original_weight is None
        and module.low_rank_a is module.low_rank_b is None and module.bias is None
        and module.weight_rotation.to_spec() == spec['weight_rotation']
        and weight is not None and weight.shape == (spec['out_features'],spec['in_features'])
        and weight.dtype == b.torch.bfloat16 and weight.is_contiguous() and not weight.requires_grad,
        'Require the actual original-coordinate native BF16 cache: '+spec['name'])
    return weight


@b.torch.no_grad()
def capture_source(model,specs):
    return {spec['name']:native_weight(model,spec).detach().cpu() for spec in specs}


@b.torch.no_grad()
def endpoint_motion(model,specs,source):
    rows = []
    for spec in specs:
        name = spec['name']
        current, initial = native_weight(model,spec),source[name]
        delta_sq, source_sq, changed = 0.,0.,0
        for start in range(0,current.shape[0],256):
            old = initial[start:start+256].to(current.device).float()
            delta = current.detach()[start:start+256].float()-old
            delta_sq += float(delta.double().square().sum())
            source_sq += float(old.double().square().sum())
            changed += int(b.torch.count_nonzero(delta))
        rows.append(dict(name=name,elements=current.numel(),changed_elements=changed,
            delta_L2=math.sqrt(delta_sq),source_L2=math.sqrt(source_sq),
            relative_L2=math.sqrt(delta_sq/source_sq) if source_sq else None))
    total_delta = math.sqrt(math.fsum(row['delta_L2']**2 for row in rows))
    total_source = math.sqrt(math.fsum(row['source_L2']**2 for row in rows))
    source.clear()
    return dict(semantics='Actual native BF16 original-coordinate W(endpoint)-W(G09); not a kappa-matched step.',
        LM_forwards_added=0,transient_CPU_reference_released=True,
        aggregate=dict(projections=len(rows),elements=sum(row['elements'] for row in rows),
            changed_elements=sum(row['changed_elements'] for row in rows),delta_L2=total_delta,
            source_L2=total_source,relative_L2=total_delta/total_source),per_projection=rows)


@b.torch.no_grad()
def parameter_motion(active,before):
    totals = {key:dict(squared=0.,changed_elements=0) for key in ('FP32','BF16_decoder')}
    for name,p in active.items():
        old = before[name].to(p.device)
        for key,delta in (('FP32',p.detach()-old),
                ('BF16_decoder',p.detach().bfloat16().float()-old.bfloat16().float())):
            totals[key]['squared'] += float(delta.double().square().sum())
            totals[key]['changed_elements'] += int(b.torch.count_nonzero(delta))
    return {key:dict(delta_L2=math.sqrt(value['squared']),changed_elements=value['changed_elements'],
        total_elements=sum(p.numel() for p in active.values())) for key,value in totals.items()}
