"""Keep the original native rollout plus Muon/group/kappa state guards."""
import bindings as b
from optimizer import metadata

original = b.module(b.OPD / 'rollout.py', '_G09_Muon_original_native_rollout')
memory, target = original.memory, original.target
teacher_stamp, move_teacher = original.teacher_stamp, original.move_teacher


def collect(model,teacher,tokenizer,cases,modules,active,optimizer,record,deadline):
    before = metadata(optimizer)
    initialized = {name:optimizer.state[p] for name,p in active.items() if optimizer.state.get(p)}
    steps = sorted({int(value['step']) for value in initialized.values()})
    b.require(len(initialized) in (0,len(active)) and len(steps) <= 1
        and (not initialized or optimizer.kappa is not None), 'Keep either fresh or complete continuous Muon/Adam state.')
    result = original.collect(model,teacher,tokenizer,cases,modules,active,optimizer,record,deadline)
    b.require(metadata(optimizer) == before, 'Native generation changed Muon kappa or either group LR/settings.')
    transition = result[2]
    transition.update(Muon_group_metadata_exact=True,Muon_fixed_kappa=optimizer.kappa,
        group_LRs={group['group_name']:group['lr'] for group in optimizer.param_groups},
        Muon_momentum_initialized_leaves=sum('momentum_buffer' in value for value in initialized.values()),
        auxiliary_Adam_initialized_leaves=sum('exp_avg' in value for value in initialized.values()),
        continuous_optimizer_steps=steps)
    return result
