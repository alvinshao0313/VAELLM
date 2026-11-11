"""Independent actual BF16 snapshots without disturbing the live learning state."""
import gc
import random
import time

import numpy as np

from . import bindings as b


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), CPU=b.torch.get_rng_state().clone(),
        CUDA=[value.clone() for value in b.torch.cuda.get_rng_state_all()])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    b.torch.set_rng_state(state['CPU'])
    b.torch.cuda.set_rng_state_all(state['CUDA'])


def rng_exact(state):
    actual = rng()
    return actual['python'] == state['python'] and actual['numpy'][0] == state['numpy'][0] \
        and np.array_equal(actual['numpy'][1],state['numpy'][1]) and actual['numpy'][2:] == state['numpy'][2:] \
        and b.torch.equal(actual['CPU'],state['CPU']) \
        and len(actual['CUDA']) == len(state['CUDA']) \
        and all(b.torch.equal(left,right) for left,right in zip(actual['CUDA'],state['CUDA'],strict=True))


def live_stamp(active, optimizer):
    return dict(parameters={name:(id(p),p.data_ptr(),p._version,str(p.dtype)) for name,p in active.items()},
        optimizer={name:dict(identity=id(optimizer.state[p]), values={key:(value.data_ptr(),value._version,
            str(value.dtype),tuple(value.shape)) for key,value in optimizer.state[p].items() if b.torch.is_tensor(value)})
            for name,p in active.items()}, lr=optimizer.param_groups[0]['lr'])


@b.torch.no_grad()
def save(model, arm, active, optimizer, identity, step, root, non_mlp, protection, references, fixture):
    directory = root/'snapshots'/f'step_{step:06d}'
    b.require(not directory.exists(), 'Never overwrite a learning snapshot.')
    state, stamp, training = rng(), live_stamp(active,optimizer), model.training
    started = time.monotonic()
    deployment = handles = None
    try:
        directory.mkdir(parents=True)
        live_logits = b.run_short.probe(model,fixture)
        deployment, export_audit = b.export.build(model,arm,identity['source_binding'],active,non_mlp,protection)
        handles, device_map = b.life.apply_map(deployment)
        prewarm = b.core.prewarm_deployment(deployment)
        expected_logits = b.run_short.probe(deployment,fixture)
        b.export.protection_proof(deployment,protection,deployed=True)
        motion = b.life.weight_motion(deployment,references)
        input_fixture = dict(fixture, logits=expected_logits, live_logits=live_logits)
        b.torch.save(input_fixture,directory/'same_input_output.pt')
        provenance = dict(schema='all_MLP_learning_snapshot_provenance_v1', arm=arm, actual_updates=step,
            planned_updates=identity['planned_updates'], validation_only=identity['validation_only'],
            diagnostic_overbudget=arm=='dense', source_binding=identity['source_binding'],
            source_C0_binding=identity['source_C0_binding'], training_contract=b.CONTRACT,
            source_non_mlp=non_mlp, protection_records=protection, execution=identity)
        meta,budget,values = b.checkpoint.save(deployment,directory/'final_model',provenance=provenance)
        expected = {name:b.life.tensor_record(value) for name,value in values.items()}
        b.write(directory/'expected_state.json',expected)
        del values
        report = dict(schema='all_MLP_learning_snapshot_v1', status='BF16_snapshot_saved_fresh_pending',
            arm=arm, actual_updates=step, planned_updates=identity['planned_updates'],
            validation_only=identity['validation_only'], diagnostic_overbudget=arm=='dense',
            source_binding=identity['source_binding'], source_C0_binding=identity['source_C0_binding'],
            training_contract=b.CONTRACT, physical_GPUs=identity['physical_GPUs'], source_sha256=identity['source_sha256'],
            source_non_mlp=non_mlp, protection_records=protection, export_audit=export_audit,
            deployment_device_map=device_map, prewarm=prewarm, actual_BF16_W_motion=motion,
            live_to_deployed_probe=b.run_short.comparison(live_logits,expected_logits),
            export=dict(directory=str(directory/'final_model'),checkpoint_id=meta['checkpoint_id'],bit_budget=budget,
                expected_state=b.binding(directory/'expected_state.json'), fixture=b.binding(directory/'same_input_output.pt')))
    finally:
        if handles is not None:
            for handle in handles: handle.remove()
        del deployment, handles
        gc.collect()
        for device in range(2):
            with b.torch.cuda.device(device): b.torch.cuda.empty_cache()
        restore_rng(state)
    b.require(rng_exact(state) and live_stamp(active,optimizer) == stamp and model.training == training,
        'Snapshot changed Python/NumPy/CPU/CUDA RNG, live FP32 masters, Adam state, LR or training mode.')
    report.update(live_master_and_Adam_unchanged=True, CPU_CUDA_Python_NumPy_RNG_preserved=True,
        live_training_mode_preserved=True, seconds=time.monotonic()-started, memory=b.life.memory())
    b.write(directory/'report.json',report)
    return b.binding(directory/'report.json')
