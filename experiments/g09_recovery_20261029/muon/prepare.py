"""Actual CPU layout and optimizer checks before one new Muon GPU short2."""
import ast
from copy import deepcopy
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import bindings as b
import optimizer as opt


def state_values(active,optimizer):
    return {name:{key:b.core.tensor_record(value) if isinstance(value,b.torch.Tensor) else value
        for key,value in optimizer.state.get(p,{}).items()} for name,p in active.items()}


def optimizer_checks(state,declared,layout):
    b.torch.manual_seed(137)
    active = {name:state[name].float().clone().requires_grad_() for name in declared}
    matrices,affine = opt.split(active,layout)
    shapes = sorted({tuple(row['matrix_shape']) for row in layout.values()})
    isolation = []
    for shape in shapes:
        value = b.torch.randn(shape)
        original = opt.zeropower(value)
        changed = value.clone()
        changed[0] = b.torch.randn_like(changed[0])
        modified = opt.zeropower(changed)
        b.require(b.torch.equal(original[1:],modified[1:]), 'NS mixed independent stage matrices.')
        reordered = opt.zeropower(value.flip(0)).flip(0)
        b.require(b.torch.equal(original,reordered), 'NS failed actual stage-order independence.')
        isolation.append(dict(matrix_shape=list(shape),perturb_other_stage_exact=True,stage_reordering_exact=True))
    optimizer = opt.MuonAdam(active,layout)
    for p in active.values(): p.grad = b.torch.randn_like(p)*1e-4
    clip = float(b.torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True))
    before = {name:b.core.tensor_record(p) for name,p in active.items()}
    gradients = {name:(id(p.grad),p.grad.data_ptr(),p.grad._version,b.core.tensor_record(p.grad)) for name,p in active.items()}
    calibration = optimizer.calibrate()
    b.require(before == {name:b.core.tensor_record(p) for name,p in active.items()}
        and gradients == {name:(id(p.grad),p.grad.data_ptr(),p.grad._version,b.core.tensor_record(p.grad)) for name,p in active.items()}
        and not optimizer.state, 'Fresh calibration mutated real parameters/gradients/state.')
    rates1 = opt.schedule(optimizer,1)
    optimizer.step()
    step1 = opt.state_steps(active,optimizer,1)
    identities = {name:id(optimizer.state[p]) for name,p in active.items()}
    optimizer.zero_grad(set_to_none=True)
    for p in active.values(): p.grad = b.torch.randn_like(p)*1e-4
    b.torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True)
    rates2 = opt.schedule(optimizer,2)
    optimizer.step()
    b.require(identities == {name:id(optimizer.state[p]) for name,p in active.items()}, 'Existing state objects were replaced at update2.')
    step2 = opt.state_steps(active,optimizer,2)
    buffer = io.BytesIO()
    b.torch.save(optimizer.state_dict(),buffer)
    buffer.seek(0)
    loaded = b.torch.load(buffer,map_location='cpu',weights_only=True)
    copies = {name:p.detach().clone().requires_grad_() for name,p in active.items()}
    restored = opt.MuonAdam(copies,layout)
    restored.load_state_dict(loaded)
    b.require(state_values(copies,restored) == state_values(active,optimizer)
        and opt.metadata(restored) == opt.metadata(optimizer), 'Muon momentum/aux Adam/kappa/group LR state roundtrip failed.')
    b.require(rates1['aux_adam'] == 1e-6 and rates2['aux_adam'] == 1e-6*(1-1/500)
        and rates1['muon_matrix'] == 1e-6*optimizer.kappa
        and rates2['muon_matrix'] == rates2['aux_adam']*optimizer.kappa,
        'Both optimizer groups must follow the original500 prefix.')
    return dict(scope='CPU optimizer math on actual G09 parameter shapes with deterministic random gradients; not GPU training or reusable kappa.',
        matrix_leaves=len(matrices),auxiliary_Adam_leaves=len(affine),
        actual_matrix_stages=sum(row['stages'] for row in layout.values()),
        matrix_elements=sum(p.numel() for p in matrices.values()),
        affine_elements=sum(p.numel() for p in affine.values()),total_D_elements=sum(p.numel() for p in active.values()),
        norm_affine_kept_out_of_Muon=all('norm_out' not in name for name in matrices),
        stage_independence=isolation,CPU_only_clip_before_calibration=clip,
        CPU_only_calibration_do_not_reuse=calibration,rates_step1=rates1,rates_step2=rates2,
        state1=step1,state2=step2,state_object_identity_continuous=True,state_roundtrip_exact=True)


def main():
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not b.torch.cuda.is_initialized(), 'Prepare in activated bitvae with no CUDA devices.')
    b.require(not (b.OUTPUT / 'prepared.json').exists(), 'Do not overwrite an existing Muon preparation.')
    b.torch.set_num_threads(4)
    started = time.monotonic()
    parent,parent_record = b.direct.read_prepared()
    directory,meta,audit = b.source()
    state = b.load_state(directory,meta)
    record = deepcopy(parent_record)
    layout = opt.derive_layout(meta,state)
    declared,frozen = b.hot.state_inventory(state,record['student_projection_names'])
    b.require(declared == record['decoder_parameters'] and frozen == record['validated_frozen_state'],
        'Actual G09 source D/frozen inventory differs from verified direct OPD.')
    checked = optimizer_checks(state,declared,layout)
    b.require(checked['matrix_leaves'] == 504 and checked['actual_matrix_stages'] == 1008
        and checked['matrix_elements'] == 4552704 and checked['total_D_elements'] == 4762368,
        'The actual G09 source does not match its declared matrix inventory.')
    record['matrix_layout'] = layout
    import run
    import export
    b.require(run.runtime.fit is run.fit and Path(run.runtime.rollout.__file__).resolve() == b.HERE / 'rollout.py',
        'The reused pair loader must route through the isolated Muon lifecycle.')
    inspect.signature(export.export).bind(b.OUTPUT / 'short2/report.json',b.OUTPUT / 'short2/native_final/final_model')
    inspect.signature(b.read_prepared).bind(b.OUTPUT / 'prepared.json')
    stat = b.file_stat(directory / meta['state_dict_file'])
    b.require(stat == parent['source_weight_stat'], 'Exporter source file_stat differs from the immutable G09 source.')
    files = list(b.HERE.glob('*.py'))+list(b.HERE.glob('*.sh'))
    for path in b.HERE.glob('*.py'): ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    for path in b.HERE.glob('*.sh'): subprocess.run(['bash','-n',str(path)],check=True)
    files += [b.DIRECT/'bindings.py',b.DIRECT/'export.py',b.MATH_SOURCE,
        b.R/'llama_sparse005_adam_muon32_pair_20261025/pair_optimizer.py']
    files += [Path(item['path']) for item in parent['bindings'].values()]
    bound = {str(path.resolve()):b.binding(path) for path in files}
    result = dict(schema='G09_Muon_OPD_prepared_v1',status='CPU_verified_new_Muon_GPU_short2_pending',
        source_checkpoint=str(directory),source_checkpoint_id=b.SOURCE_ID,source_audit=b.binding(b.SOURCE_AUDIT),
        source_weight_stat=stat,source_bit_budget=audit['bit_budget'],source_geometry=parent['source_geometry'],
        direct_prepared=b.binding(b.direct.OUTPUT / 'prepared.json'),contract=b.CONTRACT,phases=b.STEPS,
        record=record,decoder_parameters=declared,dataset_binding=parent['dataset_binding'],
        optimizer_CPU_validation=checked,bindings=bound,CUDA_initialized=b.torch.cuda.is_initialized(),
        source_and_data_validation='Same actual G09/147 exact teacher backbone/once-mixed full-F M and original QA as verified direct prepared; no data selection change.',
        exporter_API_validation=dict(actual_export_signature_bound=True,read_prepared_path_argument_bound=True,
            actual_source_file_stat_exact=True),elapsed_seconds=time.monotonic()-started)
    b.require(not b.torch.cuda.is_initialized(), 'CPU validation initialized CUDA.')
    b.OUTPUT.mkdir(parents=True,exist_ok=True)
    b.core.write(b.OUTPUT / 'prepared.json',result)
    loaded,_ = b.read_prepared(b.OUTPUT / 'prepared.json')
    b.require(loaded == result, 'Actual prepared save/read/exporter-path contract failed.')
    print(json.dumps(dict(status=result['status'],matrix_layout_counts={key:checked[key] for key in
        ('matrix_leaves','actual_matrix_stages','matrix_elements','total_D_elements')},
        prepared_sha256=b.sha256(b.OUTPUT / 'prepared.json'),elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()
