"""CPU-only check of original64 payload, exact identity, and cursor panel labels."""
import ast
from collections import Counter
from copy import deepcopy
import os
import time
from . import bindings as b,core,state


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation must hide CUDA.')
    b.require(not b.RUN.exists(),'Never overwrite an existing continuation attempt.')
    began=time.monotonic()
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    original=b.read(b.PREDECESSOR/'request.json')
    b.parent.s.check_pins(original['source_sha256'])
    result=b.read(b.PREDECESSOR/'result.json');training=b.read(b.PREDECESSOR/'training/training.json')
    b.require(result['checkpoint_id']==b.PREDECESSOR_ID and training['exit_code']==0
        and training['completed_updates']==64,'Original completed attention64 endpoint required.')
    saved=b.torch.load(b.PREDECESSOR_STATE,map_location='cpu',weights_only=False)
    b.require(saved['schema']=='S4_scope_named_training_state_v1' and saved['identity']==state.identity(original)
        and saved['completed_updates']==saved['next_cursor']==saved['next_group_index']==64,
        'Original state identity/cursor changed.')
    b.require(saved['parameter_names']==list(saved['masters']) and len(saved['masters'])==864
        and all(value.dtype==b.torch.float32 and b.torch.isfinite(value).all() for value in saved['masters'].values()),
        'Original ordered FP32 masters differ.')
    b.require(len(saved['optimizer']['state'])==864 and all(int(item['step'])==64
        and item['exp_avg'].dtype==item['exp_avg_sq'].dtype==b.torch.float32
        and b.torch.isfinite(item['exp_avg']).all() and b.torch.isfinite(item['exp_avg_sq']).all()
        for item in saved['optimizer']['state'].values()),'Original complete FP32 Adam step64 required.')
    b.require(set(saved['rng'])=={'python','numpy','CPU','CUDA'} and len(saved['rng']['CUDA'])==1
        and saved['rng']['CPU'].dtype==b.torch.uint8 and saved['rng']['CUDA'][0].dtype==b.torch.uint8,
        'Original single-GPU Python/NumPy/CPU/CUDA RNG fields incomplete.')
    validation=b.read(b.prior.RUN/'validation/result.json')
    b.require(validation['status']=='verified_attention_scope_cross_process_2plus1'
        and validation['fresh']['full_state_exact'] and validation['fresh']['BF16_forward_finite']
        and validation['first_frozen_exact'] and validation['resumed_frozen_exact'],
        'Original applicable GPU2+1 recovery evidence missing.')
    dataset,targets,evidence=core.load_data()
    b.require(evidence['actual_input_target_records_sha256']==original['data_digest'],'Original data digest differs.')
    labels={str(cursor):core.exposure_labels(dataset,cursor) for cursor in (64,128)}
    b.require(labels['64']['early']['trained_by_cursor'] and not labels['64']['late']['trained_by_cursor']
        and all(labels['128'][name]['trained_by_cursor'] for name in ('early','late'))
        and not labels['128']['heldout']['trained_by_cursor'],'Cursor-based panel exposure labels failed.')
    counts=Counter(key for group in dataset['groups'] for key in group['case_ids'])
    for name in ('webinstruct','medmcqa'):
        found=[key for key in counts if key.startswith(name+'/')]
        b.require(len(found)==2048 and all(counts[key]==1 for key in found),'Cumulative2048/source once required.')
    qa=[case for case in dataset['training_cases'] if case['domain']!='mmlu']
    b.require(len(qa)==128 and all(counts[case['case_id']]==8 for case in qa),'OriginalQA128 each8 required.')
    request=deepcopy(original)
    request['continuation']=dict(contract=b.CONTRACT,source_sha256=b.pins(),
        original_request=b.binding(b.PREDECESSOR/'request.json'),predecessor_state=b.binding(b.PREDECESSOR_STATE),
        predecessor_result=b.binding(b.PREDECESSOR/'result.json'),
        original_validation=b.binding(b.prior.RUN/'validation/result.json'),
        learning_operators='Original train.build/update and state.restore; no learning-algorithm change',
        validation_reuse='Original GPU2+1 state/gradient/shape/export evidence retained unchanged; new labels checked on CPU; actual65/66 checked at startup')
    b.check_request(request)
    b.require(not b.torch.cuda.is_initialized(),'CPU preparation initialized CUDA.')
    b.RUN.mkdir(parents=True)
    b.write(b.RUN/'request.json',request)
    b.write(b.RUN/'CPU_validation.json',dict(status='verified_original64_state_and_continuation_binding',
        request=b.binding(b.RUN/'request.json'),masters=864,Adam_step=64,cursor=64,next_group=64,
        original_identity_preserved=state.identity(request)==saved['identity'],panel_labels=labels,
        new_exposure=core.data.exposure(dataset,targets,list(range(64,128))),
        cumulative_exposure=core.data.exposure(dataset,targets,list(range(128))),
        step65_lr=5e-7*(1-64/500),step128_lr=5e-7*(1-127/500),
        original_GPU_validation=request['continuation']['original_validation'],
        CUDA_initialized=False,seconds=time.monotonic()-began))
    print(dict(status='verified_original64_state_and_continuation_binding',run=str(b.RUN)),flush=True)


if __name__=='__main__':main()
