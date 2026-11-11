"""CPU-bind donors, frozen codec and exact endpoint reconstruction before eval."""
import ast
import math
import os
import time
from . import bindings as b,merge
from experiments.s4_scope_objective_20261107 import state as prior_state


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not b.RUN.exists(),'Fresh CPU preparation required.')
    began=time.monotonic()
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    complete=b.read(b.prior.RUN/'completion_verification.json')
    b.require(complete['status']=='verified_complete_four_arm128_eight_endpoints' and complete['all40_stages_exit0'],
        'Completed donor batch evidence required.')
    original=b.read(b.SOURCE.parent/'expected_state.json');request=b.read(b.DONOR_ROOT/'request.json')
    prior_state.b.check_request(request);saved={};donors={};endpoint_checks={};frozen=None;names=None
    for step,spec in b.DONORS.items():
        directory=b.donor_directory(step);meta=b.read(directory/'post/model/checkpoint_meta.json')
        result=b.read(directory/'result.json');fresh=b.read(directory/'post/evaluation/fresh.json')
        b.require(meta['checkpoint_id']==result['checkpoint_id']==fresh['checkpoint_id']==spec['checkpoint_id']
            and result['paired_parent']['tasks']['mmlu']['right_correct']==spec['correct'] and fresh['full_state_exact']
            and fresh['BF16_forward_finite'],'Wrong or unverified donor.')
        payload=prior_state.check_payload(b.torch.load(directory/'training/training_state.pt',map_location='cpu',weights_only=False,mmap=True),request)
        b.require(payload['completed_updates']==step,'Wrong donor training step.')
        if names is None:names=payload['parameter_names']
        else:b.require(names==payload['parameter_names'],'Donor master order differs.')
        expected=b.read(directory/'post/expected_state.json');mutable=merge.active_keys(names)
        b.require(expected.keys()==original.keys() and len(names)==1764 and len(mutable)==1764,'Donor topology differs.')
        part={key:value for key,value in expected.items() if key not in mutable}
        b.require(part=={key:value for key,value in original.items() if key not in mutable},
            'Donor maincodes/scale/index/rotation/backbone differ from common parent.')
        if frozen is None:frozen=part
        else:b.require(part==frozen,'Donor frozen complement differs.')
        reconstructed={}
        for name,value in payload['masters'].items():
            key,hard=merge.native(name,value)
            record=b.old.old_life.tensor_record(hard)
            b.require(record==expected[key],'FP32 donor state does not reconstruct actual native donor: '+key)
            reconstructed[key]=record
        endpoint_checks[str(step)]=dict(native_mutable_fields_exact=len(reconstructed),frozen_fields_exact=len(part),
            all3927_fields_reconstructed=True)
        saved[step]=payload['masters']
        donors[str(step)]=dict(checkpoint=b.event(directory/'post/model'),state=b.binding(directory/'training/training_state.pt'),
            request=b.binding(b.DONOR_ROOT/'request.json'),result=b.binding(directory/'result.json'),
            expected_state=b.binding(directory/'post/expected_state.json'),fresh=b.binding(directory/'post/evaluation/fresh.json'),
            evaluation={part:b.binding(directory/'post/evaluation'/(part+'.json')) for part in ('mmlu','seven')})
    a,bv=saved[64],saved[128]
    b.require(list(a)==list(bv)==names,'Original FP32 master order changed.')
    differences=dict(decoder_elements=0,q_hard_elements=0)
    for name in names:
        for ratio,expected in ((0,a[name]),(1,bv[name])):
            b.require(b.torch.equal(merge.blend(a[name],bv[name],ratio),expected),'Merge endpoints are not exact.')
        key,left=merge.native(name,a[name]);_,right=merge.native(name,bv[name])
        kind='decoder_elements' if '._parallel_stage_decoder.' in name else 'q_hard_elements'
        differences[kind]+=int(b.torch.count_nonzero(left!=right))
    pins=b.pins();requests={}
    for arm,spec in b.ARMS.items():
        expected=dict(original);motion=dict(decoder_vs64=0,decoder_vs128=0,q_vs64=0,q_vs128=0)
        for name in names:
            value=merge.blend(a[name],bv[name],spec['ratio_late']);key,hard=merge.native(name,value)
            expected[key]=b.old.old_life.tensor_record(hard)
            for step in (64,128):
                _,reference=merge.native(name,saved[step][name]);kind='decoder' if '._parallel_stage_decoder.' in name else 'q'
                motion[f'{kind}_vs{step}']+=int(b.torch.count_nonzero(hard!=reference))
        req=dict(schema='S4_model_merge_request_v1',arm=arm,recipe=spec,contract=b.CONTRACT,parent=b.source(),
            donors=donors,source_sha256=pins,parameter_names=names,fixture=request['fixture'],
            expected_native_state=expected,motion=motion,frozen_complement=payload['frozen_complement'],
            training_data_used=False,inference_optimizer_state=False)
        b.check(req);requests[arm]=req
    b.require(not b.torch.cuda.is_initialized(),'CPU merge preparation initialized CUDA.')
    b.RUN.mkdir(parents=True)
    for arm,req in requests.items():b.write(b.RUN/arm/'request.json',req)
    b.write(b.RUN/'CPU_validation.json',dict(status='verified_aligned_donors_endpoints_and_three_native_merges',
        donors=donors,donor_endpoint_checks=endpoint_checks,donor_differences=differences,
        frozen_fields=3927-1764,request_bindings={a:b.binding(b.RUN/a/'request.json') for a in requests},
        three_ratios_fixed=True,CPU_only=True,no_training=True,seconds=time.monotonic()-began))
    print(dict(status='CPU_merge_prepared',run=str(b.RUN)),flush=True)


if __name__=='__main__':main()
