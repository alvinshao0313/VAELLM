"""Only the added dense-plan/source and genuine sample-reader boundaries."""
import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import subprocess
import sys

from .protocol import baseline_receipt,binding, code_identity, read, sample_evidence, write
from .evaluate import make_bf16_plan, verify_plan, authorization,child_environment
from .results import load_worker_request,extract_bsq_diagnostics
from .validate_cpu import INVENTORY,PILOT,ROOT,rejects,tiny_check


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    started=time.time();baseline_path=args.output.parent/'bf16_quality_receipt.json'
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text())
    plan=make_bf16_plan(inventory_receipt=INVENTORY,baseline_path=baseline_path,output=args.output.parent/'not_started')
    baseline,_=verify_plan(plan)
    assert plan['endpoint']['kind']=='BF16' and 'checkpoint_id' not in plan['endpoint'] and plan['authorization'] is None
    rejects(authorization,plan)
    old=ROOT/'result/qwen3_8b_mmlu65_20261004/up_vae_muon_cmcc_20261104/control/raw_full8'
    paths=[old/'mmlu.json',old/'seven.json'];samples=sample_evidence(paths)
    rejects(sample_evidence,paths[:1])
    with tempfile.TemporaryDirectory(prefix='cat_offline_binding_CPU_') as root:
        bad=deepcopy(read(INVENTORY));bad['inventory']['model_identity']['weight_shards_sha256'][next(iter(bad['inventory']['model_identity']['weight_shards_sha256']))]='0'*64
        wrong=Path(root)/'different_weights.json';write(wrong,bad)
        rejects(baseline_receipt,wrong)
        native=tiny_check(Path(root))
    live=ROOT/'artifacts/cat_up_rate_profiles_cmcc_20261104/batch_20261104_1538/k32/formal/completed.json'
    registry,request=load_worker_request(live)
    assert registry.name=='formal_request.json' and registry.parent==live.parent.parent.resolve() and request['phase']=='formal'
    rejects(load_worker_request,live,dict(path=str(registry),sha256='0'*64))
    fit=read(PILOT)['candidates'][0]['fit_report']
    assert extract_bsq_diagnostics(fit,False) is None
    rejects(extract_bsq_diagnostics,fit,True)
    inherited=os.environ.get('ACCELERATE_USE_CPU');os.environ['ACCELERATE_USE_CPU']='true'
    try:
        environment=child_environment(5)
        observed=subprocess.check_output([sys.executable,'-B','-c','import os;print(os.environ["ACCELERATE_USE_CPU"]);print(os.environ["CUDA_VISIBLE_DEVICES"])'],env=environment,text=True).splitlines()
        assert observed==['false','5']
    finally:
        if inherited is None:os.environ.pop('ACCELERATE_USE_CPU',None)
        else:os.environ['ACCELERATE_USE_CPU']=inherited
    assert not torch.cuda.is_initialized()
    value=dict(status='CPU_added_dense_source_sample_and_native_boundaries_verified',CUDA_initialized=False,elapsed_seconds=time.time()-started,
        source_sha256=code_identity(),reused_reference_worker_CPU_evidence=binding(args.output.parent/'native_reader_cpu.json'),
        BF16_source_files_rehashed=len(plan['endpoint']['source_files_sha256']),BF16_has_no_fake_checkpoint_UUID=True,
        quality_reference_verified=baseline['quality_reference_verified'],ordered_BF16_pairing_verified=False,
        genuine_existing_sample_reader=dict(leaf_count=samples['leaf_count'],sample_count=samples['sample_count'],ordered_input_identity=samples['ordered_input_identity'],
            source_files=[binding(path) for path in paths],scope='Actual existing N/mixed full8 serialized samples only; not a PUBLIC runtime/metric claim'),
        missing_shard_rejected=True,missing_GPU_authorization_rejected=True,latest_tiny_native=native,
        actual_live_registry_binding=dict(path=str(registry),sha256=binding(registry)['sha256'],source_process=binding(live.parent/'process.json')),
        wrong_registry_SHA_rejected=True,same_config_different_weights_rejected=True,
        missing_formal_BSQ_diagnostics_rejected=True,technical_BSQ_explicitly_unavailable=True,actual_child_ACCELERATE_USE_CPU_false=True,
        limitations='Actual real-model native/BF16 full8 GPU entry and positive native complete-score receipt await separately authorized pilot.')
    write(args.output,value);print({key:value[key] for key in ('status','CUDA_initialized','elapsed_seconds')},flush=True)


if __name__=='__main__':main()
