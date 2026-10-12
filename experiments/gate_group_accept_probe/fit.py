"""Fit the accepted-group gate codes with the unchanged original decoder500 recipe."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch

from experiments.gate_group_accept_probe.run import BASE, check
from experiments.gate_local_delta_probe.run import load_assets, read
from experiments.gate_student_context_probe.core import measure, ready, records, train
from experiments.gate_student_context_probe.run import paired
from experiments.joint_reconstruction_probe.core import require
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record


def prepare(output):
    require(not output.exists() and not torch.cuda.is_initialized(), 'Use a fresh CPU-only preparation.')
    torch.set_num_threads(4)
    source=BASE/'gate_group_accept_probe'
    previous=read(source/'preparation.json'); check(previous)
    report,review=read(source/'run/report.json'),read(source/'review.json')
    require(report['status']=='completed' and review['status']=='verified'
            and review['report_sha256']==sha256(source/'run/report.json')
            and (source/'run.exit').read_text().strip()=='0', 'Group acceptance must have a verified natural terminal.')
    state_path=Path(report['saved']['state_path']); mask_path=Path(report['saved']['mask_path'])
    require(sha256(state_path)==report['saved']['state_sha256']
            and sha256(mask_path)==report['saved']['mask_sha256'], 'The accepted endpoint/mask changed.')
    state=torch.load(state_path,map_location='cpu',weights_only=True,mmap=True)
    layer,_,_,_,_=load_assets(previous['paths'])
    original=records(layer)
    layer.load_state_dict(state,strict=True); ready(layer)
    expected=report['search']['final_state']
    require(records(layer)==expected and len(expected)==13
            and all(expected[k]==v for k,v in original.items() if k not in {'vq_weight','vq_weight_s1'})
            and all(p.dtype==torch.bfloat16 for p in layer._parallel_stage_decoder.parameters()),
            'Start with exactly the accepted codes and original f927 BF16 decoder values.')
    controls=read(previous['paths']['prior_report'])
    a,b=controls['arms']['original_codes'],controls['arms']['exact64_codes']
    require(a['training']['steps']==b['training']['steps']==500
            and a['training']['settings']==b['training']['settings'], 'The two historical controls need the same completed decoder500 recipe.')
    sources=dict(previous['source_sha256']);sources[str(Path(__file__).resolve())]=sha256(__file__)
    evidence=dict(previous['input_sha256'])
    for path in (source/'preparation.json',source/'run/report.json',source/'review.json',source/'run.exit',state_path,mask_path):
        evidence[str(path)]=sha256(path)
    require(not torch.cuda.is_initialized(),'CPU wiring validation initialized CUDA.')
    p={'status':'cpu_ready','output_root':str(output),'python':sys.executable,'torch':torch.__version__,
        'source_sha256':sources,'input_sha256':evidence,'protected_stats':previous['protected_stats'],
        'paths':previous['paths'],'accepted_state_path':str(state_path),'initial_state':expected,
        'before_endpoint':report['endpoint'],'expected_training_settings':a['training']['settings'],
        'expected_persistent_bits':a['training']['persistent_bits'],
        'controls':{'original_codes500':a['endpoint'],'all_weight_mse_codes500':b['endpoint']},
        'cuda_initialized':False,
        'protocol':{'steps':500,'method':'Unchanged experiments.gate_student_context_probe.core.train(steps=500)',
            'initialization':'Fresh original f927 decoder values with the saved201 accepted-group codes; no state from either decoder500 control.',
            'fixed_codes':'Both uint8 packed banks remain exactly the accepted endpoint throughout all500 updates.',
            'updates':'Only six FP32 decoder masters/FP32 Adam, native BF16 decode and MLP, original1e-5 linear500 recipe.',
            'data':'Same4289 train32 student tokens/conditional HF_MLP target; heldout and replay used only for final endpoint measurement.',
            'comparison':'Historical original-code500 and all-weight-MSE-code500 endpoints are reused, not retrained. Decoder update budgets match, but candidate adds the384-group acceptance pass on top of the original weight-neighbor search.',
            'save':'One BF16 single-layer13-tensor endpoint only; mask stays in the accepted-group source. No complete model export or downstream evaluation.',
            'wall_budget_seconds':900,'kill_grace_seconds':20,'estimate':'Existing same-scope decoder500 took about50seconds; allow a few minutes within900s.'},
        'command':[sys.executable,'-m','experiments.gate_group_accept_probe.fit','--preparation',str(output/'preparation.json')]}
    output.mkdir(parents=True);write_json(output/'preparation.json',p)
    print(json.dumps({'status':'cpu_ready','preparation_sha256':sha256(output/'preparation.json'),
        'source_pins':len(sources),'input_pins':len(evidence),'cuda_initialized':False}))


def run(path):
    p=read(path);require(p['status']=='cpu_ready','Use immutable CPU preparation.');check(p)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one externally assigned GPU is required.')
    output=Path(p['output_root'])/'run';require(not output.exists(),'No overwrite or automatic retry.');output.mkdir()
    started=time.time()
    report={'status':'running','pid':os.getpid(),'started_unix':started,
        'process_start_ticks':int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
        'preparation_sha256':sha256(path),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'protocol':p['protocol']}
    write_json(output/'report.json',report)
    try:
        from train_utils.utils import configure_deterministic_mode,set_seed
        torch.set_num_threads(4);torch.cuda.set_device(0);configure_deterministic_mode(True);set_seed(31)
        layer,archive,cache,_,_=load_assets(p['paths'])
        state=torch.load(p['accepted_state_path'],map_location='cpu',weights_only=True,mmap=True)
        layer.load_state_dict(state,strict=True);layer.to('cuda:0');ready(layer)
        require(records(layer)==p['initial_state'],'Training did not start from the real accepted endpoint.')
        training=train(layer,archive,cache,steps=500,deadline=started+p['protocol']['wall_budget_seconds'],
            observer=lambda row:logging.info('decoder500 %s',json.dumps(row)))
        require(training['initial_payload']==p['initial_state']
                and training['settings']==p['expected_training_settings']
                and training['persistent_bits']==p['expected_persistent_bits']
                and training['frozen_state_exact'],'Recipe, initialization, codes or persistent bits changed.')
        endpoint=measure(layer,archive,cache,torch.device('cuda:0'))
        state_path=output/'gate_state.pt'
        torch.save({k:v.detach().cpu() for k,v in layer.state_dict().items()},state_path)
        saved=torch.load(state_path,map_location='cpu',weights_only=True,mmap=True)
        require({k:tensor_record(v) for k,v in saved.items()}==training['final_payload'],
                'CPU saved endpoint differs from the measured native final state.')
        # The same strict reconstruction/save path was already exercised by this
        # group endpoint and both old500 controls; do not repeat a GPU reload test.
        require(all(torch.equal(saved[k],state[k]) for k in ('vq_weight','vq_weight_s1')),
                'A fixed packed bank changed in the saved endpoint.')
        report.update(training=training,endpoint=endpoint,before_endpoint=p['before_endpoint'],
            versus_accepted_initial=paired(p['before_endpoint'],endpoint),
            versus_controls={k:paired(v,endpoint) for k,v in p['controls'].items()},
            saved={'state_path':str(state_path),'state_sha256':sha256(state_path),
                'saved_state_exact':True,'fixed_banks_exact':True,
                'native_save_reload_path_evidence':'Completed accepted-group strict native reload and original same-scope decoder500 controls; not rerun here.'})
        check(p)
        report.update(status='completed',finished_unix=time.time(),elapsed_seconds=time.time()-started)
        write_json(output/'report.json',report)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',finished_unix=time.time())
        write_json(output/'report.json',report);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--preparation',type=Path)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    if args.prepare:
        require(args.output_root is not None,'--prepare needs --output-root.');prepare(args.output_root.resolve())
    else:run(args.preparation.resolve())


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');main()
