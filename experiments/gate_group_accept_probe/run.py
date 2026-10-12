"""Prepare/run one isolated fixed-order384-group native conditional-MLP search."""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch

from experiments.gate_group_accept_probe.core import BANKS, apply_choices, group_inventory, search
from experiments.gate_local_delta_probe.run import load_assets, read
from experiments.gate_student_context_probe.core import measure, ready, records
from experiments.gate_student_context_probe.run import paired
from experiments.joint_reconstruction_probe.core import require
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'result/qwen3_8b_mmlu65_20261004'


def check(p):
    check_hashes(p['source_sha256']); check_hashes(p['input_sha256'])
    for record in p['protected_stats']: check_stat(record)


def prepare(output):
    require(not output.exists() and not torch.cuda.is_initialized(), 'Use a fresh CPU-only preparation.')
    torch.set_num_threads(4)
    diagnostic=BASE/'gate_local_delta_probe_r1'
    previous=read(diagnostic/'preparation.json')
    report,review=read(diagnostic/'run/report.json'),read(diagnostic/'review.json')
    require(report['status']=='completed' and review['status']=='verified'
            and review['report_sha256']==sha256(diagnostic/'run/report.json')
            and (diagnostic/'run.exit').read_text().strip()=='0', 'Require the completed corrected native16 diagnostic.')
    check(previous)
    layer,_,_,_,_=load_assets(previous['paths'])
    choices=torch.load(previous['paths']['choices'],map_location='cpu',weights_only=True,mmap=True)
    inventory=group_inventory(choices)
    cpu_counts=[]
    for stage in range(2):
        bank=layer.get_stage_part_vq_storage(stage,0)
        trial=bank.clone()
        apply_choices(trial,choices,stage)
        expected=sum(row['stage_xor'][stage] for row in inventory)
        require(int((trial!=bank).sum())==expected, 'One saved choice must change exactly one packed byte.')
        changed=(choices>stage*32)&(choices<=(stage+1)*32)
        bits=choices[changed].long()-1-stage*32
        rows=changed.nonzero().flatten()
        require(torch.equal((trial^bank)[rows,0,bits//8],(1 << (bits%8)).to(torch.uint8)),
                'CPU group replay uses an incorrect bit order.')
        apply_choices(trial,choices,stage)
        require(torch.equal(trial,bank), 'CPU XOR undo must restore the original bank exactly.')
        cpu_counts.append(expected)
    sources=dict(previous['source_sha256']); evidence=dict(previous['input_sha256'])
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and path.suffix=='.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'result'):
                digest=sha256(path)
                require(str(path) not in sources or sources[str(path)]==digest,'An inherited source changed.')
                sources[str(path)]=digest
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path.resolve())]=sha256(path)
    for path in (diagnostic/'preparation.json',diagnostic/'run/report.json',diagnostic/'review.json',diagnostic/'run.exit'):
        evidence[str(path)]=sha256(path)
    require(not torch.cuda.is_initialized() and records(layer)==previous['initial_state'], 'CPU feasibility altered source or initialized CUDA.')
    p={'status':'cpu_ready','python':sys.executable,'torch':torch.__version__,'cuda_initialized':False,
       'output_root':str(output),'source_sha256':sources,'input_sha256':evidence,
       'protected_stats':previous['protected_stats'],'paths':previous['paths'],
       'initial_state':previous['initial_state'],'group_inventory':inventory,
       'cpu_proposed_stage_xor':cpu_counts,'cpu_packed_apply_undo_exact':True,
       'protocol':{'source_checkpoint_id':'f927ffda-dd5b-4d00-b63a-5e839a182d70','target':'model.layers.18.mlp.gate_proj',
           'groups':384,'original_outputs_per_group':32,'vectors_per_group':4032,
           'order':'Exactly J=0..383, one pass. Previous accepted groups remain in the candidate; no order search.',
           'proposal':'For each fixed J, simultaneously apply its existing saved weight-only single-bit winners to the two original banks. No newly searched bit or decoder update.',
           'acceptance':'Full native BF16 gate decode and existing Objective.prediction over all4289 train32 tokens; sum((prediction.float()-target.float()).double()**2). Accept iff strictly less than current SSE; otherwise restore both group slices. No proxy, epsilon or early stopping.',
           'frozen':'All decoder/protection/rotation/up/down values fixed; only accepted packed-code bits differ.',
           'endpoint_measurement':'After all384 decisions, existing per-document measure evaluates train/heldout/replay32 exactly once. Heldout and replay never select codes or stopping.',
           'comparison':'Current original source and completed old all-weight-MSE bank endpoint before decoder fitting, plus their retained raw metrics. No claim of a fresh heldout population.',
           'save':'Exactly one13-tensor single-layer endpoint and one mask file (384 accepted groups plus effective original choices), no whole model or optimizer.',
           'wall_budget_seconds':900,'kill_grace_seconds':20,
           'estimate':'Native16 observed4.252s total/~.07s per trial;384 trials roughly27s before overhead. Allow1-3minutes including endpoint metrics and save/reload, with900s cap; no budget extension.',
           'boundary':'Isolated new-method prototype; the greedy training loss is monotone by construction, not a guarantee of heldout, downstream or nine-task benefit.'},
       'command':[sys.executable,'-m','experiments.gate_group_accept_probe.run','--preparation',str(output/'preparation.json')]}
    output.mkdir(parents=True); write_json(output/'preparation.json',p)
    print(json.dumps({'status':'cpu_ready','preparation_sha256':sha256(output/'preparation.json'),
          'source_pins':len(sources),'input_pins':len(evidence),'groups':len(inventory),'stage_proposals':cpu_counts,'cuda_initialized':False}))


def run(path):
    p=read(path); require(p['status']=='cpu_ready','Use the immutable CPU record.'); check(p)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one externally assigned GPU is required.')
    output=Path(p['output_root'])/'run'; require(not output.exists(),'Do not overwrite or retry an existing run.'); output.mkdir()
    started=time.time()
    report={'status':'running','pid':os.getpid(),'started_unix':started,
        'process_start_ticks':int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
        'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'preparation_sha256':sha256(path),'protocol':p['protocol']}
    write_json(output/'report.json',report)
    try:
        from train_utils.utils import configure_deterministic_mode,set_seed
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(31)
        layer,archive,cache,_,_=load_assets(p['paths'])
        require(records(layer)==p['initial_state'],'Original gate values changed.')
        choices=torch.load(p['paths']['choices'],map_location='cpu',weights_only=True,mmap=True)
        require(group_inventory(choices)==p['group_inventory'],'Original group choices changed.')
        layer.to('cuda:0'); ready(layer)
        prior=read(p['paths']['prior_report'])
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            original_endpoint=measure(layer,archive,cache,torch.device('cuda:0'))
            report['original_endpoint']=original_endpoint
            report['original_vs_historical_original']=paired(prior['before_original_codes'],original_endpoint)
            report['historical_all_weight_mse_endpoint']=prior['before_exact64_codes']
            write_json(output/'report.json',report)
            masks,report['search']=search(layer,archive,cache,choices,deadline=started+p['protocol']['wall_budget_seconds'],
                observer=lambda row:logging.info('group %s',json.dumps(row)))
            # Endpoint scores are evaluated only after the accepted mask is final.
            endpoint=measure(layer,archive,cache,torch.device('cuda:0'))
            report['endpoint']=endpoint
            report['versus_original']=paired(original_endpoint,endpoint)
            report['versus_historical_all_weight_mse']=paired(prior['before_exact64_codes'],endpoint)
            state_path,mask_path=output/'gate_state.pt',output/'accepted_mask.pt'
            torch.save({k:v.detach().cpu() for k,v in layer.state_dict().items()},state_path)
            torch.save(masks,mask_path)
            saved=torch.load(state_path,map_location='cpu',weights_only=True,mmap=True)
            saved_masks=torch.load(mask_path,map_location='cpu',weights_only=True,mmap=True)
            require({k:tensor_record(v) for k,v in saved.items()}==report['search']['final_state']
                    and all(torch.equal(masks[k],saved_masks[k]) for k in masks),'Saved state or mask changed.')
            rebuilt,_,_,_,_=load_assets(p['paths'])
            rebuilt.load_state_dict(saved,strict=True); rebuilt.to('cuda:0'); ready(rebuilt)
            require(records(rebuilt)==report['search']['final_state']
                    and torch.equal(rebuilt._decode_weight(torch.bfloat16),layer._decode_weight(torch.bfloat16)),
                    'Fresh strict single-layer reload changed persistent state or native BF16 weight.')
            report['saved']={'state_path':str(state_path),'state_sha256':sha256(state_path),
                'mask_path':str(mask_path),'mask_sha256':sha256(mask_path),'strict_state_and_native_weight_exact':True}
        check(p)
        report.update(status='completed',finished_unix=time.time(),elapsed_seconds=time.time()-started)
        write_json(output/'report.json',report)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',finished_unix=time.time())
        write_json(output/'report.json',report); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--preparation',type=Path)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    if args.prepare:
        require(args.output_root is not None,'--prepare requires --output-root.'); prepare(args.output_root.resolve())
    else: run(args.preparation.resolve())


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s'); main()
