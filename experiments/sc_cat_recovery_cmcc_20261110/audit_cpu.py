"""Rehash the actual BF16 root and first-class banks; exercise finite CPU decisions."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import time

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.adapter import file_sha
from experiments.cat_auto_protocol_cmcc_20261104.nodes import NodeScore
from experiments.cat_auto_protocol_cmcc_20261104.validate_recovery_cpu import score, rejects
from . import io, search
from .identity import source_pins
from .first_batch import record
from .processes import Budget, Processes, command
from .selection import TrajectoryResult
from .settings import RecoverySettings


def decision_checks(handoff):
    state = search.new_state(handoff,'cpu_fixture')
    category = state['current_category']
    state['approved_categories']=[category]
    state['categories'][category]=dict(default_winner_raw='raw',rounds=[])
    base = RecoverySettings()
    inventory=dict(prefix=['a'],current=['a'],protected=['a'],future=['b'])
    row = search.begin_round(state,category=category,parameter='combination',base=base,raw_id='raw',inventory=inventory,
        data_identity='cpu_data',code_identity='cpu_code',completed={})
    assert row['first_calibration'] and not row['short_screen'] and len(row['candidates'])==7
    values={}
    for index,c in enumerate(row['candidates']):
        identifier=c['trajectory_id']
        values[identifier]=TrajectoryResult(category,'parent','raw','recipe','candidate',identifier,1,1,
            RecoverySettings(**c['settings']),score(.1,'raw'),
            tuple(NodeScore(s,score(.3+index/100,identifier+str(s)),identifier+str(s)) for s in (125,250,375,500)))
    chosen=search.finish_round(state,category,values)
    assert chosen.groups=='DPL' and state['short_screen']['combination']['status']=='passed'
    assert all(v['status']=='uncalibrated' for k,v in state['short_screen'].items() if k!='combination')
    scope=search.begin_round(state,category=category,parameter='scope',base=chosen,raw_id='raw',inventory=inventory,
        data_identity='cpu_data',code_identity='cpu_code',completed=values)
    assert len(scope['candidates'])==1 and scope['candidates'][0]['reuse'] and not scope['first_calibration']
    assert record(next(iter(values.values())))['status']=='complete'
    with tempfile.TemporaryDirectory() as folder:
        authorization=dict(gpus=[4,5,6,7],total_GPU_hours=1.,budget_by_component=dict(technical=.1,training=.9),
            maximum_raws=2,maximum_trajectories=1,maximum_KD_updates=500,maximum_raw_full8=2,maximum_node_full8=4)
        budget=Budget(Path(folder)/'budget.json',authorization,[],[])
        budget.reserve({'one/train':dict(component='training',seconds=900)},counts=dict(trajectories=1,KD_updates=500))
        rejects(budget.reserve,{'second/train':dict(component='training',seconds=900)},counts=dict(trajectories=1))
        processes=Processes(folder,budget,{})
        assert [processes.gpus.get_nowait() for _ in range(4)]==[4,5,6,7]
        assert 'experiments.sc_cat_recovery_cmcc_20261110.worker' in command('worker',Path(folder)/'request.json')
    return dict(default_winner_only=True,per_item_calibration_state=True,first_scope_equivalence=True,
        reserved_count_and_GPUh_bounds=True,one_shared_four_GPU_pool=True,actual_GPU_processes_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU/bitvae only.')
    started=time.monotonic()
    handoff=io.read(args.handoff)
    inventory=io.read(io.bound_path(handoff['inventory']))['inventory']
    checks=decision_checks(handoff)
    evidence=inventory['source_evidence']
    files=[]
    for name,expected in evidence['files'].items():
        path=Path(evidence['path'])/name
        require(path.stat().st_size==expected['bytes'] and file_sha(path)==expected['sha256'],'BF16 source changed: '+name)
        stat=path.stat()
        files.append(dict(path=str(path),**expected,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns))
    banks=[]
    for identifier in handoff['categories'][handoff['frozen_order'][0]]['top2']:
        candidate=io.read(io.bound_path(handoff['candidates'][identifier]['canonical']))
        bank=candidate['bank']
        require(file_sha(bank['state_file'])==bank['state_sha256'],'First-class bank changed.')
        banks.append(dict(candidate_id=identifier,path=bank['state_file'],sha256=bank['state_sha256'],
            bytes=Path(bank['state_file']).stat().st_size,complete_identity=candidate['run_identity']==handoff['run_identity']))
    require(all(b['complete_identity'] for b in banks),'Bank belongs to a different full fit identity.')
    for path in Path(__file__).parent.glob('*.py'):
        compile(path.read_text(),str(path),'exec')
    io.write(args.output,dict(status='CPU_verified',source_files=files,bank_files=banks,decision_checks=checks,
        source_pins=source_pins(),elapsed_seconds=time.monotonic()-started,GPUh=0,formal_chain_verified=False))
    print('BF16 bytes, both banks, finite decisions and all Python modules verified.',flush=True)


if __name__=='__main__':
    main()
