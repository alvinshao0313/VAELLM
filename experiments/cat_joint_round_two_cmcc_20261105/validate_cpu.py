"""Actual profile-pool rules and explicit mathematical integer-cap tests only."""
import argparse
from copy import deepcopy
from fractions import Fraction
import os
from pathlib import Path
import time

from . import driver as d
from .rules import allocate_Nc,nominations,stable,rank_actual


def refuses(function):
    try:function()
    except (ValueError,KeyError):return True
    raise AssertionError('Invalid rule input accepted.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--r1-batch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    d.require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Use caller bitvae and hidden CUDA.')
    began=time.monotonic();batch,first,coverage,inventory,pool,common=d.context(args.r1_batch.resolve())
    assert common['hard_feasible_count']==1062 and common['common_set_count']==667 and common['common_limit_bits']==17364418560
    originals=[d.bound(row['proposal']) for row in batch['arms']];cases=[]
    for arm,parent in zip(batch['arms'],originals):
        chosen=nominations(pool,parent,originals)
        assert len(chosen)<=4 and len({row['proposal_id'] for row in chosen})==len(chosen)
        for row in chosen:
            differences=[dict(stable(row))[category]-rate for category,rate in stable(parent)]
            assert sorted(value for value in differences if value)==[-Fraction(1,2),Fraction(1,2)]
            assert stable(row) not in {stable(old) for old in originals} and row['logical_bits']<=common['common_limit_bits']
        cases.append(dict(hypothetical_parent_label=arm['label'],proposals=[dict(proposal_id=row['proposal_id'],rate_vector=row['rate_vector'],
            logical_bits=row['logical_bits'],packed_bits=row['packed_bits'],signed_profile_loss_exact=str(row['signed_profile_loss'])) for row in chosen]))
    # Explicit arithmetic fixture: stable-ID remainder goes to a, although c has
    # the largest fractional remainder. These are not model results or fees.
    tiny=dict(category_ids=['c','b','a'],total_target_numel=7,projections=[dict(actual_name='a',category_id='a',numel=1),
        dict(actual_name='b',category_id='b',numel=2),dict(actual_name='c',category_id='c',numel=4)])
    ledger=dict(N=7,logical_bits=3,per_category={category:dict(N=n,logical_bits=1) for category,n in [('a',1),('b',2),('c',4)]})
    caps,evidence=allocate_Nc(tiny,ledger);assert caps=={'a':4,'b':5,'c':9} and evidence['floor_remainder_bits']==1
    bad=deepcopy(ledger);bad['N']=8;assert refuses(lambda:allocate_Nc(tiny,bad))
    bad=deepcopy(ledger);bad['logical_bits']=19;assert refuses(lambda:allocate_Nc(tiny,bad))
    terminal=all((Path(d.bound(arm['proposal'])['output_root']).parent/'controller.exit').exists() for arm in batch['arms'])
    pending=dict(status='already_terminal_reader_not_replayed_in_CPU_scope') if terminal else d.measured(batch['arms'])
    if not terminal:assert pending['status']=='pending' and pending['results']==[] and pending['winner'] is None
    import torch
    assert not torch.cuda.is_initialized()
    d.p.write(args.output,dict(exit_code=0,elapsed_seconds=time.monotonic()-began,CUDA_initialized=False,source_sha256=d.own_sources(),
        actual_profile_pool=common,mechanism_cases=cases,case_scope='Hypothetical parent vectors only; no future joint score/quality has been invented.',
        mathematical_Nc_fixture=dict(caps=caps,evidence=evidence,scope='Pure integer rule fixture; not measured model fees.'),
        actual_R1_barrier=pending,cap_or_R2_generated=False,GPU_execution=False))
    print(dict(exit_code=0,elapsed_seconds=time.monotonic()-began,CUDA_initialized=False,pending_R1=len(pending.get('pending',[]))),flush=True)


if __name__=='__main__':main()
